#!/usr/bin/env python3
"""
bqsm_serve_int8.py — drop-in replacement for bqsm_infer.py on port 8781,
backed by the resident int8 engine instead of the bqsm_llama path.

Same API the dashboard already speaks, so no dashboard change is needed:
    GET  /health
    POST /generate   {"prompt": ..., "n": 4096}  -> 202 {"job": id, "poll": ...}
                     generation stops on EOS; n is a runaway guard, not a cap
    GET  /jobs/<id>  -> {"state","tokens":[{"id","text"}],"text","elapsed"}

The dashboard's chat worker polls for up to 900 s because the old path ran at
~70 s/token. This one runs at ~0.7 s/token, so a 32-token reply lands in ~25 s.

    python3 bqsm_serve_int8.py --port 8781
"""
import argparse, http.server, json, os, re, sys, threading, time, uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqsm_int8 as E
import bqsm_full_settle as FS
from bqsm_llama import Safetensors, BASE

JOBS, JLOCK = {}, threading.Lock()
GEN = threading.Lock()          # one settle at a time: the engine holds one KV cache


# ── what the model may change about itself ──────────────────────────────
# Allow-list, not deny-list. Everything here is reversible, bounded, and
# logged. Deliberately ABSENT and unreachable from any tool:
#   PHOX_ALLOW_SHELL   a model that can enable its own shell is not sandboxed
#   PHOX_ROOTS         a model that can widen its own roots is not sandboxed
#   PHOX_HOME          moving state out from under the running process
# Those stay operator-only, in the environment, where the model cannot see or
# reach them.
SELF_SETTABLE = {
    # name            (lo,  hi,  cast,  note)
    "temperature":    (0.0, 2.0, float, "sampling randomness"),
    "top_p":          (0.0, 1.0, float, "nucleus cutoff"),
    "top_k":          (0,   200, int,   "0 = off"),
    "repeat_penalty": (1.0, 2.0, float, "1.0 = off"),
    "repeat_window":  (8,  1024, int,   "tokens the penalty looks back over"),
    "hvm_gain":       (0.0, 1.0, float, "associative memory mixture weight"),
    "hvm_top_k":      (1,    64, int,   "memory hits considered"),
    "hvm_sharp":      (1.0, 8.0, float, "memory distribution sharpening"),
    "voice_period":   (1.0, 120.0, float, "seconds between monologue bursts"),
    "voice_idle":     (0.0, 300.0, float, "quiet seconds after a keystroke"),
    "voice_temp":     (0.0, 2.0, float, "monologue sampling"),
    "voice_gain":     (0.0, 0.05, float, "residual injection; >0.05 rewrites output"),
}
SELF_LOG = None      # set in Backend.__init__


def clamp(name, value):
    lo, hi, cast, _ = SELF_SETTABLE[name]
    v = cast(value)
    return max(lo, min(hi, v)), (v < lo or v > hi)


def system_text():
    """The persona, from PHOX_HOME/system.md. Read fresh each call so editing
    the file takes effect on the next turn rather than needing a restart --
    it costs one stat() against a ~400 token prefix that is then cached.

    Identity is appended rather than baked into the file, so one canonical
    persona can serve several instances that differ only in who they are.
    """
    home = os.environ.get("PHOX_HOME") or os.path.expanduser("~/.phox")
    fp = os.path.join(home, "system.md")
    try:
        txt = open(fp).read().strip()
    except Exception:
        return ""
    name = os.environ.get("PHOX_NAME")
    if txt and name:
        txt += f"\n\nYou are running as {name}."
    return txt


class Backend:
    def __init__(self):
        self.st = Safetensors(BASE)
        tok = json.load(open(os.path.join(BASE, "tokenizer.json")))
        self.vocab = tok["model"]["vocab"]
        self.inv = {v: k for k, v in self.vocab.items()}
        from bqsm_tokenizer import Tokenizer          # real byte-level BPE
        self.bpe = Tokenizer(BASE)
        self.wnorm = self.st.get("model.norm.weight")
        name = ("model.embed_tokens.weight" if FS.CFG.get("tie_word_embeddings")
                else "lm_head.weight")
        self.eraw, self.esh = E.bf16_view(self.st, "model.embed_tokens.weight")
        self.hraw, self.hsh = E.bf16_view(self.st, name)
        # This is a ChatML model. generation_config lists the Llama-3 stops
        # [128001, 128008, 128009], but the token this model actually emits to
        # end a turn is <|im_end|> = 128039 -- which is exactly what config.json
        # says and what generation_config omits. Union both; a missing stop
        # token is why generation ran to the guard instead of ending.
        gp = os.path.join(BASE, "generation_config.json")
        e = json.load(open(gp))["eos_token_id"] if os.path.exists(gp) else []
        self.eos = set(e if isinstance(e, list) else [e])
        c = FS.CFG.get("eos_token_id")
        self.eos |= set(c if isinstance(c, list) else ([c] if c is not None else []))
        at = {t["content"]: t["id"] for t in tok.get("added_tokens", [])}
        self.sp = at
        for name in ("<|im_end|>", "<|end_of_text|>", "<|eot_id|>"):
            if name in at:
                self.eos.add(at[name])
        t0 = time.time()
        self.eng = E.Engine(FS.make_invf())
        self.load_s = round(time.time() - t0, 1)
        self.cyl = {"step": 0, "n_prompt": 0, "rings": [],
                    "plugins": [{"name": "adjacency", "p": [0.35], "on": True},
                                {"name": "rope-phase", "p": [], "on": True},
                                {"name": "int8-percol", "p": [], "on": True}]}
        self.settles = 0
        self.last_prefill = {}
        self.spt = None            # MEASURED sec/token (EMA of real generation); None until first token
        self.live = {}          # 8 live dimensions, refreshed every token
        # Sampling. Greedy argmax is deterministic and reproducible, which is
        # what the golden-token check needs — so it stays the default. Anything
        # non-zero here makes output non-reproducible, by design.
        self.params = {"temperature": 0.0, "top_p": 1.0, "top_k": 0,
                       "repeat_penalty": 1.0, "repeat_window": 64,
                       # Associative memory -> logits. 0.0 = off, which is the
                       # default because it is the only setting whose effect on
                       # output has been measured (none). Raise it and the
                       # golden-token check stops being meaningful.
                       # A MIXTURE weight in [0,1], not a logit gain. An
                       # additive gain cannot express "influence": the gap
                       # between the model's top-1 and the memory's pick was
                       # measured at 3.8 to 13.1 nats depending on how
                       # confident the model is, so any fixed addend either
                       # does nothing or overrides. A mixture is bounded by
                       # construction -- at 0.3 the memory owns at most 30% of
                       # the mass, whatever the logits look like.
                       "hvm_gain": float(os.environ.get("PHOX_HVM_GAIN", "0.0")),
                       "hvm_window": 24,
                       "hvm_top_k": 8, "hvm_sharp": 3.0}
        self._hvm = None
        self._hvm_tried = False
        self._hvm_err = None
        # ── plasticity: Hebbian associative memory ON THE RESIDUAL (settle→recall→learn) ──
        # Wired but OFF by default: gain 0.0 => recall injects nothing and no teaching
        # happens, so the engine is byte-identical to the un-wired one. Toggle live via
        # /plastic?gain=... (clamped to the measured-safe [0,0.05], same route as the voice).
        from plasticity import PlasticField
        self.PlasticField = PlasticField
        self.plastic = None                                              # lazy-init at first settle
        self.plastic_gain = float(os.environ.get("PHOX_PLASTIC_GAIN", "0.0"))
        self.plastic_lr = float(os.environ.get("PHOX_PLASTIC_LR", "0.3"))
        # ── taught FACTS: content-addressed recall that steers the LOGITS (not the
        # residual). A fact = (unit context-key, target token ids, text). At generation,
        # if the context matches a taught key, the output is biased toward the target —
        # bounded mixture, so it's visible and controllable, never degenerate. ──
        self.facts = []
        self.fact_keys = []            # aligned with self.facts: unit mean-embedding key per fact (ASSOCIATIVE recall)
        self.facts_path = os.path.expanduser(os.environ.get("PHOX_FACTS", "~/.phox/plastic_facts.jsonl"))
        self.fact_gain = float(os.environ.get("PHOX_FACT_GAIN", "0.9"))    # LIVE by default; inert until a fact matches
        self.fact_thresh = float(os.environ.get("PHOX_FACT_THRESH", "0.7"))  # high: only near-matching contexts trigger
        # ASSOCIATIVE (near-match) fact recall: fire a fact when the context is geometrically
        # NEAREST in embedding space, not just token-identical. Off by default (0.0) => exact-
        # match behavior byte-unchanged; opt in + tune with PHOX_FACT_ASSOC (e.g. 0.85).
        self.fact_assoc = float(os.environ.get("PHOX_FACT_ASSOC", "0.0"))
        self.last_fact_assoc = None    # measured: {score, target} of the last near-match fire (observability)
        self._rc_ids = None; self._rc_pos = 0
        # NUDGE  finish the sentence you are on, then stop.
        # GAG    stop now, mid-word if that is where you are.
        self.interrupt = None
        # ── the second stream: same weights, own KV, runs only in dead air ──
        from inner_voice import InnerVoice
        self.voice = InnerVoice(self.eng, FS.make_invf(), self.wnorm,
                                self.eraw, self.esh, self.hraw, self.hsh,
                                self.dec, lock=GEN, eos=self.eos)
        self.voice.start()      # period from PHOX_VOICE_PERIOD, default 6 s
        threading.Thread(target=self._ground_voice_loop, daemon=True).start()  # feed the voice its real state
        self._load_facts()      # restore taught facts from disk -> they survive a restart (real memory)

    def log_self_change(self, changed, rejected):
        """Every self-modification is recorded. A system that can change itself
        without a record cannot be debugged after it does."""
        try:
            home = os.environ.get("PHOX_HOME") or os.path.expanduser("~/.phox")
            os.makedirs(home, exist_ok=True)
            with open(os.path.join(home, "self_changes.jsonl"), "a") as f:
                f.write(json.dumps({"ts": time.time(), "changed": changed,
                                    "rejected": rejected}) + "\n")
        except Exception:
            pass

    def ring(self, z, tid, nosc=16):
        """Real telemetry, not decoration. The engine already treats channel
        pairs (j, j+D/2) as one complex amplitude -- that is exactly what
        rope_phase does -- so the oscillator phase is atan2(x[j+h], x[j]) of the
        settled state, and coherence is the Kuramoto order parameter |<e^{i0}>|
        over those pairs. Both are measured, neither is generated for the view."""
        v = np.asarray(z[-1], np.float64)
        h = v.size // 2
        th = np.angle(v[:h] + 1j * v[h:])
        idx = np.linspace(0, h - 1, nosc).astype(int)
        return {"t": int(tid), "lab": self.dec(tid),
                "th": [round(float(x), 4) for x in th[idx]],
                "coh": round(float(abs(np.exp(1j * th).mean())), 4)}

    def encode(self, text):
        """Real BPE. The old whitespace splitter fell back to one token per
        CHARACTER for anything not in the vocab as a whole word, so 'Hello!'
        became H-e-l-l-o-! and the model was reading spelled-out text."""
        return self.bpe.encode(text)

    def dec(self, i):
        return self.bpe.decode_one(i)

    def voice_state_readout(self):
        """Ground the inner voice's next thought in its OWN measured state AND the correct
        interpretive FRAME: it is a FRUSTRATED CRITICAL system, so health is not high coherence
        — full phase-lock (order parameter -> 1) freezes it and structureless noise dissolves it;
        it must stay on the EDGE between them. Stops the over/under-coupled false axis and points
        it at distance-from-the-edge instead. All numbers from voice_state_dict — measured, never
        invented. None before the voice has any state (falls back to the seed)."""
        d = self.voice_state_dict()
        if d is None:
            return None
        return (f"I am a frustrated critical coupled-oscillator system of {d['dim']} phase units, "
                f"coupled to {d['neighbors']} neighbors. Health is NOT high coherence: full phase-lock "
                f"(order parameter -> 1.0) freezes me, structureless noise dissolves me — I must stay "
                f"CRITICAL, on the edge, off both attractors. My measured state right now: coherence "
                f"r1={d['r1']} (lock sits at 1.0), harmonic couplings r2={d['r2']} r3={d['r3']} "
                f"r4={d['r4']}, participation ratio {d['pr']} (uniform noise sits at 1.0), amplitude "
                f"{d['amp']}; by these measures I am {d['regime']}. What I notice about my distance "
                f"from the edge is")

    def voice_state_dict(self):
        """The inner voice's OWN measured couplings + criticality, as numbers. r_p = Kuramoto
        harmonic order parameters (0 = incoherent/noise side, 1 = fully phase-LOCKED). pr =
        participation ratio of the state's energy (fraction of units carrying it: low = sparse/
        structured, 1 = uniform/spread). The two are the distances from the two death attractors —
        lock (r1->1) and structureless noise (pr->1). regime is DERIVED from them, not asserted.
        All measured from self.voice.state; midpoint (0.5) splits each axis — a coarse label over
        continuous distances, which are the real signal. Never invented."""
        st = getattr(self.voice, "state", None)
        if st is None:
            return None
        v = np.asarray(st, np.float64).ravel()
        h = v.size // 2
        if h == 0:
            return None
        th = np.angle(v[:h] + 1j * v[h:])
        r = {p: float(abs(np.exp(1j * p * th).mean())) for p in (1, 2, 3, 4)}
        e = v * v
        pr = float((e.sum() ** 2) / (v.size * (e * e).sum() + 1e-30))   # participation ratio in (0,1]
        if r[1] > 0.5:
            regime = "locked (over-coupled, frozen)"
        elif pr > 0.5:
            regime = "noise (under-structured)"
        else:
            regime = "critical (on the edge)"
        return {"dim": h, "neighbors": len(self.voice.eng.ids),
                "amp": round(float(np.linalg.norm(v)), 2),
                "r1": round(r[1], 3), "r2": round(r[2], 3),
                "r3": round(r[3], 3), "r4": round(r[4], 3),
                "pr": round(pr, 3), "regime": regime}

    def _ground_voice_loop(self):
        """Reseed the inner voice with its OWN measured state on a slow cadence, so each new
        thought starts grounded in real numbers. Queued (not now=True) so it never cuts a
        thought off mid-stream — the voice adopts it when it next restarts."""
        while True:
            time.sleep(float(os.environ.get("PHOX_GROUND_PERIOD", "25")))
            try:
                txt = self.voice_state_readout()
                if txt:
                    self.voice.seed(self.encode(txt))
            except Exception:
                pass

    def chat_ids(self, user_text, system=None, history=None):
        """ChatML, as the tokenizer_config template specifies:
        <|im_start|>system\n{persona}<|im_end|>\n
        [<|im_start|>{role}\n{text}<|im_end|>\n  for each prior turn]
        <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n

        The system turn goes FIRST and is identical every turn, so it extends
        the reusable prefix rather than breaking it -- the opposite of the
        inner-voice line, which mutated and had to be removed from the stream.

        MULTI-TURN: `history` is the prior [{role,text},...] turns, laid down in
        order between system and the current user turn. Because the whole prior
        transcript is IDENTICAL to what the last turn already settled, prefill's
        prefix reuse matches all of it in the warm KV -- so only the new user
        turn is settled, and context is held for ~free (the warm-start retention).

        Fed raw prose instead, the model does free continuation -- which has no
        natural ending, so it writes until it hits the runaway guard. The format
        is what puts it in a state where emitting a stop token is correct."""
        ims, ime = self.sp.get("<|im_start|>"), self.sp.get("<|im_end|>")
        nl = self.vocab.get("Ċ")
        if ims is None or ime is None or nl is None:
            return self.encode(user_text)
        out = [self.vocab.get("<|begin_of_text|>", 128000)]
        sys_txt = system if system is not None else system_text()
        if sys_txt:
            out += [ims] + self.encode("system")[1:] + [nl]
            out += self.encode(sys_txt)[1:] + [ime, nl]
        for turn in (history or []):        # prior turns -> real multi-turn context
            role = "assistant" if turn.get("role") == "assistant" else "user"
            txt = (turn.get("text") or "").strip()
            if not txt:
                continue
            out += [ims] + self.encode(role)[1:] + [nl]
            out += self.encode(txt)[1:] + [ime, nl]
        out += [ims] + self.encode("user")[1:] + [nl]
        out += self.encode(user_text)[1:] + [ime, nl]
        out += [ims] + self.encode("assistant")[1:] + [nl]
        return out

    def hvm(self):
        """The associative memory, loaded on first use.

        Not imported at startup: building the oscillator cache costs ~260 MB
        and a corpus burn-in, and with hvm_gain at 0 that is paid for nothing.
        A failure here is not fatal -- the engine simply generates without it.
        """
        if self._hvm is None and not self._hvm_tried:
            self._hvm_tried = True
            try:
                import hyper_vocab_memory as h
                self._hvm = h
            except Exception as e:
                self._hvm_err = f"{type(e).__name__}: {e}"
                print(f"  hvm unavailable: {self._hvm_err}", flush=True)
        return self._hvm

    def memory_bias(self, recent, n):
        """Additive logit bias from associative recall over the recent context.

        This is the fusion the HVM docstring has always described and never
        performed: `query_sparse` returns (token, score) pairs for what the
        memory associates with the current context, and those scores are added
        to the logits before sampling. Scores are normalised to unit max so the
        gain means the same thing regardless of how much has been burned in.
        """
        h = self.hvm()
        if h is None or not recent:
            return None
        try:
            ctx = "".join(self.dec(t) for t in recent[-int(
                self.params.get("hvm_window", 24)):])
            hits = h.query_sparse(ctx, top_k=int(self.params.get("hvm_top_k", 8)))
        except Exception:
            return None
        if not hits:
            return None
        # Sharpen. Spread over 32 tokens the memory's best carries only ~0.044
        # of its own mass, against a model top of 0.11-0.51, so the crossover
        # sits above mixture 0.7 -- past the point where "influence" is honest.
        # Fewer hits raised to a power concentrates it into a usable range.
        top = max(abs(s) for _, s in hits) or 1.0
        pw = float(self.params.get("hvm_sharp", 3.0))
        b = np.zeros(n, np.float32)
        for tid, sc in hits:
            if 0 <= tid < n:
                b[tid] = (max(sc, 0.0) / top) ** pw
        return b

    def pick(self, lg, recent, override=None):
        """Greedy when temperature is 0; otherwise penalise repeats, cut the
        tail with top-k/top-p, then sample. Memory bias, if any, is applied to
        the logits FIRST so it participates in every downstream decision rather
        than only in the argmax."""
        # Per-request sampling. The board ran through this path at the global
        # default of temperature 0, and greedy argmax maps a repeating context
        # to byte-identical replies -- Office-Phox produced 4 distinct messages
        # across 20 turns, two of them 18 times between them. A conversation
        # needs a decoder that can vary; the golden-token check needs one that
        # cannot. Per-request override lets both be true.
        P = dict(self.params)
        if override:
            P.update({k: v for k, v in override.items() if v is not None})
        t = float(P.get("temperature", 0.0))
        g = min(max(float(P.get("hvm_gain", 0.0)), 0.0), 1.0)
        if g > 0.0:
            b = self.memory_bias(recent, lg.shape[0])
            if b is not None and b.sum() > 0:
                # blend in probability space, then return to logits so every
                # downstream step (top-k, top-p, argmax) sees one distribution
                e = np.exp(lg.astype(np.float64) - lg.max())
                pm = e / e.sum()
                pmem = b.astype(np.float64) / b.sum()
                mix = (1.0 - g) * pm + g * pmem
                lg = np.log(np.maximum(mix, 1e-30)).astype(np.float32)
        rp = float(P.get("repeat_penalty", 1.0))
        if rp != 1.0 and recent:
            w = int(P.get("repeat_window", 64))
            for tok in set(recent[-w:]):
                lg[tok] = lg[tok] / rp if lg[tok] > 0 else lg[tok] * rp
        if t <= 0.0:
            return int(np.argmax(lg))
        x = lg.astype(np.float64) / max(t, 1e-6)
        k = int(P.get("top_k", 0))
        if k > 0 and k < x.size:
            cut = np.partition(x, -k)[-k]
            x = np.where(x < cut, -np.inf, x)
        e = np.exp(x - x.max()); p = e / e.sum()
        tp = float(P.get("top_p", 1.0))
        if 0.0 < tp < 1.0:
            order = np.argsort(p)[::-1]
            keep = np.cumsum(p[order]) <= tp
            keep[0] = True                      # never empty
            mask = np.zeros_like(p, bool); mask[order[keep]] = True
            p = np.where(mask, p, 0.0); p /= p.sum()
        return int(np.random.choice(p.size, p=p))

    def probe(self, lg, nxt, z, k=5):
        """Eight dimensions, all read off the state that already exists.

        The routes NOT taken are the interesting half: a 0.3-nat margin means
        the model nearly said something else, and that is invisible in the text.
        """
        top = np.argpartition(lg, -k)[-k:]
        top = top[np.argsort(lg[top])[::-1]]
        m = float(lg.max())
        e = np.exp((lg - m).astype(np.float64))
        p = e / e.sum()
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        v = np.asarray(z[-1], np.float64)
        h = v.size // 2
        th = np.angle(v[:h] + 1j * v[h:])
        return {
            "token":    self.dec(nxt),
            "margin":   round(float(lg[top[0]] - lg[top[1]]), 3),
            "entropy":  round(ent, 3),
            "top1_p":   round(float(p[top[0]]), 4),
            "routes":   [{"t": self.dec(int(i)), "p": round(float(p[i]), 4)} for i in top],
            "coh":      round(float(abs(np.exp(1j * th).mean())), 4),
            "amp":      round(float(np.linalg.norm(v)), 2),
            "ctx":      len(self.eng.ids),
            "tok_per_s": round(1.0 / self.spt, 2) if self.spt else 0.0,  # MEASURED (0 until first token timed)
        }

    # ── Hermes agentic framework: tools -> <tool_call> -> execute -> <tool_response> -> loop ──
    AGENT_TOOLS = [
        {"name": "add", "description": "Add two numbers and return the sum.",
         "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                        "required": ["a", "b"]}},
        {"name": "now", "description": "Return the current server date and time.",
         "parameters": {"type": "object", "properties": {}}},
        {"name": "recall", "description": "Look up a fact you were taught earlier, by query.",
         "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    ]

    def agent_system(self):
        """Hermes function-calling system prompt, built from AGENT_TOOLS."""
        tools = "\n".join(json.dumps({"type": "function", "function": t}) for t in self.AGENT_TOOLS)
        return ("You are a function calling AI model. You are provided with function signatures within "
                "<tools></tools> XML tags. You may call one or more functions to assist with the user query. "
                "Don't make assumptions about what values to plug into functions. Here are the available tools:\n"
                f"<tools>\n{tools}\n</tools>\n"
                "For each function call return a json object with function name and arguments within "
                '<tool_call></tool_call> XML tags:\n<tool_call>{"name": <function-name>, "arguments": <args-dict>}</tool_call>')

    @staticmethod
    def parse_tool_call(text):
        """Tolerant Hermes <tool_call> parser: the abliterated model sometimes drops the outer
        braces or pretty-prints, so repair before parsing, then fall back to regex extraction."""
        m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
        if not m:
            return None
        body = m.group(1).strip()
        if not body.startswith("{"):
            body = "{" + body + "}"
        try:
            o = json.loads(body)
            return {"name": o.get("name"), "arguments": o.get("arguments", {}) or {}}
        except Exception:
            nm = re.search(r'"name"\s*:\s*"([^"]+)"', body)
            if not nm:
                return None
            am = re.search(r'"arguments"\s*:\s*(\{.*\})', body, re.DOTALL)
            try:
                args = json.loads(am.group(1)) if am else {}
            except Exception:
                args = {}
            return {"name": nm.group(1), "arguments": args}

    def exec_tool(self, name, args):
        """Execute a registered tool. Pure/safe only: arithmetic, clock, taught-fact recall."""
        try:
            if name == "add":
                return (args.get("a") or 0) + (args.get("b") or 0)
            if name == "now":
                return time.strftime("%Y-%m-%d %H:%M:%S %Z")
            if name == "recall":
                q = str(args.get("query", "")).lower()
                for sids, tids, target in self.facts:
                    src = "".join(self.dec(i) for i in sids).lower()
                    if q and (q in src or src in q or any(w and w in src for w in q.split())):
                        return target
                taught = [t for _, _, t in self.facts]
                return ("(no taught fact matches; taught: " + "; ".join(taught[:5]) + ")"
                        if taught else "(nothing taught yet)")
        except Exception as ex:
            return f"(tool error: {type(ex).__name__}: {ex})"
        return f"(unknown tool: {name})"

    def agent_run(self, query, max_rounds=4, on_round=None):
        """Hermes agent loop: generate -> parse <tool_call> -> execute -> feed <tool_response> back ->
        repeat until a call-free final answer. Reuses multi-turn history + warm-start KV — each round is
        a turn, so prior rounds stay resident and only the new round settles."""
        sysp = self.agent_system()
        history, trace, prompt = [], [], query
        reply = ""
        for rnd in range(max_rounds):
            _, reply = self.generate(prompt, 512, (lambda *a: None), system=sysp, history=history)
            tc = self.parse_tool_call(reply)
            if not tc or not tc.get("name"):
                return {"answer": reply.strip(), "trace": trace, "rounds": rnd + 1}
            result = self.exec_tool(tc["name"], tc.get("arguments", {}))
            step = {"call": tc, "result": result}
            trace.append(step)
            if on_round:
                on_round(step)
            history += [{"role": "user", "text": prompt}, {"role": "assistant", "text": reply}]
            prompt = "<tool_response>\n" + json.dumps({"name": tc["name"], "content": result}) + "\n</tool_response>"
        return {"answer": reply.strip(), "trace": trace, "rounds": max_rounds, "note": "hit max rounds"}

    def generate(self, prompt, n, on_token, system=None, sampling=None, history=None):
        self.voice.paused = True          # the quieter thought yields the cores
        try:
            return self._generate(prompt, n, on_token, system, sampling, history)
        finally:
            self.voice.paused = False

    def _generate(self, prompt, n, on_token, system=None, sampling=None, history=None):
        with GEN:
            # Seed with the PLAIN text, never chat_ids. chat_ids is now ~375
            # tokens of system prompt + ChatML markers, and seed() keeps only
            # the last max_ctx of it -- which lands mid-tool-protocol with no
            # sentence to continue. Measured result: the monologue free-ran on
            # protocol text and produced 'ACTION Searching Sure /action' salad,
            # degenerating every ~26 tokens into the loop guard.
            self.voice.seed(self.encode(prompt))
            # The voice reaches the model ONLY through the residual, never the
            # token stream. Splicing its line into the prompt put a string that
            # mutates every 1.5 s at position 0, and prefix matching runs from
            # token 0 -- measured 10 of 93 positions reused, so 83 re-settles a
            # turn. Weighting cannot fix that: the cache is keyed on token
            # identity, not importance. Injection is the correct route, and it
            # is applied in append() only -- never in prefill(), so the cached
            # prefix stays a pure function of the ids and reuse stays sound.
            ids = self.chat_ids(prompt, system=system, history=history)
            emb = lambda t: E.bf16_row(self.eraw, self.esh, t)
            inject, gain, at = self.voice.injection()
            # Prefix reuse: only the tokens that differ from the cached sequence
            # are settled, and those go through in ONE batched pass.
            t0 = time.time()
            zb, reused, settled = self.eng.prefill(ids, emb, self.wnorm)
            self.settles += settled
            self.last_prefill = {"tokens": len(ids), "reused": reused,
                                 "settled": settled, "sec": round(time.time() - t0, 2)}
            z = zb[-1:]
            # ── FACT RECALL: if this context matches a taught fact, steer output to its target ──
            self._rc_ids = None; self._rc_pos = 0
            if self.facts and self.fact_gain > 0.0:
                p_ids = list(self.encode(prompt))
                for sids, tids, _ in self.facts:
                    n_ = len(sids)
                    if 0 < n_ <= len(p_ids) and tuple(p_ids[-n_:]) == sids:
                        self._rc_ids = list(tids); break      # prompt ends with a taught question
                # ── ASSOCIATIVE recall: no exact hit -> fire the fact whose taught context is
                # geometrically NEAREST to this one (embedding cosine), if above threshold.
                # Runs ONLY when the exact path found nothing, ONLY when opted in (fact_assoc>0),
                # and only ADDS the same bounded steer the exact path uses. Pure encode: the
                # query key touches no KV/cache, so live generation is never perturbed.
                if self._rc_ids is None and self.fact_assoc > 0.0 and self.fact_keys:
                    q = self._fact_key(tuple(p_ids))
                    if q is not None:
                        best_i, best_s = -1, 0.0
                        for i, kv in enumerate(self.fact_keys):
                            if kv is None:
                                continue
                            s = float(q @ kv)
                            if s > best_s:
                                best_s, best_i = s, i
                        if best_i >= 0 and best_s >= self.fact_assoc:
                            self._rc_ids = list(self.facts[best_i][1])
                            self.last_fact_assoc = {"score": round(best_s, 3),
                                                    "target": self.facts[best_i][2]}
            # ── plasticity RECALL: nudge the residual toward what this context settled to before ──
            # key is the layer-`at` hidden (aligned to the injection space), captured in prefill
            pkey = self.eng._at.copy()
            if self.plastic is None:
                self.plastic = self.PlasticField(pkey.shape[-1])
            if self.plastic_gain > 0.0 and self.plastic.n > 0:
                zr = self.plastic.recall(pkey)
                if float(np.linalg.norm(zr)) > 1e-4:
                    nudge = (self.plastic_gain * float(np.linalg.norm(pkey))) * zr
                    if inject is not None and gain != 0.0:
                        v = np.asarray(inject); v = v[-1] if v.ndim == 2 else v
                        inject, gain = (gain * v + nudge).astype(np.float32), 1.0
                    else:
                        inject, gain = nudge.astype(np.float32), 1.0
            rings = [self.ring(zb[i:i+1], ids[reused + i]) for i in range(zb.shape[0])]
            self.cyl.update(rings=list(rings[-64:]), n_prompt=len(rings[-64:]),
                            step=self.settles)
            out = []
            self.interrupt = None
            gen_t0 = time.time()                               # time the real per-token rate
            for _ in range(n):
                if self.interrupt == "gag":
                    on_token(-1, ""); break
                lg = E.bf16_logits(self.hraw, self.hsh, z[-1])
                if self._rc_ids is not None and self._rc_pos < len(self._rc_ids):
                    t_ = self._rc_ids[self._rc_pos]                    # bias toward the taught answer's next token
                    lg = lg.astype(np.float32)
                    lg[t_] = lg[t_] + self.fact_gain * (float(lg.max()) - float(lg[t_]) + 8.0)
                    self._rc_pos += 1
                nxt = self.pick(lg, ids, override=sampling)
                self.live = self.probe(lg, nxt, z)
                if nxt in self.eos:
                    break
                if self.interrupt == "nudge" and self.dec(nxt).strip() in (".", "!", "?"):
                    out.append(self.dec(nxt)); on_token(nxt, self.dec(nxt)); break
                s = self.dec(nxt)
                out.append(s); ids.append(nxt)
                on_token(nxt, s)
                z = self.eng.append(nxt, emb, self.wnorm,
                                    inject=inject, g=gain, at=at)
                self.settles += 1
                rings.append(self.ring(z, nxt))
                self.cyl.update(rings=list(rings[-64:]), step=self.settles,
                                n_prompt=min(self.cyl["n_prompt"], len(rings[-64:])))
            if out:                                            # update the MEASURED sec/token (EMA)
                spt = (time.time() - gen_t0) / len(out)
                self.spt = spt if self.spt is None else 0.6 * self.spt + 0.4 * spt
            # (auto-teach removed: binding context->own-response every turn was a
            #  self-reinforcing loop that accumulated and degraded live generation.
            #  Learning is now EXPLICIT taught facts only — stable, bounded, always-safe.)
            return ids, "".join(out)

    def _fact_key(self, sids):
        """Unit mean-embedding of the source tokens — the ASSOCIATIVE key for a fact.
        Pure embedding lookup (no prefill, no KV/cache disturbance), so both teaching and
        matching stay side-effect free. Query keys are built the same way, in the same space,
        so cosine measures how near the current context sits to a taught one."""
        if not sids:
            return None
        # bf16_row returns a 2-D (1, dim) row -> ravel to 1-D so the mean and the cosine
        # q@kv stay vectors (a 2-D key turns the dot into a shape-mismatched matmul).
        rows = [np.asarray(E.bf16_row(self.eraw, self.esh, int(t)), dtype=np.float32).ravel() for t in sids]
        k = np.mean(np.stack(rows), axis=0)
        nrm = float(np.linalg.norm(k))
        return (k / (nrm + 1e-8)).astype(np.float32) if nrm > 0.0 else None

    def plastic_learn(self, source, target, lr=1.0):
        """Teach a FACT: 'when the prompt ends with `source`, answer `target`.'
        Matched on the source's actual TOKENS — reliable, stable, zero false-triggers
        on unrelated text, inert until the taught question is really asked. Inline,
        no retrain, no cache disturbance (pure encode)."""
        sids = tuple(self.encode(source))
        self.facts.append((sids, list(self.encode(target)), target))
        self.fact_keys.append(self._fact_key(sids))          # associative key, aligned by index
        self._persist_fact(source, target)                  # durable: written to disk, survives restart
        return {"ok": True, "facts": len(self.facts), "target": target,
                "source_tokens": len(sids)}

    def _persist_fact(self, source, target):
        """Append the taught fact (source/target TEXT) to the on-disk store. Text, not
        token ids, so it survives tokenizer specifics — re-encoded on load."""
        try:
            os.makedirs(os.path.dirname(self.facts_path), exist_ok=True)
            with open(self.facts_path, "a") as f:
                f.write(json.dumps({"source": source, "target": target}) + "\n")
        except Exception:
            pass

    def _load_facts(self):
        """Restore taught facts from the on-disk store into self.facts at startup."""
        n = 0
        try:
            with open(self.facts_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    self.facts.append((tuple(self.encode(d["source"])),
                                       list(self.encode(d["target"])), d["target"]))
                    self.fact_keys.append(self._fact_key(self.facts[-1][0]))   # rebuild associative key
                    n += 1
        except FileNotFoundError:
            pass
        except Exception:
            pass
        return n


    def save_state(self, path):
        """A brain file that means something: the token sequence AND the settled
        KV cache. Reload it and the model resumes mid-thought with no re-settle
        -- you are storing the equilibrium, not a transcript.

        BQSM_PACK_STATE=1 stores the KV with the sparse+bulk packer (state_pack)
        at BQSM_PACK_TOL (default 0.001, float16-equivalent) instead of flat
        float16 -- 1.22x smaller at that fidelity, more at a looser tol, measured
        on real KV. Default off keeps the file byte-identical to before. Packed
        files self-identify, so load_state reads either format regardless of the
        flag."""
        pack = os.environ.get("BQSM_PACK_STATE", "0") == "1"
        tol = float(os.environ.get("BQSM_PACK_TOL", "0.001"))
        kv = {}
        for L, e in enumerate(self.eng.kv):
            if e is None:
                continue
            if pack:
                import state_pack as SP
                for tag, arr in (("k", e[0]), ("v", e[1])):
                    for kk, vv in SP.pack_kv(arr, tol).items():
                        kv[f"P_{tag}{L}__{kk}"] = vv
            else:
                kv[f"k{L}"], kv[f"v{L}"] = e[0].astype(np.float16), e[1].astype(np.float16)
        np.savez_compressed(path, ids=np.array(self.eng.ids, np.int32),
                            packed=np.array([1 if pack else 0], np.int8),
                            voice=np.array([self.voice.thought], object), **kv)
        return {"tokens": len(self.eng.ids), "bytes": os.path.getsize(path),
                "packed": pack, "tol": tol if pack else None}

    def load_state(self, path):
        z = np.load(path, allow_pickle=True)
        self.eng.ids = [int(t) for t in z["ids"]]
        self.eng.kv = [None] * E.NL
        packed = "packed" in z.files and int(z["packed"][0]) == 1
        if packed:
            import state_pack as SP
            for L in range(E.NL):
                if f"P_k{L}__q" not in z.files:
                    continue
                krec = {kk: z[f"P_k{L}__{kk}"] for kk in ("q", "scale", "shape")}
                vrec = {kk: z[f"P_v{L}__{kk}"] for kk in ("q", "scale", "shape")}
                self.eng.kv[L] = (SP.unpack_kv(krec).astype(np.float32),
                                  SP.unpack_kv(vrec).astype(np.float32))
        else:
            for L in range(E.NL):
                if f"k{L}" in z.files:
                    self.eng.kv[L] = (z[f"k{L}"].astype(np.float32),
                                      z[f"v{L}"].astype(np.float32))
        try:
            self.voice.thought = str(z["voice"][0])
        except Exception:
            pass
        return {"tokens": len(self.eng.ids), "packed": packed}


class Handler(http.server.BaseHTTPRequestHandler):
    backend = None

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _html(self, fname):
        """Serve a static file that ships beside the server (the showcase page)."""
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fname), "rb") as f:
                b = f.read()
        except OSError:
            return self._json({"error": f"{fname} not found"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/" or self.path == "/showcase":
            return self._html("showcase.html")
        if self.path == "/health":
            return self._json({"ok": True, "engine": "int8 resident",
                               "model": os.path.basename(BASE),
                               "weights_gb": round(self.backend.eng.blob.nbytes / 1e9, 2),
                               "load_s": self.backend.load_s,
                               "sec_per_token": (round(self.backend.spt, 3)
                                                 if self.backend.spt is not None else None)})
        if self.path.startswith("/state/save") or self.path.startswith("/state/load"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            name = q.get("name", ["session"])[0]
            path = os.path.join(os.path.expanduser("~/brains"), name + ".pbrain.npz")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                with GEN:
                    r = (self.backend.save_state(path) if "save" in self.path
                         else self.backend.load_state(path))
                return self._json({"ok": True, "path": path, **r})
            except Exception as ex:
                return self._json({"ok": False, "error": f"{type(ex).__name__}: {ex}"}, 400)
        if self.path.startswith("/params"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            for k, cast in (("temperature", float), ("top_p", float), ("top_k", int),
                            ("repeat_penalty", float), ("repeat_window", int),
                        ("hvm_gain", float), ("hvm_window", int),
                        ("hvm_top_k", int), ("hvm_sharp", float)):
                if k in q:
                    try: self.backend.params[k] = cast(q[k][0])
                    except ValueError: pass
            return self._json({"ok": True, "params": self.backend.params})
        if self.path.startswith("/interrupt"):
            from urllib.parse import urlparse, parse_qs
            a = parse_qs(urlparse(self.path).query).get("action", ["nudge"])[0]
            if a not in ("nudge", "gag", "clear"):
                return self._json({"error": "action must be nudge|gag|clear"}, 400)
            self.backend.interrupt = None if a == "clear" else a
            return self._json({"ok": True, "interrupt": self.backend.interrupt})
        if self.path.startswith("/teach"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            txt = (q.get("text") or [""])[0]
            if not txt:
                return self._json({"error": "text= required"}, 400)
            h = self.backend.hvm()
            if h is None:
                return self._json({"error": "memory unavailable"}, 503)
            st = q.get("strength")
            return self._json({"ok": True,
                               **h.burn(txt, strength=float(st[0]) if st else None)})
        if self.path.startswith("/plastic"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            b = self.backend
            if "gain" in q: b.plastic_gain = max(0.0, min(0.3, float(q["gain"][0])))  # residual route
            if "lr" in q: b.plastic_lr = float(q["lr"][0])
            if "fact_gain" in q: b.fact_gain = max(0.0, min(1.0, float(q["fact_gain"][0])))  # logit fact route
            if "fact_thresh" in q: b.fact_thresh = float(q["fact_thresh"][0])
            # DIAL: associative near-match threshold. 0.0 = off (exact-match only); ~0.85 typical.
            # Live — no restart. Env PHOX_FACT_ASSOC sets the persistent boot default.
            if "fact_assoc" in q: b.fact_assoc = max(0.0, min(1.0, float(q["fact_assoc"][0])))
            if q.get("reset"): b.plastic = None; b.facts = []; b.fact_keys = []   # in-memory only; reloads from disk on restart
            if q.get("wipe"):                                    # forget permanently: clear memory AND disk
                b.facts = []; b.fact_keys = []                   # keep the aligned key list in step
                try: open(b.facts_path, "w").close()
                except Exception: pass
            return self._json({"enabled": b.plastic_gain > 0.0, "gain": b.plastic_gain,
                               "lr": b.plastic_lr,
                               "n_taught": (b.plastic.n if b.plastic is not None else 0),
                               "dim": (b.plastic.dim if b.plastic is not None else None),
                               "fact_gain": b.fact_gain, "facts": len(b.facts),
                               "fact_thresh": b.fact_thresh,
                               "fact_assoc": b.fact_assoc,          # the dial's current value
                               "last_fact_assoc": b.last_fact_assoc})  # measured: last near-match {score,target}
        if self.path.startswith("/learn"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            src = (q.get("source") or [""])[0]; tgt = (q.get("target") or [""])[0]
            if not (src and tgt):
                return self._json({"error": "source= and target= required"}, 400)
            lr = float((q.get("lr") or ["1.0"])[0])
            return self._json(self.backend.plastic_learn(src, tgt, lr=lr))
        if self.path.startswith("/self"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            b = self.backend
            v = b.voice
            def snapshot():
                return {"temperature": b.params["temperature"],
                        "top_p": b.params["top_p"], "top_k": b.params["top_k"],
                        "repeat_penalty": b.params["repeat_penalty"],
                        "repeat_window": b.params["repeat_window"],
                        "hvm_gain": b.params["hvm_gain"],
                        "hvm_top_k": b.params["hvm_top_k"],
                        "hvm_sharp": b.params["hvm_sharp"],
                        "voice_period": getattr(v, "period", None),
                        "voice_idle": v.idle_s, "voice_temp": v.temp,
                        "voice_gain": v.gain, "pinned": v.pinned}
            if not q:
                return self._json({"settable": {k: {"min": a, "max": bb,
                                                    "note": n}
                                   for k, (a, bb, c, n) in SELF_SETTABLE.items()},
                                   "current": snapshot()})
            changed, rejected = {}, {}
            for k, vals in q.items():
                if k not in SELF_SETTABLE:
                    rejected[k] = "not settable"
                    continue
                try:
                    val, was_clamped = clamp(k, vals[0])
                except Exception:
                    rejected[k] = "bad value"
                    continue
                if k.startswith("voice_"):
                    attr = {"voice_period": "period", "voice_idle": "idle_s",
                            "voice_temp": "temp", "voice_gain": "gain"}[k]
                    setattr(v, attr, val)
                else:
                    b.params[k] = val
                changed[k] = {"value": val, "clamped": was_clamped}
            b.log_self_change(changed, rejected)
            return self._json({"ok": True, "changed": changed,
                               "rejected": rejected, "current": snapshot()})
        if self.path.startswith("/voice/pause"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            on = (q.get("on") or ["1"])[0] not in ("0", "false", "no", "off")
            self.backend.voice.paused = on          # on=1 stops the monologue, on=0 resumes
            return self._json({"ok": True, "paused": self.backend.voice.paused})
        if self.path.startswith("/voice/log"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            n = int((q.get("n") or ["20"])[0])
            reason = (q.get("reason") or [None])[0]
            import inner_voice as IV
            rows = []
            try:
                with open(IV.VOICE_LOG) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        if reason and r.get("reason") != reason:
                            continue
                        rows.append(r)
            except FileNotFoundError:
                pass
            return self._json({"path": IV.VOICE_LOG, "total": len(rows),
                               "rows": rows[-n:]})
        if self.path.startswith("/voice/think"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            about = (q.get("about") or [""])[0]
            if not about:
                return self._json({"error": "about= required"}, 400)
            v = self.backend.voice
            if about.lower() in ("-", "clear", "off"):
                v.unpin()
                return self._json({"ok": True, "pinned": False,
                                   "note": "conversation seeding resumed"})
            v.seed(self.backend.encode(about), now=True, pin=True,
                   text_hint=about)
            return self._json({"ok": True, "thinking_about": about,
                               "pinned": True, "restarted": True})
        if self.path == "/voice/touch":
            # A keystroke, not a submit. Silences the voice immediately and
            # keeps it silent until the human has been still for idle_s.
            self.backend.voice.touch()
            return self._json({"ok": True,
                               "quiet_for": round(self.backend.voice.quiet_for(), 1)})
        if self.path == "/voice":
            v = self.backend.voice
            return self._json({"running": v.running, "tokens": v.tokens,
                               "gain": v.gain, "layer": v.at,
                               "quiet_for": round(v.quiet_for(), 1),
                               "pinned": getattr(v, "pinned", False),
                               "restarts": getattr(v, "restarts", 0),
                               "loops": getattr(v, "loops", 0),
                               "temp": getattr(v, "temp", 0.0),
                               "thought": v.thought[-400:]})
        if self.path == "/cyl":
            b = self.backend
            h = b._hvm
            hv = {"loaded": h is not None, "tried": b._hvm_tried,
                  "err": getattr(b, "_hvm_err", None)}
            if h is not None:
                hv["assoc"] = len(getattr(h, "following", {}) or {})
                hv["cache"] = len(getattr(h, "cache", {}) or {})
            return self._json({"hvm": hv, "params": self.backend.params,
                               "live": self.backend.live,
                               "cyl": self.backend.cyl,
                               "voice": getattr(self.backend.voice, "thought", ""),
                               "voice_state": self.backend.voice_state_dict(),
                               "cycles": self.backend.settles,
                               "prefill": self.backend.last_prefill})
        if self.path.startswith("/jobs/"):
            with JLOCK:
                j = JOBS.get(self.path.split("/")[-1])
            return self._json(j or {"error": "no such job"}, 200 if j else 404)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/agent":
            ln = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(ln) or b"{}")
            except Exception as ex:
                return self._json({"error": f"bad json: {ex}"}, 400)
            query = req.get("prompt") or req.get("query") or ""
            jid = uuid.uuid4().hex[:8]
            job = {"id": jid, "state": "running", "query": query, "trace": [],
                   "answer": "", "started": time.time()}
            with JLOCK:
                JOBS[jid] = job

            def run_agent():
                try:
                    def on_round(step):
                        with JLOCK:
                            job["trace"] = job["trace"] + [step]
                            job["elapsed"] = round(time.time() - job["started"], 1)
                    res = self.backend.agent_run(query, on_round=on_round)
                    with JLOCK:
                        job.update(state="done", answer=res["answer"], trace=res["trace"],
                                   rounds=res.get("rounds"),
                                   elapsed=round(time.time() - job["started"], 1))
                except Exception as ex:
                    with JLOCK:
                        job.update(state="error", error=f"{type(ex).__name__}: {ex}")

            threading.Thread(target=run_agent, daemon=True).start()
            return self._json({"job": jid, "poll": f"/jobs/{jid}"}, 202)
        if self.path != "/generate":
            return self._json({"error": "not found"}, 404)
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as ex:
            return self._json({"error": f"bad json: {ex}"}, 400)
        prompt = req.get("prompt", "The capital of France is")
        # Optional per-request persona. The board needs a discussion prompt
        # WITHOUT the tool protocol: with it, each side emits ACTION lines that
        # nothing executes and the other side then invents their results.
        sys_override = req.get("system")
        hist = req.get("history")          # [{role,text},...] prior turns -> multi-turn context
        samp = {k: req.get(k) for k in
                ("temperature", "top_p", "top_k", "repeat_penalty",
                 "repeat_window", "hvm_gain")
                if req.get(k) is not None} or None
        # No product-level length cap: the EOS token is the stop condition.
        # 4096 is a runaway guard only -- at 224 KB/token that is 917 MB of KV,
        # which still fits alongside the 2.82 GB of weights.
        cnt = max(1, min(int(req.get("n", 4096)), 4096))

        jid = uuid.uuid4().hex[:8]
        job = {"id": jid, "state": "running", "prompt": prompt, "n": cnt,
               "tokens": [], "text": "", "started": time.time()}
        with JLOCK:
            JOBS[jid] = job

        def run():
            try:
                def on_tok(tid, s):
                    with JLOCK:
                        job["tokens"].append({"id": tid, "text": s})
                        job["text"] = "".join(t["text"] for t in job["tokens"])
                        job["elapsed"] = round(time.time() - job["started"], 1)
                        job["live"] = self.backend.live      # 8 live signals for the showcase
                _, text = self.backend.generate(prompt, cnt, on_tok,
                                                system=sys_override,
                                                sampling=samp, history=hist)
                with JLOCK:
                    job.update(state="done", text=text, full=prompt + text,
                               elapsed=round(time.time() - job["started"], 1))
            except Exception as ex:
                with JLOCK:
                    job.update(state="error", error=f"{type(ex).__name__}: {ex}")

        threading.Thread(target=run, daemon=True).start()
        return self._json({"job": jid, "poll": f"/jobs/{jid}",
                           "note": (f"~{self.backend.spt:.2f} s/token"
                                    if self.backend.spt is not None else "rate not yet measured")}, 202)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8781)
    ap.add_argument("--host", default="127.0.0.1")   # bind addr; pass the tailnet IP to expose across tailscale
    a = ap.parse_args()
    # Make the engine the preferred OOM victim. On the 7 GB laptop, if memory
    # still runs out, the kernel should take THIS process (restartable in ~15s)
    # rather than the desktop session it is running under. A process may raise
    # its own oom_score_adj without privilege; lowering it would need root.
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write(os.environ.get("BQSM_OOM_ADJ", "600"))
    except Exception:
        pass
    Handler.backend = Backend()
    b = Handler.backend
    print(f"  int8 engine: {b.eng.blob.nbytes/1e9:.2f} GB resident, loaded in {b.load_s}s")
    print(f"  serving on http://{a.host}:{a.port}  (/health /generate /jobs/<id>)")
    http.server.ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
