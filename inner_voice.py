#!/usr/bin/env python3
"""
inner_voice.py — a second, slower stream of thought that influences the reply
without competing for it.

Two mechanisms, both live at once:

  A. TEXT   the background stream's latest thought is folded into the next
            prompt as one short line. Coarse, safe, and the model simply reads
            it. This is influence by suggestion.

  B. STATE  the background stream's hidden vector is injected into the main
            stream's residual after layer `at`:  x += g * z_background.
            Continuous and sub-symbolic -- it colours the state rather than
            adding words. `g` is the entire "not as pressing" dial.

Both share ONE 2.82 GB weight blob. A second stream costs only its own KV cache
(224 KB per token), not another copy of the model.

The background thread NEVER holds the generation lock against the user: it takes
the lock only if it is free, generates one short burst, and releases. A user
message always wins.

Nothing here is trained. The model has never seen an injected residual, so the
usable range of `g` is measured, not assumed:

    python3 inner_voice.py --sweep      # token agreement vs g (needs ~3 GB free)
"""
import argparse, json, os, sys, threading, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqsm_int8 as E
import bqsm_full_settle as FS
from bqsm_llama import Safetensors, BASE

# MEASURED, not guessed (inner_voice.py --sweep, background state norm 90.7):
#   g <= 0.02  output identical to g=0        -- influence, no override
#   g  = 0.05  33% token agreement            -- past the knee
#   g  = 0.20  17%, and the background topic ("waves") leaks into the text
#   g >= 0.50  degenerate repetition
# 0.02 is the largest gain that still agrees 100% with the uncoupled run.
DEFAULT_GAIN = float(os.environ.get("PHOX_VOICE_GAIN", "0.02"))
DEFAULT_AT = int(os.environ.get("PHOX_VOICE_LAYER", "14"))
# How long the voice stays quiet after the last sign of a human. Typing is the
# signal, not sending -- by the time a message is sent the thought has already
# stolen the core the reply needs.
IDLE_S = float(os.environ.get("PHOX_VOICE_IDLE", "8"))
# 1.5 s was effectively flat out: at ~0.45 tok/s a burst does not finish before
# the next one is due, so the voice held a core continuously for 25 h.
DEFAULT_PERIOD = float(os.environ.get("PHOX_VOICE_PERIOD", "6"))
# Greedy argmax on a free-running stream has no turn structure to end it and
# no randomness to escape a cycle, so it settles into a fixed point and repeats
# one token forever -- observed as "PhPhPhPhPh..." and "AsAsAs". The main
# generate path can afford greedy because a user prompt anchors it; a
# background monologue cannot.
VOICE_TEMP = float(os.environ.get("PHOX_VOICE_TEMP", "0.9"))
VOICE_TOPP = float(os.environ.get("PHOX_VOICE_TOPP", "0.92"))
# Cycle guard: if the last LOOP_WIN tokens hold LOOP_UNIQ or fewer distinct
# values, the stream is chewing rather than thinking.
LOOP_WIN, LOOP_UNIQ = 24, 4
# Everything the voice thinks is appended here. It used to live only in a
# 400-character RAM buffer that _restart() cleared, so 33 completed thoughts on
# Oracle were gone before anyone read them. A thought is written when it ends
# (eos / loop / context-full) AND periodically while still running, so a
# process death loses at most FLUSH_EVERY tokens rather than the whole thing.
_HOME = os.environ.get("PHOX_HOME") or os.path.expanduser("~/.phox")
VOICE_LOG = os.path.join(_HOME, "voice.jsonl")
# A pinned topic lived only in RAM, so an engine restart silently dropped it
# and the voice went back to the default SEED without saying so.
PIN_FILE = os.path.join(_HOME, "voice_pin.json")
FLUSH_EVERY = int(os.environ.get("PHOX_VOICE_FLUSH", "20"))
# PHOX_VOICE_TRACE=1 additionally logs every single token with its timing.
TRACE = os.environ.get("PHOX_VOICE_TRACE", "0") not in ("0", "", "off", "no")
SEED = ("I am a wave-interference engine. My weights are coupling strengths and "
        "my state is a settled equilibrium. What I notice right now is")


class InnerVoice:
    """A second engine on the same weights, thinking quietly in the background."""

    def __init__(self, main_engine, invf, wnorm, eraw, esh, hraw, hsh, dec,
                 gain=DEFAULT_GAIN, at=DEFAULT_AT, lock=None, max_ctx=192,
                 eos=None):
        self.eng = E.Engine(invf, blob=main_engine.blob)      # SHARED weights
        self.wnorm, self.eraw, self.esh = wnorm, eraw, esh
        self.hraw, self.hsh, self.dec = hraw, hsh, dec
        self.gain, self.at, self.max_ctx = gain, at, max_ctx
        self.lock = lock                                       # the main GEN lock
        # Without these the voice runs off the end of a turn and then feeds the
        # stop token back to itself forever. Observed on Oracle as a monologue
        # of nothing but repeated <|im_end|>.
        self.eos = set(eos or ())
        self.temp, self.top_p = VOICE_TEMP, VOICE_TOPP
        self._recent = []          # for the cycle guard
        self.loops = 0             # times a repetition loop was broken
        self.thought, self.state = "", None
        self.paused = False        # stand down while a user reply is in flight
        self.idle_s = IDLE_S
        self.last_touch = 0.0      # wall time of the last human keystroke
        self.tokens, self.running = 0, False
        self._stop = threading.Event()
        self._seed_ids = None
        self._seed_cache = None
        self.pinned = False        # an explicit redirect outranks conversation
        self._logged_at = 0        # tokens at the last periodic flush
        self._load_pin()
        self._t_start = time.time()

    def emb(self, t):
        return E.bf16_row(self.eraw, self.esh, t)

    def _log(self, reason, text=None, **extra):
        """Append one record. Never raises: losing a log line must not stop
        the stream, but nothing is dropped silently either -- a failure here
        shows up as a gap in the token counter, which is monotonic."""
        rec = {"ts": time.time(), "reason": reason,
               "text": text if text is not None else self.thought,
               "tokens": self.tokens, "restarts": getattr(self, "restarts", 0),
               "loops": self.loops, "pinned": self.pinned,
               "ctx": len(self.eng.ids),
               "uptime_s": round(time.time() - self._t_start, 1)}
        if self.state is not None:
            rec["state_norm"] = round(float(np.linalg.norm(self.state)), 3)
        if self._seed_ids:
            rec["seed"] = "".join(self.dec(t) for t in self._seed_ids[:24])
        rec.update(extra)
        try:
            os.makedirs(os.path.dirname(VOICE_LOG), exist_ok=True)
            with open(VOICE_LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def _default_seed(self):
        if self._seed_cache is None:
            try:
                from bqsm_tokenizer import Tokenizer
                self._seed_cache = Tokenizer(BASE).encode(SEED)
            except Exception:
                self._seed_cache = [128000]
        return self._seed_cache

    def seed(self, ids, now=False, pin=False, force=False, text_hint=None):
        """Point the background stream at some context.

        By default this QUEUES: _step() only reads _seed_ids when the KV is
        empty or has hit max_ctx, so a new topic waits for the current thought
        to finish -- up to 192 tokens, which at the idle rate is ~21 minutes.
        That is why a redirect appeared to be ignored and then took effect
        several minutes later.

        now=True drops the current thought and starts on the new seed at the
        next step.
        """
        # A pinned topic is not overwritten by the ordinary per-turn seeding
        # from _generate, which is what made an explicit redirect last only
        # until the next chat message.
        if self.pinned and not (pin or force):
            return
        if pin:
            self.pinned = True
            self._pin_text = text_hint or getattr(self, "_pin_text", "")
            self._save_pin(self._pin_text)
        self._seed_ids = list(ids)[-self.max_ctx:]
        if now:
            self.eng.kv = [None] * E.NL
            self.eng.ids = []
            self.thought = ""
            self._last = 128000
            self._recent = []

    def _load_pin(self):
        """Restore a pinned topic across restarts."""
        try:
            d = json.load(open(PIN_FILE))
            if d.get("pinned") and d.get("text"):
                from bqsm_tokenizer import Tokenizer
                self._seed_ids = Tokenizer(BASE).encode(d["text"])[-self.max_ctx:]
                self.pinned = True
                self._pin_text = d["text"]
        except Exception:
            pass

    def _save_pin(self, text=None):
        try:
            os.makedirs(_HOME, exist_ok=True)
            json.dump({"pinned": self.pinned, "text": text or
                       getattr(self, "_pin_text", ""), "ts": time.time()},
                      open(PIN_FILE, "w"))
        except Exception:
            pass

    def unpin(self):
        """Release a pinned topic; conversation seeding resumes."""
        self.pinned = False
        self._save_pin("")

    def touch(self):
        """A human is at the keyboard. Drop whatever we are doing."""
        self.last_touch = time.time()

    def quiet_for(self):
        """Seconds still owed to the human before thinking may resume."""
        return max(0.0, self.idle_s - (time.time() - self.last_touch))

    def _step(self):
        """One short burst, only if the main stream is idle and nobody is typing."""
        if self.paused or self.quiet_for() > 0:
            return False
        if self.lock is not None and not self.lock.acquire(blocking=False):
            return False                                       # user is talking; yield
        try:
            if self.eng.ids and len(self.eng.ids) < self.max_ctx:
                z = self.eng.settle_batch(self.emb(self._last), len(self.eng.ids),
                                          self.wnorm)
                self.eng.ids.append(self._last)
            else:
                if self.thought.strip():
                    self._log("context_full")
                # SEED was written as the default and never wired in: the
                # fallback was a bare <|begin_of_text|>, which is an
                # unconditioned sample from the prior. That is why an idle
                # monologue read like recalled training data -- "Question: Let
                # g(f) = 28*f**2 - 2*f + 1" is a maths-corpus continuation, not
                # a thought. A real sentence gives it something to continue.
                ids = self._seed_ids or self._default_seed()
                z, _, _ = self.eng.prefill(ids, self.emb, self.wnorm)
            nxt = self._sample(E.bf16_logits(self.hraw, self.hsh, z[-1]))
            self.state = z[-1:].copy()                         # what B injects
            if nxt in self.eos:
                # The thought finished. Start a new one rather than chewing on
                # the stop token; a background stream has no turn to end.
                self._log("eos")
                self._restart()
                return True
            self._recent.append(nxt)
            if len(self._recent) > LOOP_WIN:
                self._recent.pop(0)
            if (len(self._recent) == LOOP_WIN
                    and len(set(self._recent)) <= LOOP_UNIQ):
                self.loops += 1
                self._log("loop", degenerate="".join(
                    self.dec(t) for t in self._recent[-12:]))
                self._restart()
                return True
            self._last = nxt
            self.thought = (self.thought + self.dec(nxt))[-400:]
            self.tokens += 1
            if TRACE:
                self._log("token", text=self.dec(nxt), tid=int(nxt))
            elif self.tokens - self._logged_at >= FLUSH_EVERY:
                self._logged_at = self.tokens
                self._log("partial")
            return True
        finally:
            if self.lock is not None:
                self.lock.release()

    def _sample(self, lg):
        """Temperature + nucleus. Sampling is not decoration here: it is the
        only thing that lets the stream leave a basin it has fallen into."""
        if self.temp <= 0:
            return int(np.argmax(lg))
        x = lg.astype(np.float64) / max(self.temp, 1e-6)
        e = np.exp(x - x.max()); p = e / e.sum()
        if 0.0 < self.top_p < 1.0:
            order = np.argsort(p)[::-1]
            keep = np.cumsum(p[order]) <= self.top_p
            keep[0] = True
            mask = np.zeros_like(p, bool); mask[order[keep]] = True
            p = np.where(mask, p, 0.0); p /= p.sum()
        return int(np.random.choice(p.size, p=p))

    def _restart(self):
        """Drop the finished thought and begin again from the seed."""
        self.eng.kv = [None] * E.NL
        self.eng.ids = []
        self.thought = ""
        self._last = 128000
        self._recent = []
        self.restarts = getattr(self, "restarts", 0) + 1

    def _run(self, period):
        self._last = 128000
        while not self._stop.is_set():
            try:
                if not self._step():
                    time.sleep(0.25)                           # main stream busy
                    continue
            except Exception:
                self.thought = ""
                self.eng.kv = [None] * E.NL
                self.eng.ids = []
            self._stop.wait(getattr(self, "period", period))

    def start(self, period=DEFAULT_PERIOD):
        self.period = period          # settable at runtime via /self
        self.running = True
        threading.Thread(target=self._run, args=(period,), daemon=True).start()

    def stop(self):
        self._stop.set(); self.running = False

    # ── A: text influence ────────────────────────────────────────────────────
    def context_line(self):
        t = self.thought.strip().replace("\n", " ")
        return f"(a quieter thought running underneath: {t[-160:]})\n" if t else ""

    # ── B: state influence ───────────────────────────────────────────────────
    def injection(self):
        return (self.state, self.gain, self.at) if self.state is not None else (None, 0.0, self.at)


def _sweep():
    """Where does g stop colouring the output and start destroying it?"""
    st = Safetensors(BASE)
    tok = json.load(open(os.path.join(BASE, "tokenizer.json")))
    vocab = tok["model"]["vocab"]; inv = {v: k for k, v in vocab.items()}
    dec = lambda i: inv.get(i, f"[{i}]").replace("Ġ", " ").replace("Ċ", "\n")
    wnorm = st.get("model.norm.weight")
    eraw, esh = E.bf16_view(st, "model.embed_tokens.weight")
    emb = lambda t: E.bf16_row(eraw, esh, t)

    t0 = time.time()
    main = E.Engine(FS.make_invf())
    back = E.Engine(FS.make_invf(), blob=main.blob)          # shared: +0 GB
    print(f"  two engines, one 2.82 GB blob, loaded in {time.time()-t0:.1f}s\n")

    prompt = [128000] + [vocab[("Ġ" + w) if i else w]
                         for i, w in enumerate("The capital of France is".split())]
    # background stream thinks about something unrelated
    other = [128000] + [vocab[("Ġ" + w) if i else w]
                        for i, w in enumerate("Ocean waves interfere and cancel".split())]
    zb, _, _ = back.prefill(other, emb, wnorm)
    inject = zb[-1:].copy()
    print(f"  background state norm {float(np.linalg.norm(inject)):.2f}")

    base = None
    print(f"\n  {'gain':>7}{'tokens':>34}{'agree w/ g=0':>14}")
    for g in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        main.kv = [None] * E.NL; main.ids = []
        z, _, _ = main.prefill(prompt, emb, wnorm)
        out = []
        for _ in range(6):
            nxt = int(np.argmax(E.bf16_logits(eraw, esh, z[-1])))
            out.append(nxt)
            z = main.append(nxt, emb, wnorm, inject=inject, g=g, at=DEFAULT_AT)
        if base is None:
            base = out
        agree = sum(a == b for a, b in zip(out, base)) / len(base)
        txt = "".join(dec(t) for t in out)
        print(f"  {g:>7.2f}{txt[:32]!r:>34}{agree*100:>13.0f}%")
    print("\n  pick the largest g that still agrees ~100%: that is influence.")
    print("  past the knee it is not a quieter thought, it is a different model.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    if a.sweep:
        _sweep()
    else:
        print(__doc__)
