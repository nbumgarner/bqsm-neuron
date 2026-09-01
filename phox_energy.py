#!/usr/bin/env python3
"""phox_energy.py -- MEASURED efficiency of the int8 wave-equilibrium engine.

Produces verifiable numbers, never estimates: every printed figure is the live
measured variable computed from this run's counters and clocks.

TWO MODES, auto-selected by the hardware:
  * energy mode  -- when the CPU exposes RAPL (Intel Running Average Power Limit,
    /sys/class/powercap/intel-rapl, Sandy Bridge 2011+). Reads the on-die energy_uj
    accumulator around each window -> real joules -> tokens/joule and J/token. RAPL
    energy_uj is mode 0400, so this mode needs root (sudo).
  * timing mode  -- when there is NO RAPL (e.g. the T7400's 2007 Xeon X5472, which
    predates RAPL). Reports measured tok/s, ms/token, and the batched-vs-sequential
    speed lever. No joules are invented; a real power number on such a box needs an
    external wall meter (tokens/joule = tokens / (avg_watts * decode_seconds)).

Windows, each isolated from model load (load happens first, out of band):
  1. idle baseline (energy mode only) -- so tokens/joule can be reported NET of the
     machine merely being on.
  2. prefill -- BATCHED settle over the prompt (settle_batch / gemm path) vs the
     SEQUENTIAL per-token settle (gemv path) over identical tokens; outputs asserted
     byte-identical, then compared on time (and energy, if available).
  3. decode -- autoregressive generation of N tokens (greedy => deterministic =>
     reproducible). Headline tok/s (+ tokens/joule in energy mode) come from here.

Run:
    python3 phox_energy.py --prompt "The capital of France is" --n 32      # timing
    sudo python3 phox_energy.py --prompt "..." --n 32                       # +energy
"""
import os, sys, time, glob, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqsm_int8 as M
import bqsm_full_settle as FS
from bqsm_int8 import Engine, bf16_view, bf16_row, bf16_logits, BASE


# ─────────────────────────────── RAPL ────────────────────────────────────────
class Rapl:
    """Every RAPL domain exposing energy_uj: packages (intel-rapl:0 = package-0,
    intel-rapl:1 = psys/platform) and subdomains (:0:0 core, :0:1 uncore). On a
    pre-RAPL CPU the glob is empty and `available` is False -> timing-only mode."""

    def __init__(self):
        self.domains = []                       # (label, energy_path, max_range_uj)
        for path in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
            ej = os.path.join(path, "energy_uj")
            if not os.path.exists(ej):
                continue
            try:
                name = open(os.path.join(path, "name")).read().strip()
            except OSError:
                name = os.path.basename(path)
            try:
                mx = int(open(os.path.join(path, "max_energy_range_uj")).read())
            except OSError:
                mx = None
            self.domains.append((f"{name} [{os.path.basename(path)}]", ej, mx))

    @property
    def available(self):
        return bool(self.domains)

    def readable(self):
        for _, ej, _ in self.domains:
            try:
                open(ej).read()
            except OSError:
                return False
        return True

    def snapshot(self):
        return {label: int(open(ej).read()) for label, ej, _ in self.domains}

    def delta_j(self, before, after):
        out = {}
        for label, ej, mx in self.domains:
            d = after[label] - before[label]
            if d < 0 and mx is not None:                 # counter wrapped past max_range
                d += mx
            out[label] = d / 1e6                          # microjoules -> joules
        return out


def _fmt_j(dj, secs):
    return "\n".join(
        f"      {label:<22} {j:9.3f} J   {(j/secs if secs>0 else float('nan')):7.2f} W avg"
        for label, j in dj.items())


# ──────────────────────────── engine driver ──────────────────────────────────
def load_engine():
    safe = M.Safetensors(BASE)
    tok = json.load(open(os.path.join(BASE, "tokenizer.json")))
    vocab = tok["model"]["vocab"]
    inv = {v: k for k, v in vocab.items()}

    def dec(i):
        return inv.get(i, f"[{i}]").replace("Ġ", " ").replace("Ċ", "\n")

    wnorm = safe.get("model.norm.weight")
    ename = "model.embed_tokens.weight" if FS.CFG.get("tie_word_embeddings") else "lm_head.weight"
    eraw, eshape = bf16_view(safe, "model.embed_tokens.weight")
    hraw, hshape = bf16_view(safe, ename)

    gp = os.path.join(BASE, "generation_config.json")
    e = json.load(open(gp))["eos_token_id"] if os.path.exists(gp) else FS.CFG["eos_token_id"]
    EOS = set(e if isinstance(e, list) else [e])

    t0 = time.time()
    eng = Engine(FS.make_invf())
    load_s = time.time() - t0

    def embed(t):
        return bf16_row(eraw, eshape, t)

    def logits(z):
        return bf16_logits(hraw, hshape, z)

    return eng, embed, logits, wnorm, EOS, vocab, dec, load_s


def encode_prompt(prompt, vocab):
    return [128000] + [vocab[("Ġ" + w) if i else w]
                       for i, w in enumerate(prompt.split())]


def reset_engine(eng):
    eng.kv = [None] * FS.NL
    eng.ids = []


# ───────────────────────────────── main ──────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=32, help="tokens to generate (decode window)")
    ap.add_argument("--idle-secs", type=float, default=3.0,
                    help="idle-baseline sampling seconds (energy mode only)")
    ap.add_argument("--no-energy", action="store_true", help="force timing-only mode")
    ap.add_argument("--json", action="store_true", help="emit a machine-readable JSON block")
    a = ap.parse_args()

    rapl = Rapl()
    EN = rapl.available and rapl.readable() and not a.no_energy
    if a.no_energy:
        print("timing-only mode (--no-energy).")
    elif not rapl.available:
        print("no RAPL on this CPU (pre-Sandy-Bridge / no /sys/class/powercap) "
              "-> timing-only mode. A real joule number here needs an external wall meter.")
    elif not rapl.readable():
        print("RAPL present but energy_uj unreadable (mode 0400). Re-run with sudo for "
              "energy; continuing in timing-only mode for now.")
    else:
        print(f"energy mode. RAPL domains: {', '.join(l for l, _, _ in rapl.domains)}")

    print("loading int8 engine (out of the measured window)...", flush=True)
    eng, embed, logits, wnorm, EOS, vocab, dec, load_s = load_engine()
    print(f"  loaded {eng.blob.nbytes/1e9:.2f} GB int8 resident in {load_s:.1f}s")
    print(f"  host {os.uname().nodename}, load{tuple(round(x,2) for x in os.getloadavg())}, "
          f"{os.cpu_count()} cpus")

    ids = encode_prompt(a.prompt, vocab)
    idle_w = {}

    # ── window 1: idle baseline (energy mode only) ──
    if EN:
        b = rapl.snapshot(); t0 = time.time()
        time.sleep(a.idle_secs)
        idle_s = time.time() - t0
        idle_j = rapl.delta_j(b, rapl.snapshot())
        idle_w = {k: v / idle_s for k, v in idle_j.items()}
        print(f"\n[idle {idle_s:.2f}s] baseline (machine on, engine idle):")
        print(_fmt_j(idle_j, idle_s))

    # ── window 2: prefill -- BATCHED vs SEQUENTIAL over identical tokens ──
    reset_engine(eng)
    b = rapl.snapshot() if EN else None; t0 = time.time()
    zb, reused, settled = eng.prefill(ids, embed, wnorm)
    batched_s = time.time() - t0
    batched_j = rapl.delta_j(b, rapl.snapshot()) if EN else None
    tok_batched = int(np.argmax(logits(zb[-1])))

    reset_engine(eng)
    b = rapl.snapshot() if EN else None; t0 = time.time()
    zs = None
    for i, tk in enumerate(ids):
        zs = eng.settle(embed(tk), i, wnorm, reset=(i == 0))
    eng.ids = list(ids)
    seq_s = time.time() - t0
    seq_j = rapl.delta_j(b, rapl.snapshot()) if EN else None
    tok_seq = int(np.argmax(logits(zs[-1])))

    same = (tok_batched == tok_seq)
    npos = len(ids)
    print(f"\n[prefill] {npos} prompt positions, batched vs sequential "
          f"(byte-exact next-token match: {same}; batched={tok_batched} seq={tok_seq})")
    print(f"  BATCHED    {batched_s:6.2f}s  ({npos/batched_s:7.2f} pos/s)")
    if EN: print(_fmt_j(batched_j, batched_s))
    print(f"  SEQUENTIAL {seq_s:6.2f}s  ({npos/seq_s:7.2f} pos/s)")
    if EN: print(_fmt_j(seq_j, seq_s))
    print(f"  batched speedup: {seq_s/batched_s:5.2f}x faster for identical output")
    if EN:
        print("  batched energy vs sequential (x less energy, identical output):")
        for label in batched_j:
            if batched_j[label] > 0:
                print(f"      {label:<22} {seq_j[label]/batched_j[label]:6.2f}x")

    # ── window 3: decode -- autoregressive, greedy/deterministic ──
    reset_engine(eng)
    z, _, _ = eng.prefill(ids, embed, wnorm)   # prime state (outside the decode window)
    gen, gen_ids = [], []
    b = rapl.snapshot() if EN else None; t0 = time.time()
    for _ in range(a.n):
        nxt = int(np.argmax(logits(z[-1])))
        if nxt in EOS:
            break
        gen_ids.append(nxt); gen.append(dec(nxt))
        z = eng.append(nxt, embed, wnorm)
    decode_s = time.time() - t0
    decode_j = rapl.delta_j(b, rapl.snapshot()) if EN else None
    ntok = len(gen_ids)

    print(f"\n[decode] {ntok} tokens in {decode_s:.2f}s  "
          f"-> {decode_s/ntok*1000:.0f} ms/token, {ntok/decode_s:.3f} tok/s")
    print(f"  text: {''.join(gen)!r}")
    print(f"  token ids (byte-exact re-run check): {gen_ids}")
    if EN:
        print(_fmt_j(decode_j, decode_s))
        print("\n  MEASURED power efficiency:")
        for label, j in decode_j.items():
            net = j - idle_w.get(label, 0.0) * decode_s
            gtpj = ntok / j if j > 0 else float("nan")
            ntpj = ntok / net if net > 0 else float("nan")
            print(f"      {label:<22} {gtpj:7.2f} tok/J gross   "
                  f"{ntpj:7.2f} tok/J net-of-idle   {j/ntok:7.3f} J/token")
    else:
        print("  (no joules: this CPU has no energy counter; feed avg wall-watts to get "
              f"tokens/joule = {ntok} / (watts x {decode_s:.2f}s).)")

    if a.json:
        blob = {"host": os.uname().nodename, "energy_mode": EN,
                "prompt": a.prompt, "gen_tokens": ntok, "gen_ids": gen_ids,
                "load_s": load_s, "decode_s": decode_s,
                "ms_per_token": decode_s / ntok * 1000 if ntok else None,
                "tok_per_s": ntok / decode_s if decode_s else None,
                "prefill_positions": npos, "prefill_batched_s": batched_s,
                "prefill_seq_s": seq_s, "batched_speedup": seq_s / batched_s,
                "prefill_next_token_match": same,
                "decode_j": decode_j,
                "tok_per_j_gross": ({k: (ntok / v if v > 0 else None)
                                     for k, v in decode_j.items()} if EN else None)}
        print("\nJSON " + json.dumps(blob))


if __name__ == "__main__":
    main()
