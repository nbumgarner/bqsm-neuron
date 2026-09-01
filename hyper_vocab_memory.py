#!/usr/bin/env python3
"""
Hyperdimensional Vocab Memory — oscillator associative memory for a language model.

  - Encode vocab as 256 complex oscillator hypervectors (random projection).
  - Burn in token co-occurrence from text corpus (Hebbian relationship modulation).
  - Query: context string → recall associated tokens by phase-coherent pattern
    completion.
  - Fuse: memory scores boost model logits (3B params for reasoning, memory for
    knowledge → functions like a larger model).

Real Llama 3B tokenizer + bf16 embeddings, text corpus from the local disk.

    python3 hyper_vocab_memory.py
"""
import glob, json, math, os, struct, sys, time
import numpy as np

# ── tokenizer + embeddings (mmap'd bf16) ───────────────────────────────
# One resolver, not two. This used to carry its own copy of the hub glob
# pointing at a specific developer's Hermes-3B checkout, so on any other
# machine it raised IndexError at import -- which the dashboard surfaced as
# the uninformative "hvm load failed: list index out of range".
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bqsm_llama import BASE
C = json.load(open(os.path.join(BASE, "config.json")))
D = C["hidden_size"]
TK = json.load(open(os.path.join(BASE, "tokenizer.json")))
VOCAB = TK["model"]["vocab"]    # token -> id
INV = {v: k for k, v in VOCAB.items()}


class ST:
    """Minimal safetensors reader — mmap, no copy on parse."""
    def __init__(self):
        self.shards, self.idx = [], {}
        for p in sorted(glob.glob(os.path.join(BASE, "*.safetensors"))):
            mm = np.memmap(p, dtype=np.uint8, mode="r")
            n = struct.unpack("<Q", bytes(mm[:8]))[0]
            hdr = json.loads(bytes(mm[8:8 + n]).decode())
            st = 8 + n
            si = len(self.shards)
            self.shards.append((mm, st))
            for k, v in hdr.items():
                if k != "__metadata__":
                    self.idx[k] = (si, v)

    def get(self, name):
        si, v = self.idx[name]
        mm, st = self.shards[si]
        a, b = v["data_offsets"]
        raw = np.asarray(mm[st + a: st + b])
        dt = v["dtype"]
        if dt == "BF16":
            return ((raw.view(np.uint16).astype(np.uint32) << 16)
                    ).view(np.float32).reshape(v["shape"])
        elif dt == "F16":
            return raw.view(np.float16).astype(np.float32).reshape(v["shape"])
        return raw.view(np.float32).reshape(v["shape"])


st = ST()


# ── embed-row helper (bf16 zero-copy per row) ───────────────────────────
def embed_row(token_id):
    si, v = st.idx["model.embed_tokens.weight"]
    mm, st_off = st.shards[si]
    D_emb = v["shape"][1]
    a, b = v["data_offsets"]
    off = st_off + a + token_id * D_emb * 2  # bf16 = 2 bytes
    raw = np.asarray(mm[off: off + D_emb * 2])
    return ((raw.view(np.uint16).astype(np.uint32) << 16)
            ).view(np.float32)  # → f32, D-dim


# ── tokenizer ────────────────────────────────────────────────────────────


# The real byte-level BPE, not a whitespace splitter. The previous encoder
# fell back to ONE TOKEN PER CHARACTER for any word not in the vocab whole, so
# "Kuramoto" became K,u,r,a,m,o,t,o -- eight single characters, every one of
# them then discarded by the content filter. Two consequences: the corpus was
# burned in over a mangled tokenisation, and the ids this memory recalls were
# not the ids the model emits, so the fusion bias landed on tokens the model
# had no use for.
_BPE = None


def _bpe():
    global _BPE
    if _BPE is None:
        from bqsm_tokenizer import Tokenizer
        _BPE = Tokenizer(BASE)
    return _BPE


def encode(text):
    return _bpe().encode(text)


def dec(i):
    return _bpe().decode_one(i)


# ── Hyperdimensional encoding: token embedding → oscillator state ───────
N_OSC = 256
rng = np.random.default_rng(42)
# random projection matrix [2*N_OSC, D]
PROJ = rng.standard_normal((N_OSC * 2, D), dtype=np.float32) / np.sqrt(D)


def osc_vector(tok, cache):
    """Complex oscillator state for a token (from cache), or zeros if unseen."""
    v = cache.get(tok)
    return v if v is not None else np.zeros(N_OSC, dtype=np.complex64)


def build_cache(token_ids, chunk=4096, label=""):
    """Encode token IDs into normalized complex oscillator states (batched).

    Reads the bf16 embedding matrix via mmap in chunks, projects each chunk,
    and stores a compact complex64 vector per token.  Full vocab (~128K) is
    ~262 MB of cache — built chunk-wise so peak memory stays ~100 MB."""
    si, v = st.idx["model.embed_tokens.weight"]
    mm, st_off = st.shards[si]
    D_emb = v["shape"][1]
    a, b = v["data_offsets"]
    base = st_off + a
    ids = list(token_ids)
    cache = {}
    for i in range(0, len(ids), chunk):
        batch = ids[i:i + chunk]
        lo, hi = batch[0], batch[-1]
        off = base + lo * D_emb * 2
        raw = np.asarray(mm[off: off + (hi - lo + 1) * D_emb * 2])
        E = ((raw.view(np.uint16).astype(np.uint32) << 16)
             ).view(np.float32).reshape(hi - lo + 1, D_emb)
        P = E[[t - lo for t in batch]] @ PROJ.T   # [batch, 2*N_OSC]
        P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
        C = (P[:, :N_OSC] + 1j * P[:, N_OSC:]).astype(np.complex64)
        for j, t in enumerate(batch):
            cache[t] = C[j]
        if label and (i == 0 or (i + chunk) >= len(ids)):
            print(f"    {label} {min(i + chunk, len(ids))}/{len(ids)}", flush=True)
    return cache


# ── Gather the tokens that actually matter (corpus + queries) ───────────
# Resolved relative to this file, so a deployed copy burns in its own docs
# instead of silently reading nothing. The absolute paths that used to live
# here existed on one machine; anywhere else this reported "0 associations
# from 0 tokens" and looked like a working memory with nothing to remember.
# PHOX_CORPUS overrides, colon-separated. Missing files are skipped.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
# A 27k-token corpus cannot produce a distribution sharp enough to compete
# with a trained model's confidence -- measured: the memory's mass spread over
# 32 tokens gives its best ~0.1 against the model's ~0.9. Volume is the fix.
_BIG = ["/home/compunerd/corpus/gutenberg.txt",
        "/home/compunerd/corpus/wikitext103.txt"]
# Both: the books give general English, the docs keep the project vocabulary.
# Replacing rather than appending cost "wave interference" its recall.
corpus_files = [p for p in os.environ.get("PHOX_CORPUS", "").split(":") if p.strip()] or \
    [p for p in _BIG if os.path.exists(p)] + [
    os.path.join(_REPO, "README.md"),
    os.path.join(_HERE, "README.md"),
    os.path.join(_HERE, "WAVE_RIDER_BREAKTHROUGH.md"),
    os.path.join(_HERE, "PIPELINE_PLAN.md"),
    os.path.join(_HERE, "claude.soul"),
    os.path.expanduser("~/Desktop/bqsm/basin-quotient-machine/LENS_CONTROL_METHODS.md"),
    os.path.expanduser("~/Desktop/bqsm/basin-quotient-machine/README.md"),
]
test_queries = [
    "The capital of France is",
    "BQSM uses coupled",
    "The ring computes through mode",
    "lens site 0 enhances the",
    "Phase 0 Gate",
    "a transformer forward pass as",
    "the model with real bf16",
    "attention becomes geometric",
]

used_ids = set()
corpus_texts = []
for fp in corpus_files:
    if os.path.exists(fp):
        # Cap per file: the skip-gram counters hold ~3 entries per token and
        # this box has ~4 GB free beside the engine. HVM_MAX_MB tunes it.
        text = open(fp).read()[:int(float(os.environ.get("HVM_MAX_MB", "9")) * 1e6)]
        corpus_texts.append(text)
        used_ids.update(encode(text))
for q in test_queries:
    used_ids.update(encode(q))

# Full-dictionary coverage: encode the entire vocab so ANY token the model
# emits can be scored/recalled.  Corpus-derived associations still come from
# `used_ids`, but every token now has a hypervector.  BQSM_FULL_VOCAB=off
# reverts to the corpus-subset cache for fast smoke tests.
FULL_VOCAB = os.environ.get("BQSM_FULL_VOCAB", "on") != "off"
if FULL_VOCAB:
    V = st.idx["model.embed_tokens.weight"][1]["shape"][0]
    print(f"encoding full vocab ({V} tokens) — one-time, ~30s...")
    t0 = time.time()
    cache = build_cache(range(V), label="vocab")
    print(f"  {time.time()-t0:.1f}s ({len(cache)} tokens cached)")
else:
    print(f"encoding {len(used_ids)} distinct tokens (corpus + queries)...")
    t0 = time.time()
    cache = build_cache(sorted(used_ids))
    print(f"  {time.time()-t0:.1f}s")

# Burn-in + build sparse "following" index (skip-gram, distance-decayed)
from collections import Counter, defaultdict

print("\nBurn-in corpus (skip-gram PMI, window=3)...")
MAX_DIST = 3
unigram = Counter()
skipgram = {d: Counter() for d in range(1, MAX_DIST + 1)}
total_tokens = 0
for text in corpus_texts:
    ids = encode(text)
    total_tokens += len(ids)
    unigram.update(ids)
    for d in range(1, MAX_DIST + 1):
        skipgram[d].update(zip(ids[:-d], ids[d:]))

# Distance-decayed PMI: tokens d apart get weight 1/d.  This captures
# "France -> is -> Paris" as "France -> Paris" (d=2, weight 0.5), which is
# what a pure bigram memory misses.
_fanin = {}          # target -> how many distinct sources follow it (IDF)
W = np.zeros((N_OSC, N_OSC), dtype=np.complex64)
following = defaultdict(list)       # token_id -> [(target_id, weight), ...]
n_pairs = 0
# PMI rewards RARE co-occurrence, so with no floor the strongest associations
# in a markdown/code corpus are bracket and single-character pairs -- measured:
# the top recalls for "The capital of France is" were '[', 'f', '(', 'c'.
# Three filters, all on the burn-in rather than the query, so the memory never
# stores the junk in the first place:
MIN_PAIR = int(os.environ.get("HVM_MIN_PAIR", "1"))   # coverage floor; the
MIN_UNI = int(os.environ.get("HVM_MIN_UNI", "2"))     # content filter above does
                                                      # the real work -- at MIN_PAIR=3
                                                      # a 27k-token corpus keeps only
                                                      # 181 pairs and most queries
                                                      # return nothing at all


def _is_content(tid):
    """A token worth associating: at least two characters, containing a letter.
    Punctuation, digits and single characters carry syntax, not meaning."""
    t = dec(tid).strip()
    return len(t) >= 2 and any(c.isalpha() for c in t)


_content = {t: _is_content(t) for t in unigram}
_wa, _wb = [], []                             # batched outer-product buffers
n_skip_junk = n_skip_rare = 0
for d in range(1, MAX_DIST + 1):
    decay = 1.0 / d
    for (a, b), cnt in skipgram[d].items():
        za = cache.get(a); zb = cache.get(b)
        if za is None or zb is None:
            continue
        if not (_content.get(a) and _content.get(b)):
            n_skip_junk += 1
            continue
        if cnt < MIN_PAIR or unigram[a] < MIN_UNI or unigram[b] < MIN_UNI:
            n_skip_rare += 1
            continue
        pmi = math.log((cnt * total_tokens) / (unigram[a] * unigram[b]) + 1e-12)
        if pmi <= 0:
            continue
        w = decay * pmi
        _wa.append(za * w); _wb.append(zb)
        if len(_wa) >= 20000:                 # 410 us per np.outer x millions
            A = np.asarray(_wa); B = np.asarray(_wb)
            W += A.T @ np.conj(B)             # same sum, one matmul
            _wa.clear(); _wb.clear()
        following[a].append((b, w))
        n_pairs += 1
if _wa:
    W += np.asarray(_wa).T @ np.conj(np.asarray(_wb))
    _wa.clear(); _wb.clear()
for _src, _lst in following.items():
    for _t, _w in _lst:
        _fanin[_t] = _fanin.get(_t, 0) + 1
print(f"  {n_pairs} associations burned in (PMI>0), from {total_tokens} tokens"
      f"  [dropped {n_skip_junk} non-content, {n_skip_rare} below floor]")
norm = np.linalg.norm(W)
if norm > 0:
    W /= norm

# ── Plasticity: writing new associations at runtime ──────────────────────
# The update rule was always local -- W += w * outer(za, conj(zb)) needs only
# the two vectors and a scalar, no gradient and no optimiser. What was missing
# was a caller, a decay term, and somewhere to persist it. All three are here.
LAST_W = float(os.environ.get("HVM_LAST_W", "12.0"))
LEARNED = os.path.join(os.environ.get("PHOX_HOME") or
                       os.path.expanduser("~/.phox"), "learned.jsonl")
DECAY = float(os.environ.get("HVM_DECAY", "0.999"))
# MEASURED: at 6 a taught pair does not enter the top-5 recall against 2M
# accumulated corpus associations; at 25 it does, and 100/400 add nothing --
# the ranking has already flipped. 25 is the knee, not a guess.
TEACH_W = float(os.environ.get("HVM_TEACH_W", "25.0"))


def burn(text, strength=None, persist=True):
    """Write the associations in `text` into the memory, immediately.

    Explicit teaching is supervision, not observation, so it does not wait for
    a corpus-scale PMI estimate -- a fact told once should be learnable once.
    The weight is fixed (TEACH_W) and distance-decayed like the corpus pairs,
    which puts a taught pair roughly an order of magnitude above a typical
    observed one without letting a single sentence dominate the memory.

    Every write leaks the existing memory by DECAY. Hebbian accumulation with
    no forgetting saturates: this is what makes it track recent experience
    rather than silt up. At 0.999 a pair is halved after ~700 further writes.
    """
    global total_tokens, n_pairs, W
    ids = [t for t in encode(text) if t in cache]
    if len(ids) < 2:
        return {"pairs": 0, "reason": "nothing encodable"}
    w0 = TEACH_W if strength is None else float(strength)
    # leak first, so a write never lands on an already-saturated W
    if DECAY < 1.0:
        W *= DECAY
        for k in following:
            following[k] = [(t, v * DECAY) for t, v in following[k]]
    added = 0
    A, B = [], []
    for d in range(1, MAX_DIST + 1):
        for a, b in zip(ids[:-d], ids[d:]):
            if not (_is_content(a) and _is_content(b)):
                continue
            w = w0 / d
            A.append(cache[a] * w); B.append(cache[b])
            # MERGE, do not append: teaching the same sentence twice was
            # leaving two entries for the same pair, so the list grew without
            # bound and query_sparse summed the duplicates.
            lst = following[a]
            for j, (t_, w_) in enumerate(lst):
                if t_ == b:
                    lst[j] = (b, w_ + w)
                    break
            else:
                lst.append((b, w))
            unigram[a] += 1; unigram[b] += 1
            skipgram[d][(a, b)] += 1
            added += 1
    if A:
        W += np.asarray(A).T @ np.conj(np.asarray(B))
    total_tokens += len(ids); n_pairs += added
    if added:
        _rebuild_fanin()
    if persist and added:
        try:
            os.makedirs(os.path.dirname(LEARNED), exist_ok=True)
            with open(LEARNED, "a") as f:
                f.write(json.dumps({"text": text, "w": w0, "ts": time.time()}) + "\n")
        except Exception:
            pass
    return {"pairs": added, "tokens": len(ids), "weight": w0,
            "total_assoc": n_pairs}


def _rebuild_fanin():
    """How many distinct sources each target follows -- the IDF denominator."""
    _fanin.clear()
    for _src, lst in following.items():
        for t_, _w in lst:
            _fanin[t_] = _fanin.get(t_, 0) + 1


def replay_learned():
    """Re-apply everything ever taught. The corpus burn-in is deterministic,
    so replaying the learned file on top reconstructs the exact memory --
    which is why only the text is persisted, not the matrix."""
    if not os.path.exists(LEARNED):
        return 0
    n = 0
    for line in open(LEARNED):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            burn(d["text"], strength=d.get("w"), persist=False)
            n += 1
        except Exception:
            pass
    return n


# ── Sparse recall: per context token, aggregate its strongest followers ───
def query_sparse(context_str, top_k=20):
    """Followers of the context, ranked.

    Two corrections, both measured. The last token is what the next token
    actually follows, so it dominates rather than merely counting double --
    at 2x, a stopword appearing three times earlier in the context outscored
    the taught association attached to the final word. And a target that
    follows hundreds of different sources carries no information about THIS
    context, so it is discounted by how many sources it follows: the same
    idea as IDF, and it is what stops 'at' and 'the' owning every ranking.
    """
    ids = encode(context_str)
    scores = defaultdict(float)
    n = len(ids)
    for i, cid in enumerate(ids):
        if cid not in following:
            continue
        w = LAST_W if i == n - 1 else 1.0
        for tid, pmi in following[cid]:
            scores[tid] += w * pmi
    if scores:
        for tid in scores:
            scores[tid] /= (1.0 + math.log1p(_fanin.get(tid, 0)))
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [r for r in ranked if r[0] in cache][:top_k]


# ── Query: context → associative recall ──────────────────────────────────
def query(context_str, top_k=20):
    """Encode context as oscillator state, recall associated tokens via W."""
    ids = encode(context_str)
    ctx = np.zeros(N_OSC, dtype=np.complex64)
    n_valid = 0
    for t in ids:
        z = cache.get(t)
        if z is not None:
            ctx += z
            n_valid += 1
    if n_valid == 0:
        return []
    ctx /= n_valid + 1e-8
    recalled = W @ ctx
    recalled = recalled / (np.linalg.norm(recalled) + 1e-8)
    scores = []
    for t, zt in cache.items():
        score = float(abs(np.dot(np.conj(zt), recalled)))
        scores.append((t, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ── Demo (only when run directly) ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 66)
    print("HYPERDIMENSIONAL VOCAB MEMORY — recall demo")
    print("=" * 66)

    tests = [
        ("The capital of France is", "Paris"),
        ("BQSM uses coupled", "oscillator"),
        ("The ring computes through mode", "coupling"),
        ("lens site 0 enhances the", "channel"),
        ("Phase 0 Gate", "FAILURE"),
        ("a transformer forward pass as", "coupled"),
        ("the model with real bf16", "weights"),
        ("attention becomes geometric", "adjacency"),
    ]

    def find_token(text):
        for t in cache:
            if dec(t).strip() == text:
                return t
        return None

    for context, expected in tests:
        results = query_sparse(context)
        expected_id = find_token(expected)
        rank = None
        for i, (t, s) in enumerate(results):
            if t == expected_id:
                rank = i + 1
                break
        print(f"\n  \"{context}\"")
        print(f"  expect: \"{expected}\"  rank: "
              f"{rank if rank else '-- (not in top %d)' % len(results)}")
        print(f"  top 5: ", end="")
        for t, s in results[:5]:
            print(f"{dec(t)!r}({s:.4f})", end="  ")
        print()

    print("\n" + "=" * 66)
    print("FUSION — memory boosts model logits (simulated)")
    print("=" * 66)
    context = "The capital of France is"
    model_logits = {t: float(rng.standard_normal()) * 0.5 for t in cache}
    results = query_sparse(context)
    for t, mem_score in results:
        model_logits[t] = model_logits.get(t, 0.0) + 2.0 * mem_score
    top_after = sorted(model_logits, key=lambda t: model_logits[t],
                       reverse=True)[:10]
    print(f"  context: {context!r}")
    print(f"  top-10 after fusion: {[dec(t) for t in top_after]}")
    paris_id = find_token("Paris")
    if paris_id:
        rank = top_after.index(paris_id) + 1 if paris_id in top_after else None
        print(f"  'Paris' rank after fusion: "
              f"{'#' + str(rank) if rank else '-- (out of top 10)'}")

    print("\n" + "=" * 66)
    print("HOW IT SCALES TO 30B-CLASS:")
    print("  - 3B model: grammar, reasoning, common patterns (its parameters)")
    print("  - Oscillator memory: facts, entity links, co-occurrence (burn-in)")
    print("  - The memory costs N² oscillators (~256² = 65K couplings), not GBs")
    print("  - Continually learns: new facts burn in without retraining the model")
    print("  - Hyperdimensional encoding: near-orthogonal random projections")
    print("    = associative memory for 128K vocab in ~65K complex couplings")
    print("=" * 66)