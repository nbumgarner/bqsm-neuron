"""
PLASTICITY  —  clean-room Hebbian associative memory for the wave-transformer.

Inline one-shot learning, no retraining, graceful degradation (no catastrophic
forgetting within capacity). Classic linear associative memory:

    teach(key, value):  W += value ⊗ key        (synaptic plasticity — fire together, wire together)
    recall(query):      r = W · query            (associative retrieval), decode to nearest value

Keys/values in production are the wave-transformer's settled hidden state; here a
deterministic random projection stands in for a symbol so the layer is testable
standalone. Additive superposition means new associations never overwrite old ones —
they accumulate in one matrix, so there is no catastrophic forgetting until the keys
run out of room to be distinguished. A plain Hebbian outer-product associative
memory — nothing but numpy and the model's own hidden state.
"""
import numpy as np, hashlib


class PlasticMemory:
    def __init__(self, dim=1024, decay=1.0):
        self.dim = int(dim)
        self.W = np.zeros((self.dim, self.dim), dtype=np.float32)   # the synaptic matrix
        self.values = {}                                           # value -> unit vector (codebook)
        self.decay = float(decay)                                  # 1.0 = perfect retention
        self.n_taught = 0

    def vec(self, x):
        """Unit vector for a symbol (str) — or pass an ndarray to use an engine state directly."""
        if isinstance(x, np.ndarray):
            v = x.astype(np.float32)
        else:
            seed = int.from_bytes(hashlib.blake2b(str(x).encode(), digest_size=8).digest(), "big")
            v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    def teach(self, key, value, lr=1.0):
        k = self.vec(key)
        v = self.values.get(value)
        if v is None:
            v = self.vec(value); self.values[value] = v
        if self.decay < 1.0:
            self.W *= self.decay                       # optional slow forgetting (default off)
        self.W += lr * np.outer(v, k)                  # one-shot Hebbian update — O(dim^2), no retrain
        self.n_taught += 1

    def recall(self, query, top_k=1):
        r = self.W @ self.vec(query)                   # associative retrieval
        r /= (np.linalg.norm(r) + 1e-8)
        scored = sorted(((val, float(r @ vv)) for val, vv in self.values.items()),
                        key=lambda t: -t[1])
        return scored[:top_k]


class PlasticField:
    """Vector-space Hebbian plasticity for the ENGINE residual. Keys and values are
    hidden states (the settled 'thought'). teach binds a key-state to a value-state;
    recall returns a hidden vector to add into the residual at layer `at`, exactly the
    route the inner voice already uses. Empty memory recalls the zero vector, so an
    untaught engine is byte-identical to the un-wired one — plasticity only ever ADDS."""
    def __init__(self, dim):
        self.dim = int(dim)
        self.W = np.zeros((self.dim, self.dim), dtype=np.float32)
        self.n = 0

    @staticmethod
    def _u(v):
        v = np.asarray(v, dtype=np.float32).ravel()
        return v / (np.linalg.norm(v) + 1e-8)

    def teach(self, key, value, lr=1.0):
        self.W += lr * np.outer(self._u(value), self._u(key))   # one-shot Hebbian, no retrain
        self.n += 1

    def recall(self, query):
        """Raw associative retrieval. Magnitude carries match CONFIDENCE (~lr when the
        query matches a stored key, ~0 otherwise) — do NOT renormalize it away. The
        caller scales by the query's hidden magnitude and the small injection gain."""
        q = np.asarray(query, dtype=np.float32).ravel()
        return (self.W @ (q / (np.linalg.norm(q) + 1e-8))).astype(np.float32)  # 0 when untaught


# ─────────────────────────────────────────────────────────────────────────────
def _harness():
    D = 1024; rng = np.random.default_rng(7)
    print("=" * 74)
    print(f"  PLASTICITY — Hebbian associative memory  (dim={D})")
    print("=" * 74)

    def corr_keys(n, rank):
        """n unit keys inside a rank-`rank` subspace — lower rank = more correlated,
        the realistic condition (engine states share structure, aren't random)."""
        B = rng.standard_normal((rank, D)).astype(np.float32)
        K = rng.standard_normal((n, rank)).astype(np.float32) @ B
        return K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-8)

    def acc(m, keys, truevals):                        # batched, vectorized recall accuracy
        K = keys if isinstance(keys, np.ndarray) else np.stack([m.vec(k) for k in keys])
        K = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-8)
        R = K @ m.W.T; R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
        vals = list(m.values); V = np.stack([m.values[v] for v in vals])
        pred = np.asarray(vals, dtype=object)[(R @ V.T).argmax(1)]
        return float(np.mean(pred == np.asarray(truevals, dtype=object)))

    # 1a. RANDOM keys — capacity is near-unlimited (codebook decode with distinguishable
    # keys). Honest, but NOT the operating limit; shown so the next test is in context.
    print("  RANDOM keys (idealized): near-unlimited via codebook decode")
    for load in (256, 1024):
        m = PlasticMemory(D)
        for i in range(load): m.teach(f"k{i}", f"v{i}")
        a = acc(m, [f"k{i}" for i in range(load)], [f"v{i}" for i in range(load)])
        print(f"      load {load:>5}  ->  {a*100:5.1f}%")

    # 1b. THE REAL LIMIT — correlated keys. Fix load, shrink the key subspace rank so
    # keys overlap; capacity degrades as rank falls below the load. Discriminative.
    print("  CORRELATED keys (realistic): load=400, shrinking key-subspace rank")
    print(f"  {'key rank':>10}{'recall acc':>13}   note")
    for rank in (128, 64, 32, 16, 8, 4):
        m = PlasticMemory(D); K = corr_keys(400, rank)
        for i in range(400): m.teach(K[i], f"v{i}")
        a = acc(m, K, [f"v{i}" for i in range(400)])
        tag = "" if a > 0.95 else ("← degrading" if a > 0.5 else "← saturated: rank < load")
        print(f"  {rank:>10}{a*100:>11.1f}%   {tag}")

    # 2. NO CATASTROPHIC FORGETTING
    m = PlasticMemory(D); m.teach("first-fact", "remembered")
    for i in range(400): m.teach(f"noise{i}", f"n{i}")
    keep = m.recall("first-fact")[0]
    print("-" * 74)
    print(f"  no catastrophic forgetting: 'first-fact' after +400 teaches -> "
          f"{keep[0]!r}  {'✓ retained' if keep[0]=='remembered' else '✗ lost'}")

    # 3. ASSOCIATIVE GENERALIZATION — recall from a noisy/partial cue
    m = PlasticMemory(D)
    for i in range(200): m.teach(f"key{i}", f"val{i}")
    K = np.stack([m.vec(f"key{i}") for i in range(200)])
    noisy = K + 0.6 * rng.standard_normal((200, D)).astype(np.float32) / np.sqrt(D)
    print(f"  associative generalization: noisy-cue recall (200) -> "
          f"{acc(m, noisy, [f'val{i}' for i in range(200)])*100:.1f}% correct")

    # 4. CONTROL — a never-taught cue must not yield a confident false memory
    taught = np.mean([m.recall(f"key{i}")[0][1] for i in range(50)])
    novel  = np.mean([m.recall(f"UNSEEN-{i}")[0][1] for i in range(50)])
    print(f"  control (discriminative): taught-cue {taught:.2f}  vs  novel-cue {novel:.2f}  "
          f"{'✓ separable' if taught > novel + 0.2 else '✗ not separable'}")
    print("=" * 74)


if __name__ == "__main__":
    _harness()
