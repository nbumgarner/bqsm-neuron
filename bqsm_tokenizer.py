#!/usr/bin/env python3
"""
bqsm_tokenizer.py — real byte-level BPE for the Llama-3 vocabulary.

The engine's previous encoder split on whitespace and looked the whole word up
in the vocab, falling back to ONE TOKEN PER CHARACTER when that missed. So:

    'Hello! How are you?'   -> H e l l o ! ' How' ' are' ' ' y o u ?
    'interference'          -> i n t e r f e r e n c e

Anything with attached punctuation, any contraction, and any word not present as
a single vocab entry got spelled out. The model then had to work from a prompt
that no longer looked like text, which is why generation degraded into spelling.

This does what the tokenizer.json actually specifies:
  1. pre-tokenise with the Llama-3 regex (contractions, letters, digits 1-3,
     punctuation runs, whitespace)
  2. GPT-2 byte->unicode mapping, so every byte is representable
  3. greedy BPE by merge rank over the 280,147 merges
  4. vocab lookup

Python's `re` has no \\p{L}, and the `regex` module is not installed here, so the
unicode classes are expressed as [^\\W\\d_] (letters) and \\d (numbers) with
re.UNICODE. That is exact for ASCII and correct for the Latin text this model is
used on; it can differ from the reference on exotic scripts, which is stated
rather than hidden.

    python3 bqsm_tokenizer.py --selftest
"""
import functools, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Python's re has no \p{L}. Translations used below:
#   \p{L}                 -> [^\W\d_]        (word char that is not digit/underscore)
#   \p{N}                 -> \d
#   [^\r\n\p{L}\p{N}]     -> (?:[^\r\n\w]|_) (not CR/LF, not letter, not digit)
#   [^\s\p{L}\p{N}]       -> (?:[^\s\w]|_)   (not space, not letter, not digit)
# Negating a class by stripping its brackets does NOT work and silently drops
# punctuation -- an earlier version did exactly that and ate every '!' and '?'.
PAT = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|(?:[^\r\n\w]|_)?[^\W\d_]+"
    r"|\d{1,3}"
    r"| ?(?:[^\s\w]|_)+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+",
    re.UNICODE)


@functools.lru_cache(maxsize=1)
def _byte_map():
    """GPT-2 byte<->unicode table: keeps every byte printable and BPE-able."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    enc = {b: chr(c) for b, c in zip(bs, cs)}
    return enc, {v: k for k, v in enc.items()}


class Tokenizer:
    def __init__(self, base):
        tk = json.load(open(os.path.join(base, "tokenizer.json")))
        self.vocab = tk["model"]["vocab"]
        self.inv = {v: k for k, v in self.vocab.items()}
        # model.vocab stops at 128000; every special (<|im_end|>, <|im_start|>,
        # the reserved block) lives in added_tokens. Without these, any special
        # token decodes as a bare "[128256]" -- which is exactly what the inner
        # monologue panel showed when the voice ran off the end of a turn.
        self.special = {t["id"]: t["content"] for t in tk.get("added_tokens", [])}
        for i, c in self.special.items():
            self.inv.setdefault(i, c)
        merges = tk["model"].get("merges", [])
        # tokenizer.json stores merges as "a b" strings or ["a","b"] pairs
        self.ranks = {}
        for i, m in enumerate(merges):
            a, b = (m.split(" ", 1) if isinstance(m, str) else m)
            self.ranks[(a, b)] = i
        self.bos = self.vocab.get("<|begin_of_text|>", 128000)
        self.b2u, self.u2b = _byte_map()
        self._cache = {}

    def _bpe(self, piece):
        if piece in self._cache:
            return self._cache[piece]
        parts = list(piece)
        while len(parts) > 1:
            best, at = None, -1
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, at = r, i
            if at < 0:
                break
            parts[at:at + 2] = [parts[at] + parts[at + 1]]
        self._cache[piece] = parts
        return parts

    def encode(self, text, bos=True):
        ids = [self.bos] if bos else []
        for m in PAT.findall(text):
            if not m:
                continue
            piece = "".join(self.b2u[b] for b in m.encode("utf-8"))
            for tokn in self._bpe(piece):
                v = self.vocab.get(tokn)
                if v is not None:
                    ids.append(v)
                else:                       # unreachable for a complete BPE vocab
                    for ch in tokn:
                        if ch in self.vocab:
                            ids.append(self.vocab[ch])
        return ids

    def decode_one(self, i):
        if i in getattr(self, "special", ()):
            return self.special[i]
        s = self.inv.get(i)
        if s is None:
            return f"[{i}]"
        try:
            return bytes(self.u2b[c] for c in s).decode("utf-8", "replace")
        except KeyError:
            return s.replace("Ġ", " ").replace("Ċ", "\n")

    def decode(self, ids):
        return "".join(self.decode_one(i) for i in ids)


def _selftest():
    from bqsm_llama import BASE
    t = Tokenizer(BASE)
    cases = ["The capital of France is",
             "Explain wave interference, briefly.",
             "What's the difference between int8 and bf16?",
             "Hello! How are you?",
             "  leading and   repeated   spaces\nand a newline"]
    ok = True
    for c in cases:
        ids = t.encode(c)
        rt = t.decode(ids[1:])
        good = rt == c
        ok &= good
        print(f"  {c!r}\n    {len(ids)-1:>3} tokens  {[t.decode_one(i) for i in ids[1:]][:12]}")
        print(f"    round-trip {'OK' if good else 'MISMATCH: ' + repr(rt)}")
    print(f"\n  {'all round-trips exact' if ok else '*** ROUND-TRIP FAILURE ***'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)
