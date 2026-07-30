"""compare.py — compare two Legendre-pair datasets by a COMPLETE INVARIANT.

The difficulty the user flagged: an inequivalent class has no canonical
representative on paper — the same class appears in many guises (independent
cyclic shifts, independent reversal, a common decimation, and swapping u<->v).
So you cannot compare two datasets by string-matching their stored sequences.

The fix is a complete invariant. ``symmetry.canonical_pair(u, v)`` returns the
SAME bytes key for every pair in an orbit and different keys for different
orbits. Feeding every stored pair through it turns each dataset into a *set of
keys*; then

    equivalent-as-collections  <=>  equal key sets,

and set differences pinpoint exactly which classes are unique to each side.

Parsing is convention-robust:
  * sequences may be '0/1' or '+/-' strings;
  * each sequence is normalized to weight (L+1)/2 ones (row sum +1) by negating
    if needed — negation leaves the PAF (hence the pair) unchanged and matches
    the fixed-weight convention under which the FGS / Kotsireas counts are
    defined;
  * every pair is verified to actually satisfy PAF_u + PAF_v = -2.

The comparison is only as trustworthy as canonical_pair being a genuine
invariant for the intended group; if two datasets disagree, dump_diff() prints
representatives so the specific pairs can be inspected by hand.
"""
from __future__ import annotations

import csv
from typing import Dict, List, Tuple

import numpy as np

from .paf import paf_naive
from .symmetry import canonical_pair

Pair = Tuple[np.ndarray, np.ndarray]
PairKey = Tuple[bytes, bytes]


def parse_seq(s: str) -> np.ndarray:
    s = s.strip()
    chars = set(s)
    if chars <= {"0", "1"}:
        return np.array([1 if c == "1" else -1 for c in s], dtype=np.int8)
    if chars <= {"+", "-"}:
        return np.array([1 if c == "+" else -1 for c in s], dtype=np.int8)
    raise ValueError(f"unrecognized sequence encoding: {s!r}")


def normalize(v: np.ndarray) -> np.ndarray:
    """Force row sum +1 (weight (L+1)/2 ones). Negation preserves the PAF."""
    return (-v).astype(np.int8) if int(v.sum()) < 0 else v


def load_pairs(path: str, seq_cols=(0, 1), has_header: bool = True) -> List[Pair]:
    """Load stored (u, v) pairs, normalized to ±1 with row sum +1."""
    with open(path) as fh:
        rows = list(csv.reader(fh))
    if has_header and rows:
        rows = rows[1:]
    out: List[Pair] = []
    i, j = seq_cols
    for row in rows:
        if not row or i >= len(row) or not row[i].strip():
            continue
        out.append((normalize(parse_seq(row[i])), normalize(parse_seq(row[j]))))
    return out


def keyset(pairs: List[Pair], verify: bool = True):
    """Map pairs -> {canonical_key: representative}; count non-LP and duplicate rows."""
    keys: Dict[PairKey, Pair] = {}
    bad = 0
    dup = 0
    for u, v in pairs:
        if verify and not np.all(paf_naive(u) + paf_naive(v) == -2):
            bad += 1
            continue
        k = canonical_pair(u, v)
        if k in keys:
            dup += 1
        else:
            keys[k] = (u, v)
    return keys, bad, dup


def compare(path_a: str, path_b: str, a_kw=None, b_kw=None, verify: bool = True) -> Dict:
    A = load_pairs(path_a, **(a_kw or {}))
    B = load_pairs(path_b, **(b_kw or {}))
    ka, bad_a, dup_a = keyset(A, verify)
    kb, bad_b, dup_b = keyset(B, verify)
    sa, sb = set(ka), set(kb)
    return {
        "rows_a": len(A), "rows_b": len(B),
        "classes_a": len(sa), "classes_b": len(sb),
        "shared": len(sa & sb),
        "a_only": sa - sb, "b_only": sb - sa,
        "bad_a": bad_a, "bad_b": bad_b, "dup_a": dup_a, "dup_b": dup_b,
        "keys_a": ka, "keys_b": kb,
    }


def _sign(v) -> str:
    return "".join("+" if x > 0 else "-" for x in v)


def dump_diff(res: Dict, label_a="A", label_b="B", limit: int = 5):
    for side, keys, tag in (("a_only", res["keys_a"], label_a),
                            ("b_only", res["keys_b"], label_b)):
        only = res[side]
        if not only:
            continue
        print(f"  {len(only)} class(es) only in {tag}:")
        for k in list(only)[:limit]:
            u, v = keys[k]
            print(f"    u={_sign(u)}  v={_sign(v)}")
        if len(only) > limit:
            print(f"    ... (+{len(only) - limit} more)")


if __name__ == "__main__":
    import argparse
    import os

    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OURS = os.path.join(HERE, "results", "lps", "LP{L}.csv")            # index,u,v  (+/-)
    FRIEND = os.path.join(
        os.path.dirname(HERE),  # .../legendre_pairs
        "Legendre_database", "database", "LegendrePairs", "legendre_pair{L}.csv")  # seq1,seq2 (0/1)

    ap = argparse.ArgumentParser(description="Compare our LP tables vs the friend database.")
    ap.add_argument("Ls", type=int, nargs="*", help="odd lengths (default 3..31)")
    ap.add_argument("--diff", action="store_true", help="print representatives of differing classes")
    args = ap.parse_args()

    Ls = args.Ls or list(range(3, 32, 2))
    ah, bh = "ours-fr", "fr-ours"
    print(f"{'L':>3} {'ours':>6} {'friend':>7} {'shared':>7} {ah:>8} {bh:>8}  {'dupF':>5} {'badF':>5}")
    for L in Ls:
        ours = OURS.format(L=L)
        friend = FRIEND.format(L=L)
        if not (os.path.exists(ours) and os.path.exists(friend)):
            print(f"{L:>3}  (missing file)")
            continue
        res = compare(ours, friend,
                      a_kw={"seq_cols": (1, 2), "has_header": True},
                      b_kw={"seq_cols": (0, 1), "has_header": True})
        print(f"{L:>3} {res['classes_a']:>6} {res['classes_b']:>7} {res['shared']:>7} "
              f"{len(res['a_only']):>8} {len(res['b_only']):>8}  {res['dup_b']:>5} {res['bad_b']:>5}")
        if args.diff and (res["a_only"] or res["b_only"]):
            dump_diff(res, "ours", "friend")
