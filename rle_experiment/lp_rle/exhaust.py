"""exhaust.py — the RLE exhaustive driver and the §3a self-validation.

run_exhaustive(L): enumerate canonical runs -> PSD filter -> hash-match into
LPs -> dedup to inequivalent classes.

validate(L): run BOTH the RLE driver and the independent pm1 brute force and
assert they agree on (i) the set of feasible rotation classes, (ii) the exact
rotation-multiplicity relation on survivor counts, and (iii) the set of LP
equivalence classes. Any discrepancy is our bug (spec §3).
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np

from .conventions import check_odd
from .filter import collect_survivors
from .match import find_pairs, inequivalent_classes
from .runs import seq_to_runs, canonical
from .bruteforce import collect_survivors_brute


RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
LP_DIR = os.path.join(RESULTS, "lps")


def _sign_str(seq) -> str:
    """Compact +/- rendering of a pm1 sequence (LP-literature convention)."""
    return "".join("+" if x > 0 else "-" for x in seq)


def write_lps(L: int, classes: Dict, outdir: str = LP_DIR) -> str:
    """Write every inequivalent Legendre pair of length L to outdir/LP{L}.csv.

    One row per inequivalent class: ``index,u,v`` with u, v as +/- sign
    strings of length L. Returns the path written.
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"LP{L}.csv")
    with open(path, "w") as fh:
        fh.write("index,u,v\n")
        for i, (u, v) in enumerate(classes.values()):
            fh.write(f"{i},{_sign_str(u)},{_sign_str(v)}\n")
    return path


def _canon_class_keys(buckets) -> set:
    """Set of canonical-run byte keys over all sequences in the buckets."""
    keys = set()
    for vs in buckets.values():
        for v in vs:
            keys.add(canonical(seq_to_runs(v)).tobytes())
    return keys


def _reduce_to_canonical(buckets):
    """Collapse buckets to one representative per canonical rotation class.

    The brute force enumerates all L rotations of every class; matching those
    all-vs-all is O((L*n)^2). Reducing to canonical reps first (post-processing
    on pm1, allowed by spec §1.3) makes the independent match as cheap as the
    RLE one while keeping the enumeration itself fully independent.
    """
    from collections import defaultdict

    out = defaultdict(list)
    seen = set()
    for key, vs in buckets.items():
        for v in vs:
            ck = canonical(seq_to_runs(v)).tobytes()
            if ck in seen:
                continue
            seen.add(ck)
            out[key].append(v)
    return out


def run_exhaustive(L: int, tol: float = 1e-6) -> Dict:
    """Full RLE exhaustive search; returns survivors, LPs, and class dedup."""
    check_odd(L)
    buckets, stats = collect_survivors(L, tol)
    pairs = find_pairs(buckets)
    classes = inequivalent_classes(pairs)
    return {
        "L": L,
        "buckets": buckets,
        "total_classes": stats["classes"],
        "survivors": stats["survivors"],
        "per_k": stats["per_k"],
        "raw_pairs": len(pairs),
        "lp_classes": len(classes),
        "classes": classes,
    }


def validate(L: int, tol: float = 1e-6) -> Dict:
    """Cross-check the RLE driver against the independent pm1 brute force."""
    check_odd(L)
    ex = run_exhaustive(L, tol)
    bb, bstats = collect_survivors_brute(L, tol)

    # (i) identical set of feasible rotation classes
    rle_keys = _canon_class_keys(ex["buckets"])
    brute_keys = _canon_class_keys(bb)
    assert rle_keys == brute_keys, (
        f"L={L}: RLE and brute force disagree on feasible rotation classes "
        f"({len(rle_keys)} vs {len(brute_keys)})"
    )

    # (ii) rotation-multiplicity: every orbit has size exactly L (freeness)
    assert bstats["survivors"] % L == 0, f"L={L}: brute survivors not divisible by L"
    assert bstats["survivors"] // L == len(rle_keys), (
        f"L={L}: brute survivors/L = {bstats['survivors'] // L} "
        f"!= #canonical survivors {len(rle_keys)}"
    )
    assert ex["survivors"] == len(rle_keys), (
        f"L={L}: RLE survivor count {ex['survivors']} != #canonical classes {len(rle_keys)}"
    )

    # (iii) identical set of LP equivalence classes
    bpairs = find_pairs(_reduce_to_canonical(bb))
    bclasses = inequivalent_classes(bpairs)
    assert set(ex["classes"].keys()) == set(bclasses.keys()), (
        f"L={L}: RLE and brute force disagree on LP equivalence classes "
        f"({ex['lp_classes']} vs {len(bclasses)})"
    )

    return {
        "L": L,
        "ex": ex,
        "rle_survivors": ex["survivors"],
        "brute_survivors": bstats["survivors"],
        "lp_classes": ex["lp_classes"],
        "total_classes": ex["total_classes"],
        "brute_total": bstats["total"],
    }


if __name__ == "__main__":
    import sys

    # Dump the full list of inequivalent Legendre pairs, one file per length:
    # results/lps/LP{L}.csv. Uses run_exhaustive (the code path the test
    # suite validates against brute force + FGS-2001 literature counts).
    Ls = [int(a) for a in sys.argv[1:]] or list(range(3, 26, 2))
    print(f"{'L':>3} {'classes':>9} {'survivors':>10} {'LPs':>5}  file")
    for L in Ls:
        ex = run_exhaustive(L)
        path = write_lps(L, ex["classes"])
        print(
            f"{ex['L']:>3} {ex['total_classes']:>9} {ex['survivors']:>10} "
            f"{ex['lp_classes']:>5}  {path}"
        )
