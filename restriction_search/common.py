"""common.py — shared PAF-bucket "first hit" matcher for restricted exhaustive search.

Both restricted_binary.py and restricted_rle.py enumerate candidate sequences
in different parametrizations, but both ultimately yield literal length-ell
tuples of +-1. This module holds the one true-early-stop matching routine so
neither has to reimplement it: for each candidate, check whether its PAF
complement bucket is already populated (a Legendre pair found), else bucket it
and keep scanning. Mirrors src/legendre.py's find_legendre_pairs_incremental,
just parametrized over an arbitrary candidate iterator instead of
_iter_candidates.
"""
from __future__ import annotations

import os
import random
import sys
import time
from typing import Iterable, Iterator

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
from legendre import _paf_key  # noqa: E402


def find_first_pair_random_order(candidates: Iterable[tuple[int, ...]], ell: int,
                                  seed: int) -> dict:
    """Like find_first_pair, but scans the candidate set in a seeded random
    order first. A *deterministic* enumeration order (plain combinations(),
    or run-length compositions()) makes "time to first hit" pure artifact of
    where the real solutions happen to fall in that arbitrary order -- not a
    fair, seed-averaged comparison against the metaheuristics, which already
    explore in a seeded-random way. Materializing + shuffling is only
    tractable because restriction already shrinks the candidate space (this
    is not meant for the unrestricted, tens-of-millions-candidate case).

    NOTE: materializes `candidates` freshly every call. For multiple seeds
    against the same (ell,i,j), build the list once and call
    find_first_pair_multi_seed instead -- RLE's composition-based generator
    in particular is expensive enough that re-materializing per seed would
    dominate the whole benchmark's runtime.
    """
    pool = list(candidates)
    return find_first_pair_multi_seed(pool, ell, [seed])[0]


def find_first_pair_multi_seed(pool: list[tuple[int, ...]], ell: int,
                                seeds: list[int]) -> list[dict]:
    """Run find_first_pair under `len(seeds)` independent shuffles of the
    SAME already-materialized `pool` (mutates a shuffled copy per seed, never
    the caller's list)."""
    out = []
    for seed in seeds:
        shuffled = pool.copy()
        random.Random(seed).shuffle(shuffled)
        result = find_first_pair(shuffled, ell)
        result["candidate_space"] = len(shuffled)
        out.append(result)
    return out


def find_first_pair(candidates: Iterable[tuple[int, ...]], ell: int) -> dict:
    """Scan `candidates` once, stopping the instant a Legendre pair is seen.

    Returns dict(found, a, b, seconds, scanned).
    """
    half = (ell - 1) // 2
    buckets: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    t0 = time.perf_counter()
    n = 0
    for seq in candidates:
        n += 1
        key = _paf_key(seq, half)
        comp = tuple(-2 - x for x in key)
        partners = buckets.get(comp)
        if partners:
            return {"found": True, "a": seq, "b": partners[0],
                     "seconds": time.perf_counter() - t0, "scanned": n}
        if key == comp:
            return {"found": True, "a": seq, "b": seq,
                     "seconds": time.perf_counter() - t0, "scanned": n}
        buckets.setdefault(key, []).append(seq)
    return {"found": False, "a": None, "b": None,
             "seconds": time.perf_counter() - t0, "scanned": n}
