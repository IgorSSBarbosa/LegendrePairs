"""restricted_binary.py — binary-representation restricted exhaustive search.

Forces the first i literal positions to +1 and the last j literal positions
to -1, and enumerates only the sum-normalized candidates consistent with
that (same closed form as canonical_orbits/restriction_matrix.py's
`denominator`), then hands them to common.find_first_pair for a genuine
early-stopping brute-force scan -- this is the "restricted binary
brute-force" arm of the search-strategy comparison.
"""
from __future__ import annotations

import os
import sys
from itertools import combinations
from typing import Iterator

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "canonical_orbits"))
from restriction_matrix import denominator  # noqa: E402

from common import find_first_pair, find_first_pair_random_order, find_first_pair_multi_seed  # noqa: E402


def iter_restricted_candidates(ell: int, i: int, j: int) -> Iterator[tuple[int, ...]]:
    """Sum=+1 candidates with the first i entries forced +1, last j forced -1."""
    m = ell - i - j
    if m < 0:
        return
    n_minus = (ell - 1) // 2 - j
    p_free = m - n_minus
    if n_minus < 0 or p_free < 0:
        return
    for minus in combinations(range(i, ell - j), n_minus):
        seq = [1] * ell
        for pos in range(ell - j, ell):
            seq[pos] = -1
        for pos in minus:
            seq[pos] = -1
        yield tuple(seq)


def find_first_binary_exhaustive(ell: int, i: int, j: int, seed: int | None = None) -> dict:
    """Time-to-first-hit for the restricted binary brute-force scan.

    seed=None: deterministic combinations() order (fast, but "time to first
    hit" is then pure enumeration-order artifact). seed=int: shuffled order,
    fair to compare against seeded metaheuristic runs -- use this for the
    benchmark.
    """
    if seed is None:
        result = find_first_pair(iter_restricted_candidates(ell, i, j), ell)
        result["candidate_space"] = denominator(ell, i, j)
        return result
    return find_first_pair_random_order(iter_restricted_candidates(ell, i, j), ell, seed)


def find_first_binary_exhaustive_multi_seed(ell: int, i: int, j: int, seeds: list[int]) -> list[dict]:
    """Materializes the restricted candidate list once, then runs one
    seeded-shuffle scan per seed (much cheaper than calling
    find_first_binary_exhaustive per seed)."""
    pool = list(iter_restricted_candidates(ell, i, j))
    return find_first_pair_multi_seed(pool, ell, seeds)


if __name__ == "__main__":
    import sys as _sys

    ell = int(_sys.argv[1]) if len(_sys.argv) > 1 else 27
    i = int(_sys.argv[2]) if len(_sys.argv) > 2 else 3
    j = int(_sys.argv[3]) if len(_sys.argv) > 3 else 1
    r = find_first_binary_exhaustive(ell, i, j)
    print(f"ell={ell} i={i} j={j}: found={r['found']} scanned={r['scanned']}/"
          f"{r['candidate_space']} in {r['seconds']:.4f}s")
