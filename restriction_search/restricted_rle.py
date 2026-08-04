"""restricted_rle.py — RLE-representation restricted exhaustive search.

runs_to_seq(r) always emits a sequence that STARTS at the boundary of the
first (+1) run and ENDS at the boundary of the last (-1) run (that's the
anchoring convention of rle_experiment/lp_rle/runs.py). So requiring the
first run r[0] >= i and the last run r[-1] >= j produces EXACTLY the same
literal-array space as restricted_binary.iter_restricted_candidates(ell,i,j)
-- just enumerated via a different parametrization (run-length compositions
instead of position combinations). This is verified in
canonical_orbits' restriction_exact ground truth cross-check (see
test_restricted_rle.py / benchmark.py).

Structural caveat: runs_to_seq ALWAYS has a boundary at position 0 (v[0]=+1)
and position ell-1 (v[ell-1]=-1), so this parametrization cannot represent
i=0 or j=0 (no restriction on that side) at all -- it inherently requires
i>=1 and j>=1. That's fine for this project (the whole premise is that real
canonical LPs always start with '++' and end with '-'), but callers must not
ask for i=0 or j=0 here.
"""
from __future__ import annotations

import os
import sys
from typing import Iterator

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "rle_experiment"))
from lp_rle.runs import runs_to_seq  # noqa: E402

from common import find_first_pair, find_first_pair_random_order, find_first_pair_multi_seed  # noqa: E402


def compositions(total: int, k: int, min_first: int = 1) -> Iterator[tuple[int, ...]]:
    """k-tuples of positive ints summing to total; first part >= min_first."""
    if k == 1:
        if total >= min_first:
            yield (total,)
        return
    for first in range(min_first, total - (k - 1) + 1):
        for rest in compositions(total - first, k - 1, min_first=1):
            yield (first,) + rest


def compositions_last(total: int, k: int, min_last: int = 1) -> Iterator[tuple[int, ...]]:
    """k-tuples of positive ints summing to total; LAST part >= min_last."""
    for c in compositions(total, k, min_first=min_last):
        yield tuple(reversed(c))


def iter_restricted_run_seqs(ell: int, i: int, j: int) -> Iterator[tuple[int, ...]]:
    """Restricted candidates via run-length composition (i>=1, j>=1 required)."""
    if i < 1 or j < 1:
        raise ValueError(f"RLE restriction requires i>=1 and j>=1, got i={i}, j={j}")
    P = (ell + 1) // 2
    M = (ell - 1) // 2
    kmax = min(P, M)
    for k in range(1, kmax + 1):
        for p_comp in compositions(P, k, min_first=i):
            for m_comp in compositions_last(M, k, min_last=j):
                r = np.empty(2 * k, dtype=np.int64)
                r[0::2] = p_comp
                r[1::2] = m_comp
                yield tuple(int(x) for x in runs_to_seq(r))


def find_first_rle_exhaustive(ell: int, i: int, j: int, seed: int | None = None) -> dict:
    """Time-to-first-hit for the restricted RLE brute-force scan (see
    find_first_binary_exhaustive's docstring re: seed=None vs seed=int)."""
    if seed is None:
        return find_first_pair(iter_restricted_run_seqs(ell, i, j), ell)
    return find_first_pair_random_order(iter_restricted_run_seqs(ell, i, j), ell, seed)


def find_first_rle_exhaustive_multi_seed(ell: int, i: int, j: int, seeds: list[int]) -> list[dict]:
    """Materializes the restricted run-array candidate list once (the
    expensive part -- compositions() is a pure-Python recursive generator),
    then runs one seeded-shuffle scan per seed."""
    pool = list(iter_restricted_run_seqs(ell, i, j))
    return find_first_pair_multi_seed(pool, ell, seeds)


if __name__ == "__main__":
    ell = int(sys.argv[1]) if len(sys.argv) > 1 else 27
    i = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    j = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    r = find_first_rle_exhaustive(ell, i, j)
    print(f"ell={ell} i={i} j={j}: found={r['found']} scanned={r['scanned']} "
          f"in {r['seconds']:.4f}s")
