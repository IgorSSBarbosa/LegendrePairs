"""rle.py — Phase 1: RLE bijection + canonicalization (thin reuse of lp_rle).

A row-sum-+1 sequence ``A in {-1,+1}^ell`` maps bijectively to a run array of
even length ``2k`` (even indices = +1 runs, odd = -1 runs), anchored at the start
of a +1 run, with ``sum(+runs) = (ell+1)//2`` and ``sum(-runs) = (ell-1)//2``
(PLAN §5, Phase 1). The rotation-class canonical form is the lexicographically
least rotation of the length-``2k`` pair array (Booth, O(k)).

All heavy lifting is REUSED from ``lp_rle.runs``; this module only re-exposes it
under the PLAN names and adds ``run_transfer_moves`` (a same-parity unit-transfer
neighbourhood for later local search — not used by the exhaustive path).
"""
from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

import numpy as np

from lp_rle.runs import (
    seq_to_runs as _seq_to_runs,
    runs_to_seq as _runs_to_seq,
    canonical as _canonical,
    is_canonical as _is_canonical,
    rotate_runs as _rotate_runs,
    reverse_runs as _reverse_runs,
)


def rle_encode(A) -> List[int]:
    """pm1 sequence (row sum +1) -> run array as a Python list of ints (2k runs).

    Anchors at the first +1 run; ``rle_decode(rle_encode(A))`` is the anchored
    representative of ``A`` (a cyclic rotation of ``A``), and the run array itself
    round-trips exactly: ``rle_encode(rle_decode(r)) == r``.
    """
    return _seq_to_runs(np.asarray(A, dtype=np.int8)).tolist()


def rle_decode(runs: Sequence[int]) -> np.ndarray:
    """Run array -> anchored pm1 sequence ``int8[ell]`` (starts at a +1 run)."""
    return _runs_to_seq(np.asarray(runs, dtype=np.int64))


def booth_canonical(runs: Sequence[int]) -> Tuple[int, ...]:
    """Lexicographically least pair-array rotation (Booth, O(k)) as a tuple.

    Two run arrays share this form iff they are rotations of each other, i.e. iff
    the underlying sequences are cyclic shifts (the rotation action is free at odd
    ``ell``, so orbits all have size ``ell``).
    """
    return tuple(int(x) for x in _canonical(np.asarray(runs, dtype=np.int64)))


def is_canonical(runs: Sequence[int]) -> bool:
    """True iff ``runs`` equals its own least pair-array rotation."""
    return bool(_is_canonical(np.asarray(runs, dtype=np.int64)))


def rotate_runs(runs: Sequence[int], t: int) -> np.ndarray:
    """Rotate the pair array by ``t`` pairs (a cyclic shift of the sequence)."""
    return _rotate_runs(np.asarray(runs, dtype=np.int64), int(t))


def reverse_runs(runs: Sequence[int]) -> np.ndarray:
    """Reversal ``A[i] -> A[-i mod ell]`` as a re-anchored run array."""
    return _reverse_runs(np.asarray(runs, dtype=np.int64))


def run_transfer_moves(runs: Sequence[int]) -> Iterator[np.ndarray]:
    """Yield run arrays obtained by moving one unit between two same-parity runs.

    Moving a unit from run ``i`` to run ``j`` with ``i, j`` the same parity keeps
    ``sum(+runs)`` and ``sum(-runs)`` (hence ``ell`` and row sum +1) fixed, and
    all run lengths ``>= 1``. This is a local-search neighbourhood for later
    phases; it is NOT part of the exhaustive enumeration.
    """
    r = np.asarray(runs, dtype=np.int64)
    n = r.shape[0]
    for parity in (0, 1):  # 0 -> +1 runs, 1 -> -1 runs
        idxs = [i for i in range(n) if i % 2 == parity]
        for i in idxs:
            if r[i] < 2:  # cannot donate below length 1
                continue
            for j in idxs:
                if j == i:
                    continue
                nr = r.copy()
                nr[i] -= 1
                nr[j] += 1
                yield nr
