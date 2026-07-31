"""test_rle.py — Phase 1 spec (PLAN §5). Tests are the spec.

  * round-trip bijection on ALL row-sum-+1 sequences for ell <= 15;
  * canonical form invariant under rotation;
  * two sequences share the canonical form iff related by a rotation;
  * DoD: #distinct canonical run arrays == rotation-class (necklace) count,
    which for the free Z_ell action equals C(ell,(ell-1)//2)/ell.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pytest

from lp_compress.core import paf_int
from lp_compress.rle import (
    rle_encode, rle_decode, booth_canonical, is_canonical,
    rotate_runs, run_transfer_moves,
)

SMALL = [3, 5, 7, 9, 11, 13, 15]


def all_rowsum1(ell):
    """Yield every A in {+-1}^ell with exactly (ell+1)//2 ones (row sum +1)."""
    p = (ell + 1) // 2
    base = -np.ones(ell, dtype=np.int8)
    for ones in combinations(range(ell), p):
        A = base.copy()
        A[list(ones)] = 1
        yield A


def _is_rotation(a: np.ndarray, b: np.ndarray) -> bool:
    ell = a.shape[0]
    return any(np.array_equal(np.roll(a, t), b) for t in range(ell))


@pytest.mark.parametrize("ell", SMALL)
def test_encode_decode_roundtrip(ell):
    """decode(encode(A)) is a rotation of A with identical PAF; run arrays
    round-trip exactly (encode(decode(r)) == r)."""
    for A in all_rowsum1(ell):
        r = rle_encode(A)
        assert len(r) % 2 == 0
        assert sum(r[0::2]) == (ell + 1) // 2   # +runs
        assert sum(r[1::2]) == (ell - 1) // 2   # -runs
        B = rle_decode(r)
        assert _is_rotation(A, B)
        assert np.array_equal(paf_int(A), paf_int(B))
        # exact run-array round trip
        assert rle_encode(B) == r


@pytest.mark.parametrize("ell", SMALL)
def test_canonical_invariant_under_rotation(ell):
    for A in all_rowsum1(ell):
        r = rle_encode(A)
        k = len(r) // 2
        canon = booth_canonical(r)
        for t in range(k):
            assert booth_canonical(rotate_runs(r, t)) == canon


@pytest.mark.parametrize("ell", [3, 5, 7, 9, 11])
def test_same_canonical_iff_rotation(ell):
    seqs = list(all_rowsum1(ell))
    canons = [booth_canonical(rle_encode(A)) for A in seqs]
    for i in range(len(seqs)):
        for j in range(i, len(seqs)):
            same_canon = canons[i] == canons[j]
            rotated = _is_rotation(seqs[i], seqs[j])
            assert same_canon == rotated


@pytest.mark.parametrize("ell", SMALL)
def test_canonical_count_equals_necklace_count(ell):
    canon_set = {booth_canonical(rle_encode(A)) for A in all_rowsum1(ell)}
    expected = comb(ell, (ell - 1) // 2) // ell  # free Z_ell action => orbits size ell
    assert len(canon_set) == expected
    # every canonical form is genuinely canonical
    assert all(is_canonical(list(c)) for c in canon_set)


@pytest.mark.parametrize("ell", [7, 9, 11])
def test_run_transfer_moves_preserve_invariants(ell):
    # pick a multi-run sequence (k >= 2) so a same-parity transfer exists; the
    # degenerate single-block [+^p,-^m] (k=1) legitimately has no such move.
    A = next(a for a in all_rowsum1(ell) if len(rle_encode(a)) >= 4)
    r = rle_encode(A)
    moves = list(run_transfer_moves(r))
    assert moves, "expected at least one transfer move"
    for nr in moves:
        assert np.all(nr >= 1)
        assert int(nr.sum()) == ell                 # ell preserved
        assert int(nr[0::2].sum()) == (ell + 1) // 2  # +runs preserved
        assert int(nr[1::2].sum()) == (ell - 1) // 2  # -runs preserved
        v = rle_decode(nr)
        assert int(v.sum()) == 1                     # still row sum +1
