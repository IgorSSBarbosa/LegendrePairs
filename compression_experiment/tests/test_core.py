"""test_core.py — Phase 0 spec (PLAN §5). Tests are the spec.

Anchors:
  * a hardcoded known ell=7 Legendre pair is recognized;
  * exact-integer PAF equals the FFT PAF (rounded) for random small ell;
  * Parseval: sum_k PSD_A(k) == ell^2 for A in {+-1}^ell;
  * PSD pairing: PSD_A(s)+PSD_B(s) == 2*ell+2 for all s != 0 on the known LP.
"""
from __future__ import annotations

import numpy as np
import pytest

from lp_compress.core import paf_int, paf_fft, psd, is_legendre_pair, paf_half

ODD = list(range(3, 26, 2))

# A genuine length-7 Legendre pair (extracted from the exhaustive lp_rle pipeline).
LP7_U = np.array([1, -1, 1, 1, 1, -1, -1], dtype=np.int8)
LP7_V = np.array([1, -1, 1, 1, 1, -1, -1], dtype=np.int8)


def _rand_pm1(ell, rng):
    return (rng.integers(0, 2, size=ell) * 2 - 1).astype(np.int8)


def test_known_lp7_is_recognized():
    assert is_legendre_pair(LP7_U, LP7_V)
    # PAF half must be all -1 (so P_u+P_v = -2 on every nonzero shift)
    assert paf_half(LP7_U).tolist() == [-1, -1, -1]


def test_paf_int_zero_and_symmetry():
    rng = np.random.default_rng(0)
    for ell in ODD:
        A = _rand_pm1(ell, rng)
        p = paf_int(A)
        assert p.shape == (ell,)
        assert p[0] == ell                       # P_A(0) = ell
        assert np.array_equal(p[1:], p[1:][::-1])  # P_A(s) = P_A(ell-s)


def test_paf_int_matches_fft():
    rng = np.random.default_rng(1)
    for ell in ODD:
        for _ in range(20):
            A = _rand_pm1(ell, rng)
            assert np.array_equal(paf_int(A), paf_fft(A))


def test_parseval_psd_sums_to_ell_squared():
    rng = np.random.default_rng(2)
    for ell in ODD:
        for _ in range(20):
            A = _rand_pm1(ell, rng)
            assert abs(float(psd(A).sum()) - ell * ell) < 1e-6


def test_psd_pairing_on_known_lp():
    ell = LP7_U.shape[0]
    pu = psd(LP7_U)
    pv = psd(LP7_V)
    total = pu + pv
    assert np.all(np.abs(total[1:] - (2 * ell + 2)) < 1e-6)
    assert np.all(pu[1:] <= 2 * ell + 2 + 1e-6)  # single-seq PSD bound


@pytest.mark.parametrize("ell", ODD)
def test_paf_int_matches_naive_definition(ell):
    """Cross-check the full PAF against the textbook O(ell^2) definition."""
    rng = np.random.default_rng(100 + ell)
    A = _rand_pm1(ell, rng)
    p = paf_int(A)
    Ai = A.astype(np.int64)
    for s in range(ell):
        expect = int(sum(Ai[i] * Ai[(i + s) % ell] for i in range(ell)))
        assert p[s] == expect
