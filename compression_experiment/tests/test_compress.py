"""test_compress.py — Phase 3 spec (PLAN §5): compression + compressed sieves.

Anchors / VERIFIED identities:
  * spectral sub-lattice:  DFT_m(cA)(s) = Ahat(n*s),  PSD^m_cA(s) = PSD_A(n*s);
  * compressed PAF:        P_cA(s) = sum_{t == s (mod m)} P_A(t);
  * compressed sieves (PAF exact-int, PSD float) agree on all inputs;
  * a genuine LP passes both sieves;
  * incomparable_divisors returns the maximal proper-divisor antichain;
  * ANCHOR: ell=9, m=3 compressed survivors == 72 (VERIFIED sieves, stub OFF).
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from lp_compress.core import paf_int, psd
from lp_compress.compress import (
    compress, compressed_paf, compressed_psd, incomparable_divisors,
    compressed_paf_sieve, compressed_psd_sieve, cascade, cascade_pairs,
    enumerate_compressed,
)
from lp_compress.sieve import compressed_integer_sieve

# genuine length-9 Legendre pair (from the exhaustive lp_rle pipeline)
LP9_U = np.array([1, -1, 1, 1, 1, 1, -1, -1, -1], dtype=np.int8)
LP9_V = np.array([1, -1, 1, 1, -1, -1, 1, 1, -1], dtype=np.int8)

CASES = [(9, 3), (15, 3), (15, 5), (21, 3), (21, 7), (25, 5), (27, 9)]


def _rand_pm1(ell, rng):
    return (rng.integers(0, 2, ell) * 2 - 1).astype(np.int8)


@pytest.mark.parametrize("ell,m", CASES)
def test_spectral_sublattice_identity(ell, m):
    n = ell // m
    rng = np.random.default_rng(ell * 10 + m)
    for _ in range(40):
        A = _rand_pm1(ell, rng)
        cA = compress(A, m)
        Fc = np.fft.fft(cA.astype(float))
        FA = np.fft.fft(A.astype(float))
        for s in range(m):
            assert abs(Fc[s] - FA[(n * s) % ell]) < 1e-8
        pm = compressed_psd(cA)
        pA = psd(A)
        for s in range(m):
            assert abs(pm[s] - pA[(n * s) % ell]) < 1e-6


@pytest.mark.parametrize("ell,m", CASES)
def test_compressed_paf_identity(ell, m):
    rng = np.random.default_rng(ell * 7 + m)
    for _ in range(40):
        A = _rand_pm1(ell, rng)
        pc = compressed_paf(compress(A, m))
        PA = paf_int(A)
        for s in range(1, m):
            expect = sum(int(PA[t]) for t in range(ell) if t % m == s % m)
            assert pc[s - 1] == expect


def test_incomparable_divisors():
    assert incomparable_divisors(7) == []          # prime: no compression
    assert incomparable_divisors(9) == [3]
    assert incomparable_divisors(15) == [3, 5]
    assert incomparable_divisors(25) == [5]
    assert incomparable_divisors(27) == [9]
    assert incomparable_divisors(45) == [9, 15]
    assert incomparable_divisors(105) == [15, 21, 35]


@pytest.mark.parametrize("ell,m", CASES)
def test_sieves_agree_on_reachable_pairs(ell, m):
    rng = np.random.default_rng(100 + ell + m)
    for _ in range(300):
        cA = compress(_rand_pm1(ell, rng), m)
        cB = compress(_rand_pm1(ell, rng), m)
        assert compressed_paf_sieve(cA, cB, ell, m) == compressed_psd_sieve(cA, cB, ell, m)


def test_sieves_agree_on_full_space_ell9():
    ell, m, n = 9, 3, 3
    vecs = [np.array(c) for c in itertools.product(range(-n, n + 1, 2), repeat=m)]
    for cA in vecs:
        for cB in vecs:
            assert compressed_paf_sieve(cA, cB, ell, m) == compressed_psd_sieve(cA, cB, ell, m)


def test_genuine_lp_passes_sieves():
    ell, m, n = 9, 3, 3
    cU, cV = compress(LP9_U, m), compress(LP9_V, m)
    assert compressed_paf_sieve(cU, cV, ell, m)
    assert compressed_psd_sieve(cU, cV, ell, m)
    # explicit VERIFIED values
    assert np.all(compressed_paf(cU) + compressed_paf(cV) == -2 * n)            # s != 0
    assert int(cU @ cU) + int(cV @ cV) == 2 * ell - 2 * n + 2                   # t = 0 energy


def test_anchor_ell9_compressed_survivors_72():
    res = cascade(9)
    assert res["m"] == 3 and res["n"] == 3
    assert res["n_vectors"] == 64
    assert res["n_survivor_pairs"] == 72
    # the streamed pairs agree with the count and each truly passes the sieve
    pairs = list(cascade_pairs(9))
    assert len(pairs) == 72
    assert all(compressed_paf_sieve(a, b, 9, 3) for a, b in pairs)


@pytest.mark.xfail(reason="compressed_integer_sieve is a documented pass-through STUB "
                          "(TODO cite); it must one day cut compressed pairs on its own.")
def test_integer_sieve_is_a_real_filter():
    """The STUB must eventually REJECT at least one sieve-passing pair (a real
    Diophantine cut). While it is a pass-through it accepts everything -> xfail."""
    ell, m = 9, 3
    passing = [(a, b) for a, b in cascade_pairs(ell, m)]
    rejected = [(a, b) for a, b in passing if not compressed_integer_sieve(a, b, ell, m)]
    assert rejected, "integer sieve rejected nothing (still a pass-through stub)"
