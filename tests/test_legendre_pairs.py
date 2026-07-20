"""Tests for the FFT-based Legendre-pair toolkit (``src/legendre_pairs.py``).

Written pytest-free (plain ``assert`` + a ``__main__`` runner) to match the
sibling ``test_legendre.py`` and run without extra deps; pytest still discovers
every ``test_*`` function.
"""

import os
import sys
import tempfile
from contextlib import contextmanager

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import legendre_pairs as lp  # noqa: E402


@contextmanager
def assert_raises(exc):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} was not raised")


# --------------------------------------------------------------------------- #
# PAF: FFT vs naive, index-0 identity
# --------------------------------------------------------------------------- #
def test_paf_matches_naive():
    for ell in (7, 11, 15, 21):
        rng = np.random.default_rng(ell)
        u = rng.choice([-1, 1], size=ell).astype(float)
        assert np.allclose(lp.paf(u), lp.paf_naive(u), atol=1e-9)


def test_paf_index0_is_ell():
    for ell in (7, 11, 15, 21):
        rng = np.random.default_rng(ell + 1)
        u = rng.choice([-1, 1], size=ell).astype(float)
        assert abs(lp.paf(u)[0] - ell) < 1e-9


# --------------------------------------------------------------------------- #
# PSD: Parseval, agreement with |fft|^2
# --------------------------------------------------------------------------- #
def test_parseval():
    for ell in (7, 11, 15, 21):
        rng = np.random.default_rng(ell + 2)
        u = rng.choice([-1, 1], size=ell).astype(float)
        # For +-1 vectors, sum_k |u_hat(k)|^2 = ell^2.
        assert abs(lp.psd_values(u).sum() - ell * ell) < 1e-6


def test_psd_values_matches_full_fft():
    for ell in (7, 11, 15, 21, 8, 16):        # include even ell for the mirror
        rng = np.random.default_rng(ell + 3)
        u = rng.choice([-1, 1], size=ell).astype(float)
        ref = np.abs(np.fft.fft(u)) ** 2
        assert np.allclose(lp.psd_values(u), ref, atol=1e-9)


# --------------------------------------------------------------------------- #
# Pair verification on known LPs and negative cases
# --------------------------------------------------------------------------- #
def test_known_lp_ell3():
    u = np.array([1, -1, -1])
    assert lp.is_legendre_pair(u, u, 3)


def test_known_lp_ell7():
    u = np.array([-1, -1, 1, -1, 1, 1, 1])    # from results/found_pairs.md
    assert u.sum() == 1
    assert lp.is_legendre_pair(u, u, 7)


def test_non_pair_all_ones():
    u = np.array([1, 1, 1])
    assert not lp.is_legendre_pair(u, u, 3)


def test_passes_psd_test_normalization():
    assert not lp.passes_psd_test(np.array([1, 1, 1]), 3)     # sum != 1
    assert lp.passes_psd_test(np.array([-1, 1, 1]), 3)        # sum == 1


# --------------------------------------------------------------------------- #
# Cyclotomic cosets
# --------------------------------------------------------------------------- #
def test_cyclotomic_cosets_ell7_H124():
    assert lp.cyclotomic_cosets(7, [1, 2, 4]) == [[0], [1, 2, 4], [3, 5, 6]]


def test_cyclotomic_cosets_trivial_H():
    assert lp.cyclotomic_cosets(7, [1]) == [[i] for i in range(7)]


def test_candidates_from_cosets_weight_and_constancy():
    ell, H = 7, [1, 2, 4]
    cosets = lp.cyclotomic_cosets(ell, H)
    tw = (ell - 1) // 2
    got = list(lp.candidates_from_cosets(cosets, tw))
    assert got
    for u in got:
        assert np.count_nonzero(u == -1) == tw
        for c in cosets:                       # constant on each coset
            assert len({int(u[i]) for i in c}) == 1


# --------------------------------------------------------------------------- #
# Compression: subsampling identity
# --------------------------------------------------------------------------- #
def test_compress_subsampling_identity():
    for m, n in ((3, 5), (5, 3), (3, 7), (7, 3)):
        ell = m * n
        rng = np.random.default_rng(ell + 7)
        u = rng.choice([-1, 1], size=ell).astype(float)
        c_hat = np.fft.fft(lp.compress(u, m))
        u_hat = np.fft.fft(u)
        for kappa in range(m):                 # c_hat(kappa) == u_hat(kappa*n)
            assert abs(c_hat[kappa] - u_hat[(kappa * n) % ell]) < 1e-9


def test_compress_requires_divisor():
    with assert_raises(ValueError):
        lp.compress(np.ones(7), 3)


def test_compressed_psd_bound_is_stub():
    with assert_raises(NotImplementedError):
        lp.compressed_psd_bound(15, 3)


# --------------------------------------------------------------------------- #
# Search driver
# --------------------------------------------------------------------------- #
def test_search_ell7_finds_pairs_serial():
    hits = lp.search(7, verbose=False)
    assert hits
    for u, v in hits:
        assert lp.is_legendre_pair(u, v, 7)


def test_search_ell7_parallel_matches_serial():
    a = lp.search(7, verbose=False)
    b = lp.search(7, workers=2, verbose=False)
    assert len(a) == len(b)


def test_search_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "ck.json")
        first = lp.search(7, checkpoint=ckpt, verbose=False)
        assert os.path.exists(ckpt)
        again = lp.search(7, checkpoint=ckpt, verbose=False)  # resume
        assert len(again) == len(first)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
