import numpy as np

from lp_rle.conventions import half_len
from lp_rle.paf import paf_naive, paf_fft, paf_runs, psd_from_paf
from lp_rle.runs import runs_to_seq, seq_to_runs
from lp_rle.walk import random_seq

ODD_ALL = list(range(3, 26, 2))


def test_three_way_cross_check():
    rng = np.random.default_rng(1)
    for L in ODD_ALL:
        for _ in range(40):
            v = random_seq(L, rng)
            r = seq_to_runs(v)
            va = runs_to_seq(r)  # re-anchored; PAF is rotation invariant
            pn = paf_naive(v)
            pf = paf_fft(v)
            pr = paf_runs(r)
            assert np.array_equal(pn, pf)
            assert np.array_equal(paf_naive(va), pr)


def test_paf_mod4():
    rng = np.random.default_rng(2)
    for L in ODD_ALL:
        for _ in range(20):
            v = random_seq(L, rng)
            pn = paf_naive(v)
            assert np.all((pn - L) % 4 == 0)


def test_paf0_and_sum():
    rng = np.random.default_rng(3)
    for L in ODD_ALL:
        v = random_seq(L, rng)
        pn = paf_naive(v)
        # PAF(0)=L and sum_{s=0}^{L-1} PAF = (sum v)^2 = 1  =>  2*sum_half = 1 - L
        assert 2 * int(pn.sum()) == 1 - L


def test_psd_reconstruction_nonneg():
    rng = np.random.default_rng(4)
    for L in ODD_ALL:
        v = random_seq(L, rng)
        psd = psd_from_paf(paf_naive(v), L)
        assert np.all(psd >= -1e-6)  # PSD is |vhat|^2 >= 0
