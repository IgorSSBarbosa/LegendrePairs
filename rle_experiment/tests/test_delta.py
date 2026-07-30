import numpy as np

from lp_rle.paf import paf_naive
from lp_rle.delta import (
    delta_paf_2flip, delta_paf_general, delta_f, assert_delta_valid,
)
from lp_rle.walk import random_seq, RLEMoveSet, BinaryMoveSet

ODD_ALL = list(range(3, 26, 2))


def test_2flip_vs_recompute():
    rng = np.random.default_rng(10)
    for L in ODD_ALL:
        for _ in range(50):
            v = random_seq(L, rng)
            pos = np.where(v == 1)[0]
            neg = np.where(v == -1)[0]
            a = int(pos[rng.integers(pos.shape[0])])
            b = int(neg[rng.integers(neg.shape[0])])
            v2 = v.copy()
            v2[a] = -1
            v2[b] = 1
            dp = delta_paf_2flip(v, a, b, L)
            assert np.array_equal(dp, paf_naive(v2) - paf_naive(v))
            assert_delta_valid(dp)


def test_general_matches_2flip_and_recompute():
    rng = np.random.default_rng(11)
    for L in ODD_ALL:
        moves = RLEMoveSet()
        for _ in range(60):
            v = random_seq(L, rng)
            mv = moves.propose(v, rng)
            if mv is None:
                continue
            dp = delta_paf_general(v, mv.J, mv.delta, L)
            assert np.array_equal(dp, paf_naive(mv.v_new) - paf_naive(v))
            assert_delta_valid(dp)


def test_delta_f_matches_full():
    rng = np.random.default_rng(12)
    L = 15
    for _ in range(200):
        u = random_seq(L, rng)
        v = random_seq(L, rng)
        e = paf_naive(u) + paf_naive(v) + 2
        f = int(np.dot(e, e))
        # flip in v
        pos = np.where(v == 1)[0]
        neg = np.where(v == -1)[0]
        a = int(pos[rng.integers(pos.shape[0])])
        b = int(neg[rng.integers(neg.shape[0])])
        v2 = v.copy(); v2[a] = -1; v2[b] = 1
        dp = delta_paf_2flip(v, a, b, L)
        df = int(delta_f(e, dp))
        e2 = paf_naive(u) + paf_naive(v2) + 2
        f2 = int(np.dot(e2, e2))
        assert f + df == f2
