import numpy as np

from lp_rle.conventions import P_of
from lp_rle.paf import paf_naive
from lp_rle.walk import (
    random_seq, RLEMoveSet, BinaryMoveSet, JointWalker, TwoStageSearch, Meta,
)


def test_moves_preserve_rowsum():
    rng = np.random.default_rng(20)
    for L in [13, 15, 17]:
        for MS in (RLEMoveSet(), BinaryMoveSet()):
            for _ in range(200):
                v = random_seq(L, rng)
                mv = MS.propose(v, rng)
                if mv is None:
                    continue
                assert int(mv.v_new.sum()) == 1
                assert int((mv.v_new == 1).sum()) == P_of(L)
                assert np.array_equal(np.where(v != mv.v_new)[0], np.sort(mv.J))


def test_joint_residual_stays_exact():
    # incremental e/f must match a from-scratch recompute after every accept
    rng = np.random.default_rng(21)
    L = 15
    w = JointWalker(L, RLEMoveSet(), Meta(kind="sa", T0=5.0), rng)
    for _ in range(3000):
        before = (w.u.copy(), w.v.copy())
        w.run(1)  # one step
        e_check = paf_naive(w.u) + paf_naive(w.v) + 2
        assert np.array_equal(w.e, e_check)
        assert w.f == int(np.dot(e_check, e_check))
        if w.f == 0:
            break


def test_joint_finds_lp_small_L():
    # L=13 has 4 classes and a large basin; SA should hit f=0 within budget.
    rng = np.random.default_rng(7)
    found = False
    for seed in range(30):
        w = JointWalker(13, RLEMoveSet(), Meta(kind="sa", T0=6.0, cooling=0.9997),
                        np.random.default_rng(seed))
        st = w.run(30000)
        if st.found:
            u, v = w.solution()
            assert np.all(paf_naive(u) + paf_naive(v) == -2)
            found = True
            break
    assert found, "joint SA failed to find any LP at L=13 in budget"
