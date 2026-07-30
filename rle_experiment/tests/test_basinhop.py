import numpy as np
import pytest

from lp_rle.paf import paf_naive
from lp_rle.walk import Meta, BasinHopper
from lp_rle.baseline import make_moveset


def _recompute_f(u, v):
    e = paf_naive(u) + paf_naive(v) + 2
    return int(np.dot(e, e))


@pytest.mark.parametrize("rep", ["rle", "binary"])
def test_basinhop_bookkeeping_matches_recompute(rep):
    # the incremental (e, f) state must equal a from-scratch recompute after a run
    rng = np.random.default_rng(0)
    w = BasinHopper(13, make_moveset(rep), Meta(kind="sa"), rng)
    w.run(max_steps=3000)
    assert w.f == _recompute_f(w.u, w.v)
    assert np.array_equal(w.e, paf_naive(w.u) + paf_naive(w.v) + 2)


@pytest.mark.parametrize("rep", ["rle", "binary"])
def test_basinhop_finds_lp_small_L(rep):
    # over a handful of seeds, basin-hopping should land a real LP at L=13
    L = 13
    found = None
    for s in range(12):
        rng = np.random.default_rng(s)
        w = BasinHopper(L, make_moveset(rep), Meta(kind="sa"), rng)
        st = w.run(max_steps=20000)
        if st.found:
            found = w.solution()
            break
    assert found is not None, f"basinhop/{rep} found no LP at L={L}"
    u, v = found
    assert np.all(paf_naive(u) + paf_naive(v) == -2)


@pytest.mark.parametrize("rep", ["rle", "binary"])
@pytest.mark.parametrize("mode", ["low", "equal", "both"])
def test_basinhop_energy_reg_bookkeeping(rep, mode):
    # with the energy regularizer on, incremental half-energies E(u),E(v) and f
    # must all match a from-scratch recompute after a run (incl. kick/restore).
    rng = np.random.default_rng(4)
    meta = Meta(kind="sa", energy_reg=0.05, reg_mode=mode, reg_cooling=0.999)
    w = BasinHopper(15, make_moveset(rep), meta, rng)
    w.run(max_steps=5000)
    assert w.Eu == int(np.dot(w.pu, w.pu))
    assert w.Ev == int(np.dot(w.pv, w.pv))
    assert w.f == _recompute_f(w.u, w.v)


def test_basinhop_reg_off_matches_baseline():
    # energy_reg=0.0 must reproduce the un-regularized hopper bit-for-bit.
    a = BasinHopper(15, make_moveset("rle"), Meta(kind="sa"), np.random.default_rng(5))
    a.run(max_steps=4000)
    b = BasinHopper(15, make_moveset("rle"), Meta(kind="sa", energy_reg=0.0),
                    np.random.default_rng(5))
    b.run(max_steps=4000)
    assert np.array_equal(a.u, b.u) and np.array_equal(a.v, b.v) and a.f == b.f


def test_basinhop_reports_hop_stats():
    rng = np.random.default_rng(1)
    w = BasinHopper(13, make_moveset("rle"), Meta(kind="sa"), rng)
    st = w.run(max_steps=5000)
    d = st.as_dict()
    # macro_* now count outer hops; there should be at least one hop unless solved
    assert d["macro_prop"] >= 1 or st.found
    assert 0.0 <= d["macro_acc_rate"] <= 1.0
