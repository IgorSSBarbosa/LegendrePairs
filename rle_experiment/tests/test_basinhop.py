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


def test_basinhop_reports_hop_stats():
    rng = np.random.default_rng(1)
    w = BasinHopper(13, make_moveset("rle"), Meta(kind="sa"), rng)
    st = w.run(max_steps=5000)
    d = st.as_dict()
    # macro_* now count outer hops; there should be at least one hop unless solved
    assert d["macro_prop"] >= 1 or st.found
    assert 0.0 <= d["macro_acc_rate"] <= 1.0
