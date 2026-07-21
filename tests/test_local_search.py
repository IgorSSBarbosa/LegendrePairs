"""Tests for the incremental local search (``src/local_search.py``).

Pytest-free (plain asserts + ``__main__`` runner).  The load-bearing tests check
that the O(ell) incremental PAF/E update agrees exactly with the FFT
``paf_vector``/``objective`` -- if that drifts, every strategy is meaningless.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import local_search as ls  # noqa: E402
from legendre import is_legendre_pair  # noqa: E402


# --------------------------------------------------------------------------- #
# incremental PAF / E match the FFT baseline
# --------------------------------------------------------------------------- #
def test_dpaf_matches_paf_vector():
    rng = np.random.default_rng(0)
    for ell in (5, 7, 9, 13, 15, 21):
        half = (ell - 1) // 2
        seq = ls.random_sum_one(ell, rng)
        pa = ls.paf_vector(seq, half)
        for _ in range(30):
            i, j = ls._pick_swap(seq, rng)
            d = ls._dpaf(seq, i, j, half)
            seq[i], seq[j] = -1, 1
            pa = pa + d
            assert np.array_equal(pa, ls.paf_vector(seq, half)), f"drift ell={ell}"


def test_incremental_energy_matches_objective():
    rng = np.random.default_rng(1)
    for ell in (7, 11, 13, 17):
        half = (ell - 1) // 2
        a = ls.random_sum_one(ell, rng)
        b = ls.random_sum_one(ell, rng)
        pa, pb = ls.paf_vector(a, half), ls.paf_vector(b, half)
        r = pa + pb + 2
        E = int(np.dot(r, r))
        assert E == ls.objective(pa, pb)
        for _ in range(60):
            seq = a if rng.random() < 0.5 else b
            i, j = ls._pick_swap(seq, rng)
            d = ls._dpaf(seq, i, j, half)
            dE = int(2 * np.dot(r, d) + np.dot(d, d))
            seq[i], seq[j] = -1, 1
            r = r + d
            E += dE
            # recompute from scratch and compare
            E_ref = ls.objective(ls.paf_vector(a, half), ls.paf_vector(b, half))
            assert E == E_ref, f"dE drift ell={ell}: {E} != {E_ref}"
            assert E >= 0


# --------------------------------------------------------------------------- #
# each strategy solves small ell
# --------------------------------------------------------------------------- #
def test_strategies_solve_small_ell():
    for strat in ("greedy", "sideways", "anneal", "basinhop", "threshold"):
        for ell in (5, 7, 9, 11, 13):
            res = ls.search(ell, strategy=strat, restarts=40, steps=8000, seed=0)
            assert res["solved"], f"{strat} failed ell={ell}"
            ok, _ = is_legendre_pair(res["A"], res["B"])
            assert ok


def test_auto_t0_solves():
    # t0=None triggers auto-calibration; must still solve.
    for ell in (7, 11, 13):
        res = ls.search(ell, strategy="anneal", restarts=40, steps=8000,
                        t0=None, seed=0)
        assert res["solved"], f"auto-t0 anneal failed ell={ell}"
        assert is_legendre_pair(res["A"], res["B"])[0]


def test_rejects_even_ell():
    try:
        ls.search(8, restarts=1, steps=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError for even ell")


def test_reproducible_same_seed():
    a = ls.search(13, strategy="anneal", restarts=20, steps=5000, seed=7)
    b = ls.search(13, strategy="anneal", restarts=20, steps=5000, seed=7)
    assert a["solved"] == b["solved"]
    assert a["A"] == b["A"] and a["B"] == b["B"]
    assert a["restarts_used"] == b["restarts_used"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
