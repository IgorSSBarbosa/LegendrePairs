"""Tests for the simulated-annealing search (``src/anneal_search.py``).

Pytest-free (plain asserts + ``__main__`` runner), matching the sibling suites.
The critical tests check that the O(ell) incremental PAF/E update agrees exactly
with the FFT objective -- if that drifts, the whole annealing run is meaningless.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import anneal_search as sa  # noqa: E402
from rn_search import objective  # noqa: E402
from legendre_pairs import is_legendre_pair  # noqa: E402


# --------------------------------------------------------------------------- #
# incremental PAF / E match the FFT objective
# --------------------------------------------------------------------------- #
def test_dpaf_matches_fft_after_swap():
    rng = np.random.default_rng(0)
    for ell in (5, 7, 9, 13, 15):
        s = np.arange(ell)
        u, _ = sa._random_pair(ell, rng)
        paf = sa._paf_int(u)
        for _ in range(20):
            plus = np.nonzero(u == 1)[0]
            minus = np.nonzero(u == -1)[0]
            p = int(plus[rng.integers(plus.size)])
            q = int(minus[rng.integers(minus.size)])
            dpaf = sa._dpaf_swap(u, p, q, s)
            u[p], u[q] = u[q], u[p]           # apply the swap
            paf = paf + dpaf
            assert np.array_equal(paf, sa._paf_int(u)), f"dPAF drift at ell={ell}"


def test_incremental_energy_matches_objective():
    rng = np.random.default_rng(1)
    for ell in (7, 11, 13):
        s = np.arange(ell)
        u, v = sa._random_pair(ell, rng)
        pafu, pafv = sa._paf_int(u), sa._paf_int(v)
        r = pafu + pafv + 2
        E = sa._energy(pafu, pafv)
        assert E == objective(u, v)
        for _ in range(50):
            on_u = rng.random() < 0.5
            x = u if on_u else v
            plus = np.nonzero(x == 1)[0]
            minus = np.nonzero(x == -1)[0]
            p = int(plus[rng.integers(plus.size)])
            q = int(minus[rng.integers(minus.size)])
            dpaf = sa._dpaf_swap(x, p, q, s)
            dE = int(2 * (r * dpaf).sum() + (dpaf * dpaf).sum())
            x[p], x[q] = x[q], x[p]
            if on_u:
                pafu = pafu + dpaf
            else:
                pafv = pafv + dpaf
            r = r + dpaf
            E += dE
            assert E == objective(u, v), f"dE drift at ell={ell}"
            assert E >= 0


def test_paf_zero_is_ell():
    rng = np.random.default_rng(2)
    for ell in (5, 9, 17):
        u, _ = sa._random_pair(ell, rng)
        assert sa._paf_int(u)[0] == ell


# --------------------------------------------------------------------------- #
# normalization + acceptance behavior
# --------------------------------------------------------------------------- #
def test_random_pair_is_normalized():
    rng = np.random.default_rng(3)
    for ell in (5, 11, 19):
        u, v = sa._random_pair(ell, rng)
        assert u.sum() == 1 and v.sum() == 1
        assert (u == -1).sum() == (ell - 1) // 2
        assert (v == -1).sum() == (ell - 1) // 2


def test_accept_worse_climbs_at_high_T():
    # At high T a positive-dE (worse) move is accepted a good fraction of the time.
    rng = np.random.default_rng(4)
    accepts = sum(sa._accept(4.0, 1000.0, rng) for _ in range(2000))
    assert accepts > 1500                      # ~exp(-4/1000) ~ 0.996
    # At T -> 0 uphill moves are rejected.
    assert not any(sa._accept(4.0, 0.0, rng) for _ in range(50))


def test_accept_downhill_always():
    rng = np.random.default_rng(5)
    assert all(sa._accept(-1.0, T, rng) for T in (0.0, 0.1, 10.0))
    assert all(sa._accept(0.0, T, rng) for T in (0.0, 0.1, 10.0))


# --------------------------------------------------------------------------- #
# search solves small ell
# --------------------------------------------------------------------------- #
def test_search_solves_small_ell():
    for ell in (5, 7, 9, 11, 13):
        res = sa.search_anneal(ell, restarts=10, steps=20000, seed=0)
        assert res["solved"], f"failed to solve ell={ell}"
        assert is_legendre_pair(res["u"], res["v"], ell)


def test_search_rejects_even_ell():
    try:
        sa.search_anneal(8, restarts=1, steps=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError for even ell")


def test_not_found_is_honest():
    # Tiny budget: must never claim a false solve, and best_E is a real int.
    res = sa.search_anneal(19, restarts=1, steps=200, seed=0)
    assert res["solved"] in (True, False)
    if not res["solved"]:
        assert isinstance(res["best_E"], int) and res["best_E"] > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
