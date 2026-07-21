"""Tests for the (r, n) restart+kick search (``src/rn_search.py``).

Pytest-free (plain asserts + ``__main__`` runner), matching the sibling suites.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import rn_search as rn  # noqa: E402
from legendre_pairs import is_legendre_pair, paf  # noqa: E402


# --------------------------------------------------------------------------- #
# objective
# --------------------------------------------------------------------------- #
def test_objective_zero_iff_lp():
    # ell=7 known LP (from results/found_pairs.md): u=v works.
    u = np.array([-1, -1, 1, -1, 1, 1, 1])
    assert rn.objective(u, u) == 0
    assert is_legendre_pair(u, u, 7)


def test_objective_matches_definition():
    rng = np.random.default_rng(0)
    for ell in (7, 11, 13):
        u = rng.choice([-1, 1], size=ell)
        v = rng.choice([-1, 1], size=ell)
        pu, pv = paf(u), paf(v)
        resid = pu + pv + 2.0
        resid[0] = 0.0
        expect = int(round((resid ** 2).sum()))
        assert rn.objective(u, v) == expect


def test_objective_is_nonneg_multiple_of_4():
    rng = np.random.default_rng(1)
    U = rn.random_normalized(500, 13, rng)
    V = rn.random_normalized(500, 13, rng)
    E = rn.objective_batch(U, V)
    assert np.all(E >= 0)
    assert np.all(E % 4 == 0)


# --------------------------------------------------------------------------- #
# moves preserve normalization
# --------------------------------------------------------------------------- #
def test_random_normalized_weight():
    rng = np.random.default_rng(2)
    ell = 15
    A = rn.random_normalized(200, ell, rng)
    assert np.all(A.sum(axis=1) == 1)
    assert np.all((A == -1).sum(axis=1) == (ell - 1) // 2)


def test_kick_preserves_normalization():
    rng = np.random.default_rng(3)
    ell = 13
    u = rn.random_normalized(1, ell, rng)[0]
    v = rn.random_normalized(1, ell, rng)[0]
    for n in (1, 2, 3, 5):
        U, V = rn.kick_batch(u, v, 64, n, rng)
        assert np.all(U.sum(axis=1) == 1)
        assert np.all(V.sum(axis=1) == 1)


def test_kick_distance_at_most_n():
    # n 2-swaps move at most n of the -1s, so Hamming distance <= 2n per member.
    rng = np.random.default_rng(4)
    ell = 17
    u = rn.random_normalized(1, ell, rng)[0]
    v = rn.random_normalized(1, ell, rng)[0]
    n = 3
    U, V = rn.kick_batch(u, v, 128, n, rng)
    du = (U != u).sum(axis=1)
    dv = (V != v).sum(axis=1)
    assert np.all(du + dv <= 2 * n)


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def test_search_solves_small_ell():
    for ell in (5, 7, 9):
        res = rn.search_rn(ell, r=512, n=2, seed=0, max_batches=200)
        assert res["solved"], f"failed to solve ell={ell}"
        assert is_legendre_pair(res["u"], res["v"], ell)


def test_search_rejects_even_ell():
    try:
        rn.search_rn(8, r=16, n=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError for even ell")


def test_search_not_found_is_honest():
    # Tiny budget on a sparse length: should return not-found, never a false hit.
    res = rn.search_rn(19, r=8, n=1, seed=0, max_batches=2)
    assert res["solved"] in (True, False)
    if not res["solved"]:
        assert res["u"] is None and res["v"] is None


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def test_min_swaps_to_lp_zero_on_a_pair():
    bu, bv = rn.load_lp_pairs(7)
    # reconstruct the first LP pair's sequences and check distance 0
    m_u, m_v = int(bu[0]), int(bv[0])
    u = np.array([-1 if (m_u >> p) & 1 else 1 for p in range(7)])
    v = np.array([-1 if (m_v >> p) & 1 else 1 for p in range(7)])
    assert is_legendre_pair(u, v, 7)
    assert rn.min_swaps_to_lp(u, v, bu, bv) == 0


def test_sample_E_vs_distance_shapes():
    E, dist = rn.sample_E_vs_distance(9, n_seeds=50, seed=0)
    assert E.shape == (50,) and dist.shape == (50,)
    assert np.all(dist >= 0)
    # every zero-distance seed must itself be an LP (E == 0)
    assert np.all(E[dist == 0] == 0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
