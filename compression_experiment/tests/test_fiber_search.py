"""test_fiber_search.py — Phase 5 (Q2) invariants + recovery.

The fiber search only makes sense if (a) every state stays in the fiber (moves
preserve ``compress``), (b) the incremental half-PAF equals a full recompute, and
(c) the energy is ``0`` exactly on true LPs. Then the acid test: seeded inside a
KNOWN LP's fibers, SA must actually descend to ``E=0``. If it can't reach LPs that
provably live in those fibers, the move/energy is wrong.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from lp_compress.compress import compress, cascade_pairs_vec
from lp_compress.core import paf_half, is_legendre_pair
from lp_compress.orbit import orbit_reduce_arrays
from lp_compress.fiber_search import (
    _column_slots,
    fiber_sample,
    fiber_plus_counts,
    column_swap_positions,
    _apply_swap_half,
    energy_half,
    anneal_fiber_pair,
    search_survivors,
)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(os.path.dirname(_HERE), "rle_experiment", "results", "lps")


def _pm(s: str) -> np.ndarray:
    b = np.frombuffer(s.strip().encode(), np.uint8)
    return np.where(b == ord("+"), 1, -1).astype(np.int8)


def _first_db_lp(ell: int):
    path = os.path.join(DB, f"LP{ell}.csv")
    with open(path) as f:
        f.readline()  # header index,u,v
        _, u, v = f.readline().strip().split(",")
    return _pm(u), _pm(v)


# --------------------------------------------------------------------------- #
# fiber invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ell,m", [(15, 5), (21, 7), (27, 9), (33, 11)])
def test_fiber_sample_lands_in_fiber(ell, m):
    rng = np.random.default_rng(0)
    CA, CB = cascade_pairs_vec(ell, m)
    for idx in rng.choice(CA.shape[0], size=min(30, CA.shape[0]), replace=False):
        cA = CA[idx]
        A = fiber_sample(cA, ell, m, rng)
        assert np.array_equal(compress(A, m), cA.astype(np.int64))
        assert set(np.unique(A)).issubset({-1, 1})


@pytest.mark.parametrize("ell,m", [(15, 5), (21, 7), (27, 9)])
def test_column_swap_preserves_compression_and_rowsum(ell, m):
    rng = np.random.default_rng(1)
    CA, _ = cascade_pairs_vec(ell, m)
    cA = CA[rng.integers(CA.shape[0])]
    A = fiber_sample(cA, ell, m, rng)
    for _ in range(200):
        sw = column_swap_positions(A, ell, m, rng)
        if sw is None:
            continue
        i, j = sw
        s0 = int(A.sum())
        A[i], A[j] = A[j], A[i]
        assert np.array_equal(compress(A, m), cA.astype(np.int64))
        assert int(A.sum()) == s0


# --------------------------------------------------------------------------- #
# incremental half-PAF equals full recompute
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ell,m", [(15, 5), (21, 7), (27, 9), (33, 11)])
def test_incremental_half_paf_matches_recompute(ell, m):
    rng = np.random.default_rng(2)
    half = (ell - 1) // 2
    CA, _ = cascade_pairs_vec(ell, m)
    cA = CA[rng.integers(CA.shape[0])]
    A = fiber_sample(cA, ell, m, rng)
    pa = paf_half(A).astype(np.int64)
    for _ in range(300):
        sw = column_swap_positions(A, ell, m, rng)
        if sw is None:
            continue
        i, j = sw
        _apply_swap_half(A, pa, i, j, half)
        assert np.array_equal(pa, paf_half(A).astype(np.int64))


# --------------------------------------------------------------------------- #
# energy is exactly zero on true LPs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ell", [15, 21, 27, 33])
def test_energy_zero_on_db_lp(ell):
    A, B = _first_db_lp(ell)
    assert is_legendre_pair(A, B)
    E = energy_half(paf_half(A).astype(np.int64), paf_half(B).astype(np.int64))
    assert E == 0


# --------------------------------------------------------------------------- #
# recovery: SA descends to E=0 inside a known LP's fibers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ell,m", [(15, 5), (21, 7), (27, 9)])
def test_sa_recovers_lp_from_known_fibers(ell, m):
    A, B = _first_db_lp(ell)
    cA, cB = compress(A, m), compress(B, m)
    res = anneal_fiber_pair(cA, cB, ell, m, iters=20_000, restarts=80,
                            T0=10.0, seed=12345)
    assert res["found"], f"SA failed to reach E=0 in ell={ell} fibers (E={res['E']})"
    RA, RB = res["A"], res["B"]
    assert np.array_equal(compress(RA, m), cA.astype(np.int64))
    assert np.array_equal(compress(RB, m), cB.astype(np.int64))
    assert is_legendre_pair(RA, RB)


# --------------------------------------------------------------------------- #
# global early stop: existence mode halts at the first LP
# --------------------------------------------------------------------------- #
def test_stop_on_first_halts_after_one_lp():
    """``stop_on_first`` returns as soon as ANY fiber pair yields an LP.

    Existence mode: exactly one class is emitted, ``stopped_early`` is set, and
    strictly fewer fibers are annealed than the full sample would touch. The one
    class emitted must be a genuine LP. Serial (``n_workers=1``) keeps it
    deterministic so the "searched fewer" assertion is stable.
    """
    ell, m = 15, 5
    CA, CB = cascade_pairs_vec(ell, m)
    reps = orbit_reduce_arrays(CA, CB, m, ell // m)
    assert len(reps) > 1                                # a real early-stop is possible

    res = search_survivors(ell, m, reps, max_pairs=len(reps), stop_on_first=True,
                           n_workers=1, iters=20_000, restarts=60, seed=7)
    assert res["stopped_early"] is True
    assert res["n_found"] == 1                          # halts on the first hit
    assert res["n_classes"] == 1
    assert res["n_pairs_searched"] < len(reps)          # did NOT scan the whole sample
    (sA, sB), = res["classes"].values()
    assert is_legendre_pair(_pm(sA), _pm(sB))


def test_stop_on_first_false_scans_the_whole_sample():
    """Default (coverage) mode anneals every sampled fiber and never sets the flag."""
    ell, m = 15, 5
    CA, CB = cascade_pairs_vec(ell, m)
    reps = orbit_reduce_arrays(CA, CB, m, ell // m)

    res = search_survivors(ell, m, reps, max_pairs=len(reps), stop_on_first=False,
                           n_workers=1, iters=20_000, restarts=60, seed=7)
    assert res["stopped_early"] is False
    assert res["n_pairs_searched"] == len(reps)         # scanned the full sample
