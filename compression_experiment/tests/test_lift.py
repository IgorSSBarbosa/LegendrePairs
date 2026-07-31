"""test_lift.py — Phase 4 spec (PLAN §5): the fiber lift + induced group actions.

Anchors / VERIFIED identities:
  * |fiber(cA)| = prod_j C(n,(n+cA_j)/2), and ``lift`` streams exactly that set;
  * every lifted A satisfies compress(A,m) == cA (exact left-inverse on the fiber);
  * induced equivariance (VERIFIED empirically):
        compress(roll(A,t))     == roll(compress(A), t mod m),
        compress(-A)            == -compress(A),
        compress(reverse(A))    == reverse(compress(A));
  * canonical_c is a group invariant + idempotent; canonical_compressed_pair is
    swap-invariant; orbit_reduced_pairs keeps one rep per orbit.
"""
from __future__ import annotations

import itertools
from math import comb

import numpy as np
import pytest

from lp_compress.compress import compress
from lp_compress.lift import fiber_size, lift
from lp_compress.orbit import (
    rotate_c, negate_c, reverse_c, multiply_c,
    canonical_c, canonical_compressed_pair, orbit_reduced_pairs,
)

CASES = [(9, 3), (15, 3), (15, 5), (21, 3), (25, 5), (27, 9)]


def _rand_pm1(ell, rng):
    return (rng.integers(0, 2, ell) * 2 - 1).astype(np.int8)


# --------------------------------------------------------------------------- #
# fiber size + lift
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ell,m", CASES)
def test_fiber_size_matches_multinomial(ell, m):
    n = ell // m
    rng = np.random.default_rng(ell + m)
    for _ in range(30):
        cA = compress(_rand_pm1(ell, rng), m)
        expect = 1
        for c in cA:
            expect *= comb(n, (n + int(c)) // 2)
        assert fiber_size(cA, ell, m) == expect


def test_fiber_size_vs_brute_ell9():
    ell, m = 9, 3
    from collections import Counter
    cnt = Counter()
    for bits in itertools.product((-1, 1), repeat=ell):
        cnt[tuple(compress(np.array(bits, np.int8), m))] += 1
    for cA, k in cnt.items():
        assert fiber_size(np.array(cA), ell, m) == k


@pytest.mark.parametrize("ell,m", CASES)
def test_lift_is_exact_fiber(ell, m):
    rng = np.random.default_rng(7 * ell + m)
    for _ in range(6):
        cA = compress(_rand_pm1(ell, rng), m)
        seen = set()
        count = 0
        for A in lift(cA, ell, m):
            assert set(np.unique(A)).issubset({-1, 1})
            assert np.array_equal(compress(A, m), cA)   # exact left-inverse
            seen.add(A.tobytes())
            count += 1
            if count > 4000:  # cap: only spot-check huge fibers for membership
                break
        else:
            assert count == fiber_size(cA, ell, m)      # full fiber, no dups
            assert len(seen) == count


def test_lift_unreachable_is_empty():
    # entry parity wrong for n=3 (only {-3,-1,1,3} reachable): 2 is impossible
    assert list(lift(np.array([2, 0, 1]), 9, 3)) == []
    assert fiber_size(np.array([2, 0, 1]), 9, 3) == 0


# --------------------------------------------------------------------------- #
# induced group equivariance (VERIFIED)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ell,m", CASES)
def test_induced_equivariance(ell, m):
    rng = np.random.default_rng(99 + ell + m)
    for _ in range(50):
        A = _rand_pm1(ell, rng)
        cA = compress(A, m)
        for t in range(ell):
            Ash = np.roll(A, t)
            assert np.array_equal(compress(Ash, m), rotate_c(cA, t))
        assert np.array_equal(compress(-A, m), negate_c(cA))
        Arev = A[(-np.arange(ell)) % ell]
        assert np.array_equal(compress(Arev, m), reverse_c(cA))


@pytest.mark.parametrize("ell,m", [(9, 3), (15, 5), (21, 7), (25, 5)])
def test_induced_multiplier_equivariance(ell, m):
    # decimation A_i -> A_{a*i} descends to cA -> cA[(a*j) mod m] (GATED leg)
    from math import gcd
    rng = np.random.default_rng(5 * ell + m)
    for _ in range(30):
        A = _rand_pm1(ell, rng)
        cA = compress(A, m)
        for a in range(1, ell):
            if gcd(a, ell) != 1:
                continue
            Adec = A[(a * np.arange(ell)) % ell]
            assert np.array_equal(compress(Adec, m), multiply_c(cA, a % m))


# --------------------------------------------------------------------------- #
# canonical forms
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ell,m", CASES)
def test_canonical_c_group_invariant_and_idempotent(ell, m):
    rng = np.random.default_rng(3 * ell + m)
    for _ in range(40):
        c = compress(_rand_pm1(ell, rng), m)
        key = canonical_c(c, m)
        # invariant under every generator
        for t in range(m):
            assert canonical_c(rotate_c(c, t), m) == key
        assert canonical_c(negate_c(c), m) == key
        assert canonical_c(reverse_c(c), m) == key
        # idempotent: canonicalizing the representative is a no-op
        rep = np.frombuffer(key, dtype=np.int8)
        assert canonical_c(rep, m) == key


@pytest.mark.parametrize("ell,m", CASES)
def test_canonical_pair_swap_invariant(ell, m):
    rng = np.random.default_rng(11 * ell + m)
    for _ in range(40):
        cA = compress(_rand_pm1(ell, rng), m)
        cB = compress(_rand_pm1(ell, rng), m)
        assert canonical_compressed_pair(cA, cB, m) == canonical_compressed_pair(cB, cA, m)


def test_orbit_reduced_pairs_partition():
    # every input pair must land on exactly one kept representative's orbit
    from lp_compress.compress import cascade_pairs
    ell, m = 9, 3
    pairs = list(cascade_pairs(ell, m))
    reps = orbit_reduced_pairs(pairs, m)
    rep_keys = {canonical_compressed_pair(a, b, m) for a, b in reps}
    # reps are canonically distinct
    assert len(rep_keys) == len(reps)
    # every original pair reduces to a kept rep key
    assert all(canonical_compressed_pair(a, b, m) in rep_keys for a, b in pairs)
