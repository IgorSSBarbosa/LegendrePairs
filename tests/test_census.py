"""Tests for the Legendre-pair census (``census/census.py``).

Pytest-free (plain ``assert`` + ``__main__`` runner), matching the sibling test
files and runnable without extra deps; pytest still discovers ``test_*``.
"""

import os
import sys
from itertools import combinations

import numpy as np

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(_ROOT, "census"))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import census as cs  # noqa: E402
from legendre_pairs import is_legendre_pair  # noqa: E402


def _all_normalized_masks(ell):
    neg = (ell - 1) // 2
    return [sum(1 << p for p in combo) for combo in combinations(range(ell), neg)]


# --------------------------------------------------------------------------- #
# Spectral / PAF identities
# --------------------------------------------------------------------------- #
def test_parseval():
    for ell in (5, 7, 9, 11):
        rng = np.random.default_rng(ell)
        u = rng.choice([-1, 1], size=ell).astype(float)
        p = np.abs(np.fft.fft(u)) ** 2
        assert abs(p.sum() - ell * ell) < 1e-6


def test_paf_symmetry():
    for ell in (5, 7, 9, 11):
        rng = np.random.default_rng(ell + 1)
        u = rng.choice([-1, 1], size=ell).astype(float)
        r = np.fft.ifft(np.abs(np.fft.fft(u)) ** 2).real
        for j in range(1, ell):
            assert abs(r[j] - r[ell - j]) < 1e-6      # PAF(u,j) == PAF(u,ell-j)


# --------------------------------------------------------------------------- #
# Signature census == brute force over all M^2 pairs (ell in {5,7,9})
# --------------------------------------------------------------------------- #
def test_census_matches_bruteforce():
    for ell in (5, 7, 9):
        count, _ = cs.census_signatures(ell, progress=False, resume=False)
        T_ordered, T_unordered, T_diag = cs.totals_from_count(count)

        masks = _all_normalized_masks(ell)
        seqs = [cs.mask_to_seq(m, ell) for m in masks]
        brute = 0
        for u in seqs:
            for v in seqs:
                if is_legendre_pair(u, v, ell):
                    brute += 1
        assert brute == T_ordered, f"ell={ell}: brute {brute} != {T_ordered}"
        assert T_unordered == (T_ordered + T_diag) // 2


# --------------------------------------------------------------------------- #
# Orbit-stabilizer + partition consistency
# --------------------------------------------------------------------------- #
def test_orbits_partition_and_stabilizer():
    for ell in (5, 7, 9, 11, 13):
        count, sigmap = cs.census_signatures(ell, progress=False, resume=False)
        T_ordered, _, _ = cs.totals_from_count(count)
        GG = cs.group_order(ell)
        sizes, reps, nodeset, groups = cs.gg_orbits(ell, count, sigmap)

        assert sum(sizes) == T_ordered                # partition covers the set
        assert len(nodeset) == T_ordered
        for s in sizes:
            stab = GG // s
            assert stab * s == GG                     # |Stab_i| * orbit_i == |GG|
            assert s <= GG


# --------------------------------------------------------------------------- #
# seed_expand(rep) equals its union-find orbit, for every class
# --------------------------------------------------------------------------- #
def test_seed_expand_equals_orbit():
    for ell in (5, 7, 9, 11, 13):
        count, sigmap = cs.census_signatures(ell, progress=False, resume=False)
        _, reps, _, groups = cs.gg_orbits(ell, count, sigmap)
        for root, members in groups.items():
            bfs = cs.seed_expand(reps[root], ell)
            assert bfs == set(members)                # exact set equality


# --------------------------------------------------------------------------- #
# N_classes == 1  =>  a single seed expands to the entire LP set
# --------------------------------------------------------------------------- #
def test_single_class_seed_is_complete():
    for ell in (5, 7, 9):                              # these have N_classes == 1
        count, sigmap = cs.census_signatures(ell, progress=False, resume=False)
        T_ordered, _, _ = cs.totals_from_count(count)
        sizes, reps, _, _ = cs.gg_orbits(ell, count, sigmap)
        assert len(sizes) == 1
        seed = next(iter(reps.values()))
        assert len(cs.seed_expand(seed, ell)) == T_ordered


# --------------------------------------------------------------------------- #
# Number theory + stub
# --------------------------------------------------------------------------- #
def test_group_order_and_phi():
    assert cs.euler_phi(15) == 8
    assert cs.group_order(7) == 4 * 49 * 6
    assert cs.group_order(15) == 4 * 225 * 8


def test_greedy_is_stub():
    try:
        cs.step3_greedy()
    except NotImplementedError:
        return
    raise AssertionError("step3_greedy should raise NotImplementedError")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
