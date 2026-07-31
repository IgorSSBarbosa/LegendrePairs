"""test_pipeline.py — Phase 2+4 spec (PLAN §5): Approach A ground truth and the
Approach B compression funnel (sieve -> orbit-reduce -> lift -> LP).

DoD (A): exact survivor / LP / equivalence-class counts for odd ell = 3..25 match
the INDEPENDENT pm1 brute force with zero discrepancy, and this path never touches
the Phase 3 stub. The cross-check (``crosscheck_A`` -> ``lp_rle.validate``) asserts
the agreement internally; ``slow`` gates the heavy ell=23,25 brute force.

DoD (B): the ell=9 funnel anchor 262144 raw ordered pairs -> 72 compressed ->
17496 lift candidates -> 3888 LPs -> 1 class; orbit reduction lifts far fewer
candidates yet recovers the SAME LP-class set as Approach A (zero false negatives).
"""
from __future__ import annotations

import pytest

from lp_compress.pipeline import pipeline_A, crosscheck_A, pipeline_B, crosscheck_B
from lp_rle.exhaust import run_exhaustive

FAST = list(range(3, 22, 2))   # 3..21
SLOW = [23, 25]

# Confirmed ground-truth anchors (RLE == brute force).
ANCHORS = {
    13: {"lp_classes": 4,  "total_classes": 132,   "survivors": 18},
    19: {"lp_classes": 9,  "total_classes": 4862,  "survivors": 443},
    21: {"lp_classes": 22, "total_classes": 16796, "survivors": 1214},
    23: {"lp_classes": 28, "total_classes": 58786, "survivors": 2939},
    25: {"lp_classes": 46, "total_classes": 208012, "survivors": 7560},
}


@pytest.mark.parametrize("ell", FAST)
def test_pipeline_A_matches_brute(ell):
    v = crosscheck_A(ell)  # asserts RLE == independent brute force internally
    a = pipeline_A(ell)
    assert a["lp_classes"] == v["lp_classes"]
    assert a["survivors"] == v["rle_survivors"]
    assert v["brute_survivors"] == ell * a["survivors"]  # free Z_ell action


@pytest.mark.slow
@pytest.mark.parametrize("ell", SLOW)
def test_pipeline_A_matches_brute_slow(ell):
    v = crosscheck_A(ell)
    a = pipeline_A(ell)
    assert a["lp_classes"] == v["lp_classes"]
    assert v["brute_survivors"] == ell * a["survivors"]


@pytest.mark.parametrize("ell", sorted(k for k in ANCHORS))
def test_ground_truth_anchors(ell):
    a = pipeline_A(ell)
    exp = ANCHORS[ell]
    assert a["lp_classes"] == exp["lp_classes"]
    assert a["total_classes"] == exp["total_classes"]
    assert a["survivors"] == exp["survivors"]


# --------------------------------------------------------------------------- #
# Approach B: compression funnel + orbit reduction (Phase 4)
# --------------------------------------------------------------------------- #
def test_pipeline_B_ell9_full_funnel_anchor():
    """The VERIFIED ell=9 funnel with NO orbit reduction."""
    r = pipeline_B(9, orbit_reduce=False)
    assert (r["m"], r["n"]) == (3, 3)
    assert r["n_compressed_pairs"] == 72
    assert r["n_orbit_reps"] == 72            # no reduction => reps == pairs
    assert r["n_lift_candidates"] == 17496
    assert r["n_lps"] == 3888
    assert r["lp_classes"] == 1               # collapses to the single ell=9 class


def test_pipeline_B_ell9_orbit_reduced_anchor():
    """Orbit reduction lifts 243 (not 17496) candidates, same single class."""
    r = pipeline_B(9, orbit_reduce=True)
    assert r["n_compressed_pairs"] == 72
    assert r["n_orbit_reps"] == 1
    assert r["n_lift_candidates"] == 243
    assert r["n_lps"] == 54
    assert r["lp_classes"] == 1


def test_pipeline_B_multiplier_flag_is_safe():
    """The GATED multiplier leg can only shrink orbits, never lose a class."""
    base = pipeline_B(9, orbit_reduce=True, use_multipliers=False)["lp_classes"]
    mult = pipeline_B(9, orbit_reduce=True, use_multipliers=True)["lp_classes"]
    assert base == mult == 1


@pytest.mark.parametrize("ell,m", [(9, 3), (15, 5)])
def test_approach_B_recovers_A_class_set(ell, m):
    """Zero false negatives AND positives: orbit-reduced B == A on the class SET."""
    a_keys = set(run_exhaustive(ell)["classes"])
    b_keys = set(pipeline_B(ell, m=m, orbit_reduce=True)["classes"])
    assert a_keys == b_keys, f"ell={ell} m={m}: A and B disagree on LP classes"


def test_crosscheck_B_ell9():
    """crosscheck_B asserts A==B(full)==B(reduced) on lp_classes and class sets."""
    c = crosscheck_B(9)
    assert c["lp_classes"] == 1
    assert c["n_lps_full"] == 3888 and c["n_lps_reduced"] == 54
    assert c["n_lift_full"] == 17496 and c["n_lift_reduced"] == 243
    assert c["n_orbit_reps"] == 1
