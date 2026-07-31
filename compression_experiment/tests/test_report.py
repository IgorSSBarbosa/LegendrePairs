"""test_report.py — Phase 6 spec (PLAN §5): the compression cost report.

Anchors / invariants (all exact counts, no stubs):
  * ell=9,m=3 funnel: 4096 compressed pairs -> 72 sieve survivors -> 1 orbit rep
    -> 243 reduced-lift candidates (vs 17496 full, vs 262144 raw pm1 pairs);
  * monotone funnel: raw >= compressed_ordered >= survivors >= reps >= 1 and
    full_lift >= reduced_lift >= reps;
  * reported factors are exactly the stage ratios;
  * fiber_size is invariant under the orbit group (justifies orbit accounting).
"""
from __future__ import annotations

import numpy as np
import pytest

from lp_compress.report import funnel_costs, default_cases, cost_table
from lp_compress.compress import compress
from lp_compress.lift import fiber_size
from lp_compress.orbit import rotate_c, negate_c, reverse_c

# cheap cases (compressed enum + orbit reduction both sub-second)
CASES = [(9, 3), (15, 3), (15, 5), (21, 3), (25, 5)]
# heavy cases (many survivors to orbit-reduce) — behind the slow marker
SLOW_CASES = [(21, 7), (27, 9)]


def test_funnel_costs_ell9_anchor():
    r = funnel_costs(9, 3)
    assert r["raw_pairs"] == 262144                 # 2**18
    assert r["compressed_vectors"] == 64            # 4**3
    assert r["compressed_ordered_pairs"] == 4096
    assert r["sieve_survivors"] == 72
    assert r["orbit_reps"] == 1
    assert r["full_lift"] == 17496
    assert r["reduced_lift"] == 243
    assert r["orbit_factor"] == 72.0
    assert r["lift_factor"] == 72.0                 # single orbit => equals orbit_factor
    assert r["sieve_factor"] == pytest.approx(4096 / 72)
    assert r["overall_factor"] == pytest.approx(262144 / 243)


@pytest.mark.parametrize("ell,m", CASES)
def test_funnel_is_monotone(ell, m):
    r = funnel_costs(ell, m)
    assert r["raw_pairs"] >= r["compressed_ordered_pairs"]
    assert r["compressed_ordered_pairs"] >= r["sieve_survivors"] >= r["orbit_reps"] >= 1
    assert r["full_lift"] >= r["reduced_lift"] >= r["orbit_reps"]


@pytest.mark.parametrize("ell,m", CASES)
def test_reported_factors_match_ratios(ell, m):
    r = funnel_costs(ell, m)
    assert r["sieve_factor"] == pytest.approx(r["compressed_ordered_pairs"] / r["sieve_survivors"])
    assert r["orbit_factor"] == pytest.approx(r["sieve_survivors"] / r["orbit_reps"])
    assert r["lift_factor"] == pytest.approx(r["full_lift"] / r["reduced_lift"])
    assert r["overall_factor"] == pytest.approx(r["raw_pairs"] / r["reduced_lift"])


@pytest.mark.parametrize("ell,m", CASES)
def test_fiber_size_group_invariant(ell, m):
    """Orbit accounting is exact because every orbit member has the same fiber."""
    rng = np.random.default_rng(ell * 3 + m)
    for _ in range(60):
        c = compress((rng.integers(0, 2, ell) * 2 - 1).astype(np.int8), m)
        f = fiber_size(c, ell, m)
        assert fiber_size(rotate_c(c, 1), ell, m) == f
        assert fiber_size(negate_c(c), ell, m) == f
        assert fiber_size(reverse_c(c), ell, m) == f


@pytest.mark.slow
@pytest.mark.parametrize("ell,m", SLOW_CASES)
def test_funnel_is_monotone_slow(ell, m):
    r = funnel_costs(ell, m)
    assert r["raw_pairs"] >= r["compressed_ordered_pairs"]
    assert r["compressed_ordered_pairs"] >= r["sieve_survivors"] >= r["orbit_reps"] >= 1
    assert r["full_lift"] >= r["reduced_lift"] >= r["orbit_reps"]
    assert r["overall_factor"] == pytest.approx(r["raw_pairs"] / r["reduced_lift"])


def test_default_cases_budget():
    cases = default_cases(range(9, 28, 2))
    assert (9, 3) in cases and (15, 5) in cases and (27, 9) in cases
    # coarse compression of large ell blows the (n+1)**m enum budget
    assert (45, 15) not in cases


def test_cost_table_writes_csv(tmp_path):
    path = str(tmp_path / "cost.csv")
    rows = cost_table([(9, 3), (15, 5)], path)
    assert len(rows) == 2
    import os
    assert os.path.exists(path)
    with open(path) as f:
        header = f.readline().strip().split(",")
    assert "overall_factor" in header and "sieve_survivors" in header
