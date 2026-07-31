"""test_validation.py — Phase 5 spec (PLAN §5): three-pipeline agreement harness.

DoD:
  * wherever two of the three routes (A=RLE, brute=pm1, B=compression) both run,
    their inequivalent-LP-class SETS coincide exactly (complete-invariant keys);
  * route B runs on every compressible + liftable ``(ell, m)`` and equals A;
  * honest STUBS are asserted to be stubs: ``group.group_order`` is unpinned
    (returns ``None``) and MUST NOT be used to divide counts;
  * the ell=13 decimation-class merge (7 no-decimation classes -> 4 full classes)
    is real and is exactly the identification the GATED compressed multiplier leg
    would supply once the group order is pinned -> documented ``xfail``.
"""
from __future__ import annotations

import pytest

from lp_compress.validation import (
    three_pipeline_report, liftable, liftable_cases, reduced_lift_budget,
    decimation_gap,
)
from lp_compress.group import group_order, phi

# odd lengths with exhaustive ground truth; brute is cheap up to ~19.
BRUTE_RANGE = list(range(3, 20, 2))     # 3..19
LIFTABLE = liftable_cases(range(3, 28, 2))  # every compressible+liftable (ell,m)


# --------------------------------------------------------------------------- #
# three-pipeline agreement
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ell", BRUTE_RANGE)
def test_A_equals_brute(ell):
    """Route A == independent pm1 brute force (asserted inside the report)."""
    r = three_pipeline_report(ell, run_brute=True)
    assert r["brute_ran"] and r["A_classes"] >= 1


def test_liftable_set_is_expected():
    """Only small-``n`` composite cases are liftable within the default budget."""
    assert (9, 3) in LIFTABLE
    assert (15, 5) in LIFTABLE
    # prime ell -> not compressible -> not liftable
    assert all(ell not in (11, 13, 17, 19, 23) for ell, _ in LIFTABLE)
    # huge fibers excluded
    assert (25, 5) not in LIFTABLE


@pytest.mark.parametrize("ell,m", LIFTABLE)
def test_B_equals_A_on_liftable(ell, m):
    """Route B (compression funnel, orbit-reduced) == route A on the class SET."""
    r = three_pipeline_report(ell, m=m, run_brute=False)
    assert r["B_ran"] and r["B_equals_A"] is True
    assert r["B_classes"] == r["A_classes"]


def test_full_three_way_agreement_ell9_and_15():
    """All three routes agree at the two fully-liftable lengths."""
    for ell, m in [(9, 3), (15, 5)]:
        r = three_pipeline_report(ell, m=m, run_brute=True)
        assert r["brute_ran"] and r["B_ran"] and r["B_equals_A"]


# --------------------------------------------------------------------------- #
# honest stubs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ell", [9, 13, 15, 21, 25])
def test_group_order_is_unpinned_stub(ell):
    """``group_order`` is a documented STUB: it returns None, never a fake count."""
    assert group_order(ell) is None


def test_phi_is_exact():
    """Euler totient (the compressed-multiplier count) is exact, unlike group_order."""
    assert (phi(3), phi(5), phi(7), phi(9), phi(15)) == (2, 4, 6, 6, 8)


def test_liftable_predicate():
    assert liftable(9) and not liftable(13) and not liftable(25)
    assert reduced_lift_budget(9, 3) == 243


# --------------------------------------------------------------------------- #
# decimation is an ESSENTIAL identification (the gated compressed leg)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ell,full,nd", [(9, 1, 3), (13, 4, 7), (15, 8, 24), (21, 22, 126)])
def test_decimation_merges_classes(ell, full, nd):
    """Decimation genuinely merges classes (no-decimation count is strictly larger)."""
    g = decimation_gap(ell)
    assert g["full_classes"] == full
    assert g["no_decimation_classes"] == nd
    assert g["no_decimation_classes"] > g["full_classes"]


@pytest.mark.xfail(reason="Compressed multiplier/decimation leg is GATED because "
                          "group.group_order is an unpinned STUB (TODO cite). Until "
                          "it is pinned, the compressed path cannot certify the "
                          "ell=13 decimation merge (7 -> 4) on its own.")
def test_compressed_path_certifies_decimation_ell13():
    """The compressed route should one day realize decimation identifications.

    ell=13 is prime (no compression) AND the compressed multiplier group order is
    a stub, so we cannot yet certify the 7->4 merge through compression. When
    group_order is pinned this should pass; today it xfails on the stub sentinel.
    """
    assert group_order(13) is not None  # sentinel: stub not yet pinned
