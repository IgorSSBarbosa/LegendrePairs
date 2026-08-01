"""test_memory_guard.py — the pre-flight RAM guard turns an OOM into a clear error.

A bad modulus makes the lift (or sieve) astronomically large: ``ell=39, m=3`` has
``n=13``, so a single fiber is ~5e9 length-39 sequences and materializing it OOMs
the machine. :func:`lp_compress.parallel.pipeline_B_parallel` now estimates peak
memory from the exact ``fiber_size`` product BEFORE dispatching and raises a
``MemoryError`` (with a larger-modulus hint) instead of crashing.

Guard requirements:
  1. ABORTS the pathological case (ell=39, m=3) and names a feasible modulus.
  2. Is HONEST — the estimate is driven by the exact fiber cardinality.
  3. NEVER trips a feasible case; the class set is unchanged when it passes.
  4. ``max_lift_gb`` both TIGHTENS (force-abort a feasible case) and LOOSENS.
"""
from __future__ import annotations

import pytest

from lp_compress.parallel import (
    pipeline_B_parallel, _estimate_peak_lift_bytes, _check_lift_budget,
)


def test_guard_aborts_pathological_modulus_with_hint():
    """ell=39,m=3 (n=13) must raise before the sieve/lift, hinting m=13."""
    with pytest.raises(MemoryError) as ei:
        pipeline_B_parallel(39, m=3, n_workers=2, verbose=False)
    msg = str(ei.value)
    assert "m=13" in msg and "n=3" in msg, msg      # points at the feasible modulus
    assert "39" in msg


def test_estimate_tracks_the_exact_fiber_product():
    """Peak estimate scales with the summed largest fibers (exact fiber_size)."""
    # cA with n=3, all columns full (+3) -> fiber C(3,3)=1 each; and a mixed one.
    reps = [([3, 3, 3], [3, 3, 3]), ([1, 1, 1], [1, 1, 1])]  # ell=9, m=3, n=3
    peak1, big1 = _estimate_peak_lift_bytes(reps, 9, 3, n_workers=1)
    peak2, big2 = _estimate_peak_lift_bytes(reps, 9, 3, n_workers=2)
    # [1,1,1] -> C(3,2)^3 = 27 is the biggest fiber; [3,3,3] -> 1
    assert big1 == 27 and big2 == 27
    assert peak2 > peak1                             # two concurrent workers hold more


def test_feasible_case_passes_and_is_unchanged():
    """A feasible modulus is not aborted and still matches the known class count."""
    r = pipeline_B_parallel(15, m=5, n_workers=2, verbose=False)
    assert r["lp_classes"] == 8                      # ell=15 ground truth


def test_max_lift_gb_can_force_abort_a_feasible_case():
    """A tiny explicit budget aborts even a small lift (knob works both ways)."""
    with pytest.raises(MemoryError):
        pipeline_B_parallel(21, m=7, n_workers=2, max_lift_gb=1e-9, verbose=False)


def test_check_lift_budget_is_a_noop_under_budget():
    """Directly: a small rep set under a generous budget does not raise."""
    reps = [([1, 1, 1], [1, 1, 1])]                  # tiny
    _check_lift_budget(reps, 9, 3, n_workers=2, max_lift_gb=1.0, verbose=False)
