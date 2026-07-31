"""validation.py — Phase 5: the three-pipeline agreement harness.

Three INDEPENDENT routes to the inequivalent Legendre-pair classes of odd ``ell``:

  * A (RLE ground truth)      : ``pipeline_A`` -> ``lp_rle.run_exhaustive``;
  * brute (independent pm1)    : ``crosscheck_A`` -> ``lp_rle.validate`` (enumerates
                                 every row-sum-+1 sequence, no RLE/compress code);
  * B (compression funnel)     : ``pipeline_B`` -> sieve -> orbit-reduce -> lift.

A and brute run for every odd ``ell`` (brute gated by cost). B runs only where
compression exists (``incomparable_divisors(ell)`` non-empty => ``ell`` composite)
AND the orbit-reduced fiber is small enough to materialize (:func:`liftable`).

The harness asserts the class SETS (canonical keys) coincide wherever two routes
both run — a complete invariant, so agreement is exact, not just count-wise.

Honest limits (never papered over):
  * prime ``ell`` has NO compression: route B is unavailable (blind spot);
  * the compressed multiplier/decimation leg is GATED because
    :func:`lp_compress.group.group_order` is an unpinned STUB — see the ell=13
    decimation evidence in :func:`decimation_gap`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lp_rle.exhaust import run_exhaustive
from lp_rle.symmetry import least_rotation_seq, _reverse, units, decimate

from .compress import incomparable_divisors, cascade_pairs, _default_modulus
from .lift import fiber_size
from .orbit import orbit_reduced_pairs, canonical_compressed_pair
from .pipeline import pipeline_B, crosscheck_A

# Default cap on the orbit-reduced lift we are willing to materialize.
DEFAULT_MAX_LIFT = 300_000


def reduced_lift_budget(ell: int, m: int, use_multipliers: bool = False) -> int:
    """Total candidates the ORBIT-REDUCED route B would lift for ``(ell, m)``.

    Cheap: enumerates the compressed space (never ``2**ell``) and sums fiber-size
    products over orbit representatives. Used to decide :func:`liftable`.
    """
    pairs = list(cascade_pairs(ell, m))
    reps = orbit_reduced_pairs(pairs, m, use_multipliers)
    return sum(fiber_size(a, ell, m) * fiber_size(b, ell, m) for a, b in reps)


def liftable(ell: int, m: Optional[int] = None,
             max_lift: int = DEFAULT_MAX_LIFT) -> bool:
    """True if ``ell`` is compressible and route B's reduced lift fits the budget."""
    divs = incomparable_divisors(ell)
    if not divs:
        return False
    m = _default_modulus(ell, m)
    return reduced_lift_budget(ell, m) <= max_lift


def liftable_cases(Ls: Sequence[int], max_lift: int = DEFAULT_MAX_LIFT
                   ) -> List[Tuple[int, int]]:
    """All ``(ell, m)`` over ``Ls`` and every incomparable divisor that are liftable."""
    out = []
    for ell in Ls:
        for m in incomparable_divisors(ell):
            if reduced_lift_budget(ell, m) <= max_lift:
                out.append((ell, m))
    return out


def three_pipeline_report(ell: int, m: Optional[int] = None,
                          run_brute: bool = False,
                          max_lift: int = DEFAULT_MAX_LIFT) -> Dict:
    """Compare all available pipelines for one ``ell``; assert set-agreement.

    Always runs A. Runs brute (independent pm1) iff ``run_brute``. Runs B iff
    ``ell`` is compressible and liftable within ``max_lift``. Raises
    ``AssertionError`` on any disagreement between two routes that both ran.
    """
    ex = run_exhaustive(ell)           # single RLE pass; A route
    a_keys = set(ex["classes"])
    report: Dict = {
        "ell": ell,
        "A_classes": ex["lp_classes"],
        "brute_ran": False,
        "B_ran": False,
        "B_classes": None,
        "B_equals_A": None,
        "compressible": bool(incomparable_divisors(ell)),
    }

    if run_brute:
        v = crosscheck_A(ell)  # asserts A == independent brute internally
        report["brute_ran"] = True
        assert v["lp_classes"] == ex["lp_classes"]

    if report["compressible"]:
        # if the modulus is unspecified, prefer one whose reduced lift fits the
        # budget (cheapest first); fall back to the default when none fit.
        if m is None:
            divs = incomparable_divisors(ell)
            fits = [d for d in divs if reduced_lift_budget(ell, d) <= max_lift]
            mm = min(fits, key=lambda d: reduced_lift_budget(ell, d)) if fits \
                else _default_modulus(ell, None)
        else:
            mm = m
        report["m"] = mm
        if reduced_lift_budget(ell, mm) <= max_lift:
            b = pipeline_B(ell, m=mm, orbit_reduce=True)
            b_keys = set(b["classes"])
            report["B_ran"] = True
            report["B_classes"] = b["lp_classes"]
            report["B_equals_A"] = (b_keys == a_keys)
            assert b_keys == a_keys, (
                f"ell={ell} m={mm}: route B class set != route A "
                f"({b['lp_classes']} vs {a['lp_classes']})")
            report["n_orbit_reps"] = b["n_orbit_reps"]
            report["n_lift_candidates"] = b["n_lift_candidates"]
    return report


# --------------------------------------------------------------------------- #
# honest evidence for the gated decimation leg (the group_order STUB)
# --------------------------------------------------------------------------- #
def _canon_pair_no_decimation(u, v) -> Tuple[bytes, bytes]:
    """Canonical LP key WITHOUT decimation (rotation + reversal + swap only)."""
    def keyseq(x):
        x = np.asarray(x, dtype=np.int8)
        return min(least_rotation_seq(x).tobytes(),
                   least_rotation_seq(_reverse(x)).tobytes())
    a, b = keyseq(u), keyseq(v)
    return (a, b) if a <= b else (b, a)


def decimation_gap(ell: int) -> Dict:
    """How many classes decimation MERGES at ``ell`` (full group vs no-decimation).

    Expands each full-group class over its decimation orbit and counts the
    distinct no-decimation keys. ``no_decimation > full`` means decimation is an
    ESSENTIAL identification — precisely the one the compressed multiplier leg
    would supply once :func:`lp_compress.group.group_order` is pinned.
    """
    ex = run_exhaustive(ell)
    full = len(ex["classes"])
    nd = set()
    for u, v in ex["classes"].values():
        u = np.asarray(u, dtype=np.int8)
        v = np.asarray(v, dtype=np.int8)
        for d in units(ell):
            nd.add(_canon_pair_no_decimation(decimate(u, d), decimate(v, d)))
    return {"ell": ell, "full_classes": full, "no_decimation_classes": len(nd)}


if __name__ == "__main__":
    import os

    Ls = list(range(3, 26, 2))
    brute_cap = 19  # independent pm1 brute is cheap up to here
    print(f"{'ell':>3} {'A':>3} {'brute':>5} {'B(m)':>7} {'B==A':>5} "
          f"{'reps':>5} {'lift':>8}")
    rows = []
    for ell in Ls:
        r = three_pipeline_report(ell, run_brute=(ell <= brute_cap))
        b = "-" if not r["B_ran"] else f"{r['B_classes']}({r.get('m','')})"
        eq = "-" if r["B_equals_A"] is None else ("yes" if r["B_equals_A"] else "NO")
        reps = r.get("n_orbit_reps", "-")
        lift = r.get("n_lift_candidates", "-")
        print(f"{ell:>3} {r['A_classes']:>3} "
              f"{('yes' if r['brute_ran'] else '-'):>5} {b:>7} {eq:>5} "
              f"{reps:>5} {lift:>8}", flush=True)
        rows.append(r)

    print("\ndecimation gap (essential identification, gated compressed leg):")
    print(f"{'ell':>3} {'full':>5} {'no-decim':>9}")
    for ell in (9, 13, 15, 21, 25):
        g = decimation_gap(ell)
        print(f"{ell:>3} {g['full_classes']:>5} {g['no_decimation_classes']:>9}",
              flush=True)
