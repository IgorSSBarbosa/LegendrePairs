"""pipeline.py — Approach drivers.

Approach A (this phase): exhaustive RLE ground truth. Enumerate canonical
row-sum-+1 sequences -> single-sequence PSD screen -> PAF hash-match survivors
-> collect Legendre pairs and dedup to inequivalent classes. This is a thin
wrapper over the verified ``lp_rle`` exhaustive driver; it NEVER touches the
Phase 3 compression stub.

Approach B (Phase 4): compression sieve + lift — added later in this module.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Sequence

from lp_rle.exhaust import run_exhaustive, validate as _validate_rle_vs_brute
from lp_rle.symmetry import canonical_pair

from .compress import cascade_pairs, _default_modulus
from .lift import lift
from .orbit import orbit_reduced_pairs
from .core import is_legendre_pair

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def pipeline_A(ell: int) -> Dict:
    """Exhaustive RLE ground truth for odd ``ell`` (reuses ``run_exhaustive``).

    Returns the funnel counts:
      * ``survivors``      -- canonical PSD-feasible rotation classes,
      * ``raw_pairs``      -- PAF-matched Legendre pairs before class dedup,
      * ``lp_classes``     -- inequivalent Legendre-pair classes,
      * ``total_classes``  -- all canonical rotation classes enumerated.
    """
    ex = run_exhaustive(ell)
    return {
        "ell": ell,
        "survivors": ex["survivors"],
        "raw_pairs": ex["raw_pairs"],
        "lp_classes": ex["lp_classes"],
        "total_classes": ex["total_classes"],
    }


def crosscheck_A(ell: int) -> Dict:
    """Assert Approach A agrees with the INDEPENDENT pm1 brute force.

    Reuses ``lp_rle.exhaust.validate``, which enumerates all row-sum-+1
    sequences with no RLE/compress code and asserts identical feasible rotation
    classes, the rotation-multiplicity relation, and identical LP classes.
    Raises ``AssertionError`` on any discrepancy; returns the validation dict.
    """
    return _validate_rle_vs_brute(ell)


# --------------------------------------------------------------------------- #
# Approach B (Phase 4): compression sieve -> orbit reduction -> lift -> LP test
# --------------------------------------------------------------------------- #
def pipeline_B(ell: int, m: Optional[int] = None, orbit_reduce: bool = True,
               use_multipliers: bool = False, use_stub: bool = False) -> Dict:
    """Compression funnel for odd ``ell``: sieve -> (orbit-reduce) -> lift -> LP test.

    Steps (all VERIFIED except the explicitly gated stubs):
      1. ``cascade_pairs`` — surviving compressed ORDERED pairs under the exact
         compressed PAF/PSD sieve (Phase 3);
      2. orbit reduction — keep one representative compressed pair per induced-
         group orbit (rotation ``Z_m`` + reversal + negation + swap; multiplier
         GATED), so only representative fibers are lifted;
      3. ``lift`` — stream each representative's fiber back to ``{-1,+1}^ell``;
      4. exact PAF LP test, dedup to inequivalent classes via
         ``lp_rle.symmetry.canonical_pair`` (the SAME group Approach A uses).

    Orbit reduction lifts fewer candidates but STILL recovers every LP class:
    each class's compressed pair is group-equivalent to a kept representative, so
    its canonical key reappears. Returns the funnel counts; ``n_lp_classes`` is
    invariant to ``orbit_reduce`` (the correctness anchor).
    """
    m = _default_modulus(ell, m)
    n = ell // m
    pairs = list(cascade_pairs(ell, m, use_stub))
    reps = orbit_reduced_pairs(pairs, m, use_multipliers) if orbit_reduce else pairs

    n_lift = 0
    n_lps = 0
    classes: Dict = {}
    for cA, cB in reps:
        LA = list(lift(cA, ell, m))
        LB = list(lift(cB, ell, m))
        n_lift += len(LA) * len(LB)
        for A in LA:
            for B in LB:
                if is_legendre_pair(A, B):
                    n_lps += 1
                    # canonical_pair assumes row-sum +1 (it folds negation only via
                    # that normalization); our fiber contains both signs, so pin +1.
                    nA = A if int(A.sum()) > 0 else -A
                    nB = B if int(B.sum()) > 0 else -B
                    classes.setdefault(canonical_pair(nA, nB), (nA, nB))
    return {
        "ell": ell,
        "m": m,
        "n": n,
        "n_compressed_pairs": len(pairs),
        "n_orbit_reps": len(reps),
        "n_lift_candidates": n_lift,
        "n_lps": n_lps,
        "lp_classes": len(classes),
        "classes": classes,
    }


def crosscheck_B(ell: int, m: Optional[int] = None) -> Dict:
    """Assert Approach B recovers exactly Approach A's LP classes (no false negs).

    Runs ``pipeline_A`` (RLE ground truth) and both orbit-reduced and unreduced
    ``pipeline_B``; asserts all three agree on ``lp_classes`` and that the
    orbit-reduced LP-class SET equals the unreduced one (reduction loses nothing).
    Raises ``AssertionError`` on any discrepancy.
    """
    a = pipeline_A(ell)
    b_full = pipeline_B(ell, m, orbit_reduce=False)
    b_red = pipeline_B(ell, m, orbit_reduce=True)
    assert b_full["lp_classes"] == a["lp_classes"], (
        f"ell={ell}: Approach B (full) lp_classes {b_full['lp_classes']} "
        f"!= Approach A {a['lp_classes']}")
    assert set(b_red["classes"]) == set(b_full["classes"]), (
        f"ell={ell}: orbit reduction changed the LP-class set "
        f"({b_red['lp_classes']} vs {b_full['lp_classes']})")
    return {
        "ell": ell,
        "lp_classes": a["lp_classes"],
        "n_lps_full": b_full["n_lps"],
        "n_lps_reduced": b_red["n_lps"],
        "n_lift_full": b_full["n_lift_candidates"],
        "n_lift_reduced": b_red["n_lift_candidates"],
        "n_orbit_reps": b_red["n_orbit_reps"],
        "n_compressed_pairs": b_red["n_compressed_pairs"],
    }


def ground_truth_table(Ls: Sequence[int], path: Optional[str] = None) -> List[Dict]:
    """Produce (and optionally CSV-write) the Approach A funnel table."""
    rows = [pipeline_A(l) for l in Ls]
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def compression_funnel_table(cases: Sequence, path: Optional[str] = None) -> List[Dict]:
    """Approach B funnel (orbit-reduced) for ``(ell, m)`` cases; optional CSV.

    ``cases`` is a list of ``(ell, m)``; only liftable (small-``n``) cases should
    be passed since the full fiber is materialized. ``lp_classes`` here must equal
    Approach A's for the same ``ell`` (the correctness anchor).
    """
    rows = []
    for ell, m in cases:
        r = pipeline_B(ell, m=m, orbit_reduce=True)
        r.pop("classes", None)
        rows.append(r)
    if path and rows:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


if __name__ == "__main__":
    Ls = list(range(3, 26, 2))
    out = os.path.join(RESULTS, "ground_truth_A.csv")
    rows = ground_truth_table(Ls, out)
    print(f"{'ell':>3} {'survivors':>10} {'raw_pairs':>10} {'lp_classes':>11} {'total_classes':>14}")
    for r in rows:
        print(f"{r['ell']:>3} {r['survivors']:>10} {r['raw_pairs']:>10} "
              f"{r['lp_classes']:>11} {r['total_classes']:>14}", flush=True)
    print(f"\nwrote {out}")

    # Approach B compression funnel (orbit-reduced), liftable cases only.
    b_cases = [(9, 3), (15, 5)]
    bout = os.path.join(RESULTS, "compression_funnel_B.csv")
    brows = compression_funnel_table(b_cases, bout)
    print(f"\n{'ell':>3} {'m':>2} {'comp_pairs':>10} {'reps':>5} "
          f"{'lift':>8} {'LPs':>6} {'classes':>7}")
    for r in brows:
        print(f"{r['ell']:>3} {r['m']:>2} {r['n_compressed_pairs']:>10} "
              f"{r['n_orbit_reps']:>5} {r['n_lift_candidates']:>8} "
              f"{r['n_lps']:>6} {r['lp_classes']:>7}", flush=True)
    print(f"\nwrote {bout}")
