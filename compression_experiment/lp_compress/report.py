"""report.py — Phase 6: the compression cost report (funnel accounting).

For each compressible ``(ell, m)`` we account exactly how much each stage of the
funnel cuts, WITHOUT materializing any fiber (all counts are closed-form or come
from the cheap compressed-space enumeration):

    raw pm1 ordered pairs            2**(2*ell)                    (brute baseline)
      -> compressed ordered pairs    (n+1)**(2m)                   (compression)
      -> sieve survivors             |cascade_pairs|               (Phase 3 sieve)
      -> orbit representatives       |orbit_reduced_pairs|         (Phase 4 orbits)
      -> reduced lift candidates     sum_reps  fib(cA)*fib(cB)      (what B tests)

Reported factors (all exact integers upstream, floats only for display):
  * ``sieve_factor``   = compressed_ordered_pairs / survivors,
  * ``orbit_factor``   = survivors / reps,
  * ``lift_factor``    = full_lift / reduced_lift  (== orbit_factor on the fiber),
  * ``overall_factor`` = raw / reduced_lift        (brute vs Approach B workload).

Nothing here is a heuristic or a stub: every number is a count. Feasibility is
gated by ``(n+1)**m`` (the compressed enumeration size), not by the fiber.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

from .compress import cascade_pairs, incomparable_divisors
from .lift import fiber_size
from .orbit import orbit_reduced_pairs

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Cap on the compressed-space enumeration ``(n+1)**m`` we are willing to walk.
DEFAULT_MAX_ENUM = 300_000


def funnel_costs(ell: int, m: int, use_multipliers: bool = False) -> Dict:
    """Exact per-stage funnel counts and reduction factors for ``(ell, m)``."""
    n = ell // m
    n_vectors = (n + 1) ** m
    n_ordered_pairs = n_vectors * n_vectors
    raw = 1 << (2 * ell)                       # 2**(2*ell) brute pm1 ordered pairs

    pairs = list(cascade_pairs(ell, m))
    reps = orbit_reduced_pairs(pairs, m, use_multipliers)
    survivors = len(pairs)
    n_reps = len(reps)

    full_lift = sum(fiber_size(a, ell, m) * fiber_size(b, ell, m) for a, b in pairs)
    reduced_lift = sum(fiber_size(a, ell, m) * fiber_size(b, ell, m) for a, b in reps)

    def ratio(a, b):
        return (a / b) if b else float("inf")

    return {
        "ell": ell,
        "m": m,
        "n": n,
        "raw_pairs": raw,
        "compressed_vectors": n_vectors,
        "compressed_ordered_pairs": n_ordered_pairs,
        "sieve_survivors": survivors,
        "orbit_reps": n_reps,
        "full_lift": full_lift,
        "reduced_lift": reduced_lift,
        "sieve_factor": ratio(n_ordered_pairs, survivors),
        "orbit_factor": ratio(survivors, n_reps),
        "lift_factor": ratio(full_lift, reduced_lift),
        "overall_factor": ratio(raw, reduced_lift),
    }


def default_cases(Ls: Sequence[int], max_enum: int = DEFAULT_MAX_ENUM
                  ) -> List[Tuple[int, int]]:
    """All ``(ell, m)`` over ``Ls`` with ``(n+1)**m`` within the enum budget."""
    out = []
    for ell in Ls:
        for m in incomparable_divisors(ell):
            n = ell // m
            if (n + 1) ** m <= max_enum:
                out.append((ell, m))
    return out


_CSV_FIELDS = [
    "ell", "m", "n", "raw_pairs", "compressed_vectors", "compressed_ordered_pairs",
    "sieve_survivors", "orbit_reps", "full_lift", "reduced_lift",
    "sieve_factor", "orbit_factor", "lift_factor", "overall_factor",
]


def cost_table(cases: Sequence[Tuple[int, int]], path: Optional[str] = None) -> List[Dict]:
    """Compute (and optionally CSV-write) the funnel cost rows for ``cases``."""
    rows = [funnel_costs(ell, m) for ell, m in cases]
    if path and rows:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
    return rows


def _fmt(x: float) -> str:
    """Human factor: 1.2e3 style for big, plain for small."""
    if x == float("inf"):
        return "inf"
    if x >= 1000:
        return f"{x:.2e}"
    return f"{x:.1f}"


if __name__ == "__main__":
    cases = default_cases(range(9, 28, 2))
    out = os.path.join(RESULTS, "cost_report.csv")
    rows = cost_table(cases, out)
    hdr = (f"{'ell':>3} {'m':>2} {'n':>2} {'comp_pairs':>12} {'surv':>7} "
           f"{'reps':>5} {'reduced_lift':>14} {'sieve×':>8} {'orbit×':>7} "
           f"{'overall×':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['ell']:>3} {r['m']:>2} {r['n']:>2} "
              f"{r['compressed_ordered_pairs']:>12} {r['sieve_survivors']:>7} "
              f"{r['orbit_reps']:>5} {r['reduced_lift']:>14} "
              f"{_fmt(r['sieve_factor']):>8} {_fmt(r['orbit_factor']):>7} "
              f"{_fmt(r['overall_factor']):>9}", flush=True)
    print(f"\nwrote {out}")
