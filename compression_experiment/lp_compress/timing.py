"""timing.py — Phase 6b: wall-clock timing sweep over small ell.

Measures SECONDS-to-enumerate-all-Legendre-pair-classes for every available
route, and records the class counts SEPARATELY, so correctness (do the routes
agree on the count?) and cost (how long did each take?) are compared as two
independent tables:

  * ``timing_sweep_lps.csv``  : ell, route, m, lp_classes   (correctness)
  * ``timing_sweep_time.csv`` : ell, route, m, seconds_median, seconds_min, repeats

Routes:
  * ``A``     RLE exhaustive ground truth (``run_exhaustive``) — always available;
  * ``brute`` independent pm1 enumeration (``2**ell``, so cost-gated);
  * ``B``     compression funnel (orbit-reduced lift) — only where ``ell`` is
              compressible AND the reduced lift fits the budget.

The combinatorial ``cost_report.csv`` is a workload PROXY; this is its measured
counterpart. HONEST NOTE: at small ell the Python-loop constant factors of the
lift can make route B slower in wall-clock than A/brute despite its far smaller
combinatorial workload — the compression win here is asymptotic, and we report
the raw seconds rather than hide that.
"""
from __future__ import annotations

import csv
import os
import time
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lp_rle.exhaust import run_exhaustive, _reduce_to_canonical
from lp_rle.bruteforce import collect_survivors_brute
from lp_rle.match import find_pairs, inequivalent_classes

from .pipeline import pipeline_B
from .compress import incomparable_divisors
from .validation import reduced_lift_budget, DEFAULT_MAX_LIFT
from .core import paf_int, is_legendre_pair

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

TIME_FIELDS = ["ell", "route", "m", "seconds_median", "seconds_min", "repeats"]
LP_FIELDS = ["ell", "route", "m", "lp_classes"]


# --------------------------------------------------------------------------- #
# the three routes: each returns the inequivalent-LP-class count
# --------------------------------------------------------------------------- #
def route_A(ell: int) -> int:
    """RLE exhaustive ground truth."""
    return run_exhaustive(ell)["lp_classes"]


def route_brute(ell: int) -> int:
    """Independent pm1 brute force -> canonical LP classes (same invariant as A)."""
    bb, _ = collect_survivors_brute(ell)
    return len(inequivalent_classes(find_pairs(_reduce_to_canonical(bb))))


def route_B(ell: int, m: int) -> int:
    """Compression funnel: sieve -> orbit-reduce -> lift -> LP test."""
    return pipeline_B(ell, m=m, orbit_reduce=True)["lp_classes"]


# --------------------------------------------------------------------------- #
# timing helpers
# --------------------------------------------------------------------------- #
def _time(fn: Callable[[], int], repeats: int) -> Tuple[int, float, float]:
    """Run ``fn`` ``repeats`` times; return (result, median_seconds, min_seconds)."""
    ts: List[float] = []
    res = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn()
        ts.append(time.perf_counter() - t0)
    return res, median(ts), min(ts)


def _warmup() -> None:
    """Trigger numba JIT / import costs OFF the clock before any timing."""
    a = np.ones(9, dtype=np.int8)
    paf_int(a)
    is_legendre_pair(a, a)
    run_exhaustive(3)
    route_brute(3)
    pipeline_B(9, m=3, orbit_reduce=True)


def liftable_modulus(ell: int, max_lift: int = DEFAULT_MAX_LIFT) -> Optional[int]:
    """Cheapest incomparable divisor whose reduced lift fits ``max_lift`` (or None)."""
    fits = [d for d in incomparable_divisors(ell)
            if reduced_lift_budget(ell, d) <= max_lift]
    return min(fits, key=lambda d: reduced_lift_budget(ell, d)) if fits else None


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #
def sweep(Ls: Sequence[int], repeats: int = 3, brute_max: int = 21,
          max_lift: int = DEFAULT_MAX_LIFT, warmup: bool = True
          ) -> Tuple[List[Dict], List[Dict]]:
    """Time every available route over ``Ls``; return (time_rows, lp_rows).

    Correctness and timing are kept in separate row lists on purpose. Route
    ``brute`` runs only for ``ell <= brute_max`` (its ``2**ell`` cost); route
    ``B`` runs only where a liftable modulus exists.
    """
    if warmup:
        _warmup()
    time_rows: List[Dict] = []
    lp_rows: List[Dict] = []

    def record(ell, route, m, fn):
        cls, med, mn = _time(fn, repeats)
        time_rows.append({"ell": ell, "route": route, "m": m,
                          "seconds_median": med, "seconds_min": mn, "repeats": repeats})
        lp_rows.append({"ell": ell, "route": route, "m": m, "lp_classes": cls})

    for ell in Ls:
        record(ell, "A", "", lambda e=ell: route_A(e))
        if ell <= brute_max:
            record(ell, "brute", "", lambda e=ell: route_brute(e))
        m = liftable_modulus(ell, max_lift)
        if m is not None:
            record(ell, "B", m, lambda e=ell, mm=m: route_B(e, mm))
    return time_rows, lp_rows


def _write(rows: List[Dict], fields: List[str], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def write_results(time_rows, lp_rows, results_dir: str = RESULTS) -> Tuple[str, str]:
    """Write the two separate CSVs; return their paths (time, lps)."""
    tp = _write(time_rows, TIME_FIELDS, os.path.join(results_dir, "timing_sweep_time.csv"))
    lp = _write(lp_rows, LP_FIELDS, os.path.join(results_dir, "timing_sweep_lps.csv"))
    return tp, lp


def classes_agree(lp_rows: List[Dict]) -> Dict[int, bool]:
    """Per-ell: do all routes that ran report the same lp_classes?"""
    by_ell: Dict[int, List[int]] = {}
    for r in lp_rows:
        by_ell.setdefault(r["ell"], []).append(r["lp_classes"])
    return {ell: (len(set(v)) == 1) for ell, v in by_ell.items()}


if __name__ == "__main__":
    Ls = list(range(3, 22, 2))
    time_rows, lp_rows = sweep(Ls)
    tp, lp = write_results(time_rows, lp_rows)

    # correctness table
    agree = classes_agree(lp_rows)
    counts = {}
    for r in lp_rows:
        counts.setdefault(r["ell"], {})[r["route"]] = r["lp_classes"]
    print("LP classes found (correctness):")
    print(f"{'ell':>3} {'A':>4} {'brute':>6} {'B':>4} {'agree':>6}")
    for ell in Ls:
        c = counts.get(ell, {})
        print(f"{ell:>3} {c.get('A','-'):>4} {c.get('brute','-'):>6} "
              f"{c.get('B','-'):>4} {'yes' if agree.get(ell) else 'NO':>6}")

    # timing table
    secs = {}
    for r in time_rows:
        secs.setdefault(r["ell"], {})[r["route"]] = r["seconds_median"]
    print("\nwall-clock seconds (median of repeats):")
    print(f"{'ell':>3} {'A':>9} {'brute':>9} {'B':>9}")
    for ell in Ls:
        s = secs.get(ell, {})
        def f(x):
            return f"{x:.4f}" if isinstance(x, float) else "-"
        print(f"{ell:>3} {f(s.get('A')):>9} {f(s.get('brute')):>9} {f(s.get('B')):>9}",
              flush=True)
    print(f"\nwrote {tp}\nwrote {lp}")
