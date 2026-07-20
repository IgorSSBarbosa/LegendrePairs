"""Benchmark the Legendre-pair search vs length ell and extrapolate to ell=115.

What is timed
-------------
The exhaustive candidate scan: for every +-1 sequence with sum == +1 (the
canonical search space, C(ell, (ell-1)/2) of them) we compute its PAF vector.
That is the dominant, unavoidable cost of an exhaustive search -- matching and
storing pairs only add to it -- so the measured time is a *lower bound* on the
true search time. No sequences are stored, so memory stays flat even at ell=25.

Why two plots
-------------
The cost grows like C(ell, (ell-1)/2) ~ 2^ell / sqrt(ell): it is EXPONENTIAL.
  * On a semilog-y plot (log t vs ell) an exponential is a straight line.
  * On a log-log plot (log t vs log ell) it is CURVED, so a straight-line
    (power-law) fit there systematically UNDER-estimates large ell.
We draw both, fit each, and extrapolate to ell=115 so the gap is explicit.
The physically meaningful estimate is the complexity model
    t(ell) ~ k * C(ell,(ell-1)/2) * ell * (ell-1)/2,
which we fit through the origin and report as the primary prediction.

Usage
-----
    python3 benchmark_legendre.py                 # ell = 5,7,...,25
    python3 benchmark_legendre.py --max 21        # quicker
    python3 benchmark_legendre.py --min-time 0.5  # steadier small-ell timing
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(_RESULTS, exist_ok=True)

from legendre import _candidate_total, _iter_candidates, _paf_key

TARGET = 115  # smallest open Legendre-pair order (paper, ref [5])
YEAR_SECONDS = 365.25 * 24 * 3600
AGE_OF_UNIVERSE_SECONDS = 4.35e17


def scan_time(ell: int, normalized: bool, min_time: float) -> float:
    """Seconds for one exhaustive PAF scan of length ell.

    Small ell is repeated until at least ``min_time`` has elapsed, then the
    time is divided by the number of repetitions for a stable per-scan figure.
    """
    half = (ell - 1) // 2
    reps = 0
    t0 = time.perf_counter()
    while True:
        for seq in _iter_candidates(ell, normalized):
            _paf_key(seq, half)
        reps += 1
        elapsed = time.perf_counter() - t0
        if elapsed >= min_time:
            return elapsed / reps


def humanize(seconds: float) -> str:
    """Render a (possibly astronomical) number of seconds in friendly units."""
    if seconds < 60:
        return f"{seconds:.3g} s"
    if seconds < 3600:
        return f"{seconds / 60:.3g} min"
    if seconds < 86400:
        return f"{seconds / 3600:.3g} h"
    if seconds < YEAR_SECONDS:
        return f"{seconds / 86400:.3g} days"
    years = seconds / YEAR_SECONDS
    if years < 1e9:
        return f"{years:.3g} years"
    universes = seconds / AGE_OF_UNIVERSE_SECONDS
    return f"{years:.3g} years  (~{universes:.3g} x the age of the universe)"


def run_benchmark(lengths, normalized, min_time):
    rows = []
    print(f"{'ell':>4} {'candidates':>14} {'seconds':>12} {'cand/s':>12}")
    for ell in lengths:
        n = _candidate_total(ell, normalized)
        t = scan_time(ell, normalized, min_time)
        rate = n / t if t else float("inf")
        rows.append((ell, n, t))
        print(f"{ell:>4} {n:>14,} {t:>12.6f} {rate:>12,.0f}")
    return rows


def fit_and_extrapolate(rows):
    ell = np.array([r[0] for r in rows], dtype=float)
    cand = np.array([r[1] for r in rows], dtype=float)
    t = np.array([r[2] for r in rows], dtype=float)
    logt = np.log(t)

    # Exponential model (correct scaling): log t = a + b*ell  ->  t = e^a (e^b)^ell.
    b, a = np.polyfit(ell, logt, 1)
    exp_pred = math.exp(a + b * TARGET)

    # Power-law model (what a naive log-log line fit gives): t = C * ell^p.
    p, c = np.polyfit(np.log(ell), logt, 1)
    pow_pred = math.exp(c + p * math.log(TARGET))

    # Complexity model through the origin: t = k * work, work = C(ell,..)*ell*half.
    half = (ell - 1) / 2
    work = cand * ell * half
    k = float(np.sum(t * work) / np.sum(work * work))
    work_target = _candidate_total(TARGET, True) * TARGET * ((TARGET - 1) // 2)
    cx_pred = k * work_target

    return {
        "exp": {"a": a, "b": b, "base": math.exp(b), "pred": exp_pred},
        "pow": {"c": c, "p": p, "pred": pow_pred},
        "cx": {"k": k, "pred": cx_pred},
        "arrays": (ell, t),
    }


def make_plot(rows, fit, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ell, t = fit["arrays"]
    fig, (ax_ll, ax_sl) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Panel 1: log-log (requested) ---
    ax_ll.set_title("log-log:  exponential cost looks CURVED here")
    ax_ll.loglog(ell, t, "o", color="tab:blue", label="measured scan time")
    xs = np.linspace(ell.min(), TARGET, 400)
    ax_ll.loglog(
        xs, np.exp(fit["pow"]["c"]) * xs ** fit["pow"]["p"], "--", color="tab:red",
        label=f"power-law fit  t~ell^{fit['pow']['p']:.2f}  (UNDER-estimates)",
    )
    ax_ll.loglog([TARGET], [fit["pow"]["pred"]], "rx", ms=11, mew=2,
                 label=f"ell=115 (power law): {humanize(fit['pow']['pred'])}")
    ax_ll.axvline(TARGET, color="gray", ls=":", lw=1)
    ax_ll.set_xlabel("ell (log scale)")
    ax_ll.set_ylabel("scan time [s] (log scale)")
    ax_ll.grid(True, which="both", alpha=0.3)
    ax_ll.legend(fontsize=8, loc="upper left")

    # --- Panel 2: semilog-y (correct straight-line scaling) ---
    ax_sl.set_title("semilog-y:  exponential cost is a STRAIGHT line here")
    ax_sl.semilogy(ell, t, "o", color="tab:blue", label="measured scan time")
    xs2 = np.linspace(ell.min(), TARGET, 400)
    ax_sl.semilogy(
        xs2, np.exp(fit["exp"]["a"] + fit["exp"]["b"] * xs2), "--", color="tab:green",
        label=f"exponential fit  t~{fit['exp']['base']:.3f}^ell",
    )
    ax_sl.semilogy([TARGET], [fit["exp"]["pred"]], "gx", ms=11, mew=2,
                   label=f"ell=115 (exp): {humanize(fit['exp']['pred'])}")
    ax_sl.semilogy([TARGET], [fit["cx"]["pred"]], "ks", ms=8, mfc="none",
                   label=f"ell=115 (complexity model): {humanize(fit['cx']['pred'])}")
    ax_sl.axvline(TARGET, color="gray", ls=":", lw=1)
    ax_sl.set_xlabel("ell")
    ax_sl.set_ylabel("scan time [s] (log scale)")
    ax_sl.grid(True, which="both", alpha=0.3)
    ax_sl.legend(fontsize=8, loc="upper left")

    fig.suptitle("Legendre-pair exhaustive-scan cost and extrapolation to ell=115",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=130)
    print(f"\nsaved plot -> {out_png}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min", type=int, default=5, help="smallest odd ell (default 5)")
    parser.add_argument("--max", type=int, default=25, help="largest odd ell (default 25)")
    parser.add_argument("--min-time", type=float, default=0.3,
                        help="min seconds per length for stable timing (default 0.3)")
    parser.add_argument("--all-sums", action="store_true",
                        help="scan sum=+-1 space instead of the sum=+1 canonical space")
    parser.add_argument("--csv", default=os.path.join(_RESULTS, "benchmark_results.csv"))
    parser.add_argument("--png", default=os.path.join(_RESULTS, "benchmark_legendre_pairs.png"))
    args = parser.parse_args()

    normalized = not args.all_sums
    lengths = list(range(args.min | 1, args.max + 1, 2))

    rows = run_benchmark(lengths, normalized, args.min_time)

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ell", "candidates", "seconds"])
        w.writerows(rows)
    print(f"saved data -> {args.csv}")

    fit = fit_and_extrapolate(rows)

    print("\n--- extrapolation to ell = 115 ---")
    print(f"candidates at 115 : C(115,57) = {_candidate_total(TARGET, True):.3e}")
    print(f"power-law (log-log line) : {humanize(fit['pow']['pred']):<40}"
          f"  [t ~ ell^{fit['pow']['p']:.2f}]  <-- WRONG MODEL, under-estimates")
    print(f"exponential (semilog fit): {humanize(fit['exp']['pred']):<40}"
          f"  [t ~ {fit['exp']['base']:.3f}^ell]")
    print(f"complexity model (t=k*N*ell^2/2): {humanize(fit['cx']['pred']):<33}"
          f"  [PRIMARY / most principled]")

    make_plot(rows, fit, args.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
