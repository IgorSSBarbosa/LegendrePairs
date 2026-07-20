"""Compare search methods for finding ONE Legendre pair, vs length ell.

Methods
-------
* exhaustive : the "old" search -- scan every sum=+1 candidate and bucket by
               PAF vector. Deterministic; must sweep the whole space, so its
               cost is the candidate scan time (a lower bound on find-a-pair).
* greedy     : local search, accept a swap only if E strictly decreases.
* sideways   : local search, accept if E does not increase.
* anneal     : local search, simulated annealing (Metropolis acceptance).

Why the asymmetry in reporting
------------------------------
The exhaustive method gives one number per ell. The local searches are
stochastic: each trial may or may not reach E = 0 within a step/restart/time
budget. So per (method, ell) we run several trials and report

    success rate            = fraction of trials that found a pair,
    median time-to-solve    = median wall time over the SUCCESSFUL trials.

Two panels are plotted: (left) time vs ell on a log y-axis -- exhaustive scan
against each method's median time-to-solve; (right) success rate vs ell.

Usage
-----
    python3 benchmark_methods.py                 # ell = 5..25
    python3 benchmark_methods.py --max 21        # quicker
    python3 benchmark_methods.py --trials 15 --cap 6
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(_RESULTS, exist_ok=True)

from benchmark_legendre import scan_time
from local_search import search

STOCHASTIC = ("greedy", "sideways", "anneal", "threshold", "basinhop")
COLORS = {"exhaustive": "tab:blue", "greedy": "tab:red",
          "sideways": "tab:orange", "anneal": "tab:green",
          "threshold": "tab:brown", "basinhop": "tab:purple"}
MARKERS = {"exhaustive": "o", "greedy": "x", "sideways": "^", "anneal": "s",
           "threshold": "d", "basinhop": "*"}


def run_stochastic(ell, strategy, trials, restarts, steps, cap, seed):
    """Return (success_rate, median_solve_seconds_or_None, times_list).

    Trial t uses a distinct, reproducible seed derived from ``seed``."""
    solve_times = []
    for t in range(trials):
        res = search(ell, strategy=strategy, restarts=restarts, steps=steps,
                     seed=seed + 1000 * t + ell, max_seconds=cap)
        if res["solved"]:
            solve_times.append(res["seconds"])
    rate = len(solve_times) / trials
    med = statistics.median(solve_times) if solve_times else None
    return rate, med, solve_times


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--min", type=int, default=5)
    p.add_argument("--max", type=int, default=25)
    p.add_argument("--trials", type=int, default=9, help="trials per stochastic cell")
    p.add_argument("--restarts", type=int, default=120, help="restarts per trial")
    p.add_argument("--steps", type=int, default=6000, help="steps per restart")
    p.add_argument("--cap", type=float, default=4.0, help="wall-clock cap per trial (s)")
    p.add_argument("--scan-min-time", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0, help="base RNG seed (default 0)")
    p.add_argument("--csv", default=os.path.join(_RESULTS, "benchmark_methods.csv"))
    p.add_argument("--png", default=os.path.join(_RESULTS, "benchmark_methods.png"))
    args = p.parse_args()

    lengths = list(range(args.min | 1, args.max + 1, 2))

    exhaustive = {}   # ell -> seconds
    rate = {m: {} for m in STOCHASTIC}   # method -> ell -> success rate
    median = {m: {} for m in STOCHASTIC}  # method -> ell -> median solve time

    header = f"{'ell':>4} {'exhaustive':>12} | " + " | ".join(
        f"{m:>18}" for m in STOCHASTIC)
    print(header)
    print(f"{'':>4} {'scan [s]':>12} | " + " | ".join(
        f"{'rate  median[s]':>18}" for _ in STOCHASTIC))
    for ell in lengths:
        exhaustive[ell] = scan_time(ell, normalized=True, min_time=args.scan_min_time)
        cells = []
        for m in STOCHASTIC:
            r, med, _ = run_stochastic(ell, m, args.trials, args.restarts,
                                       args.steps, args.cap, args.seed)
            rate[m][ell] = r
            median[m][ell] = med
            med_s = f"{med:.3f}" if med is not None else "  --"
            cells.append(f"{r*100:4.0f}% {med_s:>11}")
        print(f"{ell:>4} {exhaustive[ell]:>12.4f} | " + " | ".join(cells))

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ell", "method", "success_rate", "median_solve_s", "exhaustive_scan_s"])
        for ell in lengths:
            w.writerow([ell, "exhaustive", 1.0, exhaustive[ell], exhaustive[ell]])
            for m in STOCHASTIC:
                w.writerow([ell, m, rate[m][ell], median[m][ell], exhaustive[ell]])
    print(f"\nsaved data -> {args.csv}")

    make_plot(lengths, exhaustive, rate, median, args)
    return 0


def make_plot(lengths, exhaustive, rate, median, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array(lengths, dtype=float)
    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Panel 1: time to find one pair (log y).
    ax_t.set_title("Time to find ONE Legendre pair (lower is better)")
    ax_t.semilogy(x, [exhaustive[e] for e in lengths], "o-",
                  color=COLORS["exhaustive"], label="exhaustive (full scan)")
    for m in STOCHASTIC:
        xs = [e for e in lengths if median[m][e] is not None]
        ys = [median[m][e] for e in xs]
        if xs:
            ax_t.semilogy(xs, ys, MARKERS[m] + "-", color=COLORS[m],
                          label=f"{m} (median time-to-solve)")
    ax_t.set_xlabel("ell")
    ax_t.set_ylabel("seconds (log scale)")
    ax_t.grid(True, which="both", alpha=0.3)
    ax_t.legend(fontsize=8)

    # Panel 2: success rate of the stochastic methods.
    ax_r.set_title(f"Success rate within budget "
                   f"({args.restarts}x{args.steps} steps, {args.cap}s cap)")
    for m in STOCHASTIC:
        ax_r.plot(x, [rate[m][e] * 100 for e in lengths], MARKERS[m] + "-",
                  color=COLORS[m], label=m)
    ax_r.set_xlabel("ell")
    ax_r.set_ylabel("success rate [%]")
    ax_r.set_ylim(-5, 105)
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=8)

    fig.suptitle("Exhaustive vs local-search methods for Legendre pairs", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.png, dpi=130)
    print(f"saved plot -> {args.png}")


if __name__ == "__main__":
    raise SystemExit(main())
