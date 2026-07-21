"""Estimate how a local-search strategy's cost-to-find-one-pair scales with ell.

The point of this experiment (unlike ``benchmark_methods.py``, which just tabulates
success rate and time per ell) is EXTRAPOLATION: fit the growth of the effort to
solve as a function of ell, then predict how long larger ell would take.

Two effort metrics are recorded per (strategy, ell):

* steps-to-solve  -- total Metropolis moves executed before E hit 0,
  ``(restarts_used - 1) * steps + steps_used``.  Hardware-independent, so this is
  what we fit and report as "the scaling".
* wall time       -- seconds to solve.  Machine-specific, reported for a concrete
  time prediction (time ~ steps / throughput).

Because the searches are stochastic, each (strategy, ell) runs ``trials`` seeds
under a fixed budget (restarts x steps, with a per-trial wall cap).  We report the
success rate and the MEDIAN effort over the SUCCESSFUL trials.  The exponential
fit ``median_steps ~ exp(a + b*ell)`` is done only over the ell whose success rate
clears ``--min-fit-rate`` -- otherwise the median is a censored (optimistic)
statistic and the fit would lie.

Honest caveats
--------------
* Median-over-winners underestimates the true effort at any ell where some trials
  fail; the fit is a LOWER bound on the exponent there.  Watch the success-rate
  column -- once it drops, believe the extrapolation less.
* Legendre-pair search is believed exponential; a straight line on a log-y axis is
  consistent with that but a handful of points cannot prove the functional form.
  The reported R^2 says how well the line fits the measured range, nothing more.

Usage
-----
    python3 scaling_time.py --strategy anneal --lmin 13 --lmax 27 --trials 15 \
        --restarts 400 --steps 20000 --auto-t0 --max-seconds 20 --predict 41 51
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(_RESULTS, exist_ok=True)

from local_search import search  # noqa: E402


def total_steps(res: dict, steps: int) -> int:
    """Total Metropolis moves executed before solving (across all restarts)."""
    return (res["restarts_used"] - 1) * steps + res["steps_used"]


def run_point(ell, strategy, trials, restarts, steps, max_seconds, seed, t0):
    """Run ``trials`` seeds at one ell.  Returns a summary dict.

    Effort statistics (median/mean) are over the SUCCESSFUL trials only."""
    solve_times, solve_steps, solve_restarts = [], [], []
    for t in range(trials):
        res = search(ell, strategy=strategy, restarts=restarts, steps=steps,
                     t0=t0, seed=seed + 1000 * t + ell, max_seconds=max_seconds)
        if res["solved"]:
            solve_times.append(res["seconds"])
            solve_steps.append(total_steps(res, steps))
            solve_restarts.append(res["restarts_used"])
    n = len(solve_times)
    return {
        "ell": ell,
        "trials": trials,
        "solved": n,
        "success_rate": n / trials,
        "median_time_s": statistics.median(solve_times) if n else None,
        "mean_time_s": statistics.mean(solve_times) if n else None,
        "median_steps": int(statistics.median(solve_steps)) if n else None,
        "median_restarts": int(statistics.median(solve_restarts)) if n else None,
    }


def fit_exponential(xs, ys):
    """Least-squares fit of ``y ~ exp(a + b*x)`` (i.e. line in log y).

    Returns ``(a, b, r2)`` or ``None`` if fewer than two positive points."""
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None and y > 0]
    if len(pts) < 2:
        return None
    x = np.array([p[0] for p in pts], dtype=float)
    ly = np.log(np.array([p[1] for p in pts], dtype=float))
    b, a = np.polyfit(x, ly, 1)               # slope, intercept
    pred = a + b * x
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(a), float(b), r2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--strategy", default="anneal",
                   choices=["greedy", "sideways", "anneal", "basinhop", "threshold"])
    p.add_argument("--lmin", type=int, default=13)
    p.add_argument("--lmax", type=int, default=27)
    p.add_argument("--trials", type=int, default=15, help="seeds per ell")
    p.add_argument("--restarts", type=int, default=400, help="restart budget per trial")
    p.add_argument("--steps", type=int, default=20000, help="steps per restart")
    p.add_argument("--max-seconds", type=float, default=20.0,
                   help="wall-clock cap per trial")
    p.add_argument("--auto-t0", action="store_true",
                   help="auto-calibrate anneal start temperature")
    p.add_argument("--t0", type=float, default=3.0)
    p.add_argument("--min-fit-rate", type=float, default=0.5,
                   help="only fit ell whose success rate clears this")
    p.add_argument("--predict", type=int, nargs="*", default=[],
                   help="odd ell values to extrapolate the fit to")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default=os.path.join(_RESULTS, "scaling_time.csv"))
    p.add_argument("--png", default=os.path.join(_RESULTS, "scaling_time.png"))
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    t0 = None if args.auto_t0 else args.t0
    lengths = list(range(args.lmin | 1, args.lmax + 1, 2))

    print(f"strategy={args.strategy}  budget={args.restarts}x{args.steps} steps, "
          f"cap={args.max_seconds}s  trials={args.trials}  "
          f"t0={'auto' if t0 is None else t0}")
    print(f"{'ell':>4} {'solved':>8} {'succ%':>6} {'med_time_s':>11} "
          f"{'med_steps':>12} {'med_restarts':>12}")
    rows = []
    for ell in lengths:
        t_wall = time.perf_counter()
        pt = run_point(ell, args.strategy, args.trials, args.restarts, args.steps,
                       args.max_seconds, args.seed, t0)
        rows.append(pt)
        mt = f"{pt['median_time_s']:.4f}" if pt["median_time_s"] is not None else "--"
        ms = f"{pt['median_steps']:,}" if pt["median_steps"] is not None else "--"
        mr = pt["median_restarts"] if pt["median_restarts"] is not None else "--"
        print(f"{ell:>4} {pt['solved']:>3}/{args.trials:<4} "
              f"{pt['success_rate']*100:5.0f} {mt:>11} {ms:>12} {mr:>12}  "
              f"({time.perf_counter() - t_wall:.1f}s)")

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ell", "strategy", "trials", "solved", "success_rate",
                    "median_time_s", "mean_time_s", "median_steps",
                    "median_restarts", "budget_restarts", "budget_steps",
                    "max_seconds"])
        for pt in rows:
            w.writerow([pt["ell"], args.strategy, pt["trials"], pt["solved"],
                        f"{pt['success_rate']:.4f}",
                        "" if pt["median_time_s"] is None else f"{pt['median_time_s']:.6f}",
                        "" if pt["mean_time_s"] is None else f"{pt['mean_time_s']:.6f}",
                        "" if pt["median_steps"] is None else pt["median_steps"],
                        "" if pt["median_restarts"] is None else pt["median_restarts"],
                        args.restarts, args.steps, args.max_seconds])
    print(f"\nsaved data -> {args.csv}")

    _report_fit(rows, args)
    if not args.no_plot:
        try:
            _make_plot(rows, args)
        except Exception as exc:              # pragma: no cover - plotting optional
            print(f"(plot skipped: {exc})")
    return 0


def _fit_rows(rows, args):
    """Rows eligible for the fit: solved, and success rate above the threshold."""
    return [r for r in rows if r["median_steps"] is not None
            and r["success_rate"] >= args.min_fit_rate]


def _report_fit(rows, args) -> None:
    usable = _fit_rows(rows, args)
    if len(usable) < 2:
        print("not enough well-sampled ell to fit a scaling "
              f"(need >=2 with success_rate>={args.min_fit_rate}).")
        return
    xs = [r["ell"] for r in usable]
    fit_steps = fit_exponential(xs, [r["median_steps"] for r in usable])
    fit_time = fit_exponential(xs, [r["median_time_s"] for r in usable])
    a, b, r2 = fit_steps
    print(f"\nexponential fit over ell in {xs} (success_rate>={args.min_fit_rate}):")
    print(f"  median_steps ~ exp({a:.3f} + {b:.4f}*ell)   R^2={r2:.4f}")
    print(f"  growth: x{math.exp(2 * b):.2f} per +2 in ell"
          + (f"; effort doubles every {math.log(2) / b:.1f} in ell" if b > 0 else ""))
    # throughput (steps/second) from the measured points, for a wall-time estimate
    thr = np.median([r["median_steps"] / r["median_time_s"]
                     for r in usable if r["median_time_s"] and r["median_time_s"] > 0])
    print(f"  measured throughput ~ {thr:,.0f} steps/s")

    for target in args.predict:
        if target % 2 == 0:
            print(f"  predict ell={target}: skipped (ell must be odd)")
            continue
        pred_steps = math.exp(a + b * target)
        pred_time_fit = math.exp(fit_time[0] + fit_time[1] * target) if fit_time else None
        pred_time_thr = pred_steps / thr
        line = (f"  predict ell={target}: ~{pred_steps:,.0f} steps"
                f"  (~{_fmt_seconds(pred_time_thr)} at measured throughput")
        if pred_time_fit is not None:
            line += f"; ~{_fmt_seconds(pred_time_fit)} by time-fit"
        print(line + ")")
    print("  NOTE: extrapolation assumes the exponential holds and the median-over-"
          "winners\n        stays representative; treat as an order-of-magnitude guide.")


def _fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.2f}s"
    if s < 3600:
        return f"{s / 60:.1f}min"
    if s < 86400:
        return f"{s / 3600:.1f}h"
    return f"{s / 86400:.1f}d"


def _make_plot(rows, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    solved = [r for r in rows if r["median_steps"] is not None]
    fig, (ax_s, ax_r) = plt.subplots(1, 2, figsize=(13, 5.2))

    ax_s.set_title(f"Effort to find one pair ({args.strategy})")
    if solved:
        xs = [r["ell"] for r in solved]
        ys = [r["median_steps"] for r in solved]
        ax_s.semilogy(xs, ys, "s", color="tab:green", label="median steps-to-solve")
        fit = fit_exponential([r["ell"] for r in _fit_rows(rows, args)],
                              [r["median_steps"] for r in _fit_rows(rows, args)])
        if fit:
            a, b, r2 = fit
            hi = max(args.predict) if args.predict else max(xs)
            xf = np.linspace(min(xs), max(hi, max(xs)), 100)
            ax_s.semilogy(xf, np.exp(a + b * xf), "--", color="tab:gray",
                          label=f"fit exp({a:.2f}+{b:.3f}·ell), R²={r2:.3f}")
            for target in args.predict:
                if target % 2:
                    ax_s.semilogy([target], [math.exp(a + b * target)], "*",
                                  color="tab:red", markersize=12)
    ax_s.set_xlabel("ell")
    ax_s.set_ylabel("steps to solve (log scale)")
    ax_s.grid(True, which="both", alpha=0.3)
    ax_s.legend(fontsize=8)

    ax_r.set_title(f"Success rate within budget "
                   f"({args.restarts}x{args.steps}, {args.max_seconds}s cap)")
    ax_r.plot([r["ell"] for r in rows], [r["success_rate"] * 100 for r in rows],
              "o-", color="tab:blue")
    ax_r.set_xlabel("ell")
    ax_r.set_ylabel("success rate [%]")
    ax_r.set_ylim(-5, 105)
    ax_r.grid(True, alpha=0.3)

    fig.suptitle(f"Scaling of {args.strategy} local search for Legendre pairs",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.png, dpi=130)
    print(f"saved plot -> {args.png}")


if __name__ == "__main__":
    raise SystemExit(main())
