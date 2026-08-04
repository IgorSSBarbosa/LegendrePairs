"""run_length_pair_scaling.py — restriction bounds at the PAIR level.

run_length_stats.py pools u- and v-slots together and reports max/mean/min of
individual leading-+/trailing-- runs. That answers "how long can one sequence's
run be" but NOT "how large a restriction (i,j) can I force on BOTH sequences of
a pair and still have that pair survive" -- which is the question that matters
for restriction_matrix.py, since the SAME (i,j) is forced on u and v together.

A pair survives leading-restriction i iff i <= min(lead_u, lead_v) = lead_pair
(already a column in run_length_rows.csv). So, per ell:

  lead_pair_max = max over canonical pairs of lead_pair
                = the LARGEST leading-+ restriction for which the restricted
                  search space still contains at least one real LP.
  lead_pair_min = min over canonical pairs of lead_pair
                = the SMALLEST leading-+ restriction that still keeps EVERY
                  canonical pair in the restricted search space (below this,
                  you start losing solutions).
                  Identity check: this equals min(pooled lead_u, lead_v),
                  i.e. the old "lead_min" column from run_length_stats.py --
                  min_p min(a_p,b_p) = min(min_p a_p, min_p b_p).

Same definitions for trail_pair_max / trail_pair_min.

Reads:  canonical_orbits/results/run_length_rows.csv
Writes: canonical_orbits/results/pair_restriction_summary.csv
        canonical_orbits/results/pair_restriction_trend.png       (linear)
        canonical_orbits/results/pair_restriction_trend_loglog.png
        canonical_orbits/results/pair_restriction_scaling_fits.csv
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
ROWS_PATH = os.path.join(RESULTS_DIR, "run_length_rows.csv")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "pair_restriction_summary.csv")
TREND_PLOT = os.path.join(RESULTS_DIR, "pair_restriction_trend.png")
LOGLOG_PLOT = os.path.join(RESULTS_DIR, "pair_restriction_trend_loglog.png")
FITS_PATH = os.path.join(RESULTS_DIR, "pair_restriction_scaling_fits.csv")


def load_rows() -> list[dict]:
    with open(ROWS_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["ell"] = int(r["ell"])
        r["lead_pair"] = int(r["lead_pair"])
        r["trail_pair"] = int(r["trail_pair"])
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    by_ell: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_ell[r["ell"]].append(r)

    out = []
    for ell in sorted(by_ell):
        rs = by_ell[ell]
        lp = [r["lead_pair"] for r in rs]
        tp = [r["trail_pair"] for r in rs]
        out.append({
            "ell": ell,
            "n_classes": len(rs),
            "lead_pair_max": max(lp), "lead_pair_min": min(lp),
            "lead_pair_mean": float(np.mean(lp)),
            "trail_pair_max": max(tp), "trail_pair_min": min(tp),
            "trail_pair_mean": float(np.mean(tp)),
        })
    return out


def write_summary(summary: list[dict]) -> None:
    fields = ["ell", "n_classes", "lead_pair_max", "lead_pair_min", "lead_pair_mean",
              "trail_pair_max", "trail_pair_min", "trail_pair_mean"]
    with open(SUMMARY_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)


def plot_trend(summary: list[dict], loglog: bool) -> None:
    ell = [s["ell"] for s in summary]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    plot_fn = "loglog" if loglog else "plot"

    ax = axes[0]
    getattr(ax, plot_fn)(ell, [s["lead_pair_max"] for s in summary], "o-",
                         label="max (largest restriction that still finds A pair)")
    getattr(ax, plot_fn)(ell, [s["lead_pair_mean"] for s in summary], "s-", label="mean")
    getattr(ax, plot_fn)(ell, [s["lead_pair_min"] for s in summary], "^-",
                         label="min (largest restriction that keeps ALL pairs)")
    ax.set_title("leading '+' restriction, PAIR level: min(lead$_u$, lead$_v$)")
    ax.set_xlabel(r"$\ell$" + (" (log)" if loglog else ""))
    ax.set_ylabel("restriction i" + (" (log)" if loglog else ""))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    getattr(ax, plot_fn)(ell, [s["trail_pair_max"] for s in summary], "o-", label="max")
    getattr(ax, plot_fn)(ell, [s["trail_pair_mean"] for s in summary], "s-", label="mean")
    getattr(ax, plot_fn)(ell, [s["trail_pair_min"] for s in summary], "^-", label="min")
    ax.set_title("trailing '-' restriction, PAIR level: min(trail$_u$, trail$_v$)")
    ax.set_xlabel(r"$\ell$" + (" (log)" if loglog else ""))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(LOGLOG_PLOT if loglog else TREND_PLOT, dpi=150)
    plt.close(fig)


def r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_models(ell: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    x_log = np.log(ell)
    a, b = np.polyfit(x_log, y, 1)
    out["log"] = {"a": a, "b": b, "r2": r_squared(y, a * x_log + b)}

    x_sqrt = np.sqrt(ell)
    a, b = np.polyfit(x_sqrt, y, 1)
    out["sqrt"] = {"a": a, "b": b, "r2": r_squared(y, a * x_sqrt + b)}

    logy = np.log(y)
    b, loga = np.polyfit(x_log, logy, 1)
    a = np.exp(loga)
    out["power"] = {"a": a, "b": b, "r2": r_squared(y, a * ell ** b)}
    return out


SERIES = ["lead_pair_max", "lead_pair_mean", "trail_pair_max", "trail_pair_mean"]


def run_fits(summary: list[dict]) -> list[dict]:
    ell = np.array([s["ell"] for s in summary], dtype=float)
    rows = []
    for name in SERIES:
        y = np.array([s[name] for s in summary], dtype=float)
        for model_name, m in fit_models(ell, y).items():
            rows.append({"series": name, "model": model_name,
                         "a": m["a"], "b": m["b"], "r2": m["r2"]})
    return rows


def write_fits(rows: list[dict]) -> None:
    fields = ["series", "model", "a", "b", "r2"]
    with open(FITS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    rows = load_rows()
    summary = summarize(rows)
    write_summary(summary)
    plot_trend(summary, loglog=False)
    plot_trend(summary, loglog=True)
    fits = run_fits(summary)
    write_fits(fits)

    print(f"{'ell':>3} {'n':>5} {'lead_pair(min/mean/max)':>24} {'trail_pair(min/mean/max)':>24}")
    print("-" * 65)
    for s in summary:
        print(f"{s['ell']:>3} {s['n_classes']:>5} "
              f"{s['lead_pair_min']:>6}/{s['lead_pair_mean']:>5.2f}/{s['lead_pair_max']:>6} "
              f"{s['trail_pair_min']:>10}/{s['trail_pair_mean']:>5.2f}/{s['trail_pair_max']:>6}")

    print(f"\n{'series':>16} {'model':>6} {'a':>9} {'b':>9} {'R2':>8}")
    print("-" * 52)
    for r in fits:
        print(f"{r['series']:>16} {r['model']:>6} {r['a']:>9.4f} {r['b']:>9.4f} {r['r2']:>8.4f}")

    print(f"\nwrote {SUMMARY_PATH}\nwrote {TREND_PLOT}\nwrote {LOGLOG_PLOT}\nwrote {FITS_PATH}")


if __name__ == "__main__":
    main()
