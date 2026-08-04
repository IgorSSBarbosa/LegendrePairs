"""run_length_stats.py — how leading-+/trailing-- run lengths grow with ell.

Reads canonical_orbits/results/canonical_dataset.csv (built by
build_canonical_dataset.py) and, for every canonical LP class, measures:

  lead_u, lead_v   = length of the leading run of '+' in canonical_u / canonical_v
  trail_u, trail_v = length of the trailing run of '-' in canonical_u / canonical_v

Since the SAME restriction (i,j) will later be applied to BOTH sequences of a
pair (restriction_matrix.py), the pair-level quantity that matters there is
  lead_pair  = min(lead_u, lead_v)     -- the largest leading-+ restriction
  trail_pair = min(trail_u, trail_v)      that still keeps this exact pair.
We report both the per-slot (u vs v) and the pooled/pair-level views.

Outputs (canonical_orbits/results/):
  run_length_rows.csv     -- one row per class, with all 6 derived columns
  run_length_summary.csv  -- one row per ell: n_classes, and max/min/mean of
                             lead_* and trail_* (pooling u and v together)
  run_length_trend.png    -- max/mean/min of leading and trailing runs vs ell
  run_length_dist.png     -- boxplot of the pair-level distributions vs ell
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from lp_group import leading_plus, trailing_minus  # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "results")
DATASET_PATH = os.path.join(RESULTS_DIR, "canonical_dataset.csv")
ROWS_PATH = os.path.join(RESULTS_DIR, "run_length_rows.csv")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "run_length_summary.csv")
TREND_PLOT = os.path.join(RESULTS_DIR, "run_length_trend.png")
DIST_PLOT = os.path.join(RESULTS_DIR, "run_length_dist.png")

ROW_FIELDS = ["ell", "canonical_u", "canonical_v", "orbit_size",
              "lead_u", "trail_u", "lead_v", "trail_v",
              "lead_pair", "trail_pair"]


def load_rows() -> list[dict]:
    with open(DATASET_PATH, newline="") as f:
        reader = csv.DictReader(f)
        base = list(reader)

    rows = []
    for r in base:
        u, v = r["canonical_u"], r["canonical_v"]
        lead_u, trail_u = leading_plus(u), trailing_minus(u)
        lead_v, trail_v = leading_plus(v), trailing_minus(v)
        rows.append({
            "ell": int(r["ell"]),
            "canonical_u": u,
            "canonical_v": v,
            "orbit_size": int(r["orbit_size"]),
            "lead_u": lead_u, "trail_u": trail_u,
            "lead_v": lead_v, "trail_v": trail_v,
            "lead_pair": min(lead_u, lead_v),
            "trail_pair": min(trail_u, trail_v),
        })
    return rows


def write_rows(rows: list[dict]) -> None:
    with open(ROWS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    by_ell: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_ell[r["ell"]].append(r)

    summary = []
    for ell in sorted(by_ell):
        rs = by_ell[ell]
        # pool u and v slots together for the "starting/ending run" distribution
        leads = [r["lead_u"] for r in rs] + [r["lead_v"] for r in rs]
        trails = [r["trail_u"] for r in rs] + [r["trail_v"] for r in rs]
        summary.append({
            "ell": ell,
            "n_classes": len(rs),
            "lead_max": max(leads), "lead_min": min(leads),
            "lead_mean": float(np.mean(leads)),
            "trail_max": max(trails), "trail_min": min(trails),
            "trail_mean": float(np.mean(trails)),
            "lead_pair_max": max(r["lead_pair"] for r in rs),
            "trail_pair_max": max(r["trail_pair"] for r in rs),
        })
    return summary


def write_summary(summary: list[dict]) -> None:
    fields = ["ell", "n_classes", "lead_max", "lead_min", "lead_mean",
              "trail_max", "trail_min", "trail_mean",
              "lead_pair_max", "trail_pair_max"]
    with open(SUMMARY_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)


def plot_trend(summary: list[dict]) -> None:
    ell = [s["ell"] for s in summary]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

    ax = axes[0]
    ax.plot(ell, [s["lead_max"] for s in summary], "o-", label="max")
    ax.plot(ell, [s["lead_mean"] for s in summary], "s-", label="mean")
    ax.plot(ell, [s["lead_min"] for s in summary], "^-", label="min")
    ax.set_title("leading '+' run length (pooled u,v)")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel("run length")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ell, [s["trail_max"] for s in summary], "o-", label="max")
    ax.plot(ell, [s["trail_mean"] for s in summary], "s-", label="mean")
    ax.plot(ell, [s["trail_min"] for s in summary], "^-", label="min")
    ax.set_title("trailing '-' run length (pooled u,v)")
    ax.set_xlabel(r"$\ell$")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(TREND_PLOT, dpi=150)
    plt.close(fig)


def plot_dist(rows: list[dict]) -> None:
    by_ell: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_ell[r["ell"]].append(r)
    ells = sorted(by_ell)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    lead_data = [[r["lead_pair"] for r in by_ell[e]] for e in ells]
    trail_data = [[r["trail_pair"] for r in by_ell[e]] for e in ells]

    axes[0].boxplot(lead_data, positions=ells, widths=1.2)
    axes[0].set_title(r"distribution of $\min$(lead$_u$, lead$_v$) per class")
    axes[0].set_ylabel("leading '+' run length")
    axes[0].grid(alpha=0.3)

    axes[1].boxplot(trail_data, positions=ells, widths=1.2)
    axes[1].set_title(r"distribution of $\min$(trail$_u$, trail$_v$) per class")
    axes[1].set_ylabel("trailing '-' run length")
    axes[1].set_xlabel(r"$\ell$")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(DIST_PLOT, dpi=150)
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    write_rows(rows)
    summary = summarize(rows)
    write_summary(summary)
    plot_trend(summary)
    plot_dist(rows)

    print(f"{'ell':>3} {'n':>5} {'lead(min/mean/max)':>20} {'trail(min/mean/max)':>20}")
    print("-" * 55)
    for s in summary:
        print(f"{s['ell']:>3} {s['n_classes']:>5} "
              f"{s['lead_min']:>5}/{s['lead_mean']:>5.1f}/{s['lead_max']:>5} "
              f"{s['trail_min']:>7}/{s['trail_mean']:>5.1f}/{s['trail_max']:>5}")
    print(f"\nwrote {ROWS_PATH}\nwrote {SUMMARY_PATH}\nwrote {TREND_PLOT}\nwrote {DIST_PLOT}")


if __name__ == "__main__":
    main()
