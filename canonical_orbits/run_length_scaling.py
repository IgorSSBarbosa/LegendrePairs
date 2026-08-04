"""run_length_scaling.py — is the leading-+/trailing-- run growth log(ell) or sqrt(ell)?

Reads canonical_orbits/results/run_length_summary.csv (built by
run_length_stats.py) and re-plots the max/mean trends from run_length_trend.png
in semilog-x and log-log axes, then fits both a log(ell) and a sqrt(ell) model
to lead_max, lead_mean, trail_max, trail_mean and compares R^2 to see which
scaling law tracks the data better.

Outputs (canonical_orbits/results/):
    run_length_trend_semilogx.png  -- x on log scale, y linear (same 2 panels)
    run_length_trend_loglog.png    -- both axes log
    run_length_scaling_fits.csv    -- per-series fit coefficients + R^2 for
                                       both candidate models
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "run_length_summary.csv")
SEMILOGX_PLOT = os.path.join(RESULTS_DIR, "run_length_trend_semilogx.png")
LOGLOG_PLOT = os.path.join(RESULTS_DIR, "run_length_trend_loglog.png")
FITS_PATH = os.path.join(RESULTS_DIR, "run_length_scaling_fits.csv")

SERIES = ["lead_max", "lead_mean", "trail_max", "trail_mean"]


def load_summary() -> list[dict]:
    with open(SUMMARY_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["ell"] = int(r["ell"])
        for k in SERIES + ["lead_min", "trail_min", "n_classes"]:
            r[k] = float(r[k])
    return rows


def plot_semilogx(summary: list[dict]) -> None:
    ell = [s["ell"] for s in summary]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

    ax = axes[0]
    ax.semilogx(ell, [s["lead_max"] for s in summary], "o-", label="max")
    ax.semilogx(ell, [s["lead_mean"] for s in summary], "s-", label="mean")
    ax.semilogx(ell, [s["lead_min"] for s in summary], "^-", label="min")
    ax.set_title("leading '+' run length (pooled u,v)")
    ax.set_xlabel(r"$\ell$ (log scale)")
    ax.set_ylabel("run length")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.semilogx(ell, [s["trail_max"] for s in summary], "o-", label="max")
    ax.semilogx(ell, [s["trail_mean"] for s in summary], "s-", label="mean")
    ax.semilogx(ell, [s["trail_min"] for s in summary], "^-", label="min")
    ax.set_title("trailing '-' run length (pooled u,v)")
    ax.set_xlabel(r"$\ell$ (log scale)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(SEMILOGX_PLOT, dpi=150)
    plt.close(fig)


def plot_loglog(summary: list[dict]) -> None:
    ell = [s["ell"] for s in summary]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

    ax = axes[0]
    ax.loglog(ell, [s["lead_max"] for s in summary], "o-", label="max")
    ax.loglog(ell, [s["lead_mean"] for s in summary], "s-", label="mean")
    ax.loglog(ell, [s["lead_min"] for s in summary], "^-", label="min")
    ax.set_title("leading '+' run length (pooled u,v)")
    ax.set_xlabel(r"$\ell$ (log scale)")
    ax.set_ylabel("run length (log scale)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.loglog(ell, [s["trail_max"] for s in summary], "o-", label="max")
    ax.loglog(ell, [s["trail_mean"] for s in summary], "s-", label="mean")
    ax.loglog(ell, [s["trail_min"] for s in summary], "^-", label="min")
    ax.set_title("trailing '-' run length (pooled u,v)")
    ax.set_xlabel(r"$\ell$ (log scale)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(LOGLOG_PLOT, dpi=150)
    plt.close(fig)


def r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_models(ell: np.ndarray, y: np.ndarray) -> dict:
    """Fit y = a*log(ell)+b, y = a*sqrt(ell)+b, y = a*ell^b (power law, via
    log-log linear regression), report slope/intercept + R^2 for each."""
    out = {}

    x_log = np.log(ell)
    a, b = np.polyfit(x_log, y, 1)
    out["log"] = {"a": a, "b": b, "r2": r_squared(y, a * x_log + b)}

    x_sqrt = np.sqrt(ell)
    a, b = np.polyfit(x_sqrt, y, 1)
    out["sqrt"] = {"a": a, "b": b, "r2": r_squared(y, a * x_sqrt + b)}

    # power law: log(y) = b*log(ell) + log(a)  -- only meaningful where y>0
    logy = np.log(y)
    b, loga = np.polyfit(x_log, logy, 1)
    a = np.exp(loga)
    pred = a * ell ** b
    out["power"] = {"a": a, "b": b, "r2": r_squared(y, pred)}

    return out


def run_fits(summary: list[dict]) -> list[dict]:
    ell = np.array([s["ell"] for s in summary], dtype=float)
    rows = []
    for name in SERIES:
        y = np.array([s[name] for s in summary], dtype=float)
        models = fit_models(ell, y)
        for model_name, m in models.items():
            rows.append({
                "series": name, "model": model_name,
                "a": m["a"], "b": m["b"], "r2": m["r2"],
            })
    return rows


def write_fits(rows: list[dict]) -> None:
    fields = ["series", "model", "a", "b", "r2"]
    with open(FITS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    summary = load_summary()
    plot_semilogx(summary)
    plot_loglog(summary)
    fits = run_fits(summary)
    write_fits(fits)

    print(f"{'series':>10} {'model':>6} {'a':>9} {'b':>9} {'R2':>8}")
    print("-" * 46)
    for r in fits:
        print(f"{r['series']:>10} {r['model']:>6} {r['a']:>9.4f} {r['b']:>9.4f} {r['r2']:>8.4f}")

    print(f"\nwrote {SEMILOGX_PLOT}\nwrote {LOGLOG_PLOT}\nwrote {FITS_PATH}")


if __name__ == "__main__":
    main()
