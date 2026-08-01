#!/usr/bin/env python3
"""Regenerate the comparison figures for docs/STRATEGY_SUMMARY.md.

Reads the timing CSVs produced across the subprojects and writes three PNGs
into results/:
  - strategy_exhaustive.png   exhaustive wall-clock by representation + extrapolation
  - strategy_findsome.png     restart-until-found scaling (anneal vs basinhop) + extrapolation
  - strategy_rep_compare.png  find-some success by representation (sa/basinhop x rle/binary)

Run from the repo root:  python3 experiments/plot_strategy_summary.py
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def col(rows, key, cast=float):
    return [cast(r[key]) for r in rows]


# ---------------------------------------------------------------- exhaustive
def fig_exhaustive():
    # RLE small-l (timing_sweep route A) + large-l (ttf_scaling exhaust_s)
    sweep = load(os.path.join(ROOT, "compression_experiment/results/timing_sweep_time.csv"))
    rle = {int(r["ell"]): float(r["seconds_median"]) for r in sweep if r["route"] == "A"}
    brute = {int(r["ell"]): float(r["seconds_median"]) for r in sweep if r["route"] == "brute"}
    ttf = load(os.path.join(ROOT, "rle_experiment/results/ttf_scaling.csv"))
    for r in ttf:
        rle[int(r["n"])] = float(r["exhaust_s"])  # extends RLE to 23,25,27
    # compressed parallel (best/min per ell)
    par = load(os.path.join(ROOT, "compression_experiment/results/parallel_B_timing.csv"))
    comp = {}
    for r in par:
        e = int(r["ell"]); t = float(r["total_s"])
        comp[e] = min(comp.get(e, 1e18), t)
    # binary bucket ordered pairs
    bm = load(os.path.join(ROOT, "results/benchmark_methods.csv"))
    bucket = {int(r["ell"]): float(r["exhaustive_scan_s"])
              for r in bm if r["method"] == "exhaustive"}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for data, lab, mk in [(rle, "RLE (E3)", "o"), (brute, "binary brute (E5)", "s"),
                          (bucket, "binary bucket, ordered (E1)", "^"),
                          (comp, "compressed parallel (E6)", "D")]:
        xs = sorted(data)
        ax.plot(xs, [data[x] for x in xs], mk + "-", label=lab)

    # exponential extrapolation of RLE to l=115
    xs = np.array(sorted(rle))
    ys = np.array([rle[x] for x in xs])
    b, a = np.polyfit(xs, np.log(ys), 1)  # log t = a + b*l
    xe = np.array([xs.min(), 115])
    ax.plot(xe, np.exp(a + b * xe), "k--", alpha=0.6,
            label=f"RLE fit  t~exp({b:.3f}·ℓ)")
    t115 = np.exp(a + b * 115)
    ax.annotate(f"ℓ=115 → ~{t115:.1e}s\n(~{t115/3.15e7:.1e} yr)",
                xy=(115, t115), xytext=(60, t115 * 1e-3),
                arrowprops=dict(arrowstyle="->", alpha=0.5), fontsize=9)

    ax.set_yscale("log")
    ax.set_xlabel("ℓ")
    ax.set_ylabel("wall-clock seconds (median)")
    ax.set_title("Exhaustive enumeration: wall-clock by representation")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(RES, "strategy_exhaustive.png")
    fig.savefig(out, dpi=130)
    print("wrote", out, f"| RLE extrapolation ℓ=115 ≈ {t115:.2e}s")


# ---------------------------------------------------------------- find-some scaling
def fig_findsome():
    an = load(os.path.join(ROOT, "results/scaling_time.csv"))
    bh = load(os.path.join(ROOT, "results/scaling_basinhop.csv"))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for rows, lab, mk in [(an, "anneal (S3)", "o"), (bh, "basinhop (S1)", "s")]:
        xs = col(rows, "ell", int)
        ys = col(rows, "median_time_s")
        sr = col(rows, "success_rate")
        ax.plot(xs, ys, mk + "-", label=lab)
        for x, y, s in zip(xs, ys, sr):
            if s < 1.0:
                ax.annotate(f"{s:.0%}", (x, y), fontsize=7, color="red")
        # extrapolate each
        xa = np.array(xs); ya = np.array(ys)
        b, a = np.polyfit(xa, np.log(ya), 1)
        xe = np.array([xa.min(), 115])
        ax.plot(xe, np.exp(a + b * xe), "--", alpha=0.4)
        print(f"{lab}: median-ttf ~ exp({b:.3f}·ℓ); ℓ=115 ≈ {np.exp(a+b*115):.2e}s")

    ax.set_yscale("log")
    ax.set_xlabel("ℓ")
    ax.set_ylabel("median seconds to first pair (restart-until-found)")
    ax.set_title("Find-some scaling: basin-hopping vs annealing (red = success<100%)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(RES, "strategy_findsome.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


# ---------------------------------------------------------------- rep comparison
def fig_rep_compare():
    base = os.path.join(ROOT, "rle_experiment/results/head_to_head_solvers.csv")
    ext = os.path.join(ROOT, "rle_experiment/results/head_to_head_solvers_ext.csv")
    rows = load(base)
    if os.path.exists(ext):
        rows += load(ext)
    # key -> {ell: success_frac}
    series = {}
    for r in rows:
        k = (r["formulation"], r["representation"])
        series.setdefault(k, {})[int(r["L"])] = float(r["success_frac"])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    style = {("sa", "binary"): "o-", ("sa", "rle"): "o--",
             ("basinhop", "binary"): "s-", ("basinhop", "rle"): "s--"}
    for k, d in sorted(series.items()):
        xs = sorted(d)
        ax.plot(xs, [d[x] for x in xs], style.get(k, "x-"),
                label=f"{k[0]}/{k[1]}")
    ax.set_xlabel("ℓ")
    ax.set_ylabel("success fraction (equal step budget)")
    ax.set_title("Find-some: representation × solver (binary ≫ RLE, basinhop ≫ SA)")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(RES, "strategy_rep_compare.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)
    return series


if __name__ == "__main__":
    fig_exhaustive()
    fig_findsome()
    fig_rep_compare()
