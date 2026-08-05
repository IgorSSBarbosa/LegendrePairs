"""sweep_methods.py — method comparison across ell=3..27, inside the restricted
search space, with an adaptive per-ell time ceiling.

DO NOT RUN AUTOMATICALLY -- written for review; run by hand once approved
(see the runtime-estimate note near the bottom of this docstring).

Modeled on experiments/benchmark_methods.py's figure (time-to-solve + success
rate vs ell, one line per method), but two things differ:

1. Every method operates INSIDE the restricted space, using the same
   (i*, j*) = argmax cell from canonical_orbits/restriction_exact/ell{ell}.csv
   that restriction_search/benchmark.py already validated as a real win for
   exhaustive search. This is the natural next step after that benchmark: now
   that we know restriction helps, compare methods *within* the restricted
   regime rather than the raw one.

2. The time budget per (method, ell) cell is not a fixed constant. It's set
   adaptively per ell: run the REFERENCE method first (RLE exhaustive, the
   same restricted brute force validated in benchmark.py) to get a baseline
   time for that ell, then cap every other method's trials at
   PRUNE_FACTOR (default 3x) that baseline. A method that blows past 3x the
   reference's own time for that ell is pruned (counted as not-found within
   budget) rather than left to run to a generic fixed cap -- which is what
   produced a lot of uninformative 0.00-success cells in benchmark.py's fixed
   1.5s cap at ell=23,27. Scaling the cap to the reference's own time keeps
   small ell cheap and gives large ell proportionally more room.

Methods plotted (mirroring benchmark_methods.py's "one exhaustive baseline +
several stochastic methods" structure):
    rle_exhaustive   -- THE baseline/reference (bold line, like the old plot's
                        "exhaustive"). Restricted to (i*,j*), falling back to
                        (max(i*,1), max(j*,1)) where the argmax cell is (0,0)
                        for very small ell, since RLE structurally requires
                        i>=1,j>=1 (see restricted_rle.py's docstring).
    binary_exhaustive -- same (i*,j*) restriction, binary parametrization.
                        Already shown statistically indistinguishable from
                        rle_exhaustive once both are fairly (seeded) compared
                        -- included here mainly as a sanity check that still
                        holds across the full ell=3..27 range, not just the
                        {15,19,23,27} spot-check.
    tabu_binary, tabu_rle, sa_binary, sa_rle, basinhop_binary, basinhop_rle
                     -- restricted_local_search.run_trial, same (i*,j*)
                        (0 is fine here; only the RLE EXHAUSTIVE enumeration
                        needs the >=1 fallback, not the RLE move set).

Runtime estimate (for the approval decision, not a guarantee): the adaptive
ceiling should make most of ell=3..21 cheap (sub-second baselines), with the
per-ell cost dominated by ell>=23 where ceiling ~= 3 * (0.03s..0.4s) and 7
methods x N_SEEDS trials each can spend up to that ceiling when they fail to
find a hit. Rough order of magnitude: a few minutes total, similar to or
somewhat less than benchmark.py's ~440s run across just 4 ell values, since
most of this sweep's 13 ell values are far cheaper than ell=27. RLE's
composition-based candidate materialization (the slow part, ~24s at ell=27)
still happens once per ell for the baseline/binary_exhaustive/tabu_rle/
sa_rle/basinhop_rle series that need it -- see NOTE in bench_one_ell below.

Output: results/sweep_methods.csv, results/sweep_methods.png
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "canonical_orbits"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark import load_best_ij  # noqa: E402
from restricted_binary import find_first_binary_exhaustive_multi_seed  # noqa: E402
from restricted_rle import find_first_rle_exhaustive_multi_seed  # noqa: E402
from restricted_local_search import run_trial  # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "sweep_methods.csv")
PLOT_PATH = os.path.join(RESULTS_DIR, "sweep_methods.png")

LADDER = list(range(3, 24, 2))          # ell = 3, 5, ..., 23 (first pass; 25,27 held back
                                         # for a later, longer run -- see runtime table
                                         # discussed before this run)
BASELINE_SEEDS = [0, 1, 2]              # seeds used to establish the per-ell ceiling
N_SEEDS = 8                             # seeds for every other method
PRUNE_FACTOR = 3.0                      # ceiling = PRUNE_FACTOR * baseline_time
MIN_CEILING = 0.5                       # seconds; floor so a tiny baseline doesn't
                                         # produce an unreasonably tight ceiling.
                                         # (0.05s was too tight: it dominated the
                                         # ceiling across ell~9..21, starving the
                                         # pure-Python metaheuristics of enough
                                         # steps and making the success-rate crash
                                         # there a floor artifact, not a real signal)
BASELINE_HARD_CAP = 30.0                # seconds; safety net on the baseline itself
MAX_STEPS = 2_000_000                   # metaheuristic step cap (time_budget binds first)

META_ALGOS = ("tabu", "sa", "basinhop")
META_REPS = ("binary", "rle")

COLORS = {"rle_exhaustive": "black", "binary_exhaustive": "tab:blue",
          "tabu_binary": "tab:red", "tabu_rle": "tab:orange",
          "sa_binary": "tab:green", "sa_rle": "tab:olive",
          "basinhop_binary": "tab:purple", "basinhop_rle": "tab:pink"}
MARKERS = {"rle_exhaustive": "o", "binary_exhaustive": "o",
           "tabu_binary": "x", "tabu_rle": "x",
           "sa_binary": "s", "sa_rle": "s",
           "basinhop_binary": "*", "basinhop_rle": "*"}
METHODS = ["rle_exhaustive", "binary_exhaustive",
           "tabu_binary", "tabu_rle", "sa_binary", "sa_rle",
           "basinhop_binary", "basinhop_rle"]


def _agg(records: list[dict]) -> dict:
    times = [r["seconds"] for r in records if r.get("found") and r.get("seconds") is not None]
    return {
        "seeds": len(records),
        "success_frac": float(np.mean([bool(r.get("found")) for r in records])) if records else 0.0,
        "median_ttf": float(np.median(times)) if times else None,
    }


def bench_one_ell(ell: int) -> tuple[list[dict], float, float]:
    """Returns (rows, baseline_time, ceiling) for this ell.

    NOTE on cost: iter_restricted_run_seqs' candidate list (RLE) is
    materialized independently for the baseline call AND again inside
    bench_meta's rle-rep... no -- bench_meta uses run_trial (the stochastic
    walkers), which does NOT enumerate the RLE candidate space at all, only
    the *_exhaustive_multi_seed calls do. So the RLE materialization happens
    exactly twice per ell here: once for the baseline (rle_exhaustive) and
    implicitly nowhere else, since binary_exhaustive uses the binary
    enumeration instead. Good -- no redundant materialization.
    """
    i_star, j_star = load_best_ij(ell)
    i_rle, j_rle = max(i_star, 1), max(j_star, 1)

    baseline_recs = find_first_rle_exhaustive_multi_seed(
        ell, i_rle, j_rle, BASELINE_SEEDS, time_budget=BASELINE_HARD_CAP)
    baseline_agg = _agg(baseline_recs)
    baseline_time = baseline_agg["median_ttf"]
    if baseline_time is None:
        baseline_time = BASELINE_HARD_CAP  # no successful baseline trial -> be generous
    ceiling = max(PRUNE_FACTOR * baseline_time, MIN_CEILING)

    rows = []
    rows.append({"ell": ell, "i": i_rle, "j": j_rle, "method": "rle_exhaustive",
                  "ceiling": ceiling, **baseline_agg})

    binary_recs = find_first_binary_exhaustive_multi_seed(
        ell, i_star, j_star, list(range(N_SEEDS)), time_budget=ceiling)
    rows.append({"ell": ell, "i": i_star, "j": j_star, "method": "binary_exhaustive",
                  "ceiling": ceiling, **_agg(binary_recs)})

    for algo in META_ALGOS:
        for rep in META_REPS:
            recs = [run_trial(algo, rep, ell, i_star, j_star, seed=s,
                               max_steps=MAX_STEPS, time_budget=ceiling)
                    for s in range(N_SEEDS)]
            rows.append({"ell": ell, "i": i_star, "j": j_star, "method": f"{algo}_{rep}",
                          "ceiling": ceiling, **_agg(recs)})

    return rows, baseline_time, ceiling


def run_all() -> list[dict]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []
    for ell in LADDER:
        t0 = time.time()
        rows, baseline_time, ceiling = bench_one_ell(ell)
        all_rows.extend(rows)
        print(f"=== ell={ell}: baseline={baseline_time:.4g}s  "
              f"ceiling={ceiling:.4g}s  ({time.time()-t0:.1f}s wall) ===", flush=True)
        for r in rows:
            ttf = f"{r['median_ttf']:.4g}" if r["median_ttf"] is not None else "None"
            print(f"  {r['method']:>18} (i={r['i']},j={r['j']}): "
                  f"succ={r['success_frac']:.2f} median_ttf={ttf}s", flush=True)
    return all_rows


FIELDS = ["ell", "i", "j", "method", "seeds", "success_frac", "median_ttf", "ceiling"]


def write_csv(rows: list[dict]) -> None:
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def make_plot(rows: list[dict]) -> None:
    by_method = {m: {r["ell"]: r for r in rows if r["method"] == m} for m in METHODS}

    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(13, 5.2))

    ax_t.set_title("Time to first Legendre pair, inside the restricted space")
    for m in METHODS:
        xs = [e for e in LADDER if by_method[m].get(e, {}).get("median_ttf") is not None]
        ys = [by_method[m][e]["median_ttf"] for e in xs]
        if not xs:
            continue
        lw = 2.5 if m == "rle_exhaustive" else 1.5
        ax_t.semilogy(xs, ys, MARKERS[m] + "-", color=COLORS[m], linewidth=lw,
                      label=f"{m} (median)" if m != "rle_exhaustive" else f"{m} (baseline)")
    ax_t.set_xlabel(r"$\ell$")
    ax_t.set_ylabel("seconds (log scale)")
    ax_t.grid(True, which="both", alpha=0.3)
    ax_t.legend(fontsize=8)

    ax_r.set_title(f"Success rate within the adaptive ceiling "
                   f"({PRUNE_FACTOR:g}x baseline, {N_SEEDS} seeds)")
    for m in METHODS:
        xs = [e for e in LADDER if e in by_method[m]]
        ys = [by_method[m][e]["success_frac"] * 100 for e in xs]
        lw = 2.5 if m == "rle_exhaustive" else 1.5
        ax_r.plot(xs, ys, MARKERS[m] + "-", color=COLORS[m], linewidth=lw, label=m)
    ax_r.set_xlabel(r"$\ell$")
    ax_r.set_ylabel("success rate [%]")
    ax_r.set_ylim(-5, 105)
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=8)

    fig.suptitle("Restricted-space method comparison, ell=3..27 "
                 "(restriction = argmax cell of restriction_exact)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PLOT_PATH, dpi=130)


def main() -> None:
    rows = run_all()
    write_csv(rows)
    make_plot(rows)
    print(f"\nwrote {CSV_PATH}\nwrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
