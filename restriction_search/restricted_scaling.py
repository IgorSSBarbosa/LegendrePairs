"""restricted_scaling.py — does p75_leading keep the restricted search fast
out to larger ell, specifically ell=3..37?

Scope, deliberately narrower than sweep_methods.py/benchmark.py:

  - RESTRICTED ONLY, exhaustive only (binary + RLE), no unrestricted
    comparison and no metaheuristics. An earlier run in this project needed
    ~5h on 12 cores to find LP pairs at ell=37 by (near-)exhaustive means
    (see git log "LP 37 pairs 5h computation in 12 cores my pc") -- running
    the unrestricted case here would just reproduce that cost. Metaheuristics
    were already unreliable by ell=23 in sweep_methods_p75_24seeds.png
    (tabu/sa success rates crash well before ell=27); re-testing them out to
    37 would mostly burn time on expected failures rather than answer the
    question this script is for.
  - Single DETERMINISTIC run per (ell, rep), not multi-seed. At ell=37 the
    restricted per-sequence candidate count is ~5.7e8 (p75_leading gives
    (i,j)=(4,1) there) -- far too large to materialize into a Python list
    and shuffle per seed the way find_first_*_exhaustive_multi_seed does.
    Both find_first_binary_exhaustive and find_first_rle_exhaustive support
    seed=None: a lazy scan straight over the generator (itertools.combinations
    for binary, the recursive run-length composition generator for RLE),
    early-stopping the instant a PAF-complement match is seen, with no
    materialization. time_budget caps the worst case per run.

The question this answers: does the ~1000x-and-growing space reduction from
p75_leading (see compare_rules.png) translate into the restricted search
actually finding a pair in seconds rather than hours, all the way to ell=37.

Output: results/restricted_scaling.csv, results/restricted_scaling.png
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

from restriction_matrix import denominator  # noqa: E402
from restricted_binary import iter_restricted_candidates  # noqa: E402
from restricted_rle import iter_restricted_run_seqs  # noqa: E402
from restriction_rule import p75_leading_ij  # noqa: E402
from common import find_first_pair  # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "restricted_scaling.csv")
PLOT_PATH = os.path.join(RESULTS_DIR, "restricted_scaling.png")

LADDER = list(range(3, 38, 2))   # ell = 3, 5, ..., 37
TIME_BUDGET = 60.0               # seconds/run safety cap


def run_all() -> list[dict]:
    rows = []
    for ell in LADDER:
        i, j = p75_leading_ij(ell)
        d = denominator(ell, i, j)
        print(f"=== ell={ell:>3}: p75_leading (i,j)=({i},{j}), "
              f"per-seq candidate space D={d:,} ===", flush=True)

        r_bin = find_first_pair(iter_restricted_candidates(ell, i, j), ell, time_budget=TIME_BUDGET)
        rows.append({"ell": ell, "rep": "binary", "i": i, "j": j,
                      "candidate_space": d, "found": r_bin["found"],
                      "seconds": r_bin["seconds"], "scanned": r_bin["scanned"],
                      "pruned": r_bin.get("pruned", False)})
        print(f"  binary: found={r_bin['found']} scanned={r_bin['scanned']:,}/{d:,} "
              f"in {r_bin['seconds']:.4g}s{' (PRUNED)' if r_bin.get('pruned') else ''}",
              flush=True)

        i_rle, j_rle = max(i, 1), max(j, 1)
        r_rle = find_first_pair(iter_restricted_run_seqs(ell, i_rle, j_rle), ell, time_budget=TIME_BUDGET)
        rows.append({"ell": ell, "rep": "rle", "i": i_rle, "j": j_rle,
                      "candidate_space": d, "found": r_rle["found"],
                      "seconds": r_rle["seconds"], "scanned": r_rle["scanned"],
                      "pruned": r_rle.get("pruned", False)})
        print(f"  rle:    found={r_rle['found']} scanned={r_rle['scanned']:,} "
              f"in {r_rle['seconds']:.4g}s{' (PRUNED)' if r_rle.get('pruned') else ''}",
              flush=True)
    return rows


FIELDS = ["ell", "rep", "i", "j", "candidate_space", "found", "seconds", "scanned", "pruned"]


def write_csv(rows: list[dict]) -> None:
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_plot(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for rep, color, marker in [("binary", "tab:blue", "o"), ("rle", "tab:orange", "s")]:
        xs = [r["ell"] for r in rows if r["rep"] == rep and r["found"]]
        ys = [r["seconds"] for r in rows if r["rep"] == rep and r["found"]]
        ax.semilogy(xs, ys, marker + "-", color=color, label=f"{rep} exhaustive (deterministic)")
        pruned_xs = [r["ell"] for r in rows if r["rep"] == rep and not r["found"]]
        for px in pruned_xs:
            ax.axvline(px, color=color, linestyle=":", alpha=0.3)

    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(f"time to first pair (s, log scale); dotted = not found within {TIME_BUDGET:g}s")
    ax.set_title("Restricted (p75_leading) exhaustive search, ell=3..37\n"
                 "single deterministic run, no unrestricted comparison (see docstring)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = run_all()
    write_csv(rows)
    make_plot(rows)
    print(f"\nwrote {CSV_PATH}\nwrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
