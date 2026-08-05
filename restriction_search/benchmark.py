"""benchmark.py — does the restriction strategy actually pay off?

For each ell in a small ladder, picks a restriction (i*, j*) and compares
time-to-first-Legendre-pair, restricted vs unrestricted, across:

  - binary exhaustive brute-force  (restricted_binary.py)
  - RLE exhaustive brute-force     (restricted_rle.py; i,j>=1 only, so no
    unrestricted row for this one -- see its module docstring)
  - tabu / simulated annealing / basin-hopping, each over both the binary and
    RLE move sets (restricted_local_search.py)

Restriction rule: p75_leading (restriction_rule.py) -- i(ell) fit to the 75th
percentile of real canonical LPs' leading-run length, j=1 fixed. This
replaces the original argmax-yield-cell rule (still available as
load_best_ij below, unused by run_all now): compare_rules.py showed that rule
freezes at a fixed (2,1) for every ell past 27 (flat ~57-60x space reduction,
no asymptotic win, and it even erodes across ell=13..27 before that), while
p75_leading tracks its live search speed almost exactly and keeps growing to
~4300x reduction by ell=57. See restriction_search/results/compare_rules.png.

Exhaustive trials use a seeded-random scan order (fair, like the seeded
metaheuristic trials) EXCEPT the unrestricted binary case, which is too large
to materialize+shuffle repeatedly (~2*10^7 candidates at ell=27) -- that one
reports a single deterministic-order run, clearly flagged via seeds=1.

Output: results/benchmark_p75.csv, results/benchmark_p75_ttf.png
(original argmax-rule outputs, results/benchmark.csv + benchmark_ttf.png,
left untouched on disk -- see restriction_rule.py for why the rule changed)
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

from restricted_binary import find_first_binary_exhaustive, find_first_binary_exhaustive_multi_seed  # noqa: E402
from restricted_rle import find_first_rle_exhaustive_multi_seed  # noqa: E402
from restricted_local_search import run_trial  # noqa: E402
from restriction_rule import p75_leading_ij  # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "results")
EXACT_DIR = os.path.join(_HERE, "..", "canonical_orbits", "results", "restriction_exact")
CSV_PATH = os.path.join(RESULTS_DIR, "benchmark_p75.csv")
PLOT_PATH = os.path.join(RESULTS_DIR, "benchmark_p75_ttf.png")

LADDER = [15, 19, 23, 27]
N_SEEDS_EXH = 12
N_SEEDS_META = 10
MAX_STEPS = 100_000
TIME_BUDGET = 1.5  # seconds/trial cap for metaheuristics
UNRESTRICTED_BINARY_MAX_ELL = 21  # beyond this, shuffled unrestricted exhaustive is too slow


def load_best_ij(ell: int) -> tuple[int, int]:
    """(i*, j*) = argmax fraction cell in restriction_exact/ell{ell}.csv.

    Superseded by restriction_rule.p75_leading_ij (see module docstring);
    kept here only for reference / re-running the original comparison."""
    path = os.path.join(EXACT_DIR, f"ell{ell}.csv")
    best = (0, 0, -1.0)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            fr = row["fraction"]
            if fr == "":
                continue
            fr = float(fr)
            if fr > best[2]:
                best = (int(row["i"]), int(row["j"]), fr)
    if best[2] < 0:
        raise ValueError(f"no cell with a valid fraction found for ell={ell}")
    return best[0], best[1]


def _agg(records: list[dict], extra: dict) -> dict:
    times = [r["seconds"] for r in records if r.get("found") and r.get("seconds") is not None]
    d = {
        "seeds": len(records),
        "success_frac": float(np.mean([bool(r.get("found")) for r in records])) if records else 0.0,
        "median_ttf": float(np.median(times)) if times else None,
    }
    d.update(extra)
    return d


def bench_exhaustive(ell: int, i: int, j: int, label: str) -> list[dict]:
    rows = []

    if i == 0 and j == 0 and ell > UNRESTRICTED_BINARY_MAX_ELL:
        r = find_first_binary_exhaustive(ell, i, j, seed=None)
        rows.append({"ell": ell, "i": i, "j": j, "restriction": label,
                      "algo": "exhaustive", "rep": "binary",
                      **_agg([r], {"candidate_space": r["candidate_space"],
                                    "note": "deterministic order (space too large to shuffle)"})})
    else:
        recs = find_first_binary_exhaustive_multi_seed(ell, i, j, list(range(N_SEEDS_EXH)))
        rows.append({"ell": ell, "i": i, "j": j, "restriction": label,
                      "algo": "exhaustive", "rep": "binary",
                      **_agg(recs, {"candidate_space": recs[0]["candidate_space"]})})

    if i >= 1 and j >= 1:
        recs = find_first_rle_exhaustive_multi_seed(ell, i, j, list(range(N_SEEDS_EXH)))
        rows.append({"ell": ell, "i": i, "j": j, "restriction": label,
                      "algo": "exhaustive", "rep": "rle",
                      **_agg(recs, {"candidate_space": recs[0].get("candidate_space")})})
    return rows


def bench_meta(ell: int, i: int, j: int, label: str) -> list[dict]:
    rows = []
    for algo in ("sa", "tabu", "basinhop"):
        for rep in ("binary", "rle"):
            recs = [run_trial(algo, rep, ell, i, j, seed=s,
                               max_steps=MAX_STEPS, time_budget=TIME_BUDGET)
                    for s in range(N_SEEDS_META)]
            rows.append({"ell": ell, "i": i, "j": j, "restriction": label,
                          "algo": algo, "rep": rep, **_agg(recs, {})})
    return rows


def run_all() -> list[dict]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []
    for ell in LADDER:
        i_star, j_star = p75_leading_ij(ell)
        print(f"=== ell={ell}: p75_leading restriction (i*,j*)=({i_star},{j_star}) ===", flush=True)
        for i, j, label in [(0, 0, "unrestricted"), (i_star, j_star, "restricted")]:
            t0 = time.time()
            rows = bench_exhaustive(ell, i, j, label) + bench_meta(ell, i, j, label)
            all_rows.extend(rows)
            for r in rows:
                ttf = f"{r['median_ttf']:.4g}" if r["median_ttf"] is not None else "None"
                print(f"  {label:>12} {r['algo']:>10} {r['rep']:>6}: "
                      f"succ={r['success_frac']:.2f} median_ttf={ttf}s", flush=True)
            print(f"  ({label} block took {time.time()-t0:.1f}s)", flush=True)
    return all_rows


FIELDS = ["ell", "i", "j", "restriction", "algo", "rep", "seeds",
          "success_frac", "median_ttf", "candidate_space", "note"]


def write_csv(rows: list[dict]) -> None:
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def plot_ttf(rows: list[dict]) -> None:
    configs = sorted({(r["algo"], r["rep"]) for r in rows})
    fig, axes = plt.subplots(1, len(LADDER), figsize=(4.2 * len(LADDER), 4.5), sharey=True)
    if len(LADDER) == 1:
        axes = [axes]

    for ax, ell in zip(axes, LADDER):
        by_cfg = {(r["algo"], r["rep"], r["restriction"]): r for r in rows if r["ell"] == ell}
        xs, un_vals, re_vals = [], [], []
        for algo, rep in configs:
            un = by_cfg.get((algo, rep, "unrestricted"))
            re = by_cfg.get((algo, rep, "restricted"))
            if un is None and re is None:
                continue
            xs.append(f"{algo}\n{rep}")
            un_vals.append(un["median_ttf"] if un and un["median_ttf"] is not None else np.nan)
            re_vals.append(re["median_ttf"] if re and re["median_ttf"] is not None else np.nan)

        pos = np.arange(len(xs))
        ax.bar(pos - 0.2, un_vals, width=0.4, label="unrestricted")
        ax.bar(pos + 0.2, re_vals, width=0.4, label="restricted (p75_leading)")
        ax.set_yscale("log")
        ax.set_xticks(pos)
        ax.set_xticklabels(xs, fontsize=7, rotation=45, ha="right")
        ax.set_title(rf"$\ell$={ell}")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("median time-to-first-hit (s, log scale)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    rows = run_all()
    write_csv(rows)
    plot_ttf(rows)
    print(f"\nwrote {CSV_PATH}\nwrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
