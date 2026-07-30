"""bench.py — the measurement apparatus (spec §5).

Two benchmarks:

  1. crossover_paf(): wall-clock of paf_naive / paf_fft / paf_runs across L, to
     report where the O(k*L) run-merge method crosses the O(L^2) naive and the
     FFT path (spec §2.2). At the regime it will actually run in, a random
     sequence has ~L/2 runs, so k ~ L/4.

  2. head_to_head(): the six configurations
        {joint, two-stage} x {rle, binary}   (binary = baseline)
     on equal accepted-move budgets, over many seeds, reporting time-to-first-LP,
     fraction of runs succeeding, distinct LP classes recovered per CPU-second,
     final-objective distribution, moves/sec, and micro vs macro acceptance.
     The headline is the FRACTION OF TOTAL LP EQUIVALENCE CLASSES each method
     recovers per unit time -- possible only because exhaustive ground truth is
     available for the whole ladder.

Run:  python3 lp_rle/bench.py            # modest default sweep -> results/
      python3 lp_rle/bench.py --paf      # only the PAF crossover
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np

# allow `python3 lp_rle/bench.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lp_rle.paf import paf_naive, paf_fft, paf_runs
from lp_rle.runs import runs_to_seq, seq_to_runs
from lp_rle.walk import Meta
from lp_rle.baseline import make_walker
from lp_rle.symmetry import canonical_pair
from lp_rle.exhaust import run_exhaustive

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# --------------------------------------------------------------------------- #
# 1. PAF crossover
# --------------------------------------------------------------------------- #
def crossover_paf(Ls: List[int], repeats: int = 200) -> List[Dict]:
    rows = []
    for L in Ls:
        rng = np.random.default_rng(0)
        # a representative random sequence and its run array
        from lp_rle.walk import random_seq
        v = random_seq(L, rng)
        r = seq_to_runs(v)
        v = runs_to_seq(r)  # re-anchor so runs and seq match
        k = r.shape[0] // 2
        # warm up numba
        paf_naive(v); paf_fft(v); paf_runs(r)

        def timeit(fn, arg):
            t0 = time.perf_counter()
            for _ in range(repeats):
                fn(arg)
            return (time.perf_counter() - t0) / repeats

        t_naive = timeit(paf_naive, v)
        t_fft = timeit(paf_fft, v)
        t_runs = timeit(paf_runs, r)
        rows.append({"L": L, "k": k, "naive_us": t_naive * 1e6,
                     "fft_us": t_fft * 1e6, "runs_us": t_runs * 1e6})
    return rows


def report_crossover(rows: List[Dict]) -> str:
    lines = [f"{'L':>3} {'k':>3} {'naive(us)':>10} {'fft(us)':>9} {'runs(us)':>9}"]
    for r in rows:
        lines.append(f"{r['L']:>3} {r['k']:>3} {r['naive_us']:>10.2f} "
                     f"{r['fft_us']:>9.2f} {r['runs_us']:>9.2f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2. head-to-head search benchmark
# --------------------------------------------------------------------------- #
def _agg(stats_list, extra: Dict) -> Dict:
    tf = [s.time_to_first for s in stats_list if s.time_to_first is not None]
    d = {
        "seeds": len(stats_list),
        "success_frac": np.mean([s.found for s in stats_list]),
        "median_ttf": float(np.median(tf)) if tf else None,
        "mean_best_f": float(np.mean([s.best_f for s in stats_list])),
        "moves_per_s": float(np.mean([s.steps / (s.time_to_first or 1e-9)
                                      if s.time_to_first else 0 for s in stats_list])),
        "micro_acc_rate": float(np.mean([s.as_dict()["micro_acc_rate"] for s in stats_list])),
        "macro_acc_rate": float(np.mean([s.as_dict()["macro_acc_rate"] for s in stats_list])),
        "mean_df_macro": float(np.mean([s.as_dict()["mean_df_macro"] for s in stats_list])),
    }
    d.update(extra)
    return d


def bench_joint(representation, L, seeds, max_steps, meta, base_seed) -> Dict:
    stats_list = []
    found_classes = set()
    t0 = time.perf_counter()
    for s in range(seeds):
        rng = np.random.default_rng(base_seed + s)
        w = make_walker("joint", representation, L, meta, rng)
        st = w.run(max_steps)
        stats_list.append(st)
        sol = w.solution()
        if sol is not None:
            found_classes.add(canonical_pair(*sol))
    wall = time.perf_counter() - t0
    return _agg(stats_list, {"classes_found": len(found_classes),
                             "wall_s": wall,
                             "classes_per_s": len(found_classes) / wall if wall else 0.0})


def bench_two_stage(representation, L, seeds, max_steps, meta, base_seed) -> Dict:
    from lp_rle.walk import TwoStageSearch
    from lp_rle.baseline import make_moveset
    stats_list = []
    t0 = time.perf_counter()
    search = TwoStageSearch(L, make_moveset(representation), meta,
                            np.random.default_rng(base_seed))
    for s in range(seeds):
        search.rng = np.random.default_rng(base_seed + s)
        st = search.stage1(max_steps)
        stats_list.append(st)
    classes = search.stage2()
    wall = time.perf_counter() - t0
    return _agg(stats_list, {"classes_found": len(classes),
                             "wall_s": wall,
                             "classes_per_s": len(classes) / wall if wall else 0.0})


def head_to_head(Ls, seeds=50, max_steps=20000, base_seed=1234, meta=None) -> List[Dict]:
    meta = meta or Meta(kind="sa", T0=6.0, cooling=0.9997)
    rows = []
    for L in Ls:
        gt = run_exhaustive(L)
        total = gt["lp_classes"]
        for formulation in ("joint", "two-stage"):
            for representation in ("rle", "binary"):
                if formulation == "joint":
                    r = bench_joint(representation, L, seeds, max_steps, meta, base_seed)
                else:
                    r = bench_two_stage(representation, L, seeds, max_steps, meta, base_seed)
                r.update({"L": L, "formulation": formulation,
                          "representation": representation,
                          "total_classes": total,
                          "recovered_frac": r["classes_found"] / total if total else 0.0})
                rows.append(r)
                print(f"L={L:>2} {formulation:>9}/{representation:<6} "
                      f"succ={r['success_frac']:.2f} "
                      f"classes={r['classes_found']}/{total} "
                      f"cls/s={r['classes_per_s']:.3f} "
                      f"macroAcc={r['macro_acc_rate']:.3f}")
    return rows


def write_csv(rows: List[Dict], path: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


def plot_headline(rows: List[Dict], path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(matplotlib unavailable, skipping plot: {e})")
        return
    configs = sorted({(r["formulation"], r["representation"]) for r in rows})
    Ls = sorted({r["L"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5))
    for form, rep in configs:
        ys = []
        for L in Ls:
            match = [r for r in rows if r["L"] == L and r["formulation"] == form
                     and r["representation"] == rep]
            ys.append(match[0]["recovered_frac"] if match else np.nan)
        ax.plot(Ls, ys, marker="o", label=f"{form}/{rep}")
    ax.set_xlabel("L")
    ax.set_ylabel("fraction of LP classes recovered")
    ax.set_title("RLE vs binary: LP classes recovered at fixed budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paf", action="store_true", help="only the PAF crossover")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--Ls", type=int, nargs="*", default=[13, 15, 17, 19])
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    paf_rows = crossover_paf([13, 15, 17, 19, 21, 23, 25])
    print("\n== PAF crossover ==")
    print(report_crossover(paf_rows))
    write_csv(paf_rows, os.path.join(RESULTS, "paf_crossover.csv"))
    if args.paf:
        return

    print("\n== head-to-head ==")
    rows = head_to_head(args.Ls, seeds=args.seeds, max_steps=args.steps)
    write_csv(rows, os.path.join(RESULTS, "head_to_head.csv"))
    plot_headline(rows, os.path.join(RESULTS, "headline_recovery.png"))


if __name__ == "__main__":
    main()
