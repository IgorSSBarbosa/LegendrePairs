"""ttf_scaling.py — time-to-first-LP vs length for the three search algorithms.

Fair, budget-robust comparison of the *first* Legendre pair found by each
algorithm as a function of odd length n:

  * exhaustive  -- full RLE sieve + match wall (no incremental first-LP: the
                   sieve must complete, so its time-to-first == full wall).
  * sa/binary   -- simulated annealing, plain 2-flip move set.
  * basinhop/binary -- basin-hopping, plain 2-flip move set.

Each stochastic solver gets a large per-seed step budget (+ a wall cap) so that
success_frac is high; we then report the MEDIAN per-seed time-to-first-LP, which
is budget-independent when the solver almost always finds one. success_frac is
recorded so any low-power point can be flagged rather than trusted blindly.

Run:  python3 ttf_scaling.py
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np

from lp_rle.walk import Meta, BasinHopper, JointWalker
from lp_rle.baseline import make_moveset
from lp_rle.filter import collect_survivors
from lp_rle.match import find_pairs

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

Ls = [13, 15, 17, 19, 21, 23, 25, 27]
SEEDS = 50
MAX_STEPS = 400_000
TIME_BUDGET = 20.0          # per-seed wall cap (s)
BASE_SEED = 1234
META = Meta(kind="sa", T0=6.0, cooling=0.9997)


def exhaustive_wall(n: int) -> float:
    t0 = time.perf_counter()
    buckets, _ = collect_survivors(n)
    find_pairs(buckets)
    return time.perf_counter() - t0


def stochastic_ttf(kind: str, n: int) -> dict:
    moves = make_moveset("binary")
    ttfs = []
    n_found = 0
    for s in range(SEEDS):
        rng = np.random.default_rng(BASE_SEED + s)
        if kind == "sa":
            w = JointWalker(n, moves, META, rng)
        else:
            w = BasinHopper(n, moves, META, rng)
        st = w.run(max_steps=MAX_STEPS, time_budget=TIME_BUDGET)
        if st.found and st.time_to_first is not None:
            n_found += 1
            ttfs.append(st.time_to_first)
    return {
        "success_frac": n_found / SEEDS,
        "median_ttf": float(np.median(ttfs)) if ttfs else float("nan"),
        "mean_ttf": float(np.mean(ttfs)) if ttfs else float("nan"),
        "n_found": n_found,
    }


def main():
    rows = []
    hdr = (f"{'n':>3} {'exhaust_s':>10} {'sa med_ttf':>11} {'sa succ':>8} "
           f"{'bh med_ttf':>11} {'bh succ':>8}")
    print(hdr)
    for n in Ls:
        ex = exhaustive_wall(n)
        sa = stochastic_ttf("sa", n)
        bh = stochastic_ttf("basinhop", n)
        rows.append({
            "n": n, "exhaust_s": ex,
            "sa_median_ttf": sa["median_ttf"], "sa_success": sa["success_frac"],
            "bh_median_ttf": bh["median_ttf"], "bh_success": bh["success_frac"],
        })
        print(f"{n:>3} {ex:>10.3f} {sa['median_ttf']:>11.4f} {sa['success_frac']:>8.2f} "
              f"{bh['median_ttf']:>11.4f} {bh['success_frac']:>8.2f}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "ttf_scaling.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
