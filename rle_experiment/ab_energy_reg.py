"""ab_energy_reg.py — A/B the energy regularizer against ground truth.

For each L and each {sa,basinhop}x{rle,binary} config, run the walk twice on an
equal step budget and identical seeds:
    base -> Meta with energy_reg=0.0   (the un-regularized control)
    reg  -> Meta with energy_reg>0     (low-energy bias, annealed to 0)
and compare success rate and distinct LP classes recovered (of the exhaustive
total). recovered_frac is the diversity signal; success_frac the solve signal.

Run:  python3 ab_energy_reg.py
"""
from __future__ import annotations

import csv
import os

from lp_rle.walk import Meta
from lp_rle.bench import bench_joint, bench_basinhop
from lp_rle.exhaust import run_exhaustive

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

Ls = [13, 15, 17, 19, 21]
SEEDS = 40
STEPS = 20000
BASE_SEED = 1234

BASE = Meta(kind="sa", T0=6.0, cooling=0.9997)
REG = Meta(kind="sa", T0=6.0, cooling=0.9997,
           energy_reg=0.1, reg_mode="low", reg_cooling=0.999)


def main():
    rows = []
    print(f"{'L':>3} {'config':>16} {'succ base':>9} {'succ reg':>9} "
          f"{'cls base':>9} {'cls reg':>9} {'total':>6}")
    for L in Ls:
        total = run_exhaustive(L)["lp_classes"]
        for solver in ("sa", "basinhop"):
            fn = bench_joint if solver == "sa" else bench_basinhop
            for rep in ("rle", "binary"):
                rb = fn(rep, L, SEEDS, STEPS, BASE, BASE_SEED)
                rr = fn(rep, L, SEEDS, STEPS, REG, BASE_SEED)
                cfg = f"{solver}/{rep}"
                rows.append({
                    "L": L, "solver": solver, "rep": rep, "total_classes": total,
                    "succ_base": rb["success_frac"], "succ_reg": rr["success_frac"],
                    "classes_base": rb["classes_found"], "classes_reg": rr["classes_found"],
                    "recovered_base": rb["classes_found"] / total if total else 0.0,
                    "recovered_reg": rr["classes_found"] / total if total else 0.0,
                })
                print(f"{L:>3} {cfg:>16} {rb['success_frac']:>9.2f} "
                      f"{rr['success_frac']:>9.2f} {rb['classes_found']:>9} "
                      f"{rr['classes_found']:>9} {total:>6}")
    path = os.path.join(RESULTS, "ab_energy_reg.csv")
    os.makedirs(RESULTS, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
