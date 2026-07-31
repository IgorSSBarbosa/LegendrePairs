"""ab_energy_reg.py — A/B the energy regularizer against ground truth.

For each n and each {sa,basinhop}x{rle,binary} config, run the walk twice on an
equal step budget and identical seeds:
    base -> Meta with energy_reg=0.0   (the un-regularized control)
    reg  -> Meta with energy_reg>0     (low-energy bias, annealed to 0)
and compare three signals, all at matched seeds so the pairing is exact:
    success_frac  -- did the seed find any LP within the budget (solve signal)
    median_ttf    -- median time-to-first-LP over the finders (speed signal)
    classes_found -- distinct inequivalent LP classes recovered (diversity)

Lengths and budget are chosen so success sits in the discriminating 0.2-0.8 band
(saturated small-n points carry no A/B power). Run:  python3 ab_energy_reg.py
"""
from __future__ import annotations

import csv
import os

from lp_rle.walk import Meta
from lp_rle.bench import bench_joint, bench_basinhop
from lp_rle.exhaust import run_exhaustive

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

Ls = [19, 21, 23, 25]
SEEDS = 100
STEPS = 40000
BASE_SEED = 1234

BASE = Meta(kind="sa", T0=6.0, cooling=0.9997)
REG = Meta(kind="sa", T0=6.0, cooling=0.9997,
           energy_reg=0.1, reg_mode="low", reg_cooling=0.999)


def _ttf(r):
    return r["median_ttf"] if r["median_ttf"] is not None else float("nan")


def main():
    rows = []
    hdr = (f"{'n':>3} {'config':>16} {'succ b':>7} {'succ r':>7} "
           f"{'ttf b':>8} {'ttf r':>8} {'cls b':>6} {'cls r':>6} {'tot':>4}")
    print(hdr)
    for L in Ls:
        total = run_exhaustive(L)["lp_classes"]
        for solver in ("sa", "basinhop"):
            fn = bench_joint if solver == "sa" else bench_basinhop
            for rep in ("rle", "binary"):
                rb = fn(rep, L, SEEDS, STEPS, BASE, BASE_SEED)
                rr = fn(rep, L, SEEDS, STEPS, REG, BASE_SEED)
                cfg = f"{solver}/{rep}"
                rows.append({
                    "n": L, "solver": solver, "rep": rep, "total_classes": total,
                    "succ_base": rb["success_frac"], "succ_reg": rr["success_frac"],
                    "ttf_base": _ttf(rb), "ttf_reg": _ttf(rr),
                    "classes_base": rb["classes_found"], "classes_reg": rr["classes_found"],
                    "recovered_base": rb["classes_found"] / total if total else 0.0,
                    "recovered_reg": rr["classes_found"] / total if total else 0.0,
                })
                print(f"{L:>3} {cfg:>16} {rb['success_frac']:>7.2f} "
                      f"{rr['success_frac']:>7.2f} {_ttf(rb):>8.4f} {_ttf(rr):>8.4f} "
                      f"{rb['classes_found']:>6} {rr['classes_found']:>6} {total:>4}",
                      flush=True)
    path = os.path.join(RESULTS, "ab_energy_reg.csv")
    os.makedirs(RESULTS, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
