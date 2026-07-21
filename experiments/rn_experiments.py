"""Phase sweep + lower-bound calibration for the (r, n) restart+kick search.

Two experiments, both written to ``results/``:

* ``sweep``  -- for each odd ell and each (r, n) on a grid, run T independent
  trials and record the success rate, median evals-to-solve, and crucially how
  often Step 2 (the kick phase) is even entered / is what solves it. This
  answers "which (r, n) find an LP for ell up to 20", and exposes whether the
  kicks contribute at all.

* ``calibrate`` -- for each odd ell, sample random seeds and measure the TRUE
  swap-distance to the nearest Legendre pair (exhaustive over the enumerated LP
  set). This is a lower bound on the kick radius any such method needs, plus the
  plateau statistics (how often a random seed lands near the E-floor).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rn_search import sample_E_vs_distance, search_rn  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def sweep(lmin, lmax, rs, ns, trials, max_batches):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "rn_phase_sweep.csv")
    header = ["ell", "r", "n", "trials", "solved", "success_rate",
              "median_evals", "step2_entered", "solved_in_step2"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for ell in range(lmin | 1, lmax + 1, 2):
            for r in rs:
                for n in ns:
                    solved = 0
                    step2_entered = 0
                    solved_in_step2 = 0
                    evals_ok = []
                    t0 = time.perf_counter()
                    for t in range(trials):
                        res = search_rn(ell, r, n, seed=1000 * t + 7,
                                        max_batches=max_batches)
                        if res.get("entered_step2"):
                            step2_entered += 1
                        if res["solved"]:
                            solved += 1
                            evals_ok.append(res["evals"])
                            if res["rounds"] > 0:
                                solved_in_step2 += 1
                    med = int(np.median(evals_ok)) if evals_ok else ""
                    w.writerow([ell, r, n, trials, solved,
                                f"{solved / trials:.3f}", med,
                                step2_entered, solved_in_step2])
                    print(f"ell={ell:2d} r={r:5d} n={n}: "
                          f"solved {solved}/{trials}  "
                          f"med_evals={med}  step2_entered={step2_entered}  "
                          f"solved_in_step2={solved_in_step2}  "
                          f"({time.perf_counter() - t0:.1f}s)")
    print(f"\nwrote {path}")


def calibrate(lmin, lmax, seeds):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "rn_lowerbound.csv")
    header = ["ell", "seeds", "E_min", "E_median", "dist_min", "dist_median",
              "dist_max", "corr_E_dist", "frac_E_le_64", "min_n_for_gate"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for ell in range(lmin | 1, lmax + 1, 2):
            E, dist = sample_E_vs_distance(ell, n_seeds=seeds, seed=0)
            frac_lo = float(np.mean(E <= 64))
            # smallest n whose gate tau=n*ell/2 admits any sampled seed
            min_n = ""
            for n in range(1, 2 * ell):
                if np.any(E <= n * ell / 2.0):
                    min_n = n
                    break
            corr = float(np.corrcoef(E, dist)[0, 1]) if E.std() and dist.std() else 0.0
            w.writerow([ell, seeds, int(E.min()), int(np.median(E)),
                        int(dist.min()), int(np.median(dist)), int(dist.max()),
                        f"{corr:.3f}", f"{frac_lo:.4f}", min_n])
            print(f"ell={ell:2d}: E_min={E.min():4d} E_med={int(np.median(E)):4d} "
                  f"dist_med={int(np.median(dist))} dist_max={dist.max()} "
                  f"frac(E<=64)={frac_lo:.4f} min_n_for_gate={min_n}")
    print(f"\nwrote {path}")


def main() -> int:
    p = argparse.ArgumentParser(description="(r,n) search experiments.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("sweep")
    a.add_argument("--lmin", type=int, default=5)
    a.add_argument("--lmax", type=int, default=19)
    a.add_argument("--rs", type=int, nargs="+", default=[256, 1024])
    a.add_argument("--ns", type=int, nargs="+", default=[1, 2, 3, 4])
    a.add_argument("--trials", type=int, default=10)
    a.add_argument("--max-batches", type=int, default=200)

    b = sub.add_parser("calibrate")
    b.add_argument("--lmin", type=int, default=5)
    b.add_argument("--lmax", type=int, default=19)
    b.add_argument("--seeds", type=int, default=400)

    args = p.parse_args()
    if args.cmd == "sweep":
        sweep(args.lmin, args.lmax, args.rs, args.ns, args.trials, args.max_batches)
    else:
        calibrate(args.lmin, args.lmax, args.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
