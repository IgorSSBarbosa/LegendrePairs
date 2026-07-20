"""Diagnose the PAF objective landscape to guide method choice.

We ask two empirical questions about E(A,B) = sum_s (PAF_A(s)+PAF_B(s)+2)^2:

1. How rugged is it?  From many random starts, run STEEPEST DESCENT -- at each
   step take the single swap (one +1 <-> one -1, within A or B) that lowers E
   the most, until no swap improves. The E value where it halts is a true local
   minimum of the swap neighbourhood. Their distribution is the ruggedness
   fingerprint: if descent reaches E=0 often, a gradient-guided move will shine;
   if it stalls at shallow E>0 minima, the escape mechanism (annealing) is what
   matters.

2. What do trajectories look like?  We record E vs iteration for a few steepest
   descent runs (which stall) and a few annealing runs (which can escape), and
   measure plateau lengths in the annealing runs.

Steepest descent here IS the discrete gradient: it scans the full swap
neighbourhood and computes each move's exact dE. So this script doubles as a
proof-of-concept for the gradient-guided idea.

Usage
-----
    python3 analyze_landscape.py                 # ell=23
    python3 analyze_landscape.py --ell 25 --starts 200
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter

import numpy as np

from local_search import _accept, objective, paf_vector, random_sum_one


def best_improving_swap(a, b, pa, pb, E, half):
    """Return (dE, tag, i, j, pnew) for the most-improving swap, or None if the
    configuration is a local minimum (no swap strictly lowers E)."""
    best = None
    for seq, pother, tag in ((a, pb, "A"), (b, pa, "B")):
        plus = np.flatnonzero(seq == 1)
        minus = np.flatnonzero(seq == -1)
        for i in plus:
            for j in minus:
                seq[i], seq[j] = -1, 1
                pnew = paf_vector(seq, half)
                dE = objective(pnew, pother) - E
                seq[i], seq[j] = 1, -1
                if dE < 0 and (best is None or dE < best[0]):
                    best = (dE, tag, i, j, pnew)
    return best


def steepest_descent(ell, rng, max_steps=2000):
    """Descend to a local minimum. Returns (local_min_E, trajectory, steps)."""
    half = (ell - 1) // 2
    a = random_sum_one(ell, rng)
    b = random_sum_one(ell, rng)
    pa = paf_vector(a, half)
    pb = paf_vector(b, half)
    E = objective(pa, pb)
    traj = [E]
    for step in range(max_steps):
        if E == 0:
            break
        mv = best_improving_swap(a, b, pa, pb, E, half)
        if mv is None:
            break  # true local minimum
        dE, tag, i, j, pnew = mv
        seq, pself = (a, pa) if tag == "A" else (b, pb)
        seq[i], seq[j] = -1, 1
        pself[:] = pnew
        E += dE
        traj.append(E)
    return E, traj, step


def anneal_trajectory(ell, rng, steps, t0, t_end):
    """Single annealing run; returns the full E-trajectory (may reach 0)."""
    half = (ell - 1) // 2
    a = random_sum_one(ell, rng)
    b = random_sum_one(ell, rng)
    pa = paf_vector(a, half)
    pb = paf_vector(b, half)
    E = objective(pa, pb)
    traj = [E]
    cool = (t_end / t0) ** (1.0 / max(steps, 1))
    T = t0
    for _ in range(steps):
        if E == 0:
            break
        seq, pself, pother = (a, pa, pb) if rng.random() < 0.5 else (b, pb, pa)
        plus = np.flatnonzero(seq == 1)
        minus = np.flatnonzero(seq == -1)
        i = plus[rng.integers(plus.size)]
        j = minus[rng.integers(minus.size)]
        seq[i], seq[j] = -1, 1
        pnew = paf_vector(seq, half)
        E_new = objective(pnew, pother)
        dE = E_new - E
        if _accept(dE, T, "anneal", rng):
            pself[:] = pnew
            E = E_new
        else:
            seq[i], seq[j] = 1, -1
        T *= cool
        traj.append(E)
    return traj


def plateau_lengths(traj):
    """Run-length of consecutive equal-E stretches in a trajectory."""
    lengths = []
    run = 1
    for prev, cur in zip(traj, traj[1:]):
        if cur == prev:
            run += 1
        else:
            lengths.append(run)
            run = 1
    lengths.append(run)
    return lengths


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ell", type=int, default=23)
    p.add_argument("--starts", type=int, default=150, help="steepest-descent starts")
    p.add_argument("--anneal-runs", type=int, default=6)
    p.add_argument("--steps", type=int, default=6000, help="annealing steps")
    p.add_argument("--t0", type=float, default=3.0)
    p.add_argument("--t-end", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--png", default="landscape_analysis.png")
    args = p.parse_args()

    if args.ell <= 0 or args.ell % 2 == 0:
        p.error(f"ell must be a positive odd integer, got {args.ell}")

    rng = np.random.default_rng(args.seed)

    # --- 1. Steepest-descent local-minima census ---
    minima = []
    descent_trajs = []
    for k in range(args.starts):
        E, traj, steps = steepest_descent(args.ell, rng)
        minima.append(E)
        if len(descent_trajs) < 6 and E > 0:
            descent_trajs.append(traj)
    minima = np.array(minima)
    solved = int(np.sum(minima == 0))
    stuck = minima[minima > 0]

    print(f"=== steepest-descent census, ell={args.ell}, {args.starts} starts ===")
    print(f"reached global min E=0 : {solved}/{args.starts} "
          f"({100*solved/args.starts:.0f}%)")
    if stuck.size:
        print(f"stuck at local min E>0 : {stuck.size}/{args.starts}")
        print(f"  local-min E: min={stuck.min()}, median={int(np.median(stuck))}, "
              f"max={stuck.max()}, mean={stuck.mean():.1f}")
        print(f"  distinct local-min E values: {len(set(stuck.tolist()))}")
        top = Counter(stuck.tolist()).most_common(6)
        print(f"  most common stuck-E: {top}")

    # --- 2. Annealing trajectories + plateau stats ---
    anneal_trajs = []
    all_plateaus = []
    reached = 0
    for k in range(args.anneal_runs):
        traj = anneal_trajectory(args.ell, rng, args.steps, args.t0, args.t_end)
        anneal_trajs.append(traj)
        all_plateaus.extend(plateau_lengths(traj))
        if traj[-1] == 0:
            reached += 1
    print(f"\n=== annealing, ell={args.ell}, {args.anneal_runs} runs ===")
    print(f"reached E=0 : {reached}/{args.anneal_runs}")
    print(f"plateau length (consecutive equal-E steps): "
          f"median={statistics.median(all_plateaus)}, "
          f"mean={statistics.mean(all_plateaus):.1f}, max={max(all_plateaus)}")

    make_plot(args, minima, stuck, descent_trajs, anneal_trajs)
    return 0


def make_plot(args, minima, stuck, descent_trajs, anneal_trajs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_tr, ax_h) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Panel 1: trajectories
    ax_tr.set_title(f"E-trajectories, ell={args.ell}")
    for tr in descent_trajs:
        ax_tr.plot(tr, color="tab:red", alpha=0.7, lw=1.3)
    for tr in anneal_trajs:
        ax_tr.plot(tr, color="tab:green", alpha=0.5, lw=1.0)
    ax_tr.plot([], [], color="tab:red", label="steepest descent (stalls at E>0)")
    ax_tr.plot([], [], color="tab:green", label="annealing (can reach E=0)")
    ax_tr.axhline(0, color="gray", ls=":", lw=1)
    ax_tr.set_xlabel("iteration")
    ax_tr.set_ylabel("objective E")
    ax_tr.set_yscale("symlog")
    ax_tr.grid(True, which="both", alpha=0.3)
    ax_tr.legend(fontsize=8)

    # Panel 2: local-minima histogram
    ax_h.set_title(f"Steepest-descent local minima ({args.starts} starts)")
    if stuck.size:
        bins = np.arange(0, stuck.max() + 8, 4)
        ax_h.hist(stuck, bins=bins, color="tab:red", alpha=0.75,
                  label="stalled at E>0")
    solved = int(np.sum(minima == 0))
    ax_h.bar([0], [solved], width=3, color="tab:green",
             label=f"reached E=0 ({solved})")
    ax_h.set_xlabel("local-minimum objective E")
    ax_h.set_ylabel("count")
    ax_h.grid(True, alpha=0.3)
    ax_h.legend(fontsize=8)

    fig.suptitle("PAF objective landscape: ruggedness and trajectories", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.png, dpi=130)
    print(f"\nsaved plot -> {args.png}")


if __name__ == "__main__":
    raise SystemExit(main())
