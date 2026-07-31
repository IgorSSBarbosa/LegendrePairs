"""sa_restart_ext.py — honest SA time-to-first at large n via short restarts.

SA's annealing schedule freezes by ~40k steps, so a single long run mostly fails
at n>=23. The fair 'time to first LP' is therefore wall accumulated across short
restarts until one succeeds -- which reduces to a single run wherever success is
already ~1 (small n, and basin-hopping everywhere). We measure it for SA/binary
at the lengths where the single-run figure had too few finders.

Parallel over seeds (4 workers). Per seed: repeat 40k-step anneals (fresh RNG)
until an LP is found or a wall cap is hit; record cumulative wall to first.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from lp_rle.walk import Meta, JointWalker
from lp_rle.baseline import make_moveset

Ns = [23, 25, 27]
SEEDS = 12
RESTART_STEPS = 40_000
CAP_S = 30.0
BASE = 1234
META = Meta(kind="sa", T0=6.0, cooling=0.9997)


def wall_to_first(args):
    n, seed = args
    moves = make_moveset("binary")
    acc = 0.0
    r = 0
    while acc < CAP_S:
        rng = np.random.default_rng([BASE, n, seed, r])
        w = JointWalker(n, moves, META, rng)
        t0 = time.perf_counter()
        st = w.run(max_steps=RESTART_STEPS)
        dt = time.perf_counter() - t0
        if st.found:
            return n, acc + (st.time_to_first or dt)
        acc += dt
        r += 1
    return n, None  # censored (no LP within cap)


def main():
    tasks = [(n, s) for n in Ns for s in range(SEEDS)]
    results = {n: [] for n in Ns}
    censored = {n: 0 for n in Ns}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for n, ttf in ex.map(wall_to_first, tasks):
            if ttf is None:
                censored[n] += 1
            else:
                results[n].append(ttf)
    print(f"{'n':>3} {'found':>6} {'cens':>5} {'median_ttf':>11} {'mean_ttf':>10}")
    for n in Ns:
        v = results[n]
        med = f"{np.median(v):.2f}" if v else "n/a"
        mean = f"{np.mean(v):.2f}" if v else "n/a"
        print(f"{n:>3} {len(v):>6} {censored[n]:>5} {med:>11} {mean:>10}", flush=True)
    # emit copy-pastable coordinates for the poster figure
    coords = " ".join(f"({n},{np.median(results[n]):.3f})" for n in Ns if results[n])
    print("\nSA restart-until-found median coords:", coords)


if __name__ == "__main__":
    main()
