"""search.py — one-command runner for the stochastic LP solvers.

Two solvers, two representations, driven on the joint (u, v) objective
f = sum_s (PAF_u + PAF_v + 2)^2 (f = 0  <=>  Legendre pair):

  * sa        -- simulated annealing (the existing JointWalker Metropolis walk)
  * basinhop  -- basin-hopping (local descent + kick + Metropolis on minima)

  * rle       -- run-length move set (micro transfers + macro run moves)
  * binary    -- plain pm1 2-flip baseline (the control)

Examples
--------
    python3 -m lp_rle.search --solver sa       --rep rle    --L 19 --seeds 10
    python3 -m lp_rle.search --solver basinhop --rep binary --L 17 --seeds 10
    python3 -m lp_rle.search --solver basinhop --rep rle    --L 21 --time 5
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .paf import paf_naive
from .walk import Meta, JointWalker, BasinHopper
from .baseline import make_moveset


def _sign(v) -> str:
    return "".join("+" if x > 0 else "-" for x in v)


def _is_lp(u, v) -> bool:
    return bool(np.all(paf_naive(u) + paf_naive(v) == -2))


def build(solver: str, rep: str, L: int, meta: Meta, rng, n_kick: int):
    moves = make_moveset(rep)
    if solver == "sa":
        return JointWalker(L, moves, meta, rng)
    if solver == "basinhop":
        return BasinHopper(L, moves, meta, rng, n_kick=n_kick)
    raise ValueError(f"unknown solver {solver!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stochastic Legendre-pair search.")
    ap.add_argument("--solver", choices=["sa", "basinhop"], default="sa")
    ap.add_argument("--rep", choices=["rle", "binary"], default="rle")
    ap.add_argument("--L", type=int, default=19)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--steps", type=int, default=200_000, help="max proposals per seed")
    ap.add_argument("--time", type=float, default=None, help="per-seed time budget (s)")
    ap.add_argument("--T0", type=float, default=4.0)
    ap.add_argument("--cooling", type=float, default=0.9995)
    ap.add_argument("--n-kick", type=int, default=3, help="macro kicks per hop (basinhop)")
    ap.add_argument("--energy-reg", type=float, default=0.0,
                    help="energy-regularizer weight lambda0 (0 = off)")
    ap.add_argument("--reg-mode", choices=["low", "equal", "both"], default="low",
                    help="regularizer flavor: low energy / equal energy / both")
    ap.add_argument("--reg-cooling", type=float, default=0.999,
                    help="per-step geometric decay of lambda toward 0")
    ap.add_argument("--quiet", action="store_true", help="only print the summary line")
    args = ap.parse_args(argv)

    meta = Meta(kind="sa", T0=args.T0, cooling=args.cooling,
                energy_reg=args.energy_reg, reg_mode=args.reg_mode,
                reg_cooling=args.reg_cooling)
    n_found = 0
    ttfs = []
    t0 = time.perf_counter()
    for s in range(args.seeds):
        rng = np.random.default_rng(s)
        w = build(args.solver, args.rep, args.L, meta, rng, args.n_kick)
        st = w.run(max_steps=args.steps, time_budget=args.time)
        if st.found:
            n_found += 1
            ttfs.append(st.time_to_first)
            sol = w.solution()
            assert sol is not None and _is_lp(*sol), "reported solution is not an LP!"
            if not args.quiet:
                u, v = sol
                print(f"seed={s:>3} FOUND  steps={st.steps:>8} ttf={st.time_to_first:.3f}s")
                print(f"          u={_sign(u)}")
                print(f"          v={_sign(v)}")
        elif not args.quiet:
            print(f"seed={s:>3} ----   steps={st.steps:>8} best_f={st.best_f}")

    dt = time.perf_counter() - t0
    med = f"{np.median(ttfs):.3f}s" if ttfs else "n/a"
    print(f"# solver={args.solver} rep={args.rep} L={args.L}: "
          f"{n_found}/{args.seeds} found  median_ttf={med}  wall={dt:.2f}s")


if __name__ == "__main__":
    main()
