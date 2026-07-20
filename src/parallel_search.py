"""Run the local-search restarts of ``local_search.search`` in parallel.

The restarts are embarrassingly parallel: each is an independent random walk on
the PAF objective, and the whole job stops as soon as any one of them hits
E = 0. So we split the ``restarts`` budget into a handful of tasks, farm them out
to a process pool (one interpreter per core, side-stepping the GIL), and return
the first solution -- cancelling the rest.

This is a constant-factor win (roughly the number of physical cores); it does
NOT change the exponential scaling of the problem. It is the cheap, certain
speed-up before the more ambitious GPU population route (see incremental_paf.py).

Determinism
-----------
Task i is seeded ``seed + 7919*i`` (7919 is prime, to spread the streams), so a
given (seed, workers, tasks) triple is reproducible. Because tasks finish in a
nondeterministic order and we return the FIRST solved one, *which* pair comes
back can vary run to run -- but every returned pair is a genuine Legendre pair.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from legendre import is_legendre_pair
from local_search import search, _fmt


def _task(args):
    """Worker: run ``count`` restarts of ``search`` with its own seed.

    Runs in a separate process, so it must be a top-level picklable function and
    take a single argument. Returns the ``search`` result dict, augmented with
    the number of restarts this task was actually given."""
    (ell, strategy, count, steps, t0, t_end, seed, max_seconds) = args
    res = search(ell, strategy=strategy, restarts=count, steps=steps,
                 t0=t0, t_end=t_end, seed=seed, max_seconds=max_seconds,
                 progress=False)
    res["restarts_given"] = count
    return res


def search_parallel(
    ell: int,
    strategy: str = "anneal",
    restarts: int = 200,
    steps: int = 20000,
    t0: float = 3.0,
    t_end: float = 0.05,
    seed: int = 0,
    workers: int | None = None,
    tasks_per_worker: int = 4,
    max_seconds: float | None = None,
):
    """Parallel local search: split ``restarts`` across a process pool.

    Returns a dict like ``local_search.search`` (solved, A, B, best_E, seconds)
    plus ``workers``, ``tasks``, and ``restarts_used`` (restarts consumed by
    tasks that had finished when we stopped). Stops early on the first solution.
    """
    if ell <= 0 or ell % 2 == 0:
        raise ValueError(f"ell must be a positive odd integer, got {ell}")

    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, min(workers, restarts))

    # More tasks than workers keeps every core busy and lets us react to a
    # solution quickly, without the overhead of one task per restart.
    n_tasks = min(restarts, max(workers * tasks_per_worker, workers))
    per = math.ceil(restarts / n_tasks)

    # Hand out restarts in chunks of `per` until the budget is exhausted.
    chunks = []
    remaining = restarts
    i = 0
    while remaining > 0:
        count = min(per, remaining)
        chunks.append((ell, strategy, count, steps, t0, t_end,
                       seed + 7919 * i, max_seconds))
        remaining -= count
        i += 1
    n_tasks = len(chunks)

    t_start = time.perf_counter()
    best_E = None
    restarts_done = 0
    solution = None

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_task, c): c for c in chunks}
        for fut in as_completed(futures):
            res = fut.result()
            restarts_done += res["restarts_given"]
            if res["best_E"] is not None:
                best_E = res["best_E"] if best_E is None else min(best_E, res["best_E"])
            if res["solved"]:
                solution = res
                # Drop the rest; don't wait for the stragglers.
                ex.shutdown(wait=False, cancel_futures=True)
                break

    seconds = time.perf_counter() - t_start
    if solution is not None:
        return {
            "solved": True,
            "A": solution["A"],
            "B": solution["B"],
            "restarts_used": restarts_done,
            "best_E": 0,
            "seconds": seconds,
            "workers": workers,
            "tasks": n_tasks,
        }
    return {
        "solved": False,
        "A": None,
        "B": None,
        "restarts_used": restarts_done,
        "best_E": best_E,
        "seconds": seconds,
        "workers": workers,
        "tasks": n_tasks,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Parallel local search for a Legendre "
                                "pair: run the restarts across a process pool.")
    p.add_argument("ell", type=int, help="odd length of the pair")
    p.add_argument("-s", "--strategy",
                   choices=["greedy", "sideways", "anneal", "basinhop", "threshold"],
                   default="anneal", help="acceptance rule (default: anneal)")
    p.add_argument("-r", "--restarts", type=int, default=200)
    p.add_argument("-n", "--steps", type=int, default=20000, help="steps per restart")
    p.add_argument("--t0", type=float, default=3.0, help="initial temperature (anneal)")
    p.add_argument("--t-end", type=float, default=0.05, help="final temperature (anneal)")
    p.add_argument("--seed", type=int, default=0,
                   help="base RNG seed for reproducibility (default 0)")
    p.add_argument("-j", "--jobs", type=int, default=None,
                   help="worker processes (default: all CPU cores)")
    p.add_argument("--tasks-per-worker", type=int, default=4,
                   help="restart-chunks per worker (default 4)")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="per-task wall-time cap between restarts")
    args = p.parse_args()

    if args.ell <= 0 or args.ell % 2 == 0:
        p.error(f"ell must be a positive odd integer, got {args.ell}")

    res = search_parallel(args.ell, args.strategy, args.restarts, args.steps,
                          args.t0, args.t_end, args.seed, workers=args.jobs,
                          tasks_per_worker=args.tasks_per_worker,
                          max_seconds=args.max_seconds)

    print(f"[{res['workers']} workers, {res['tasks']} tasks]")
    if res["solved"]:
        a, b = res["A"], res["B"]
        ok, reason = is_legendre_pair(a, b)
        print(f"SOLVED ell={args.ell}  (verified: {ok}{'' if ok else ' -- ' + reason})")
        print(f"  A = {_fmt(a)}   {a}")
        print(f"  B = {_fmt(b)}   {b}")
        print(f"  restarts used ~ {res['restarts_used']}, time = {res['seconds']:.3f}s")
        return 0

    print(f"NOT SOLVED ell={args.ell} after ~{res['restarts_used']} restarts x "
          f"{args.steps} steps  (best E = {res['best_E']}, "
          f"time = {res['seconds']:.3f}s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
