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
import multiprocessing as mp
import os
import sys
import threading
import time
from queue import Empty

from legendre import is_legendre_pair
from local_search import search, _fmt


# Set in each worker by the pool initializer when progress is on; each worker
# drops one message on this queue per completed restart, and the parent renders.
_PROGRESS_Q = None


def _init_worker(q):
    global _PROGRESS_Q
    _PROGRESS_Q = q


def _task(args):
    """Worker: run ``count`` restarts of ``search`` with its own seed.

    Runs in a separate process, so it must be a top-level picklable function and
    take a single argument. Returns the ``search`` result dict, augmented with
    the number of restarts this task was actually given."""
    (ell, strategy, count, steps, t0, t_end, seed, max_seconds) = args
    cb = None
    if _PROGRESS_Q is not None:
        q = _PROGRESS_Q
        cb = lambda r, best: q.put(best)   # one tick per completed restart
    res = search(ell, strategy=strategy, restarts=count, steps=steps,
                 t0=t0, t_end=t_end, seed=seed, max_seconds=max_seconds,
                 progress=False, on_restart=cb)
    res["restarts_given"] = count
    return res


def _bar_line(done, total, workers, tasks, best, elapsed):
    width = 28
    frac = done / max(total, 1)
    filled = int(width * min(frac, 1.0))
    bar = "#" * filled + "-" * (width - filled)
    return (f"\r[{workers}w/{tasks}t] [{bar}] restarts {done}/{total} "
            f"{100 * frac:5.1f}%  best E={best}  {elapsed:5.1f}s   ")


def _render_progress(q, total, workers, tasks, t_start, stop):
    """Parent-side listener: drain per-restart ticks and draw one stderr bar."""
    done = 0
    best = None
    while not stop.is_set():
        try:
            be = q.get(timeout=0.1)
        except (Empty, OSError, ValueError):
            continue
        done += 1
        if be is not None:
            best = be if best is None else min(best, be)
        sys.stderr.write(_bar_line(done, total, workers, tasks, best,
                                   time.perf_counter() - t_start))
        sys.stderr.flush()


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
    progress: bool = False,
):
    """Parallel local search: split ``restarts`` across a process pool.

    Returns a dict like ``local_search.search`` (solved, A, B, best_E, seconds)
    plus ``workers``, ``tasks``, and ``restarts_used`` (restarts consumed by
    tasks that had finished when we stopped). Stops early on the first solution.
    If ``progress`` is set, a live stderr bar shows the global restart count and
    best E, fed by one message per completed restart from every worker.
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

    # Optional live progress: workers push one tick per restart onto a queue and
    # a parent-side listener thread renders the bar (the main thread is busy
    # blocking on imap_unordered results).
    q = mp.Queue() if progress else None
    stop = threading.Event()
    listener = None
    if progress:
        listener = threading.Thread(
            target=_render_progress,
            args=(q, restarts, workers, n_tasks, t_start, stop),
            daemon=True,
        )
        listener.start()

    # A multiprocessing.Pool (not ProcessPoolExecutor) because we need to abandon
    # in-flight work the instant we find a solution. ``pool.terminate()`` SIGTERMs
    # the running workers immediately; the executor's cancel_futures only drops
    # *pending* tasks and then blocks on the running ones, so the process would
    # hang until a straggler chunk finished.
    pool = mp.Pool(processes=workers,
                   initializer=_init_worker if progress else None,
                   initargs=(q,) if progress else ())
    try:
        for res in pool.imap_unordered(_task, chunks):
            restarts_done += res["restarts_given"]
            if res["best_E"] is not None:
                best_E = res["best_E"] if best_E is None else min(best_E, res["best_E"])
            if res["solved"]:
                solution = res
                break
    finally:
        pool.terminate()   # kill any still-running restarts, don't wait them out
        pool.join()
        if progress:
            stop.set()
            listener.join(timeout=1.0)
            q.close()
            q.cancel_join_thread()
            # One authoritative final frame: the listener can be cut off with
            # ticks still in transit, so draw the true totals ourselves.
            final_best = 0 if solution is not None else best_E
            sys.stderr.write(_bar_line(restarts_done, restarts, workers, n_tasks,
                                       final_best, time.perf_counter() - t_start))
            sys.stderr.write("\n")
            sys.stderr.flush()

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


def _save_pair(args, res) -> None:
    """Persist a solved pair to results/found_pairs.csv (dedup per method+ell)."""
    try:
        from pairs_store import record_pair
    except Exception as exc:                 # pragma: no cover - store is optional
        print(f"  (note: could not import pairs_store: {exc})")
        return
    t0str = "auto" if args.auto_t0 else args.t0
    params = (f"restarts~{res['restarts_used']}, steps={args.steps}, seed={args.seed}, "
              f"workers={res['workers']}")
    if args.strategy == "anneal":
        params += f", t0={t0str}, t_end={args.t_end}"
    rec = record_pair(args.ell, args.strategy, res["A"], res["B"],
                      seconds=res["seconds"], params=params)
    rel = os.path.relpath(rec["path"])
    print(f"  {'saved to' if rec['written'] else 'already recorded in'} {rel}")


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
    p.add_argument("--auto-t0", action="store_true",
                   help="auto-calibrate the anneal start temperature (ignores --t0)")
    p.add_argument("--seed", type=int, default=0,
                   help="base RNG seed for reproducibility (default 0)")
    p.add_argument("-j", "--jobs", type=int, default=None,
                   help="worker processes (default: all CPU cores)")
    p.add_argument("--tasks-per-worker", type=int, default=4,
                   help="restart-chunks per worker (default 4)")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="per-task wall-time cap between restarts")
    p.add_argument("-P", "--progress", action="store_true",
                   help="show a live stderr bar: global restart count and best E")
    args = p.parse_args()

    if args.ell <= 0 or args.ell % 2 == 0:
        p.error(f"ell must be a positive odd integer, got {args.ell}")

    t0 = None if args.auto_t0 else args.t0
    res = search_parallel(args.ell, args.strategy, args.restarts, args.steps,
                          t0, args.t_end, args.seed, workers=args.jobs,
                          tasks_per_worker=args.tasks_per_worker,
                          max_seconds=args.max_seconds, progress=args.progress)

    print(f"[{res['workers']} workers, {res['tasks']} tasks]")
    if res["solved"]:
        a, b = res["A"], res["B"]
        ok, reason = is_legendre_pair(a, b)
        print(f"SOLVED ell={args.ell}  (verified: {ok}{'' if ok else ' -- ' + reason})")
        print(f"  A = {_fmt(a)}   {a}")
        print(f"  B = {_fmt(b)}   {b}")
        print(f"  restarts used ~ {res['restarts_used']}, time = {res['seconds']:.3f}s")
        _save_pair(args, res)
        return 0

    print(f"NOT SOLVED ell={args.ell} after ~{res['restarts_used']} restarts x "
          f"{args.steps} steps  (best E = {res['best_E']}, "
          f"time = {res['seconds']:.3f}s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
