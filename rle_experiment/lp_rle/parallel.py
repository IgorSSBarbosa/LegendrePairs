"""parallel.py — multiprocessing version of the exhaustive RLE driver.

The enumeration is embarrassingly parallel: canonical run arrays with k pairs
are generated from a positive-part composition `a` of P and a `b` of M
(enumerate.iter_canonical_k). We slice the work into one task per (k, a): a
worker iterates every `b`, keeps the canonical ones, PSD-filters, and returns
its survivors. The parent stitches the results back together.

Determinism / correctness anchor: tasks are emitted in exactly the serial
enumeration order (k ascending, then `a` in `compositions(P, k)` order) and the
results are consumed in that same order (Pool.map preserves order, and each
worker preserves its `b` order). So the reconstructed `buckets` dict is
BYTE-IDENTICAL to collect_survivors(), which tests/test_parallel.py asserts.
The parallel path is therefore only a speedup, never a semantic change.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

from .conventions import P_of, M_of, half_len, check_odd
from .enumerate import compositions, _interleave
from .runs import is_canonical, runs_to_seq
from .paf import paf_naive
from .filter import paf_key, is_psd_feasible, PafKey
from .match import find_pairs, inequivalent_classes


def _worker(args):
    """Process one (k, a) task: returns (k, tested, [(paf_key, v_int8), ...])."""
    L, k, a, tol = args
    M = M_of(L)
    tested = 0
    out = []
    for b in compositions(M, k):
        r = _interleave(a, b)
        if not is_canonical(r):
            continue
        tested += 1  # a canonical rotation class with k pairs
        v = runs_to_seq(r)
        pv = paf_naive(v)
        if is_psd_feasible(pv, L, tol):
            out.append((paf_key(pv), v.astype(np.int8)))
    return k, tested, out


def _tasks(L: int, tol: float):
    """Serial-order stream of (L, k, a, tol) tasks."""
    P = P_of(L)
    for k in range(1, half_len(L) + 1):
        for a in compositions(P, k):
            yield (L, k, a, tol)


def collect_survivors_parallel(L: int, tol: float = 1e-6,
                               workers: int = None, chunksize: int = 8):
    """Parallel drop-in for filter.collect_survivors; identical (buckets, stats)."""
    check_odd(L)
    tasks = list(_tasks(L, tol))
    workers = workers or os.cpu_count()
    workers = max(1, min(workers, len(tasks)))
    if workers == 1:
        results = [_worker(t) for t in tasks]
    else:
        with mp.Pool(processes=workers) as pool:
            results = pool.map(_worker, tasks, chunksize=chunksize)

    buckets: Dict[PafKey, List[np.ndarray]] = defaultdict(list)
    per_k: Dict[int, List[int]] = defaultdict(lambda: [0, 0])  # [tested, passed]
    total = 0
    survivors = 0
    # results are in task order == serial enumeration order -> identical buckets
    for k, tested, out in results:
        per_k[k][0] += tested
        total += tested
        for key, v in out:
            per_k[k][1] += 1
            survivors += 1
            buckets[key].append(v)
    stats = {
        "classes": total,
        "survivors": survivors,
        "per_k": {k: tuple(per_k[k]) for k in sorted(per_k)},
    }
    return buckets, stats


def run_exhaustive_parallel(L: int, tol: float = 1e-6, workers: int = None) -> Dict:
    """Parallel twin of exhaust.run_exhaustive (same return shape)."""
    check_odd(L)
    buckets, stats = collect_survivors_parallel(L, tol, workers=workers)
    pairs = find_pairs(buckets)
    classes = inequivalent_classes(pairs)
    return {
        "L": L,
        "buckets": buckets,
        "total_classes": stats["classes"],
        "survivors": stats["survivors"],
        "per_k": stats["per_k"],
        "raw_pairs": len(pairs),
        "lp_classes": len(classes),
        "classes": classes,
    }


if __name__ == "__main__":
    import argparse
    import time

    from .exhaust import write_lps

    ap = argparse.ArgumentParser(description="Parallel exhaustive LP search.")
    ap.add_argument("Ls", type=int, nargs="*", help="odd lengths (default 3..25)")
    ap.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    ap.add_argument("--write", action="store_true", help="write results/lps/LP{L}.csv")
    args = ap.parse_args()

    Ls = args.Ls or list(range(3, 26, 2))
    print(f"{'L':>3} {'classes':>9} {'survivors':>10} {'LPs':>5} {'wall_s':>8}  workers={args.workers or os.cpu_count()}")
    for L in Ls:
        t0 = time.perf_counter()
        ex = run_exhaustive_parallel(L, workers=args.workers)
        dt = time.perf_counter() - t0
        tail = ""
        if args.write:
            tail = "  -> " + write_lps(L, ex["classes"])
        print(f"{ex['L']:>3} {ex['total_classes']:>9} {ex['survivors']:>10} "
              f"{ex['lp_classes']:>5} {dt:>8.2f}{tail}")
