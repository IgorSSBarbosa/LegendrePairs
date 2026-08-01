"""parallel.py — multiprocess Approach B (compression funnel) for larger ell.

Route B (``pipeline_B``) has two embarrassingly-parallel stages that dominate at
larger ``ell``; this module farms both across processes while keeping the
sequential ``cascade_pairs`` sieve intact:

  1. ORBIT REDUCTION — ``canonical_compressed_pair`` is computed for every sieve
     survivor in parallel (chunked, ORDER-PRESERVING), then de-duplicated in the
     parent keeping the first occurrence (identical result to the serial
     :func:`lp_compress.orbit.orbit_reduced_pairs`, just faster);
  2. LIFT + LP TEST — each orbit representative's fiber is lifted and PAF-tested
     independently. Tasks are one-rep-each and dispatched LONGEST-FIBER-FIRST via
     ``imap_unordered`` (LPT scheduling), so the makespan approaches
     ``max(largest_rep, total_lift / n_workers)``.

Correctness is UNCHANGED from the serial route: the class set is keyed by
``lp_rle.symmetry.canonical_pair`` (the same complete invariant Approach A uses),
so ``lp_classes`` here must equal the RLE ground truth / the stored database.

This is a SPEED experiment, not a new method: same funnel, same counts, more
cores. Results (found pairs + per-stage wall-clock) are written by
:mod:`lp_compress.run_parallel`.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lp_rle.symmetry import canonical_pair

from collections import defaultdict

from .compress import cascade_pairs, cascade_pairs_vec, _default_modulus
from .orbit import canonical_compressed_pair, orbit_reduce_arrays
from .lift import lift, fiber_size
from .core import paf_half

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _chunks(seq: Sequence, n: int) -> List[List]:
    """Split ``seq`` into up to ``n`` contiguous, order-preserving chunks."""
    k, r = divmod(len(seq), n)
    out, i = [], 0
    for j in range(n):
        size = k + (1 if j < r else 0)
        if size:
            out.append(seq[i:i + size])
            i += size
    return out


def _to_str(v: np.ndarray) -> str:
    return "".join("+" if x > 0 else "-" for x in v)


# --------------------------------------------------------------------------- #
# stage 1 worker: canonical compressed keys for a chunk of survivors
# --------------------------------------------------------------------------- #
def _canon_chunk(args) -> List[Tuple[bytes, bytes]]:
    chunk, m, use_mult = args
    return [canonical_compressed_pair(cA, cB, m, use_mult) for cA, cB in chunk]


def orbit_reduce_parallel(pairs: List[Tuple[np.ndarray, np.ndarray]], m: int,
                          pool: mp.pool.Pool, n_tasks: int,
                          use_mult: bool = False
                          ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Parallel, order-stable one-rep-per-orbit reduction (matches the serial fn)."""
    chunks = _chunks(pairs, n_tasks)
    keylists = pool.map(_canon_chunk, [(c, m, use_mult) for c in chunks])
    reps: Dict[Tuple[bytes, bytes], Tuple[np.ndarray, np.ndarray]] = {}
    for chunk, keys in zip(chunks, keylists):
        for (cA, cB), key in zip(chunk, keys):
            if key not in reps:
                reps[key] = (np.asarray(cA, np.int64), np.asarray(cB, np.int64))
    return list(reps.values())


# --------------------------------------------------------------------------- #
# stage 2 worker: lift one representative's fiber and PAF-test it
# --------------------------------------------------------------------------- #
def _lift_rep(args) -> Tuple[Dict[Tuple[bytes, bytes], Tuple[str, str]], int, int]:
    """Lift one rep's fibers and PAF-HASH-JOIN them (not the O(|LA||LB|) test).

    A pair (A, B) is a Legendre pair iff ``paf_half(A) + paf_half(B) == -2``
    componentwise. So we bucket the A-fiber by its PAF key and, for each B, look
    up the required complementary key ``-2 - paf_half(B)``. This computes only
    ``|LA| + |LB|`` PAF vectors per rep instead of testing every product pair.
    """
    cA, cB, ell, m = args
    LA = list(lift(cA, ell, m))
    LB = list(lift(cB, ell, m))
    by_paf: "defaultdict[bytes, list]" = defaultdict(list)
    for A in LA:
        by_paf[paf_half(A).tobytes()].append(A)

    classes: Dict[Tuple[bytes, bytes], Tuple[str, str]] = {}
    n_lps = 0
    for B in LB:
        target = (-2 - paf_half(B)).astype(np.int64).tobytes()
        for A in by_paf.get(target, ()):
            n_lps += 1
            nA = A if int(A.sum()) > 0 else -A
            nB = B if int(B.sum()) > 0 else -B
            key = canonical_pair(nA, nB)
            if key not in classes:
                classes[key] = (_to_str(nA), _to_str(nB))
    return classes, len(LA) * len(LB), n_lps


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def pipeline_B_parallel(ell: int, m: Optional[int] = None,
                        n_workers: Optional[int] = None,
                        use_multipliers: bool = False,
                        verbose: bool = True) -> Dict:
    """Multiprocess compression funnel; returns counts, per-stage timings, classes.

    Identical LP-class set to serial route B, just parallelized over ``n_workers``
    processes for the orbit-reduction and lift stages.
    """
    m = _default_modulus(ell, m)
    n = ell // m
    n_workers = n_workers or os.cpu_count() or 1

    # Sieve: vectorized meet-in-the-middle (identical survivor set to cascade_pairs).
    t0 = time.perf_counter()
    CA_all, CB_all = cascade_pairs_vec(ell, m)
    n_pairs = CA_all.shape[0]
    t_sieve = time.perf_counter() - t0

    # Orbit reduction: the vectorized base-(2n+1) int64 path is default and far
    # faster than farming tiny per-pair canonicalizations across processes; keep
    # the process-parallel byte path only for the gated multiplier group.
    t1 = time.perf_counter()
    if use_multipliers:
        with mp.Pool(n_workers) as _pool:
            reps = orbit_reduce_parallel(list(zip(CA_all, CB_all)), m,
                                         _pool, n_workers * 8, True)
    else:
        reps = orbit_reduce_arrays(CA_all, CB_all, m, n)
    t_orbit = time.perf_counter() - t1

    with mp.Pool(n_workers) as pool:
        # LPT scheduling: dispatch the largest fibers first to minimize makespan
        reps_sorted = sorted(
            reps, key=lambda p: fiber_size(p[0], ell, m) * fiber_size(p[1], ell, m),
            reverse=True)
        tasks = [(cA, cB, ell, m) for cA, cB in reps_sorted]

        t2 = time.perf_counter()
        classes: Dict[Tuple[bytes, bytes], Tuple[str, str]] = {}
        n_lift = n_lps = 0
        for cls, nl, nlp in pool.imap_unordered(_lift_rep, tasks, chunksize=1):
            n_lift += nl
            n_lps += nlp
            for k, v in cls.items():
                classes.setdefault(k, v)
        t_lift = time.perf_counter() - t2

    total = time.perf_counter() - t0
    result = {
        "ell": ell, "m": m, "n": n, "n_workers": n_workers,
        "n_compressed_pairs": n_pairs, "n_orbit_reps": len(reps),
        "n_lift_candidates": n_lift, "n_lps": n_lps,
        "lp_classes": len(classes), "classes": classes,
        "sieve_s": t_sieve, "orbit_s": t_orbit, "lift_s": t_lift, "total_s": total,
    }
    if verbose:
        print(f"ell={ell} m={m} workers={n_workers}: "
              f"survivors={n_pairs:,} reps={len(reps):,} "
              f"lift={n_lift:,} classes={len(classes)}  "
              f"[sieve {t_sieve:.1f}s | orbit {t_orbit:.1f}s | "
              f"lift {t_lift:.1f}s | total {total:.1f}s]", flush=True)
    return result
