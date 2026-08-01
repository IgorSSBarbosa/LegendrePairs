"""fiber_search.py — Phase 5 (Q2): fiber-restricted stochastic search.

A surviving compressed pair ``(cA, cB)`` defines the fibers

    fiber(cA) = { A in {+-1}^ell : compress(A, m) == cA },

and EVERY ``A`` in ``fiber(cA)`` already satisfies all mod-``m`` spectral
constraints. So we search WITHIN the fibers, changing only the primitive-
frequency residual that compression is blind to (PLAN Q2). The move stays on the
fiber:

  * COLUMN SWAP — inside one residue class ``j (mod m)`` (the ``n`` slots
    ``{j, j+m, ..., j+(n-1)m}``) swap a ``+1`` and a ``-1``. This preserves
    ``cA_j`` (hence all of ``cA``) and the row sum ``sum A``, and perturbs only
    the uncovered spectrum.

Objective (exact integer, VERIFIED reuse of the ``lp_rle`` half PAF):

    E(A, B) = sum_{s=1}^{(ell-1)/2} (P_A(s) + P_B(s) + 2)^2  >= 0,  == 0 iff LP.

``dE`` per swap is an ``O(ell)`` rank-2 update (two sequential single-entry flips,
pre-flip values used so neighbour interactions stay exact); it is checked against
a full :func:`lp_compress.core.paf_half` recompute in ``test_fiber_search.py``.

This search is a DISCOVERY tool, honestly INCOMPLETE: simulated annealing finds
LPs at ``ell`` beyond exhaustive reach but does NOT enumerate every class or prove
non-existence. Use route B for a census, this for reach.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from lp_rle.symmetry import canonical_pair

from .core import paf_half
from .compress import compress


# --------------------------------------------------------------------------- #
# fiber geometry
# --------------------------------------------------------------------------- #
def _column_slots(ell: int, m: int) -> np.ndarray:
    """``(m, n)`` int array: row ``j`` = positions ``{j, j+m, ..., j+(n-1)m}``."""
    n = ell // m
    return (np.arange(m)[:, None] + m * np.arange(n)[None, :])


def fiber_plus_counts(cA, ell: int, m: int) -> np.ndarray:
    """Per-column plus-one count ``k_j = (n + cA_j)/2`` (VERIFIED lift geometry)."""
    n = ell // m
    cA = np.asarray(cA, dtype=np.int64)
    k = (n + cA) // 2
    if np.any((n + cA) % 2 != 0) or np.any(k < 0) or np.any(k > n):
        raise ValueError(f"cA={cA.tolist()} is not a reachable mod-{m} compression")
    return k.astype(np.int64)


def fiber_sample(cA, ell: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform-ish random ``A`` in ``fiber(cA)`` (correct column sums by construction)."""
    n = ell // m
    slots = _column_slots(ell, m)          # (m, n)
    k = fiber_plus_counts(cA, ell, m)      # (m,)
    A = -np.ones(ell, dtype=np.int8)
    for j in range(m):
        if k[j]:
            pick = rng.choice(n, size=int(k[j]), replace=False)
            A[slots[j, pick]] = 1
    return A


def column_swap_positions(A: np.ndarray, ell: int, m: int,
                          rng: np.random.Generator) -> Optional[Tuple[int, int]]:
    """Pick a mixed column and return a ``(+1 pos, -1 pos)`` to swap, or ``None``."""
    n = ell // m
    slots = _column_slots(ell, m)
    cols = rng.permutation(m)
    for j in cols:
        col = slots[j]
        pos_plus = col[A[col] > 0]
        pos_minus = col[A[col] < 0]
        if pos_plus.size and pos_minus.size:
            return int(rng.choice(pos_plus)), int(rng.choice(pos_minus))
    return None


# --------------------------------------------------------------------------- #
# incremental half-PAF (ported + test-verified against paf_half)
# --------------------------------------------------------------------------- #
def _flip_delta_half(a: np.ndarray, p: int, half: int) -> np.ndarray:
    """Change to ``[P(1..half)]`` when ``a[p]`` flips (pre-flip ``a[p]`` used)."""
    ell = a.size
    s = np.arange(1, half + 1)
    return (-2 * a[p] * (a[(p + s) % ell] + a[(p - s) % ell])).astype(np.int64)


def _apply_swap_half(a: np.ndarray, pa: np.ndarray, i: int, j: int, half: int) -> None:
    """Swap ``a[i]``,``a[j]`` (differ) and update half-PAF ``pa`` in place, exactly."""
    pa += _flip_delta_half(a, i, half); a[i] = -a[i]   # two sequential flips
    pa += _flip_delta_half(a, j, half); a[j] = -a[j]


def energy_half(pa: np.ndarray, pb: np.ndarray) -> int:
    """``E = sum_s (P_A(s)+P_B(s)+2)^2`` over the half spectrum; ``0`` iff LP."""
    r = pa + pb + 2
    return int(r @ r)


# --------------------------------------------------------------------------- #
# simulated annealing inside a fiber pair
# --------------------------------------------------------------------------- #
def anneal_fiber_pair(cA, cB, ell: int, m: int, *,
                      iters: int = 20_000, restarts: int = 60,
                      T0: float = 10.0, Tmin: float = 0.05,
                      seed: Optional[int] = None, verbose: bool = False) -> Dict:
    """Metropolis SA inside ``fiber(cA) x fiber(cB)``; stops at the first LP (E=0).

    Each restart is a full hot->cold cycle of ``iters`` steps (a reheat). On this
    plateau-y landscape MANY SHORT cycles beat few long ones: calibration on
    ell=27 known fibers gave 8/8 solves at ``(T0=10, iters=20k, restarts=60)`` vs
    1/8 at ``(4, 60k, 12)`` for a comparable total budget. Returns
    ``{found, A, B, E, iters_used, restarts_used}``; moves are column swaps, so
    ``compress`` of the returned ``A/B`` is exactly ``cA/cB``.
    """
    half = (ell - 1) // 2
    rng = np.random.default_rng(seed)
    cA = np.asarray(cA, dtype=np.int64)
    cB = np.asarray(cB, dtype=np.int64)
    cool = (Tmin / T0) ** (1.0 / max(iters, 1))

    best = None
    total_iters = 0
    for r in range(restarts):
        A = fiber_sample(cA, ell, m, rng)
        B = fiber_sample(cB, ell, m, rng)
        pa = paf_half(A).astype(np.int64)
        pb = paf_half(B).astype(np.int64)
        E = energy_half(pa, pb)
        if best is None or E < best[0]:
            best = (E, A.copy(), B.copy())
        T = T0
        for _ in range(iters):
            total_iters += 1
            T *= cool
            if E == 0:
                break
            side_A = rng.random() < 0.5
            a, pab = (A, pa) if side_A else (B, pb)
            sw = column_swap_positions(a, ell, m, rng)
            if sw is None:
                continue
            i, j = sw
            a2 = a.copy(); p2 = pab.copy()
            _apply_swap_half(a2, p2, i, j, half)
            newE = energy_half(p2, pb) if side_A else energy_half(pa, p2)
            dE = newE - E
            if dE <= 0 or rng.random() < math.exp(-dE / T):
                if side_A:
                    A, pa = a2, p2
                else:
                    B, pb = a2, p2
                E = newE
                if E < best[0]:
                    best = (E, A.copy(), B.copy())
        if verbose:
            print(f"  restart {r}: bestE={best[0]} (iters so far {total_iters})", flush=True)
        if E == 0:
            break

    E, A, B = (0, A, B) if E == 0 else best
    return {"found": E == 0, "A": A, "B": B, "E": E,
            "iters_used": total_iters, "restarts_used": r + 1,
            "cA": cA, "cB": cB}


def _to_str(v: np.ndarray) -> str:
    return "".join("+" if x > 0 else "-" for x in v)


# --------------------------------------------------------------------------- #
# parallel driver: each fiber pair is an INDEPENDENT anneal (embarrassingly ||)
# --------------------------------------------------------------------------- #
def _anneal_one(args) -> Optional[Tuple[Tuple[bytes, bytes], str, str]]:
    """Worker: anneal one fiber pair; return its LP class key + strings, or None.

    Reduced to a picklable-small result (canonical key + two ``+/-`` strings) so
    the parent only de-duplicates; the ``O(ell)`` arrays never cross the pipe.
    """
    cA, cB, ell, m, seed, sa_kw = args
    res = anneal_fiber_pair(cA, cB, ell, m, seed=seed, **sa_kw)
    if not res["found"]:
        return None
    A, B = res["A"], res["B"]
    nA = A if int(A.sum()) > 0 else -A
    nB = B if int(B.sum()) > 0 else -B
    return canonical_pair(nA, nB), _to_str(nA), _to_str(nB)


def search_survivors(ell: int, m: int, survivor_pairs, *,
                     max_pairs: int = 50, seed: Optional[int] = None,
                     n_workers: Optional[int] = None,
                     verbose: bool = False, **sa_kw) -> Dict:
    """Anneal a sample of survivor fibers; collect distinct LP CLASSES found.

    ``survivor_pairs`` is any iterable of ``(cA, cB)`` (e.g. from
    ``cascade_pairs_vec`` after orbit reduction). Each fiber pair is an INDEPENDENT
    anneal, so the sample is farmed across ``n_workers`` processes (each with its
    own reproducible seed). Returns discovered classes keyed by
    :func:`lp_rle.symmetry.canonical_pair` — a lower bound on the class count, NOT
    a census. The class set is seed-deterministic but order-independent, so it does
    not depend on ``n_workers``.

    SCALING (measured, ell=27, 40 pairs): tasks are UNIFORM (per-pair 0.3-1.5s,
    max << makespan) so there is no load imbalance; speedup is bounded by the HOST.
    On an i7-1255U (a 2 P-core + 8 E-core hybrid laptop chip, thermally throttled
    under sustained all-core load) we see ~1.6x at 6 workers and ~2.1x at 12 —
    a hardware ceiling, not an algorithmic one. On homogeneous many-core hardware
    this embarrassingly-parallel driver should approach linear scaling. (Setting
    ``OPENBLAS_NUM_THREADS=1`` made no difference — the ``r @ r`` energy dot is
    not the bottleneck.)
    """
    rng = np.random.default_rng(seed)
    pairs = list(survivor_pairs)
    if len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
    n_workers = n_workers or os.cpu_count() or 1

    # one task per fiber pair, each with its own drawn seed (reproducible)
    tasks = [(cA, cB, ell, m, int(rng.integers(1 << 30)), sa_kw)
             for cA, cB in pairs]

    classes: Dict[Tuple[bytes, bytes], Tuple[str, str]] = {}
    n_found = 0

    def _collect(t: int, out) -> None:
        nonlocal n_found
        if out is not None:
            n_found += 1
            key, sA, sB = out
            classes.setdefault(key, (sA, sB))
        if verbose:
            print(f"pair {t}/{len(tasks)}: found={out is not None} "
                  f"classes={len(classes)}", flush=True)

    if n_workers <= 1 or len(tasks) <= 1:
        for t, task in enumerate(tasks, 1):
            _collect(t, _anneal_one(task))
    else:
        with mp.Pool(min(n_workers, len(tasks))) as pool:
            for t, out in enumerate(pool.imap_unordered(_anneal_one, tasks, chunksize=1), 1):
                _collect(t, out)

    return {"n_pairs_searched": len(pairs), "n_found": n_found,
            "n_classes": len(classes), "classes": classes}
