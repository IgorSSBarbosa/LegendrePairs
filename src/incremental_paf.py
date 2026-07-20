"""Incremental PAF update: O(ell) per swap instead of an O(ell log ell) FFT.

Why
---
Every local-search step flips (swaps) two entries of a length-ell +-1 sequence
and needs the new periodic autocorrelation PAF(1..half). Recomputing it from
scratch with an FFT costs O(ell log ell). But a single-entry flip changes PAF by
a closed-form rank-2 update that costs O(ell). For the search that is a modest
constant-factor win; the real payoff is that the update is a handful of gathers
and a multiply-add, so it batches cleanly over thousands of independent walkers
on a GPU -- the population-parallel route to methods like parallel tempering /
population annealing.

The delta
---------
For a +-1 sequence a of length ell,

    PAF(s) = sum_t a[t] * a[(t+s) mod ell].

Position p contributes to PAF(s) exactly through the two terms a[p]*a[(p+s)]
and a[(p-s)]*a[p]. Flipping a[p] -> -a[p] changes a[p] by delta = -2*a[p], so

    PAF(s) += -2 * a[p] * ( a[(p+s) mod ell] + a[(p-s) mod ell] )      (s>=1)

with a[p] taken BEFORE the flip. A swap is two flips (one +1->-1, one -1->+1)
applied one after the other; doing them sequentially -- updating the array
between them -- keeps the rare case where the two flipped positions are
neighbours (|i-j| = s) exact.

This module gives a single-sequence version, a batched (W, ell) version over W
walkers, a verification against the FFT PAF, and a throughput micro-benchmark.
A real search walker holds two sequences A, B; the update is applied to each
independently, so everything here extends to the pair objective unchanged.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from local_search import paf_vector


# --------------------------------------------------------------------------- #
# Reference (FFT) PAF, single and batched.
# --------------------------------------------------------------------------- #
def paf_batch(A: np.ndarray, half: int) -> np.ndarray:
    """[PAF(1..half)] for each row of A (shape (W, ell)) via FFT -> (W, half)."""
    ell = A.shape[1]
    f = np.fft.rfft(A.astype(float), axis=1)
    ac = np.fft.irfft(f * np.conj(f), n=ell, axis=1)
    return np.rint(ac[:, 1 : half + 1]).astype(np.int64)


# --------------------------------------------------------------------------- #
# Single-sequence incremental update.
# --------------------------------------------------------------------------- #
def flip_delta(a: np.ndarray, p: int, half: int) -> np.ndarray:
    """Change to [PAF(1..half)] when a[p] is flipped (a[p] read pre-flip)."""
    ell = a.size
    s = np.arange(1, half + 1)
    plus = a[(p + s) % ell]
    minus = a[(p - s) % ell]
    return (-2 * a[p] * (plus + minus)).astype(np.int64)


def apply_flip(a: np.ndarray, pa: np.ndarray, p: int, half: int) -> None:
    """Flip a[p] and update its PAF vector pa in place (both mutated)."""
    pa += flip_delta(a, p, half)
    a[p] = -a[p]


def apply_swap(a: np.ndarray, pa: np.ndarray, i: int, j: int, half: int) -> None:
    """Swap a[i] and a[j] (they must differ) and update pa in place.

    Done as two sequential flips so neighbour interactions stay exact."""
    apply_flip(a, pa, i, half)
    apply_flip(a, pa, j, half)


# --------------------------------------------------------------------------- #
# Batched incremental update over W walkers.
# --------------------------------------------------------------------------- #
def flip_delta_batch(A: np.ndarray, p: np.ndarray, half: int) -> np.ndarray:
    """Per-walker PAF delta for flipping A[w, p[w]]. A: (W, ell), p: (W,).

    Returns (W, half). Pre-flip values are used (call before mutating A)."""
    W, ell = A.shape
    s = np.arange(1, half + 1)
    idx_plus = (p[:, None] + s[None, :]) % ell           # (W, half)
    idx_minus = (p[:, None] - s[None, :]) % ell
    plus = np.take_along_axis(A, idx_plus, axis=1)
    minus = np.take_along_axis(A, idx_minus, axis=1)
    vp = A[np.arange(W), p]                               # (W,)
    return (-2 * vp[:, None] * (plus + minus)).astype(np.int64)


def apply_flip_batch(A: np.ndarray, PA: np.ndarray, p: np.ndarray, half: int) -> None:
    """Flip A[w, p[w]] for every walker w and update PA (both (W, .) in place)."""
    PA += flip_delta_batch(A, p, half)
    W = A.shape[0]
    A[np.arange(W), p] *= -1


def apply_swap_batch(A, PA, i: np.ndarray, j: np.ndarray, half: int) -> None:
    """Per-walker swap of A[w, i[w]] with A[w, j[w]]; update PA in place."""
    apply_flip_batch(A, PA, i, half)
    apply_flip_batch(A, PA, j, half)


# --------------------------------------------------------------------------- #
# Verification and micro-benchmark.
# --------------------------------------------------------------------------- #
def verify(ell: int, walkers: int, swaps: int, seed: int = 0) -> bool:
    """Drive random swaps incrementally and check PAF matches the FFT every step,
    for both the single-sequence and batched paths. Returns True on full match."""
    half = (ell - 1) // 2
    rng = np.random.default_rng(seed)

    # single-sequence path
    a = np.where(rng.random(ell) < 0.5, 1, -1).astype(np.int64)
    pa = paf_vector(a, half)
    for _ in range(swaps):
        plus = np.flatnonzero(a == 1)
        minus = np.flatnonzero(a == -1)
        if plus.size == 0 or minus.size == 0:
            break
        i = int(plus[rng.integers(plus.size)])
        j = int(minus[rng.integers(minus.size)])
        apply_swap(a, pa, i, j, half)
        if not np.array_equal(pa, paf_vector(a, half)):
            print(f"  single: MISMATCH at ell={ell}")
            return False

    # batched path
    A = np.where(rng.random((walkers, ell)) < 0.5, 1, -1).astype(np.int64)
    PA = paf_batch(A, half)
    for _ in range(swaps):
        # one +1 and one -1 index per walker
        i = np.empty(walkers, dtype=np.int64)
        j = np.empty(walkers, dtype=np.int64)
        for w in range(walkers):
            plus = np.flatnonzero(A[w] == 1)
            minus = np.flatnonzero(A[w] == -1)
            i[w] = plus[rng.integers(plus.size)]
            j[w] = minus[rng.integers(minus.size)]
        apply_swap_batch(A, PA, i, j, half)
        if not np.array_equal(PA, paf_batch(A, half)):
            print(f"  batch: MISMATCH at ell={ell}")
            return False
    print(f"  ell={ell:>4}  W={walkers:>5}  {swaps} swaps: incremental == FFT  OK")
    return True


def _rand_swap_indices(A: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """A cheap (not-exactly-uniform) pair of differing indices per walker, used
    only to feed the throughput benchmark -- correctness is checked in verify()."""
    W, ell = A.shape
    i = rng.integers(ell, size=W)
    j = (i + 1 + rng.integers(ell - 1, size=W)) % ell
    return i, j


def benchmark(ell: int, walkers: int, steps: int, seed: int = 0) -> dict:
    """Compare walker-steps/sec: batched incremental update vs FFT recompute."""
    half = (ell - 1) // 2
    rng = np.random.default_rng(seed)
    A = np.where(rng.random((walkers, ell)) < 0.5, 1, -1).astype(np.int64)
    PA = paf_batch(A, half)
    moves = [_rand_swap_indices(A, rng) for _ in range(steps)]

    # incremental: O(ell) rank-2 update per swap
    A1, PA1 = A.copy(), PA.copy()
    t = time.perf_counter()
    for i, j in moves:
        apply_swap_batch(A1, PA1, i, j, half)
    inc_s = time.perf_counter() - t

    # baseline: apply the swap then recompute PAF from scratch with an FFT
    A2 = A.copy()
    t = time.perf_counter()
    for i, j in moves:
        A2[np.arange(walkers), i], A2[np.arange(walkers), j] = (
            A2[np.arange(walkers), j], A2[np.arange(walkers), i])
        _ = paf_batch(A2, half)
    fft_s = time.perf_counter() - t

    ws = walkers * steps
    return {
        "ell": ell, "walkers": walkers, "steps": steps,
        "inc_rate": ws / inc_s, "fft_rate": ws / fft_s,
        "speedup": fft_s / inc_s,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Incremental PAF update: verify vs FFT "
                                "and benchmark single/batched throughput.")
    p.add_argument("--verify", action="store_true", help="run correctness checks")
    p.add_argument("--bench", action="store_true", help="run the throughput benchmark")
    p.add_argument("--ells", type=int, nargs="+", default=[31, 63, 115])
    p.add_argument("--walkers", type=int, nargs="+", default=[1, 1024, 16384])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not (args.verify or args.bench):
        args.verify = args.bench = True

    if args.verify:
        print("verification (incremental PAF == FFT PAF):")
        ok = all(verify(ell, walkers=64, swaps=40, seed=args.seed) for ell in args.ells)
        print("  all verified\n" if ok else "  FAILED\n")

    if args.bench:
        print(f"throughput (walker-steps/sec), {args.steps} steps per cell:")
        print(f"  {'ell':>5} {'W':>7} {'incremental':>16} {'FFT-recompute':>16} "
              f"{'speedup':>9}")
        for ell in args.ells:
            for W in args.walkers:
                r = benchmark(ell, W, args.steps, seed=args.seed)
                print(f"  {r['ell']:>5} {r['walkers']:>7} "
                      f"{r['inc_rate']:>16,.0f} {r['fft_rate']:>16,.0f} "
                      f"{r['speedup']:>8.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
