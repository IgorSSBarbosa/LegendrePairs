"""runs.py — the RLE <-> pm1 bijection, canonicalization, and symmetry actions.

A row-sum-+1 sequence v in {-1,+1}^L is written as a run-length array anchored
at the start of a positive run:

    r = (r[0], ..., r[2k-1]),  r[i] >= 1,  even indices = +1 runs, odd = -1 runs
    sum_{i even} r[i] = P,  sum_{i odd} r[i] = M,  so sum r = L and alt-sum = 1.

Pairing consecutive runs gives w = ((r[0],r[1]), ..., (r[2k-2],r[2k-1])), an
array of k pairs over Z_{>0}^2. The group of cyclic shifts of v acts on w as
Z_k rotation, and this action is *free* (every orbit has size exactly k):

    If v had period d | L with d < L, then sum_j v[j] = (L/d)*(block sum), so
    (L/d) | 1; but L is odd => L/d >= 3, a contradiction. Hence w is aperiodic.

Therefore the canonical form -- the lexicographically least rotation of w,
computed by Booth's O(k) algorithm -- is always unique, and each of the L
cyclic shifts of v collapses to one of k distinct run arrays.

Odd rotations of r equal shift-then-negation; they are excluded by the row-sum
+1 normalization, so they never appear here.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .conventions import check_odd


# --------------------------------------------------------------------------- #
# bijection RLE <-> pm1
# --------------------------------------------------------------------------- #
def runs_to_seq(r: np.ndarray) -> np.ndarray:
    """Expand a run array into an anchored pm1 sequence (starts at a +1 run).

    Returns int8[L] with L = sum(r).
    """
    r = np.asarray(r, dtype=np.int64)
    if r.shape[0] % 2 != 0:
        raise ValueError("run array must have an even number of runs (2k)")
    if np.any(r < 1):
        raise ValueError("every run length must be >= 1")
    L = int(r.sum())
    v = np.empty(L, dtype=np.int8)
    pos = 0
    for i in range(r.shape[0]):
        val = np.int8(1) if (i % 2 == 0) else np.int8(-1)
        v[pos : pos + r[i]] = val
        pos += int(r[i])
    return v


def seq_to_runs(v: np.ndarray) -> np.ndarray:
    """Rotate v to the start of a positive run and read off run lengths.

    Requires v to be pm1 with row sum +1 (so both a +1 and a -1 run exist).
    Returns int64[2k]. Not canonicalized.
    """
    v = np.asarray(v, dtype=np.int8)
    L = v.shape[0]
    if not np.all(np.abs(v) == 1):
        raise ValueError("v must be a pm1 sequence")
    if int(v.sum()) != 1:
        raise ValueError("v must have row sum +1 (normalize by complementation)")
    # anchor: first index i with v[i]==+1 and v[i-1]==-1 (cyclic)
    anchor = -1
    for i in range(L):
        if v[i] == 1 and v[(i - 1) % L] == -1:
            anchor = i
            break
    if anchor < 0:
        raise ValueError("no positive-run boundary found (degenerate sequence)")
    runs = []
    i = 0
    while i < L:
        j = i
        while j < L and v[(anchor + j) % L] == v[(anchor + i) % L]:
            j += 1
        runs.append(j - i)
        i = j
    return np.array(runs, dtype=np.int64)


# --------------------------------------------------------------------------- #
# canonicalization (Booth on the pair array) + O(k^2) reference
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _cmp_pair(r, i, j):
    """Lexicographic compare of pair i vs pair j (indices into the pair array).

    Returns -1, 0, +1. Pair i is (r[2i], r[2i+1]).
    """
    a0 = r[2 * i]
    a1 = r[2 * i + 1]
    b0 = r[2 * j]
    b1 = r[2 * j + 1]
    if a0 < b0:
        return -1
    if a0 > b0:
        return 1
    if a1 < b1:
        return -1
    if a1 > b1:
        return 1
    return 0


@njit(cache=True)
def _booth_least_index(r):
    """Booth's O(k) least-rotation index over the pair array w (length k)."""
    n = r.shape[0] // 2
    f = np.full(2 * n, -1, dtype=np.int64)
    k = 0
    for j in range(1, 2 * n):
        sj = j % n
        idx = j - k - 1
        i = -1 if idx < 0 else f[idx]
        c = _cmp_pair(r, sj, (k + i + 1) % n)
        while i != -1 and c != 0:
            if c < 0:
                k = j - i - 1
            i = f[i]
            c = _cmp_pair(r, sj, (k + i + 1) % n)
        if c != 0:
            if _cmp_pair(r, sj, k % n) < 0:
                k = j
            f[j - k] = -1
        else:
            f[j - k] = i + 1
    return k % n


@njit(cache=True)
def _naive_least_index(r):
    """O(k^2) allocation-free reference for the least-rotation index.

    Correct and complete; used to cross-check Booth in the property tests.
    """
    n = r.shape[0] // 2
    best = 0
    for t in range(1, n):
        # compare rotation t vs rotation `best`, pair by pair
        cmp = 0
        for m in range(n):
            cmp = _cmp_pair(r, (t + m) % n, (best + m) % n)
            if cmp != 0:
                break
        if cmp < 0:
            best = t
    return best


@njit(cache=True)
def _rotate_pairs_into(r, t, out):
    """Write into `out` the pair-array rotation of r by t pairs."""
    n = r.shape[0] // 2
    for m in range(n):
        src = (t + m) % n
        out[2 * m] = r[2 * src]
        out[2 * m + 1] = r[2 * src + 1]


def canonical(r: np.ndarray) -> np.ndarray:
    """Lexicographically least rotation of the pair array (Booth, O(k))."""
    r = np.ascontiguousarray(r, dtype=np.int64)
    t = int(_booth_least_index(r))
    out = np.empty_like(r)
    _rotate_pairs_into(r, t, out)
    return out


@njit(cache=True)
def is_canonical(r) -> bool:
    """True iff r is its own least rotation. O(k^2), allocation-free.

    (An O(k) Booth check exists, but at k<=12 the quadratic test is trivially
    fast and needs no failure-function array — keeping the hot enumerate loop
    allocation-free.)
    """
    n = r.shape[0] // 2
    for t in range(1, n):
        cmp = 0
        for m in range(n):
            cmp = _cmp_pair(r, (t + m) % n, m)
            if cmp != 0:
                break
        if cmp < 0:
            return False
    return True


# --------------------------------------------------------------------------- #
# symmetry actions on runs
# --------------------------------------------------------------------------- #
def rotate_runs(r: np.ndarray, t: int) -> np.ndarray:
    """Rotate the pair array by t pairs (== an even rotation of r).

    This realizes a cyclic shift of the underlying sequence v while staying in
    the row-sum-+1 orbit. `t` is measured in *pairs* (not single runs), because
    odd run-rotations are shift-then-negation and are excluded here.
    """
    r = np.ascontiguousarray(r, dtype=np.int64)
    out = np.empty_like(r)
    _rotate_pairs_into(r, int(t), out)
    return out


def reverse_runs(r: np.ndarray) -> np.ndarray:
    """Reversal v[i] -> v[(-i) mod L], returned as a re-anchored run array.

    Implemented by round-tripping through pm1 space (the trustworthy route --
    no fragile index formula; see the property test that checks it).
    """
    v = runs_to_seq(r)
    L = v.shape[0]
    idx = (-np.arange(L)) % L
    rv = v[idx]
    return seq_to_runs(rv)
