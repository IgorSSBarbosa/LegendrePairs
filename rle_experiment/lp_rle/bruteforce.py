"""bruteforce.py — the independent pm1 ground truth (spec §3a).

This module shares NO code with the RLE layer (runs / enumerate). It walks all
C(L, P) sign sequences with row sum +1 directly in pm1 space via an integer
combination odometer, applies the same exact-integer PAF + PSD filter, and
buckets survivors by PAF key. Agreement between this and the RLE driver, for
every odd L up to 25, is the strongest correctness check we have.
"""
from __future__ import annotations

from collections import defaultdict
from math import comb
from typing import Dict, List, Tuple

import numpy as np
from numba import njit

from .conventions import P_of, M_of, half_len, check_odd
from .filter import PafKey


@njit(cache=True)
def _next_combination(c, n):
    """Advance c to the next M-subset of {0..n-1} in colex order; False at end."""
    k = c.shape[0]
    i = k - 1
    while i >= 0 and c[i] == n - k + i:
        i -= 1
    if i < 0:
        return False
    c[i] += 1
    for j in range(i + 1, k):
        c[j] = c[j - 1] + 1
    return True


@njit(cache=True)
def _paf_half_inline(v, h, L):
    out = np.zeros(h, dtype=np.int64)
    for s in range(1, h + 1):
        acc = 0
        for i in range(L):
            acc += np.int64(v[i]) * np.int64(v[(i + s) % L])
        out[s - 1] = acc
    return out


@njit(cache=True)
def _psd_ok(paf, cos_table, L, h, tol):
    bound = 2 * L + 2
    for kk in range(h):
        acc = 0.0
        for s in range(h):
            acc += paf[s] * cos_table[kk, s]
        if L + 2.0 * acc > bound + tol:
            return False
    return True


@njit(cache=True)
def _brute_count(L, cos_table, tol):
    M = (L - 1) // 2
    h = (L - 1) // 2
    c = np.arange(M)
    v = np.ones(L, dtype=np.int8)
    count = 0
    while True:
        for i in range(L):
            v[i] = 1
        for i in range(M):
            v[c[i]] = -1
        paf = _paf_half_inline(v, h, L)
        if _psd_ok(paf, cos_table, L, h, tol):
            count += 1
        if not _next_combination(c, L):
            break
    return count


@njit(cache=True)
def _brute_collect(L, cos_table, tol, n):
    M = (L - 1) // 2
    h = (L - 1) // 2
    c = np.arange(M)
    v = np.ones(L, dtype=np.int8)
    pafs = np.zeros((n, h), dtype=np.int64)
    seqs = np.zeros((n, L), dtype=np.int8)
    idx = 0
    while True:
        for i in range(L):
            v[i] = 1
        for i in range(M):
            v[c[i]] = -1
        paf = _paf_half_inline(v, h, L)
        if _psd_ok(paf, cos_table, L, h, tol):
            pafs[idx] = paf
            seqs[idx] = v
            idx += 1
        if not _next_combination(c, L):
            break
    return pafs, seqs


def _cos_table(L: int) -> np.ndarray:
    h = half_len(L)
    ks = np.arange(1, h + 1)
    ss = np.arange(1, h + 1)
    return np.cos(2.0 * np.pi * np.outer(ks, ss) / L)


def collect_survivors_brute(L: int, tol: float = 1e-6):
    """PSD survivors over ALL pm1 sequences (row sum +1), bucketed by PAF key.

    Returns (buckets, stats). buckets: PafKey -> list of pm1 sequences.
    """
    check_odd(L)
    cos_table = _cos_table(L)
    n = int(_brute_count(L, cos_table, tol))
    pafs, seqs = _brute_collect(L, cos_table, tol, n)
    buckets: Dict[PafKey, List[np.ndarray]] = defaultdict(list)
    for i in range(n):
        key: PafKey = tuple(int(x) for x in pafs[i])
        buckets[key].append(seqs[i].copy())
    stats = {"survivors": n, "total": comb(L, P_of(L))}
    return buckets, stats
