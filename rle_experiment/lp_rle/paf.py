"""paf.py — PAF and PSD, with three cross-checked PAF implementations.

Exactness rule (spec §0): PAF values are integers and are the object we hash,
pair, and optimize on. PSD values are generally irrational; they are used ONLY
as a float filter with a tolerance. Never hash a float.

All three PAF routines return the half-vector PAF_v(s) for s = 1..(L-1)/2,
exploiting PAF_v(s) = PAF_v(L-s). PAF_v(0) = L is implicit.

  1. paf_naive(v)  -- O(L^2) integer double loop; the PRODUCTION path at L<=25.
  2. paf_fft(v)    -- irfft(|rfft|^2) rounded to int; kept for larger L and as
                      an independent cross-check.
  3. paf_runs(r)   -- the run-merge reference: merge v's run partition with its
                      shift-by-s partition, accumulating +/-(overlap). This is
                      the reference implementation of the RLE idea and a
                      benchmark subject; it is NOT a production path (spec §6).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .conventions import half_len, PSD_BOUND


# --------------------------------------------------------------------------- #
# 1. naive integer PAF -- production path at L <= 25
# --------------------------------------------------------------------------- #
@njit(cache=True)
def paf_naive(v):
    """Exact integer PAF half-vector, O(L^2). v is int8[L]."""
    L = v.shape[0]
    h = (L - 1) // 2
    out = np.zeros(h, dtype=np.int64)
    for s in range(1, h + 1):
        acc = 0
        for i in range(L):
            acc += np.int64(v[i]) * np.int64(v[(i + s) % L])
        out[s - 1] = acc
    return out


# --------------------------------------------------------------------------- #
# 2. FFT PAF -- cross-check / larger-L path
# --------------------------------------------------------------------------- #
def paf_fft(v: np.ndarray) -> np.ndarray:
    """PAF half-vector via irfft(|rfft(v)|^2), rounded to int with a check."""
    v = np.asarray(v, dtype=np.float64)
    L = v.shape[0]
    F = np.fft.rfft(v)
    psd = (F * np.conjugate(F)).real
    full = np.fft.irfft(psd, n=L)
    rounded = np.rint(full)
    err = np.max(np.abs(full - rounded))
    assert err < 1e-6, f"paf_fft rounding error {err} too large"
    h = half_len(L)
    return rounded[1 : h + 1].astype(np.int64)


# --------------------------------------------------------------------------- #
# 3. run-merge PAF -- reference implementation of the RLE idea (benchmark only)
# --------------------------------------------------------------------------- #
def paf_runs(r: np.ndarray) -> np.ndarray:
    """PAF half-vector by merging run partitions of v and its shift.

    For each shift s, the circle splits into maximal segments on which both v
    and (v shifted by s) are constant; each contributes (length)*sign*sign.
    Only run boundaries are touched, so this is O(k) work per shift up to the
    boundary sort. Reference/benchmark path only -- never production (spec §6).
    """
    r = np.asarray(r, dtype=np.int64)
    L = int(r.sum())
    nruns = r.shape[0]
    starts = np.empty(nruns, dtype=np.int64)
    acc_pos = 0
    for i in range(nruns):
        starts[i] = acc_pos
        acc_pos += int(r[i])

    def sign_at(pos: int) -> int:
        idx = int(np.searchsorted(starts, pos, side="right")) - 1
        return 1 if idx % 2 == 0 else -1

    h = half_len(L)
    out = np.zeros(h, dtype=np.int64)
    for s in range(1, h + 1):
        bnds = np.unique(np.concatenate((starts % L, (starts - s) % L)))
        nb = bnds.shape[0]
        acc = 0
        for bi in range(nb):
            q = int(bnds[bi])
            qn = int(bnds[(bi + 1) % nb])
            length = (qn - q) % L
            if length == 0:
                length = L
            acc += length * sign_at(q) * sign_at((q + s) % L)
        out[s - 1] = acc
    return out


# --------------------------------------------------------------------------- #
# PSD (float filter only)
# --------------------------------------------------------------------------- #
def psd_from_paf(paf_half: np.ndarray, L: int) -> np.ndarray:
    """PSD_v(k) for k=1..(L-1)/2 from the integer PAF half-vector.

    PSD_v(k) = L + 2 * sum_{s=1}^{(L-1)/2} PAF_v(s) * cos(2*pi*s*k/L).
    """
    h = half_len(L)
    ks = np.arange(1, h + 1)
    ss = np.arange(1, h + 1)
    ang = 2.0 * np.pi * np.outer(ks, ss) / L
    return L + 2.0 * (np.cos(ang) @ paf_half.astype(np.float64))


def psd_test(psd: np.ndarray, L: int, tol: float = 1e-6) -> bool:
    """One-sequence PSD filter: PSD_v(k) <= 2L+2 for all k != 0."""
    return bool(np.all(psd <= PSD_BOUND(L) + tol))


def psd_excess(psd: np.ndarray, L: int) -> float:
    """Soft PSD-infeasibility score: sum_k max(0, PSD(k) - (2L+2))^2."""
    over = np.maximum(0.0, psd - PSD_BOUND(L))
    return float(np.sum(over * over))
