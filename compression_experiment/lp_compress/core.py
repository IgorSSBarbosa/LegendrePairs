"""core.py — Phase 0 primitives (PAF, PSD, LP test).

Exact-integer PAF is the object we hash/pair/decide on; PSD is float and used
only as a filter (PLAN §0). We REUSE the verified half-vector PAF from
``lp_rle.paf`` and expand it to the full length-``ell`` cyclic PAF that the
compression identities (PLAN §2) act on.

Conventions (PLAN §1): ``ell`` odd, ``A in {-1,+1}^ell`` indexed mod ``ell``.
    PAF:  P_A(s) = sum_i A_i * A_{(i+s) mod ell},   P_A(0) = ell,  P_A(s)=P_A(ell-s).
    PSD:  PSD_A(k) = |sum_i A_i * exp(2πi·i·k/ell)|^2,  PSD_A(0) = (sum A)^2.

Complexity: ``paf_int`` is O(ell^2) exact-integer (production path, ell<=25);
``paf_fft``/``psd`` are O(ell log ell) float (prefilter / cross-check).
"""
from __future__ import annotations

import numpy as np

# Reused, verified primitives from the sibling lp_rle package.
from lp_rle.paf import paf_naive as _paf_half_naive
from lp_rle.conventions import check_odd, half_len


def paf_half(A) -> np.ndarray:
    """Half-vector integer PAF ``P_A(s)`` for ``s = 1..(ell-1)/2`` (reused, O(ell^2))."""
    A = np.ascontiguousarray(A, dtype=np.int8)
    check_odd(A.shape[0])
    return _paf_half_naive(A).astype(np.int64)


def paf_int(A) -> np.ndarray:
    """Full cyclic integer PAF vector of length ``ell``.

    ``out[0] = ell``; ``out[s] = P_A(s)`` for ``s = 1..ell-1`` using the symmetry
    ``P_A(ell-s) = P_A(s)``. This is the vector the compression identities index.
    """
    A = np.ascontiguousarray(A, dtype=np.int8)
    ell = A.shape[0]
    check_odd(ell)
    h = half_len(ell)
    half = _paf_half_naive(A).astype(np.int64)  # s = 1..h
    out = np.empty(ell, dtype=np.int64)
    out[0] = ell
    out[1 : h + 1] = half
    out[h + 1 :] = half[::-1]  # P_A(ell-s) = P_A(s)
    return out


def paf_fft(A) -> np.ndarray:
    """Full cyclic PAF via FFT, rounded to exact integers (independent cross-check).

    O(ell log ell). Uses the full complex FFT so the result is the length-``ell``
    vector directly comparable to :func:`paf_int`.
    """
    a = np.asarray(A, dtype=np.float64)
    F = np.fft.fft(a)
    full = np.fft.ifft(F * np.conjugate(F)).real
    rounded = np.rint(full)
    err = float(np.max(np.abs(full - rounded)))
    assert err < 1e-6, f"paf_fft rounding error {err} too large"
    return rounded.astype(np.int64)


def psd(A) -> np.ndarray:
    """Full PSD vector ``PSD_A(k)`` for ``k = 0..ell-1`` (single FFT, float).

    ``PSD_A(0) = (sum A)^2`` and, by Parseval, ``sum_k PSD_A(k) = ell * sum A_i^2``
    (``= ell^2`` for a +-1 sequence). Float filter only — never hashed.
    """
    a = np.asarray(A, dtype=np.float64)
    F = np.fft.fft(a)
    return (F * np.conjugate(F)).real


def is_legendre_pair(A, B) -> bool:
    """Exact LP test (PLAN §1, VERIFIED): ``P_A(s)+P_B(s) == -2`` for all ``s != 0``.

    Uses exact-integer :func:`paf_int`; no float comparison.
    """
    A = np.ascontiguousarray(A, dtype=np.int8)
    B = np.ascontiguousarray(B, dtype=np.int8)
    if A.shape != B.shape:
        return False
    check_odd(A.shape[0])
    s = paf_int(A) + paf_int(B)
    return bool(np.all(s[1:] == -2))
