"""Fixed conventions and small helpers shared across the package (spec §0).

These are deliberately trivial functions so that every module agrees on the
same definitions of P, M, the half-length (L-1)/2, and the two magic constants
of the Legendre-pair identity.
"""
from __future__ import annotations


def check_odd(L: int) -> None:
    """Assert L is odd. Called at every public entry point (spec §6)."""
    if L % 2 == 0:
        raise ValueError(f"L must be odd, got L={L}")


def P_of(L: int) -> int:
    """Number of +1 entries with row sum +1: P = (L+1)/2."""
    check_odd(L)
    return (L + 1) // 2


def M_of(L: int) -> int:
    """Number of -1 entries with row sum +1: M = (L-1)/2."""
    check_odd(L)
    return (L - 1) // 2


def half_len(L: int) -> int:
    """Number of independent shifts/frequencies: (L-1)/2.

    PAF_v(s) = PAF_v(L-s) and PSD_v(k) = PSD_v(L-k), so we only ever store
    s, k = 1 .. (L-1)/2.
    """
    check_odd(L)
    return (L - 1) // 2


def LP_PAF_RHS() -> int:
    """Right-hand side of the Legendre-pair PAF identity: PAF_u+PAF_v = -2."""
    return -2


def PSD_BOUND(L: int) -> int:
    """One-sequence PSD upper bound: PSD_v(k) <= 2L+2 for k != 0."""
    return 2 * L + 2
