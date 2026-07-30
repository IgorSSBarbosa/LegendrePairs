"""lp_rle — Legendre Pair search via a run-length (RLE) reparametrization.

The scientific goal of this package is to *measure* whether the run-length
reparametrization helps a stochastic search for Legendre pairs, not to find a
new pair. Everything is built as a measurement apparatus: exact-integer PAF
objectives, an exhaustive ground truth cross-checked by an independent brute
force, and a head-to-head benchmark of RLE vs. binary (pm1) move sets.

Conventions (see module `conventions` and the spec §0):
  * L is odd; v in {-1,+1}^L, indices cyclic mod L.
  * Both sequences normalized to row sum +1  =>  P=(L+1)/2 ones, M=(L-1)/2 minus-ones.
  * PAF is exact integer; PSD is float and used ONLY as a filter. Never hash floats.
"""

from .conventions import (
    P_of,
    M_of,
    half_len,
    check_odd,
    LP_PAF_RHS,
    PSD_BOUND,
)

__all__ = [
    "P_of",
    "M_of",
    "half_len",
    "check_odd",
    "LP_PAF_RHS",
    "PSD_BOUND",
]
