"""litdata.py — published inequivalent Legendre-pair counts (spec §3b).

Source: R. J. Fletcher, M. Gysin, J. Seberry, "Application of the discrete
Fourier transform to the search for generalised Legendre pairs and Hadamard
matrices", Australasian J. Combinatorics 23 (2001) 75-86 -- Table 3, column
"NGL-pairs" (the number of inequivalent GL-pairs). Their equivalence convention
(swap, independent cyclic shift, independent reversal, common decimation) is the
group implemented in symmetry.canonical_pair, so our lp_classes(L) must match
these numbers.

If our count disagrees with the literature, we assume the bug is ours (spec §3b).
The PDF lives at ../papers/LegendrePairsAndHadamardMatrices2001.pdf.
"""
from __future__ import annotations

from typing import Optional

# L -> number of inequivalent Legendre (GL) pairs
NGL_PAIRS = {
    3: 1,
    5: 1,
    7: 1,
    9: 1,
    11: 2,
    13: 4,
    15: 8,
    17: 7,
    19: 9,
    21: 22,
    23: 28,
    25: 46,
}


def inequivalent_count(L: int) -> Optional[int]:
    """Published inequivalent-LP count for L, or None if not tabulated here."""
    return NGL_PAIRS.get(L)
