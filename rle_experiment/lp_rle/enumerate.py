"""enumerate.py — exhaustive generation of canonical run arrays.

Strategy at L<=25 is the honest "generate all compositions, filter by
is_canonical" (spec §3.1): the total number of rotation classes is
C(L,P)/L ~ 208k at L=25, so no CAT/prenecklace pruning is needed yet (that is
an explicitly deferred optimization, spec §6).

Orbit-count identity (spec §1.2), asserted in the tests:

    N(L, k) = (1/k) * C(P-1, k-1) * C(M-1, k-1)
    sum_{k=1}^{(L-1)/2} N(L, k) = C(L, P) / L
"""
from __future__ import annotations

from math import comb
from typing import Iterator, Tuple

import numpy as np

from .conventions import P_of, M_of, half_len, check_odd
from .runs import is_canonical


def compositions(n: int, k: int) -> Iterator[Tuple[int, ...]]:
    """Yield every composition of n into k positive parts (ordered tuples)."""
    if k <= 0 or n < k:
        return
    if k == 1:
        yield (n,)
        return
    for first in range(1, n - k + 2):
        for rest in compositions(n - first, k - 1):
            yield (first,) + rest


def _interleave(even_parts: Tuple[int, ...], odd_parts: Tuple[int, ...]) -> np.ndarray:
    k = len(even_parts)
    r = np.empty(2 * k, dtype=np.int64)
    for i in range(k):
        r[2 * i] = even_parts[i]
        r[2 * i + 1] = odd_parts[i]
    return r


def iter_canonical_k(L: int, k: int) -> Iterator[np.ndarray]:
    """Yield all canonical run arrays with exactly k pairs (2k runs)."""
    check_odd(L)
    P, M = P_of(L), M_of(L)
    for a in compositions(P, k):
        for b in compositions(M, k):
            r = _interleave(a, b)
            if is_canonical(r):
                yield r


def iter_canonical(L: int) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (k, r) over all canonical run arrays for k = 1..(L-1)/2."""
    for k in range(1, half_len(L) + 1):
        for r in iter_canonical_k(L, k):
            yield k, r


def N(L: int, k: int) -> int:
    """Number of rotation classes with exactly k pairs: (1/k)C(P-1,k-1)C(M-1,k-1)."""
    check_odd(L)
    P, M = P_of(L), M_of(L)
    num = comb(P - 1, k - 1) * comb(M - 1, k - 1)
    assert num % k == 0, "N(L,k) is not an integer — orbit freeness violated"
    return num // k


def total_classes(L: int) -> int:
    """C(L, P)/L, the total number of rotation classes across all k."""
    P = P_of(L)
    total = comb(L, P)
    assert total % L == 0
    return total // L
