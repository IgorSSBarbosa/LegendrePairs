"""lift.py — Phase 4: decompress a compressed vector back to its fiber.

Given ``ell = m*n`` and a length-``m`` compressed vector ``cA`` (entries in
``{-n,-n+2,...,n}``), the *fiber* is

    fiber(cA) = { A in {-1,+1}^ell : compress(A, m) == cA }.

Compression sums the ``n`` entries in each residue class ``j (mod m)``:
``cA_j = sum_{r=0}^{n-1} A_{j + r*m}``. The residue classes are INDEPENDENT, so
the fiber is a product of per-class choices: in class ``j`` we place
``k_j = (n + cA_j)/2`` plus-ones among the ``n`` slots, i.e. ``C(n, k_j)`` ways.
Hence (VERIFIED, exact):

    |fiber(cA)| = prod_{j=0}^{m-1} C(n, (n + cA_j)/2).

Nothing here is a heuristic: :func:`lift` streams the fiber exactly and every
yielded ``A`` satisfies ``compress(A, m) == cA`` by construction.
"""
from __future__ import annotations

from itertools import combinations, product
from math import comb
from typing import Iterator

import numpy as np


def fiber_size(cA, ell: int, m: int) -> int:
    """Exact fiber cardinality ``prod_j C(n, (n+cA_j)/2)`` (0 if ``cA`` unreachable)."""
    n = ell // m
    cA = np.asarray(cA, dtype=np.int64)
    size = 1
    for c in cA:
        c = int(c)
        if (n + c) % 2 != 0:
            return 0
        k = (n + c) // 2
        if k < 0 or k > n:
            return 0
        size *= comb(n, k)
    return size


def _class_choices(c: int, idxs, n: int):
    """All +-1 assignments to the ``n`` slots ``idxs`` summing to ``c`` (list of arrays)."""
    k = (n + c) // 2
    out = []
    for ones in combinations(range(n), k):
        vec = np.full(n, -1, dtype=np.int8)
        if ones:
            vec[list(ones)] = 1
        out.append((idxs, vec))
    return out


def lift(cA, ell: int, m: int) -> Iterator[np.ndarray]:
    """Stream every ``A in {-1,+1}^ell`` with ``compress(A, m) == cA`` (exact fiber).

    The residue classes ``j = 0..m-1`` are enumerated independently and combined
    by a Cartesian product, so the number of yields equals :func:`fiber_size`.
    Each ``A`` is a fresh ``int8`` array indexed ``0..ell-1``.
    """
    n = ell // m
    cA = np.asarray(cA, dtype=np.int64)
    per_class = []
    for j in range(m):
        c = int(cA[j])
        if (n + c) % 2 != 0 or not (0 <= (n + c) // 2 <= n):
            return  # empty fiber
        idxs = np.arange(j, ell, m)  # positions j, j+m, ..., j+(n-1)m
        per_class.append(_class_choices(c, idxs, n))
    for combo in product(*per_class):
        A = np.empty(ell, dtype=np.int8)
        for idxs, vec in combo:
            A[idxs] = vec
        yield A
