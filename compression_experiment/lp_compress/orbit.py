"""orbit.py — Phase 4: orbit reduction of compressed pairs before the lift.

Compression is EQUIVARIANT: a symmetry ``g`` of the length-``ell`` LP problem
descends to an action on the length-``m`` compressed vector. So we may quotient
the surviving compressed pairs by this induced group, lift only ONE
representative per orbit, and still recover every LP class after the group is
re-applied at the sequence level (:func:`lp_rle.symmetry.canonical_pair` folds
the sequence-level group when we finally dedup).

Induced actions on ``cA`` (each VERIFIED empirically against ``compress``; see
``test_lift.py``), for ``ell = m*n``:

  * cyclic shift ``A_i -> A_{i-t}``   =>  ``cA -> roll(cA, t mod m)``   (Z_m),
  * reversal     ``A_i -> A_{-i}``    =>  ``cA -> cA[(-j) mod m]``,
  * negation     ``A   -> -A``        =>  ``cA -> -cA``,
  * swap         ``(A,B) -> (B,A)``   =>  ``(cA,cB) -> (cB,cA)``.

Shift/reversal/negation act INDEPENDENTLY on the two sequences (PAF is
shift-invariant), so the non-swap part FACTORS per sequence:
``canonical`` of a pair is ``sorted(canonical(cA), canonical(cB))``.

GATED (behind ``use_multipliers``, default OFF): decimation ``A_i -> A_{a*i}``
(``gcd(a, ell)=1``) descends to the compressed multiplier ``cA -> cA[(a*j) mod m]``.
This is applied COMMONLY to both sequences (like sequence-level decimation) and
is tied to the UNPINNED group-order STUB (:mod:`lp_compress.group`), hence off by
default; enabling it can only shrink orbits further, never lose an LP class.
"""
from __future__ import annotations

from collections import defaultdict
from math import gcd
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# induced generators on a single compressed vector
# --------------------------------------------------------------------------- #
def rotate_c(c, t: int) -> np.ndarray:
    """Induced cyclic shift ``cA -> roll(cA, t mod m)`` (VERIFIED)."""
    c = np.asarray(c, dtype=np.int8)
    return np.roll(c, t % c.shape[0])


def negate_c(c) -> np.ndarray:
    """Induced negation ``cA -> -cA`` (VERIFIED)."""
    return -np.asarray(c, dtype=np.int8)


def reverse_c(c) -> np.ndarray:
    """Induced reversal ``cA -> cA[(-j) mod m]`` (VERIFIED)."""
    c = np.asarray(c, dtype=np.int8)
    m = c.shape[0]
    return c[(-np.arange(m)) % m]


def multiply_c(c, a: int) -> np.ndarray:
    """Induced decimation/multiplier ``cA -> cA[(a*j) mod m]`` (GATED, STUB-linked)."""
    c = np.asarray(c, dtype=np.int8)
    m = c.shape[0]
    if gcd(a, m) != 1:
        raise ValueError(f"multiplier a={a} not a unit mod m={m}")
    return c[(a * np.arange(m)) % m]


def _units(m: int) -> List[int]:
    return [a for a in range(1, m) if gcd(a, m) == 1] or [1]


# --------------------------------------------------------------------------- #
# canonical forms
# --------------------------------------------------------------------------- #
def canonical_c(c, m: Optional[int] = None) -> bytes:
    """Least byte-key of ``c`` under the per-sequence group (rotation+reversal+negation).

    Enumerates all ``4*m`` group elements (dihedral ``D_m`` times negation) and
    returns the lexicographically smallest ``int8`` byte encoding. A pure group
    minimum => a well-defined canonical representative.
    """
    base = np.asarray(c, dtype=np.int8)
    m = base.shape[0] if m is None else m
    best = None
    for v in (base, reverse_c(base)):
        for w in (v, -v):
            for t in range(m):
                key = np.roll(w, t).astype(np.int8).tobytes()
                if best is None or key < best:
                    best = key
    return best


def canonical_compressed_pair(cA, cB, m: Optional[int] = None,
                              use_multipliers: bool = False) -> Tuple[bytes, bytes]:
    """Canonical key of an ORDERED compressed pair under the induced group + swap.

    Per-sequence canonicalization then swap-ordering. With ``use_multipliers`` a
    COMMON unit ``a mod m`` is also quotiented (GATED, STUB-linked) — taking the
    min over ``a`` keeps the key a genuine group minimum.
    """
    cA = np.asarray(cA, dtype=np.int8)
    cB = np.asarray(cB, dtype=np.int8)
    m = cA.shape[0] if m is None else m
    mults = _units(m) if use_multipliers else [1]
    best = None
    for a in mults:
        ca = multiply_c(cA, a) if a != 1 else cA
        cb = multiply_c(cB, a) if a != 1 else cB
        ka, kb = canonical_c(ca, m), canonical_c(cb, m)
        pair = (ka, kb) if ka <= kb else (kb, ka)  # swap-invariance
        if best is None or pair < best:
            best = pair
    return best


def orbit_reduced_pairs(pairs: Iterable[Tuple[np.ndarray, np.ndarray]], m: int,
                        use_multipliers: bool = False
                        ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """One representative compressed pair per induced-group orbit (order-stable)."""
    reps: Dict[Tuple[bytes, bytes], Tuple[np.ndarray, np.ndarray]] = {}
    for cA, cB in pairs:
        key = canonical_compressed_pair(cA, cB, m, use_multipliers)
        if key not in reps:
            reps[key] = (np.asarray(cA, dtype=np.int64), np.asarray(cB, dtype=np.int64))
    return list(reps.values())


def orbit_sizes(pairs: Iterable[Tuple[np.ndarray, np.ndarray]], m: int,
                use_multipliers: bool = False) -> Dict[Tuple[bytes, bytes], int]:
    """Map each orbit's canonical key to how many input pairs fell into it."""
    counts: Dict[Tuple[bytes, bytes], int] = defaultdict(int)
    for cA, cB in pairs:
        counts[canonical_compressed_pair(cA, cB, m, use_multipliers)] += 1
    return dict(counts)
