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


# --------------------------------------------------------------------------- #
# VECTORIZED orbit reduction (default group only, no multipliers)
# --------------------------------------------------------------------------- #
# Correctness identity with the byte-key path above:
#   compressed entries live in ``{-n,-n+2,...,n}``, so mapping ``x -> x % (2n+1)``
#   yields a digit in ``0..2n`` whose ORDER equals the ``int8.tobytes()`` unsigned
#   order (``0<1<...<n<-n<...<-1``).  Encoding a length-``m`` row as the base-
#   ``(2n+1)`` integer with entry 0 most-significant therefore reproduces exactly
#   the lexicographic ``tobytes()`` comparison ``canonical_c`` minimizes over.  We
#   minimize over the same ``4m`` group elements (all rotations of ``{v,-v}`` and
#   of the plain reversal ``v[::-1]`` and its negation — the reversal's rotations
#   sweep the same coset ``reverse_c`` does), so the picked representative is
#   byte-for-byte the one the serial path picks.  A combined key ``lo*B+hi`` with
#   ``B=(2n+1)**m`` reproduces ``sorted((kA,kB))`` swap-folding in one int64.


def _vec_key_fits(m: int, n: int) -> bool:
    """True iff the base-``(2n+1)`` combined pair key fits a signed int64."""
    return (2 * n + 1) ** (2 * m) < 2 ** 63


def _canonical_keys_batch(C: np.ndarray, m: int, n: int) -> np.ndarray:
    """Per-row group-min integer key (min over all ``4m`` induced generators).

    ``C`` is ``(N, m)`` integer; returns ``(N,)`` int64 canonical keys matching the
    lexicographic ``canonical_c`` byte-minimum of each row (see identity note above).
    """
    base = 2 * n + 1
    powers = (base ** np.arange(m - 1, -1, -1)).astype(np.int64)  # entry 0 = MSB
    ii = np.arange(m)[:, None]
    tt = np.arange(m)[None, :]
    pmat = powers[(ii + tt) % m]                    # pmat[j,t] = base**(m-1-((j-t)%m))
    best: Optional[np.ndarray] = None
    for W in (C, -C, C[:, ::-1], -C[:, ::-1]):      # {v,-v} x {id, reversal}
        keys = (W % base).astype(np.int64) @ pmat   # (N,m): all rotations at once
        row = keys.min(axis=1)
        best = row if best is None else np.minimum(best, row)
    return best


def orbit_reduced_pairs_vec(pairs, m: int, n: int
                            ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Vectorized equivalent of :func:`orbit_reduced_pairs` (multipliers OFF).

    Identical output (same reps, same first-occurrence order) as the serial byte
    path, computed with a handful of ``(N,m)@(m,m)`` matmuls instead of ``4m``
    tiny numpy ops per sequence. Falls back to the serial path when the base-
    ``(2n+1)`` key would overflow int64 (:func:`_vec_key_fits`).
    """
    pairs = list(pairs)
    if not pairs:
        return []
    if not _vec_key_fits(m, n):
        return orbit_reduced_pairs(pairs, m, use_multipliers=False)
    CA = np.asarray([p[0] for p in pairs], dtype=np.int64)   # (N,m)
    CB = np.asarray([p[1] for p in pairs], dtype=np.int64)
    kA = _canonical_keys_batch(CA, m, n)
    kB = _canonical_keys_batch(CB, m, n)
    B = np.int64((2 * n + 1) ** m)
    lo = np.minimum(kA, kB)
    hi = np.maximum(kA, kB)
    combined = lo * B + hi                                   # swap-folded pair key
    _, first = np.unique(combined, return_index=True)        # first occurrence
    first.sort()                                             # restore input order
    return [(CA[i], CB[i]) for i in first]


def orbit_sizes(pairs: Iterable[Tuple[np.ndarray, np.ndarray]], m: int,
                use_multipliers: bool = False) -> Dict[Tuple[bytes, bytes], int]:
    """Map each orbit's canonical key to how many input pairs fell into it."""
    counts: Dict[Tuple[bytes, bytes], int] = defaultdict(int)
    for cA, cB in pairs:
        counts[canonical_compressed_pair(cA, cB, m, use_multipliers)] += 1
    return dict(counts)
