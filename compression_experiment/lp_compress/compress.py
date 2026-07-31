"""compress.py — Phase 3: mod-m compression and the VERIFIED compressed sieves.

Let ``ell = m*n``. Compress ``A`` mod ``m``:
    cA_j = sum_{r=0}^{n-1} A_{j + r*m},   j = 0..m-1
a length-``m`` integer vector with entries in ``{-n,-n+2,...,n}``.

VERIFIED identities (PLAN §2 — safe to hardcode):
  * spectral sub-lattice:  DFT_m(cA)(s) = Ahat(n*s),  so PSD^m_cA(s) = PSD_A(n*s);
  * compressed PAF:        P_cA(s) = sum_{t == s (mod m)} P_A(t);
  * compressed PSD sieve:  PSD^m_cA(s) + PSD^m_cB(s) = 2*ell + 2   (s != 0);
  * compressed PAF sieve:  P_cA(s)   + P_cB(s)       = -2*n         (s != 0 mod m).

Blind spot (VERIFIED): frequency ``k`` is constrained by mod-``m`` compression iff
``n | k``; primitive frequencies (``gcd(k,ell)=1``) are seen by no proper
compression. Divisor lattice: mod-``m`` and mod-``m'`` are independent only when
``m, m'`` are incomparable; ``incomparable_divisors`` returns the maximal antichain
(the maximal proper divisors ``ell/p``, one per prime ``p | ell``).
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from lp_rle.conventions import check_odd
from .sieve import compressed_integer_sieve


# --------------------------------------------------------------------------- #
# compression
# --------------------------------------------------------------------------- #
def compress(A, m: int) -> np.ndarray:
    """Mod-``m`` compression ``cA`` of a length-``ell`` sequence (VERIFIED §2)."""
    A = np.asarray(A, dtype=np.int64)
    ell = A.shape[0]
    if ell % m != 0:
        raise ValueError(f"m={m} does not divide ell={ell}")
    n = ell // m
    return A.reshape(n, m).sum(axis=0)  # cA_j = sum_r A[r*m + j]


def compressed_paf(c: np.ndarray) -> np.ndarray:
    """Cyclic PAF of a length-``m`` integer vector: ``P_c(s)`` for ``s=1..m-1``."""
    c = np.asarray(c, dtype=np.int64)
    m = c.shape[0]
    return np.array([int(np.dot(c, np.roll(c, -s))) for s in range(1, m)], dtype=np.int64)


def compressed_psd(c: np.ndarray) -> np.ndarray:
    """Full compressed PSD ``PSD^m_c(s) = |DFT_m(c)(s)|^2`` for ``s = 0..m-1``."""
    F = np.fft.fft(np.asarray(c, dtype=np.float64))
    return (F * np.conjugate(F)).real


# --------------------------------------------------------------------------- #
# divisor lattice
# --------------------------------------------------------------------------- #
def _prime_factors(ell: int) -> set:
    fs, x, d = set(), ell, 2
    while d * d <= x:
        while x % d == 0:
            fs.add(d)
            x //= d
        d += 1
    if x > 1:
        fs.add(x)
    return fs


def incomparable_divisors(ell: int) -> List[int]:
    """Maximal antichain of proper divisors: ``{ell/p : p prime, p | ell}``.

    Every proper divisor divides some ``ell/p``, so this set dominates all
    compression information without redundancy. Empty for prime ``ell`` (no
    nontrivial compression).
    """
    return sorted({ell // p for p in _prime_factors(ell) if 1 < ell // p < ell})


# --------------------------------------------------------------------------- #
# VERIFIED compressed sieves (a compressed ORDERED pair (cA, cB))
# --------------------------------------------------------------------------- #
def compressed_paf_sieve(cA, cB, ell: int, m: int) -> bool:
    """Exact-integer compressed PAF pairing (VERIFIED §2), ALL residues ``t=0..m-1``:

      * ``t = 0``:  ``P_cA(0)+P_cB(0) == 2*ell - 2n + 2``  (energy: ``P_c(0)=sum c_j^2``),
      * ``t != 0``: ``P_cA(t)+P_cB(t) == -2n``.

    The ``t=0`` relation is essential: it is the sum over the residue class
    ``t == 0 (mod m)`` of the LP identity (``P_A(0)=ell`` plus ``n-1`` shifts at
    ``-2``). Dropping it makes the sieve far too loose (ell=9: 504 vs 72 pairs).
    """
    cA = np.asarray(cA, dtype=np.int64)
    cB = np.asarray(cB, dtype=np.int64)
    n = ell // m
    if int(cA @ cA) + int(cB @ cB) != 2 * ell - 2 * n + 2:
        return False
    return bool(np.all(compressed_paf(cA) + compressed_paf(cB) == -2 * n))


def compressed_psd_sieve(cA, cB, ell: int, m: int, tol: float = 1e-6) -> bool:
    """Float compressed PSD pairing (VERIFIED §2), the Wiener-Khinchin dual:

      * ``s = 0``:  ``PSD^m_cA(0)+PSD^m_cB(0) == 2``   (``PSD^m_c(0)=(sum c_j)^2`` =>
        forces row sum +-1, automatic for any LP),
      * ``s != 0``: ``PSD^m_cA(s)+PSD^m_cB(s) == 2*ell+2``.

    Equivalent to :func:`compressed_paf_sieve` on all inputs (checked in the tests).
    """
    pa = compressed_psd(cA)
    pb = compressed_psd(cB)
    if abs(pa[0] + pb[0] - 2.0) >= tol:
        return False
    return bool(np.all(np.abs(pa[1:] + pb[1:] - (2 * ell + 2)) < tol))


# --------------------------------------------------------------------------- #
# enumeration of the compressed space + cascade (streamed)
# --------------------------------------------------------------------------- #
def compressed_alphabet(n: int) -> List[int]:
    """Reachable entry values of a mod-``m`` compression: ``{-n,-n+2,...,n}``."""
    return list(range(-n, n + 1, 2))


def enumerate_compressed(ell: int, m: int) -> Iterator[np.ndarray]:
    """Stream every reachable length-``m`` compressed vector (``(n+1)**m`` of them)."""
    n = ell // m
    for combo in product(compressed_alphabet(n), repeat=m):
        yield np.array(combo, dtype=np.int64)


def _paf_key(c: np.ndarray, m: int) -> Tuple[int, ...]:
    """Meet-in-the-middle key: (energy P_c(0)=sum c_j^2, P_c(1), ..., P_c(m-1))."""
    energy = int(np.dot(c, c))
    return (energy,) + tuple(int(np.dot(c, np.roll(c, -s))) for s in range(1, m))


def cascade_pairs(ell: int, m: Optional[int] = None, use_stub: bool = False
                  ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Stream surviving compressed ORDERED pairs ``(cA, cB)`` at modulus ``m``.

    Meet-in-the-middle on the compressed PAF key: a pair survives the VERIFIED
    sieve iff ``key(cB) == -2n - key(cA)``. Groups the ``(n+1)**m`` compressed
    vectors once (never the ``2**ell`` sequence space); the STUB integer sieve is
    applied per surviving pair only when ``use_stub=True`` (off by default).
    """
    check_odd(ell)
    m = _default_modulus(ell, m)
    n = ell // m
    groups: Dict[Tuple[int, ...], List[np.ndarray]] = defaultdict(list)
    for c in enumerate_compressed(ell, m):
        groups[_paf_key(c, m)].append(c)
    energy_target = 2 * ell - 2 * n + 2
    for key, clist in groups.items():
        # key = (energy, P(1), ..., P(m-1)); pair-target flips each via its RHS.
        target = (energy_target - key[0],) + tuple(-2 * n - x for x in key[1:])
        matches = groups.get(target)
        if not matches:
            continue
        for cA in clist:
            for cB in matches:
                if use_stub and not compressed_integer_sieve(cA, cB, ell, m):
                    continue
                yield cA, cB


def _default_modulus(ell: int, m: Optional[int]) -> int:
    if m is not None:
        return m
    divs = incomparable_divisors(ell)
    if not divs:
        raise ValueError(f"ell={ell} is prime: no nontrivial compression")
    return min(divs)  # cheapest (smallest m) first


def cascade(ell: int, m: Optional[int] = None, use_stub: bool = False) -> Dict:
    """Summary of the compression funnel head at modulus ``m`` (streamed count)."""
    m = _default_modulus(ell, m)
    n = ell // m
    n_vectors = (n + 1) ** m
    n_pairs = sum(1 for _ in cascade_pairs(ell, m, use_stub))
    return {
        "ell": ell,
        "m": m,
        "n": n,
        "n_vectors": n_vectors,
        "n_survivor_pairs": n_pairs,
    }
