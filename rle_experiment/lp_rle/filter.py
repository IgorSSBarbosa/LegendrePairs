"""filter.py — PSD test, exact PAF-key extraction, and survivor collection.

Survivors are keyed by the EXACT integer PAF tuple (never by a float PSD).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .conventions import half_len
from .paf import paf_naive, psd_from_paf, psd_test
from .runs import runs_to_seq
from .enumerate import iter_canonical


PafKey = Tuple[int, ...]


def paf_key(paf_half: np.ndarray) -> PafKey:
    """Hashable exact-integer key from a PAF half-vector."""
    return tuple(int(x) for x in paf_half)


def is_psd_feasible(paf_half: np.ndarray, L: int, tol: float = 1e-6) -> bool:
    """Apply the one-sequence PSD filter PSD_v(k) <= 2L+2."""
    return psd_test(psd_from_paf(paf_half, L), L, tol)


def collect_survivors(L: int, tol: float = 1e-6):
    """Enumerate canonical runs, PSD-filter, and bucket survivors by PAF key.

    Returns (buckets, stats) where
      buckets: dict PafKey -> list of pm1 sequences (int8[L]),
      stats:   dict with 'classes', 'survivors', 'per_k' (tested/passed per k).
    """
    buckets: Dict[PafKey, List[np.ndarray]] = defaultdict(list)
    per_k: Dict[int, List[int]] = defaultdict(lambda: [0, 0])  # [tested, passed]
    total = 0
    survivors = 0
    for k, r in iter_canonical(L):
        total += 1
        per_k[k][0] += 1
        v = runs_to_seq(r)
        pv = paf_naive(v)
        if is_psd_feasible(pv, L, tol):
            per_k[k][1] += 1
            survivors += 1
            buckets[paf_key(pv)].append(v)
    stats = {
        "classes": total,
        "survivors": survivors,
        "per_k": {k: tuple(per_k[k]) for k in sorted(per_k)},
    }
    return buckets, stats
