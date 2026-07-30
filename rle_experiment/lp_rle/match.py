"""match.py — hash-based pairing of PSD survivors into Legendre pairs.

(u, v) is a Legendre pair iff PAF_u(s) = -2 - PAF_v(s) for every s. So we look
up each survivor's complement PAF key in a dict; never the O(#survivors^2)
double loop (spec §3.4).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .filter import PafKey
from .symmetry import canonical_pair


def complement_key(key: PafKey) -> PafKey:
    """The PAF key an LP partner must have: -2 - key componentwise."""
    return tuple(-2 - x for x in key)


def find_pairs(buckets: Dict[PafKey, List[np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """All ordered (u, v) LPs from the survivor buckets (may include swaps)."""
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for key, us in buckets.items():
        ckey = complement_key(key)
        partners = buckets.get(ckey)
        if partners is None:
            continue
        for u in us:
            for v in partners:
                pairs.append((u, v))
    return pairs


def inequivalent_classes(pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[Tuple[bytes, bytes], Tuple[np.ndarray, np.ndarray]]:
    """Collapse raw LPs to inequivalent classes (canonical_pair key)."""
    classes: Dict[Tuple[bytes, bytes], Tuple[np.ndarray, np.ndarray]] = {}
    for u, v in pairs:
        key = canonical_pair(u, v)
        if key not in classes:
            classes[key] = (u, v)
    return classes
