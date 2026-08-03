"""lp_group.py — orbit size and run-length helpers on canonical LP pairs.

Reuses the SAME equivalence group GG as ``compare_databases.py`` (itself
matching ``rle_experiment/lp_rle/symmetry.py``): independent cyclic shift of
each sequence, a common decimation k in units(ell) with v allowed the mirror
-k, the switch (u,v)->(v,u), and negation folded away by the row-sum +1
normalization (never searched explicitly — see note in orbit_size below).

orbit_size(u, v) computes |G| / |Stab(u,v)| directly, via exact stabilizer
detection: for every one of the 4*phi(ell) (decimation, mirror, switch)
combinations, count how many independent shifts of each side land back on
(u, v) exactly, and sum. This is O(ell^2 * phi(ell)) per pair — fine up to the
ell~60 range this project's databases cover.

Why negation needs no explicit search: shifts and decimations are index
permutations, so they preserve each sequence's sum; since a genuine Legendre
pair has sum(u), sum(v) in {+1,-1} fixed by the data, the +1 representative is
already the deterministic image of whichever sign the group would need, so
folding it away up front (as compare_databases._normalize does) does not
under- or over-count the orbit relative to the same group compare_databases.py
already uses to build canonical_u/canonical_v.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "rle_experiment"))
sys.path.insert(0, os.path.join(_HERE, ".."))

from lp_rle.symmetry import units, decimate  # noqa: E402


def pm_from_str(s: str) -> np.ndarray:
    """'+'/'-' string -> +/-1 int8 array."""
    b = np.frombuffer(s.encode(), np.uint8)
    return np.where(b == ord('+'), 1, -1).astype(np.int8)


def _rot_count(cand: np.ndarray, target: np.ndarray, target_doubled: np.ndarray) -> int:
    """Number of s in [0, L) with rotate(target, s) == cand."""
    L = target.shape[0]
    count = 0
    for s in range(L):
        if np.array_equal(target_doubled[s:s + L], cand):
            count += 1
    return count


def orbit_size(u_str: str, v_str: str) -> int:
    """Exact size of the GG-orbit of (u, v), given as '+/-' strings."""
    u = pm_from_str(u_str)
    v = pm_from_str(v_str)
    L = u.shape[0]
    us = units(L)
    uu = np.concatenate([u, u])
    vv = np.concatenate([v, v])

    Ru = {k: decimate(u, k) for k in us}
    Rv = {k: decimate(v, k) for k in us}

    stab = 0
    for k in us:
        nk = (L - k) % L
        for a_arr, b_arr in ((Ru[k], Rv[k]), (Ru[k], Rv[nk]),
                             (Rv[k], Ru[k]), (Rv[nk], Ru[k])):
            nx = _rot_count(a_arr, u, uu)
            if nx == 0:
                continue
            ny = _rot_count(b_arr, v, vv)
            stab += nx * ny

    order = 4 * L * L * len(us)
    if stab == 0 or order % stab != 0:
        raise AssertionError(
            f"bad stabilizer count for L={L}: stab={stab}, order={order}")
    return order // stab


def leading_plus(s: str) -> int:
    n = 0
    for ch in s:
        if ch != '+':
            break
        n += 1
    return n


def trailing_minus(s: str) -> int:
    n = 0
    for ch in reversed(s):
        if ch != '-':
            break
        n += 1
    return n
