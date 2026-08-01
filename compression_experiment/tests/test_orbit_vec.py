"""test_orbit_vec.py — the vectorized orbit reducer must equal the serial oracle.

The serial :func:`lp_compress.orbit.orbit_reduced_pairs` (byte-key ``canonical_c``
min over the ``4m`` induced generators) is the SPEC. The vectorized
:func:`orbit_reduced_pairs_vec` uses a base-``(2n+1)`` int64 encoding instead; it
must return the SAME representatives in the SAME first-occurrence order on every
real survivor set. If they ever disagree, the fast path is wrong — trust the oracle.
"""
from __future__ import annotations

import numpy as np
import pytest

from lp_compress.compress import cascade_pairs, _default_modulus
from lp_compress.orbit import (
    orbit_reduced_pairs,
    orbit_reduced_pairs_vec,
    canonical_compressed_pair,
    _canonical_keys_batch,
    canonical_c,
    _vec_key_fits,
)

CASES = [(9, 3), (15, 5), (21, 7), (25, 5)]


def _keys(reps, m):
    return [canonical_compressed_pair(a, b, m, use_multipliers=False) for a, b in reps]


@pytest.mark.parametrize("ell,m", CASES)
def test_vec_matches_serial_reps_and_order(ell, m):
    pairs = list(cascade_pairs(ell, m))
    n = ell // m
    ser = orbit_reduced_pairs(pairs, m, use_multipliers=False)
    vec = orbit_reduced_pairs_vec(pairs, m, n)
    # same number of orbits
    assert len(vec) == len(ser), f"count ell={ell}: vec {len(vec)} != serial {len(ser)}"
    # same canonical key SET
    assert set(_keys(vec, m)) == set(_keys(ser, m)), f"orbit set differs ell={ell}"
    # SAME first-occurrence order (order-stable dedup): representative keys in sequence
    assert _keys(vec, m) == _keys(ser, m), f"order differs ell={ell}"


@pytest.mark.parametrize("ell,m", CASES)
def test_single_row_key_matches_bytekey_order(ell, m):
    """The integer key ranks rows in the same order canonical_c's byte key does."""
    pairs = list(cascade_pairs(ell, m))
    n = ell // m
    C = np.asarray([p[0] for p in pairs[:500]], dtype=np.int64)
    keys = _canonical_keys_batch(C, m, n)
    # byte canonical for the same rows
    bkeys = [canonical_c(C[i], m) for i in range(C.shape[0])]
    # equal rows -> equal int keys, and int order must match byte order pairwise
    order_int = np.argsort(keys, kind="stable")
    order_byte = sorted(range(len(bkeys)), key=lambda i: (bkeys[i], i))
    # ranks must agree up to ties; compare the induced key-equivalence classes
    assert [keys[i] for i in order_int] == sorted(keys.tolist())
    # rows the byte path calls equal must get equal int keys and vice-versa
    for i in range(0, C.shape[0], 7):
        for j in range(i + 1, min(i + 8, C.shape[0])):
            assert (keys[i] == keys[j]) == (bkeys[i] == bkeys[j])
            if keys[i] != keys[j]:
                assert (keys[i] < keys[j]) == (bkeys[i] < bkeys[j])


def test_key_fits_for_target_cases():
    # every case we actually run must take the fast int64 path, not the fallback
    for ell, m in [(15, 5), (21, 7), (27, 9), (33, 11)]:
        assert _vec_key_fits(m, ell // m), f"ell={ell} m={m} would fall back"
