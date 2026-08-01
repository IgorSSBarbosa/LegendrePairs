"""test_sieve_vec.py — the vectorized sieve must equal the serial cascade_pairs.

The serial :func:`lp_compress.compress.cascade_pairs` (Python meet-in-the-middle
over the compressed PAF key) is the SPEC. :func:`cascade_pairs_vec` computes the
same survivor SET with pure array ops; it must agree as a SET of ordered pairs
(row order may differ). If they disagree the fast path is wrong — trust the oracle.
"""
from __future__ import annotations

import numpy as np
import pytest

from lp_compress.compress import (
    cascade_pairs,
    cascade_pairs_vec,
    compressed_paf_sieve,
    _enumerate_compressed_array,
    enumerate_compressed,
)

CASES = [(9, 3), (15, 5), (21, 7), (25, 5), (27, 9), (15, 3), (21, 3)]


def _sset(pairs):
    return set((np.asarray(a, np.int8).tobytes(), np.asarray(b, np.int8).tobytes())
               for a, b in pairs)


@pytest.mark.parametrize("ell,m", CASES)
def test_vec_survivor_set_matches_serial(ell, m):
    ser = list(cascade_pairs(ell, m))
    CA, CB = cascade_pairs_vec(ell, m)
    assert CA.shape[0] == len(ser), f"count ell={ell} m={m}: {CA.shape[0]} != {len(ser)}"
    vec = [(CA[i], CB[i]) for i in range(CA.shape[0])]
    assert _sset(vec) == _sset(ser), f"survivor set differs ell={ell} m={m}"


@pytest.mark.parametrize("ell,m", CASES)
def test_vec_survivors_all_pass_the_verified_sieve(ell, m):
    """Every emitted pair must actually satisfy the VERIFIED compressed PAF sieve."""
    CA, CB = cascade_pairs_vec(ell, m)
    take = range(0, CA.shape[0], max(1, CA.shape[0] // 200))  # sample up to ~200
    for i in take:
        assert compressed_paf_sieve(CA[i], CB[i], ell, m)


@pytest.mark.parametrize("ell,m", CASES)
def test_enumeration_array_matches_itertools(ell, m):
    C = _enumerate_compressed_array(ell, m)
    ref = np.asarray(list(enumerate_compressed(ell, m)), dtype=np.int8)
    assert C.shape == ref.shape
    assert np.array_equal(C, ref), f"enumeration order/content differs ell={ell} m={m}"
