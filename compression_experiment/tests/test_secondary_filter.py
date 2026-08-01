"""test_secondary_filter.py — Q1 secondary-modulus filter is SOUND and lossless.

The secondary-modulus filter is a necessary spectral condition used to prune
fibers before the exact PAF join. Two things must hold:

  1. SOUNDNESS — it never rejects a sequence that belongs to a true LP. Checked
     directly on every database LP at several ell, for each of that ell's
     incomparable divisors.
  2. LOSSLESS — running the full funnel with the filter ON yields the EXACT same
     LP class set as with it OFF. Checked at ell=15/21 (fast, feasible).
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from lp_compress.compress import incomparable_divisors
from lp_compress.core import is_legendre_pair
from lp_compress.parallel import _passes_secondary, pipeline_B_parallel

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(os.path.dirname(_HERE), "rle_experiment", "results", "lps")


def _pm(s: str) -> np.ndarray:
    b = np.frombuffer(s.strip().encode(), np.uint8)
    return np.where(b == ord("+"), 1, -1).astype(np.int8)


def _db_lps(ell: int, limit: int = 40):
    path = os.path.join(DB, f"LP{ell}.csv")
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append((_pm(row["u"]), _pm(row["v"])))
            if len(out) >= limit:
                break
    return out


@pytest.mark.parametrize("ell", [15, 21, 33])
def test_filter_never_rejects_a_true_lp(ell):
    """Every true LP passes the secondary condition at ALL of ell's divisors."""
    sec = tuple(incomparable_divisors(ell))
    lps = _db_lps(ell)
    assert lps, f"no DB LPs for ell={ell}"
    for A, B in lps:
        assert is_legendre_pair(A, B)
        for m2 in sec:
            assert _passes_secondary(A, ell, (m2,)), f"LP A rejected at ell={ell} m2={m2}"
            assert _passes_secondary(B, ell, (m2,)), f"LP B rejected at ell={ell} m2={m2}"
        assert _passes_secondary(A, ell, sec) and _passes_secondary(B, ell, sec)


def test_a_non_lp_row_sum_is_rejected():
    """A sequence with row sum != +-1 fails the PSD(0)=1 clause."""
    ell, m2 = 15, 5
    A = np.ones(ell, dtype=np.int8)          # sum = 15 -> PSD(0)=225, must be rejected
    assert not _passes_secondary(A, ell, (m2,))


@pytest.mark.parametrize("ell,m", [(15, 5), (21, 7)])
def test_filter_is_lossless_end_to_end(ell, m):
    sec = tuple(d for d in incomparable_divisors(ell) if d != m)
    base = pipeline_B_parallel(ell, m=m, n_workers=2, verbose=False)
    filt = pipeline_B_parallel(ell, m=m, n_workers=2, secondary_moduli=sec,
                               verbose=False)
    assert set(filt["classes"]) == set(base["classes"]), \
        f"filter changed class set at ell={ell}"
    assert filt["lp_classes"] == base["lp_classes"]
    # the filter may only shrink (or equal) the number of lifted candidates tested
    assert filt["n_lift_candidates"] <= base["n_lift_candidates"]
