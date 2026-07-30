import numpy as np
import pytest

from lp_rle.exhaust import validate, run_exhaustive
from lp_rle.litdata import inequivalent_count
from lp_rle.paf import paf_naive

# brute force is C(L,P) sequences; keep the default suite <= 19 for speed.
VALIDATE_FAST = [3, 5, 7, 9, 11, 13, 15, 17, 19]
VALIDATE_SLOW = [21, 23, 25]


def _is_lp(u, v, L):
    return np.all(paf_naive(u) + paf_naive(v) == -2)


@pytest.mark.parametrize("L", VALIDATE_FAST)
def test_validate_and_literature_and_real_lps(L):
    # one exhaustive+brute pass per L, asserting all three properties
    rep = validate(L)
    # (a) rotation multiplicity of the independent brute force
    assert rep["brute_survivors"] % L == 0
    # (b) literature cross-check (FGS-2001 NGL-pairs)
    want = inequivalent_count(L)
    assert want is not None
    assert rep["lp_classes"] == want, f"L={L}: got {rep['lp_classes']}, lit {want}"
    # (c) every reported LP is a real Legendre pair
    for u, v in rep["ex"]["classes"].values():
        assert _is_lp(u, v, L)


@pytest.mark.slow
@pytest.mark.parametrize("L", VALIDATE_SLOW)
def test_rle_vs_bruteforce_slow(L):
    rep = validate(L)
    assert rep["lp_classes"] == inequivalent_count(L)
