import numpy as np
import pytest

from lp_rle.filter import collect_survivors
from lp_rle.exhaust import run_exhaustive
from lp_rle.parallel import collect_survivors_parallel, run_exhaustive_parallel

PAR_L = [3, 5, 7, 9, 11, 13, 15, 17]


def _freeze(buckets):
    """Byte-level snapshot of a survivor-bucket dict for exact comparison."""
    return {k: [v.tobytes() for v in vs] for k, vs in buckets.items()}


@pytest.mark.parametrize("L", PAR_L)
def test_parallel_survivors_identical_to_serial(L):
    sb, sstats = collect_survivors(L)
    pb, pstats = collect_survivors_parallel(L, workers=4)
    # stats agree exactly
    assert pstats == sstats
    # buckets are byte-identical (same keys, same sequences, same order)
    assert _freeze(pb) == _freeze(sb)


@pytest.mark.parametrize("L", PAR_L)
def test_parallel_classes_identical_to_serial(L):
    s = run_exhaustive(L)
    p = run_exhaustive_parallel(L, workers=4)
    assert p["survivors"] == s["survivors"]
    assert p["lp_classes"] == s["lp_classes"]
    assert set(p["classes"].keys()) == set(s["classes"].keys())


def test_parallel_single_worker_matches():
    # workers=1 takes the serial-list fast path; must still match.
    sb, _ = collect_survivors(13)
    pb, _ = collect_survivors_parallel(13, workers=1)
    assert _freeze(pb) == _freeze(sb)
