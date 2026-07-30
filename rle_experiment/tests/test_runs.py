import numpy as np
import pytest

from lp_rle.conventions import P_of, M_of, half_len
from lp_rle.runs import (
    runs_to_seq, seq_to_runs, canonical, is_canonical,
    rotate_runs, reverse_runs, _booth_least_index, _naive_least_index,
)
from lp_rle.enumerate import iter_canonical_k, iter_canonical, N, total_classes, compositions

ODD_SMALL = [3, 5, 7, 9, 11, 13, 15, 17]
ODD_ALL = list(range(3, 26, 2))


def _all_seqs_rowsum1(L):
    """All pm1 sequences of length L with exactly P ones (row sum +1)."""
    from itertools import combinations
    P = P_of(L)
    for ones in combinations(range(L), P):
        v = -np.ones(L, dtype=np.int8)
        v[list(ones)] = 1
        yield v


def test_roundtrip_and_invariants():
    for L in ODD_SMALL:
        for v in _all_seqs_rowsum1(L):
            r = seq_to_runs(v)
            assert r.sum() == L
            assert int((r[0::2]).sum()) == P_of(L)
            assert int((r[1::2]).sum()) == M_of(L)
            # RLE alt-sum condition => row sum +1, structurally
            assert int(r[0::2].sum() - r[1::2].sum()) == 1
            v2 = runs_to_seq(r)
            # v2 is v rotated to a positive-run start; same as canonical rotation of v
            assert np.array_equal(np.sort(v2), np.sort(v))
            # round trip through runs is stable
            assert np.array_equal(seq_to_runs(v2), r)


def test_all_rotations_same_canonical():
    for L in [5, 7, 9, 11, 13]:
        for v in _all_seqs_rowsum1(L):
            base = canonical(seq_to_runs(v)).tobytes()
            for t in range(L):
                vt = np.roll(v, t)
                if int(vt.sum()) != 1:
                    continue
                assert canonical(seq_to_runs(vt)).tobytes() == base


def test_booth_matches_naive():
    rng = np.random.default_rng(0)
    for L in ODD_ALL:
        for _ in range(200):
            k = rng.integers(1, half_len(L) + 1)
            a = _rand_composition(P_of(L), k, rng)
            b = _rand_composition(M_of(L), k, rng)
            r = np.empty(2 * k, dtype=np.int64)
            r[0::2] = a
            r[1::2] = b
            assert int(_booth_least_index(r)) == int(_naive_least_index(r))


def _rand_composition(n, k, rng):
    # random composition of n into k positive parts
    if k == 1:
        return np.array([n], dtype=np.int64)
    cuts = np.sort(rng.choice(np.arange(1, n), size=k - 1, replace=False))
    parts = np.diff(np.concatenate(([0], cuts, [n])))
    return parts.astype(np.int64)


def test_orbit_freeness():
    # every pair array w has rotation orbit of size exactly k (aperiodic)
    for L in [5, 7, 9, 11, 13]:
        for k, r in iter_canonical(L):
            seen = set()
            for t in range(k):
                seen.add(rotate_runs(r, t).tobytes())
            assert len(seen) == k


def test_reverse_roundtrip():
    for L in ODD_SMALL:
        for v in _all_seqs_rowsum1(L):
            r = seq_to_runs(v)
            rr = reverse_runs(r)
            # applying reversal twice returns the original canonical class
            assert canonical(reverse_runs(rr)).tobytes() == canonical(r).tobytes()
            # reverse_runs really is v[-i]
            vr = runs_to_seq(rr)
            L_ = v.shape[0]
            expected = runs_to_seq(r)[(-np.arange(L_)) % L_]
            assert canonical(seq_to_runs(vr)).tobytes() == canonical(seq_to_runs(expected)).tobytes()


def test_N_identity():
    for L in ODD_ALL:
        s = sum(N(L, k) for k in range(1, half_len(L) + 1))
        assert s == total_classes(L)


def test_N_spot_values():
    assert [N(5, k) for k in range(1, 3)] == [1, 1]
    assert total_classes(5) == 2
    assert [N(7, k) for k in range(1, 4)] == [1, 3, 1]
    assert total_classes(7) == 5


def test_enumerate_counts_match_N():
    for L in [5, 7, 9, 11, 13, 15]:
        for k in range(1, half_len(L) + 1):
            cnt = sum(1 for _ in iter_canonical_k(L, k))
            assert cnt == N(L, k)
