"""Sanity tests / demo for the Legendre-pair checker."""

import os
import sys
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from legendre import (
    find_legendre_pairs,
    find_legendre_pairs_incremental,
    find_legendre_pairs_parallel,
    find_one_legendre_pair,
    is_legendre_pair,
    paf,
    paf_profile,
)


def brute_force_pairs(ell, limit=None, require_normalization=False):
    """Enumerate all Legendre pairs of length ell (exponential; small ell only)."""
    seqs = [list(s) for s in product((1, -1), repeat=ell)]
    found = []
    for a in seqs:
        for b in seqs:
            ok, _ = is_legendre_pair(a, b, check_normalization=require_normalization)
            if ok:
                found.append((tuple(a), tuple(b)))
                if limit and len(found) >= limit:
                    return found
    return found


def test_paf_basic():
    # PAF(0) is always ell for +-1 sequences.
    for seq in ([1, 1, -1], [1, -1, 1, -1, 1]):
        assert paf(seq, 0) == len(seq)
    # PAF is symmetric: PAF(s) == PAF(ell - s).
    seq = [1, 1, -1, 1, -1]
    prof = paf_profile(seq)
    for s in range(1, len(seq)):
        assert prof[s] == prof[len(seq) - s]


def test_known_pair_ell3():
    assert is_legendre_pair([1, 1, -1], [1, 1, -1])[0]


def test_rejects_even_length():
    ok, reason = is_legendre_pair([1, -1], [1, -1])
    assert not ok and "even" in reason


def test_rejects_bad_entries():
    ok, reason = is_legendre_pair([1, 0, -1], [1, 1, -1])
    assert not ok and "outside" in reason


def test_rejects_length_mismatch():
    ok, reason = is_legendre_pair([1, 1, -1], [1, -1])
    assert not ok and "lengths differ" in reason


def test_brute_force_counts():
    # Every enumerated pair must actually satisfy the definition (consistency),
    # and there must be at least one for each valid odd length.
    for ell in (3, 5, 7):
        pairs = brute_force_pairs(ell)
        assert pairs, f"no Legendre pair found for ell={ell}"
        for a, b in pairs:
            assert is_legendre_pair(list(a), list(b))[0]
        print(f"ell={ell}: {len(pairs)} ordered Legendre pairs found")


def test_search_matches_brute_force():
    # The fast search (with sum==+-1 pruning) must find exactly the same
    # ordered pairs as the exhaustive brute force over all 2^ell sequences.
    for ell in (3, 5, 7):
        fast = set(find_legendre_pairs(ell, normalized=False))
        brute = set(brute_force_pairs(ell))
        assert fast == brute, f"mismatch at ell={ell}: {len(fast)} vs {len(brute)}"
        # Every pair the search returns is genuinely a Legendre pair.
        for a, b in fast:
            assert is_legendre_pair(list(a), list(b))[0]
        print(f"ell={ell}: search found {len(fast)} pairs (matches brute force)")


def test_find_one_for_small_odd_lengths():
    for ell in (3, 5, 7, 9, 11, 13):
        pair = find_one_legendre_pair(ell)
        assert pair is not None, f"no Legendre pair found for ell={ell}"
        a, b = pair
        assert is_legendre_pair(list(a), list(b))[0]
        # normalized=True -> canonical representatives with sum == 1.
        assert sum(a) == 1 and sum(b) == 1
        print(f"ell={ell}: sample pair A={list(a)} B={list(b)}")


def test_limit_is_respected():
    pairs = list(find_legendre_pairs(7, limit=5))
    assert len(pairs) == 5


def test_incremental_finds_valid_pairs_and_is_complete():
    # Incremental scan must (a) only emit genuine Legendre pairs and
    # (b) find a pair for exactly the lengths where the batch search does.
    for ell in (3, 5, 7, 9, 11):
        batch_has = bool(list(find_legendre_pairs(ell, limit=1)))
        inc = list(find_legendre_pairs_incremental(ell, limit=1))
        assert bool(inc) == batch_has, f"existence mismatch at ell={ell}"
        for a, b in inc:
            assert is_legendre_pair(list(a), list(b))[0]
    print("incremental scan: valid pairs, existence matches batch")


def test_incremental_early_stop_touches_few_candidates():
    # With limit=1 the scan should stop very early, not enumerate everything.
    import legendre

    calls = {"n": 0}
    real_key = legendre._paf_key

    def counting_key(seq, half):
        calls["n"] += 1
        return real_key(seq, half)

    legendre._paf_key = counting_key
    try:
        next(iter(find_legendre_pairs_incremental(15, limit=1)))
    finally:
        legendre._paf_key = real_key
    from math import comb

    total = comb(15, 7)
    assert calls["n"] < total, f"scanned {calls['n']} of {total} (no early stop)"
    print(f"incremental early stop: {calls['n']} of {total} candidates scanned")


def test_parallel_matches_serial():
    # Parallel search must return the same set of pairs as the serial version,
    # for both normalized and all-sums modes and across worker counts.
    for ell in (5, 7, 9):
        for normalized in (True, False):
            serial = set(find_legendre_pairs(ell, normalized=normalized))
            for workers in (2, 3):
                par = set(
                    find_legendre_pairs_parallel(
                        ell, normalized=normalized, workers=workers
                    )
                )
                assert par == serial, (
                    f"parallel!=serial at ell={ell}, normalized={normalized}, "
                    f"workers={workers}"
                )
        print(f"ell={ell}: parallel search matches serial")


if __name__ == "__main__":
    test_paf_basic()
    test_known_pair_ell3()
    test_rejects_even_length()
    test_rejects_bad_entries()
    test_rejects_length_mismatch()
    test_brute_force_counts()
    test_search_matches_brute_force()
    test_find_one_for_small_odd_lengths()
    test_limit_is_respected()
    test_incremental_finds_valid_pairs_and_is_complete()
    test_incremental_early_stop_touches_few_candidates()
    test_parallel_matches_serial()
    print("all tests passed")
