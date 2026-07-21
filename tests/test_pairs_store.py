"""Tests for the found-pairs store (``src/pairs_store.py``).

Pytest-free.  All writes go to a throwaway temp CSV so the real
``results/found_pairs.csv`` is never touched.
"""

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import pairs_store as ps  # noqa: E402
from legendre import find_one_legendre_pair  # noqa: E402


def _known_pair(ell):
    a, b = find_one_legendre_pair(ell)
    return list(a), list(b)


def test_records_and_dedupes():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "found.csv")
        a, b = _known_pair(13)

        r1 = ps.record_pair(13, "anneal", a, b, seconds=0.5, params="x", path=path)
        assert r1["written"] is True
        assert ps.has_pair(13, "anneal", path=path)

        # same (method, ell) -> skipped, not duplicated
        r2 = ps.record_pair(13, "anneal", a, b, seconds=9.9, path=path)
        assert r2["written"] is False and r2["reason"] == "already_recorded"

        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["A"] == ps._fmt(a) and rows[0]["B"] == ps._fmt(b)


def test_different_method_same_ell_kept():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "found.csv")
        a, b = _known_pair(11)
        assert ps.record_pair(11, "anneal", a, b, path=path)["written"]
        assert ps.record_pair(11, "basinhop", a, b, path=path)["written"]
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert {r["method"] for r in rows} == {"anneal", "basinhop"}


def test_different_ell_same_method_kept():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "found.csv")
        for ell in (7, 9, 11):
            a, b = _known_pair(ell)
            assert ps.record_pair(ell, "greedy", a, b, path=path)["written"]
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3


def test_rejects_non_pair():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "found.csv")
        bad_a = [1, 1, 1, 1, 1, 1, -1]        # not a Legendre pair with itself
        try:
            ps.record_pair(7, "greedy", bad_a, bad_a, path=path)
        except ValueError:
            assert not os.path.exists(path)   # nothing written
            return
        raise AssertionError("expected ValueError for a non-Legendre pair")


def test_header_written_once():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "found.csv")
        a7, b7 = _known_pair(7)
        a9, b9 = _known_pair(9)
        ps.record_pair(7, "anneal", a7, b7, path=path)
        ps.record_pair(9, "anneal", a9, b9, path=path)
        with open(path) as f:
            lines = f.read().strip().splitlines()
        assert lines[0].split(",")[:2] == ["ell", "method"]
        assert len(lines) == 3                # header + 2 rows


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
