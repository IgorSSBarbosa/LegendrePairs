"""test_timing.py — Phase 6b spec: the wall-clock timing sweep.

The timing module measures SECONDS-to-enumerate-all-LP-classes per route and
records the class counts SEPARATELY, so this spec guards two independent things:

  * CORRECTNESS: every route that runs at a given ``ell`` agrees on lp_classes,
    and those counts match the known small-ell ground truth (1,1,1,1,2,4,8,...);
  * SHAPE: the sweep emits two separate CSVs with the documented schemas, route
    ``brute`` is cost-gated by ``brute_max``, and route ``B`` runs only where a
    liftable modulus exists.

Timing itself (seconds) is NOT asserted — wall-clock is machine-dependent and,
as the module honestly notes, route B is *slower* than A/brute at these tiny
ell despite its smaller combinatorial workload. We assert the numbers are real
(finite, non-negative, min<=median), never that one route beats another.
"""
from __future__ import annotations

import csv

import pytest

from lp_compress.timing import (
    route_A, route_brute, route_B, liftable_modulus,
    sweep, write_results, classes_agree,
    TIME_FIELDS, LP_FIELDS,
)

# known inequivalent-LP-class counts for odd ell (RLE ground truth).
KNOWN = {3: 1, 5: 1, 7: 1, 9: 1, 11: 2, 13: 4, 15: 8, 17: 7, 19: 9, 21: 22}


# --------------------------------------------------------------------------- #
# correctness of the individual routes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ell", [3, 5, 7, 9, 11, 13, 15])
def test_route_A_matches_ground_truth(ell):
    assert route_A(ell) == KNOWN[ell]


@pytest.mark.parametrize("ell", [3, 5, 7, 9, 11, 13, 15])
def test_route_brute_matches_ground_truth(ell):
    assert route_brute(ell) == KNOWN[ell]


@pytest.mark.parametrize("ell,m", [(9, 3), (15, 5)])
def test_route_B_matches_ground_truth(ell, m):
    assert route_B(ell, m) == KNOWN[ell]


def test_liftable_modulus_picks_where_expected():
    assert liftable_modulus(9) == 3       # only incomparable divisor, fits budget
    assert liftable_modulus(15) == 5      # m=3 fiber too big; m=5 fits
    assert liftable_modulus(11) is None   # prime -> not compressible
    assert liftable_modulus(13) is None   # prime -> not compressible


# --------------------------------------------------------------------------- #
# the sweep: separate tables, gating, agreement
# --------------------------------------------------------------------------- #
def test_sweep_tables_are_separate_and_well_formed():
    Ls = [3, 5, 9, 15]
    time_rows, lp_rows = sweep(Ls, repeats=1, warmup=False)

    # schemas are exactly the two documented field sets
    assert all(set(r) == set(TIME_FIELDS) for r in time_rows)
    assert all(set(r) == set(LP_FIELDS) for r in lp_rows)

    # every timed row has a matching correctness row (same ell/route/m) 1:1
    key = lambda r: (r["ell"], r["route"], r["m"])
    assert sorted(map(key, time_rows)) == sorted(map(key, lp_rows))

    # timings are real numbers: non-negative and min <= median
    for r in time_rows:
        assert r["seconds_median"] >= 0.0
        assert r["seconds_min"] <= r["seconds_median"]
        assert r["repeats"] == 1


def test_sweep_gates_brute_and_B():
    Ls = [9, 15]
    _, lp_rows = sweep(Ls, repeats=1, warmup=False, brute_max=9)
    routes = {(r["ell"], r["route"]) for r in lp_rows}
    # brute gated off above brute_max=9
    assert (9, "brute") in routes
    assert (15, "brute") not in routes
    # B runs only where a liftable modulus exists (both 9 and 15 are liftable)
    assert (9, "B") in routes and (15, "B") in routes


def test_sweep_routes_agree_on_classes():
    Ls = [3, 5, 7, 9, 11, 13, 15]
    _, lp_rows = sweep(Ls, repeats=1, warmup=False)
    agree = classes_agree(lp_rows)
    assert all(agree.values())
    # and they agree on the *right* number
    for r in lp_rows:
        assert r["lp_classes"] == KNOWN[r["ell"]]


def test_write_results_emits_two_csvs(tmp_path):
    Ls = [3, 9]
    time_rows, lp_rows = sweep(Ls, repeats=1, warmup=False)
    tp, lp = write_results(time_rows, lp_rows, results_dir=str(tmp_path))

    with open(tp) as f:
        assert next(csv.reader(f)) == TIME_FIELDS
    with open(lp) as f:
        assert next(csv.reader(f)) == LP_FIELDS
    # the two files are distinct artifacts
    assert tp != lp
