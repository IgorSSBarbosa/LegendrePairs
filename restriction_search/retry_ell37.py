"""retry_ell37.py — re-run just the ell=37 cell of restricted_scaling.py with
a longer time budget.

The original 60s cap left ell=37 pruned at 0.2% of its candidate space
scanned (binary) -- see restricted_scaling.py's docstring for why only
binary/rle exhaustive, restricted-only, single deterministic run. Nearby ell
(29..35) needed 1-3% of their space scanned to hit a match, so a 10x longer
budget is a reasonable next try before concluding ell=37 needs something
smarter than a longer cap.

Merges the new (ell=37, rep) rows into the existing results/restricted_scaling.csv
in place (same experiment, same rule, just a completed data point) and
regenerates the plot. Not meant to be re-run routinely -- BUDGET is a
one-off parameter, not something later ell should inherit by default.
"""
from __future__ import annotations

import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "canonical_orbits"))

from restriction_matrix import denominator  # noqa: E402
from restricted_binary import iter_restricted_candidates  # noqa: E402
from restricted_rle import iter_restricted_run_seqs  # noqa: E402
from restriction_rule import p75_leading_ij  # noqa: E402
from common import find_first_pair  # noqa: E402
import restricted_scaling as rs  # noqa: E402

ELL = 37
BUDGET = 600.0  # seconds -- 10x the original 60s cap


def run() -> list[dict]:
    i, j = p75_leading_ij(ELL)
    d = denominator(ELL, i, j)
    print(f"=== ell={ELL}: p75_leading (i,j)=({i},{j}), D={d:,}, budget={BUDGET:g}s ===", flush=True)

    r_bin = find_first_pair(iter_restricted_candidates(ELL, i, j), ELL, time_budget=BUDGET)
    print(f"  binary: found={r_bin['found']} scanned={r_bin['scanned']:,}/{d:,} "
          f"in {r_bin['seconds']:.4g}s{' (PRUNED)' if r_bin.get('pruned') else ''}", flush=True)

    i_rle, j_rle = max(i, 1), max(j, 1)
    r_rle = find_first_pair(iter_restricted_run_seqs(ELL, i_rle, j_rle), ELL, time_budget=BUDGET)
    print(f"  rle:    found={r_rle['found']} scanned={r_rle['scanned']:,} "
          f"in {r_rle['seconds']:.4g}s{' (PRUNED)' if r_rle.get('pruned') else ''}", flush=True)

    return [
        {"ell": ELL, "rep": "binary", "i": i, "j": j, "candidate_space": d,
         "found": r_bin["found"], "seconds": r_bin["seconds"], "scanned": r_bin["scanned"],
         "pruned": r_bin.get("pruned", False)},
        {"ell": ELL, "rep": "rle", "i": i_rle, "j": j_rle, "candidate_space": d,
         "found": r_rle["found"], "seconds": r_rle["seconds"], "scanned": r_rle["scanned"],
         "pruned": r_rle.get("pruned", False)},
    ]


def merge(new_rows: list[dict]) -> None:
    with open(rs.CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if int(r["ell"]) != ELL]
    for r in new_rows:
        rows.append({k: r[k] for k in rs.FIELDS})
    rows.sort(key=lambda r: (int(r["ell"]), r["rep"]))
    for r in rows:
        r["ell"] = int(r["ell"])
        r["candidate_space"] = int(r["candidate_space"])
        r["found"] = r["found"] if isinstance(r["found"], bool) else r["found"] == "True"
        r["seconds"] = float(r["seconds"])
        r["scanned"] = int(r["scanned"])
        r["pruned"] = r["pruned"] if isinstance(r["pruned"], bool) else r["pruned"] == "True"
    rs.write_csv(rows)
    rs.make_plot(rows)


def main() -> None:
    new_rows = run()
    merge(new_rows)
    print(f"\nupdated {rs.CSV_PATH}\nupdated {rs.PLOT_PATH}")


if __name__ == "__main__":
    main()
