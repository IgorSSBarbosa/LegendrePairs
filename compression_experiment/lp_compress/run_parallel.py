"""run_parallel.py — run the parallel compression funnel on larger ell and log.

For each requested ``(ell, m)`` this runs :func:`pipeline_B_parallel`, then:

  * writes the FOUND PAIRS to ``results/parallel_B_LP{ell}.csv`` in the same
    ``index,u,v`` (+/-) schema as the RLE database, so it diffs directly;
  * appends a per-stage TIMING row to ``results/parallel_B_timing.csv``;
  * CROSS-CHECKS the class count (and, when a ground-truth DB file exists, the
    exact canonical class SET) against ``rle_experiment/results/lps/LP{ell}.csv``.

Route B is a speed experiment over the SAME funnel, so the count must match the
database exactly; any mismatch is flagged loudly rather than silently written.
"""
from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .parallel import pipeline_B_parallel, RESULTS
from lp_rle.symmetry import canonical_pair

# ground-truth database (RLE), if present, for the exact set cross-check
_DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "rle_experiment", "results", "lps")

TIMING_FIELDS = ["ell", "m", "n", "n_workers", "n_compressed_pairs",
                 "n_orbit_reps", "n_lift_candidates", "n_lps", "lp_classes",
                 "sieve_s", "orbit_s", "lift_s", "total_s",
                 "db_classes", "matches_db"]


def _pm(s: str) -> np.ndarray:
    b = np.frombuffer(s.encode(), np.uint8)
    return np.where(b == ord('+'), 1, -1).astype(np.int8)


def _db_class_keys(ell: int) -> Optional[set]:
    """Canonical-pair key set from the RLE database file, or None if absent."""
    path = os.path.join(_DB_DIR, f"LP{ell}.csv")
    if not os.path.exists(path):
        return None
    keys = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            u, v = _pm(row["u"]), _pm(row["v"])
            nu = u if int(u.sum()) > 0 else -u
            nv = v if int(v.sum()) > 0 else -v
            keys.add(canonical_pair(nu, nv))
    return keys


def _write_pairs(ell: int, classes: Dict) -> str:
    path = os.path.join(RESULTS, f"parallel_B_LP{ell}.csv")
    os.makedirs(RESULTS, exist_ok=True)
    rows = sorted(classes.values())  # (u_str, v_str) lexicographic, deterministic
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "u", "v"])
        for i, (u, v) in enumerate(rows):
            w.writerow([i, u, v])
    return path


def _append_timing(row: Dict) -> str:
    path = os.path.join(RESULTS, "parallel_B_timing.csv")
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TIMING_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in TIMING_FIELDS})
    return path


def run_case(ell: int, m: Optional[int] = None,
             n_workers: Optional[int] = None) -> Dict:
    """Run one parallel case, save pairs + timing, cross-check vs the DB."""
    r = pipeline_B_parallel(ell, m=m, n_workers=n_workers)

    # exact SET cross-check against the ground-truth database when available
    db_keys = _db_class_keys(ell)
    found_keys = set(r["classes"])
    if db_keys is not None:
        matches = (found_keys == db_keys)
        r["db_classes"] = len(db_keys)
        r["matches_db"] = "yes" if matches else "NO"
        if not matches:
            only_b = len(found_keys - db_keys)
            only_db = len(db_keys - found_keys)
            print(f"  !! ell={ell} MISMATCH vs DB: "
                  f"B-only={only_b} DB-only={only_db}", flush=True)
        else:
            print(f"  ok ell={ell}: {len(found_keys)} classes == DB (exact set match)",
                  flush=True)
    else:
        r["db_classes"] = ""
        r["matches_db"] = "n/a"
        print(f"  -- ell={ell}: no DB file, {len(found_keys)} classes (uncross-checked)",
              flush=True)

    p = _write_pairs(ell, r["classes"])
    _append_timing(r)
    print(f"  wrote {p}", flush=True)
    return r


# default target ladder: larger composite ell with a feasible (n<=3..5) modulus
DEFAULT_CASES: List[Tuple[int, Optional[int]]] = [
    (21, 7),   # n=3, ~2.7M lift  (DB: 22 classes)
    (27, 9),   # n=3, ~488M lift  (DB: 102 classes)
]


def main(cases: Sequence[Tuple[int, Optional[int]]] = DEFAULT_CASES,
         n_workers: Optional[int] = None) -> None:
    print(f"parallel compression funnel (route B) — {len(cases)} case(s)")
    for ell, m in cases:
        run_case(ell, m, n_workers)
    print(f"\ntiming log: {os.path.join(RESULTS, 'parallel_B_timing.csv')}")


if __name__ == "__main__":
    # optional CLI: "ell:m ell:m ..." else DEFAULT_CASES
    args = sys.argv[1:]
    if args:
        cases = []
        for a in args:
            if ":" in a:
                e, mm = a.split(":")
                cases.append((int(e), int(mm)))
            else:
                cases.append((int(a), None))
        main(cases)
    else:
        main()
