"""run_fiber.py — Q2 driver: discover LPs by fiber-restricted SA (route B head).

Pipeline: vectorized sieve -> vectorized orbit reduction -> for a SAMPLE of the
surviving compressed reps, run :func:`lp_compress.fiber_search.anneal_fiber_pair`
inside each fiber pair and collect the distinct LP classes found.

This is the SCALING route for ``ell`` where the exhaustive lift is infeasible: it
never enumerates the fiber, it searches it. Output is a LOWER BOUND on the class
count (discovery, not census). When a ground-truth DB exists we report coverage
(how many known classes were recovered) and verify every found pair is a true LP.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from typing import Dict, Optional

import numpy as np

from lp_rle.symmetry import canonical_pair

from .compress import cascade_pairs_vec, _default_modulus
from .orbit import orbit_reduce_arrays
from .core import is_legendre_pair
from .fiber_search import search_survivors
from .parallel import RESULTS

_DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "rle_experiment", "results", "lps")


def _pm(s: str) -> np.ndarray:
    b = np.frombuffer(s.encode(), np.uint8)
    return np.where(b == ord("+"), 1, -1).astype(np.int8)


def _db_class_keys(ell: int) -> Optional[set]:
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


def run_fiber(ell: int, m: Optional[int] = None, *, max_pairs: int = 60,
              iters: int = 20_000, restarts: int = 60, seed: int = 0,
              n_workers: Optional[int] = None, verbose: bool = True) -> Dict:
    m = _default_modulus(ell, m)
    n = ell // m

    t0 = time.perf_counter()
    CA, CB = cascade_pairs_vec(ell, m)
    reps = orbit_reduce_arrays(CA, CB, m, n)
    t_head = time.perf_counter() - t0

    t1 = time.perf_counter()
    res = search_survivors(ell, m, reps, max_pairs=max_pairs, iters=iters,
                           restarts=restarts, seed=seed, n_workers=n_workers,
                           verbose=verbose)
    t_search = time.perf_counter() - t1

    # every discovered pair must be a genuine LP (defensive, exact-integer)
    for u, v in res["classes"].values():
        assert is_legendre_pair(_pm(u), _pm(v)), "fiber search emitted a NON-LP!"

    db = _db_class_keys(ell)
    found = set(res["classes"])
    line = (f"ell={ell} m={m}: reps={len(reps):,} searched={res['n_pairs_searched']} "
            f"found_pairs={res['n_found']} classes={res['n_classes']}  "
            f"[head {t_head:.1f}s | search {t_search:.1f}s]")
    print(line, flush=True)
    if db is not None:
        cov = len(found & db)
        print(f"  DB: {len(db)} known classes; recovered {cov} "
              f"({100*cov/len(db):.0f}%); all found are valid LPs "
              f"{'and in DB' if found <= db else '(some NOT in DB!)'}", flush=True)
    else:
        print(f"  no DB file: {res['n_classes']} classes discovered (uncross-checked)",
              flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"fiber_LP{ell}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "u", "v"])
        for i, (u, v) in enumerate(sorted(res["classes"].values())):
            w.writerow([i, u, v])
    print(f"  wrote {out}", flush=True)
    res.update(ell=ell, m=m, head_s=t_head, search_s=t_search)
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    cases = []
    for a in args or ["27:9"]:
        if ":" in a:
            e, mm = a.split(":")
            cases.append((int(e), int(mm)))
        else:
            cases.append((int(a), None))
    print(f"fiber-restricted SA discovery — {len(cases)} case(s)")
    for ell, m in cases:
        run_fiber(ell, m, verbose=False)
