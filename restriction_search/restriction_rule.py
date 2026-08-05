"""restriction_rule.py — the validated p75-leading restriction rule.

Supersedes benchmark.py's original load_best_ij (argmax cell of
restriction_exact's yield fraction). compare_rules.py showed that rule
degenerates under extrapolation: it freezes at a fixed (2,1) for every ell
from 29 to 57 (space reduction ~57-60x, flat -- no real asymptotic win), and
even inside the exact range (ell<=27) it peaks around ell=13 and erodes back
down afterward.

p75_leading_ij(ell) instead fits i(ell) = round(a*sqrt(ell)+b) to the 75th
percentile of lead_pair = min(lead_u, lead_v) across REAL canonical LP classes
(canonical_orbits/results/run_length_rows.csv), with j fixed at 1 (trailing-run
percentiles are too noisy, R^2 ~ 0.2-0.6, to support a growing rule).

compare_rules.py validated this against a max-envelope alternative (both
lead_pair_max and trail_pair_max fit to sqrt(ell)): the max-envelope rule
produced an EMPTY restricted box (zero real pairs) at 4 of 13 tested ell in
3..27, while p75_leading never did. p75_leading also tracks the old
argmax-yield rule's live search speed almost exactly, while its closed-form
space-reduction factor keeps growing to ~4300x by ell=57 versus the old
rule's flat ~60x. See restriction_search/results/compare_rules.{csv,png}.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from math import sqrt

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RUN_LENGTH_ROWS = os.path.join(_HERE, "..", "canonical_orbits", "results", "run_length_rows.csv")

_fit_cache: tuple[float, float] | None = None


def fit_lead_p75() -> tuple[float, float]:
    """sqrt(ell) fit of the 75th percentile of lead_pair over canonical classes.
    Cached at module scope -- reads run_length_rows.csv once per process."""
    global _fit_cache
    if _fit_cache is not None:
        return _fit_cache
    rows = list(csv.DictReader(open(RUN_LENGTH_ROWS)))
    by_ell: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        by_ell[int(r["ell"])].append(int(r["lead_pair"]))
    ells = np.array(sorted(by_ell), dtype=float)
    y = np.array([np.percentile(by_ell[int(e)], 75) for e in ells])
    a, b = np.polyfit(np.sqrt(ells), y, 1)
    _fit_cache = (float(a), float(b))
    return _fit_cache


def p75_leading_ij(ell: int) -> tuple[int, int]:
    """(i, j) restriction under the p75-leading rule, clipped to this ell's grid."""
    a, b = fit_lead_p75()
    i = round(a * sqrt(ell) + b)
    imax, jmax = (ell + 1) // 2, (ell - 1) // 2
    return max(0, min(i, imax)), min(1, jmax)


if __name__ == "__main__":
    a, b = fit_lead_p75()
    print(f"p75_leading fit: i(ell) = round({a:.4f}*sqrt(ell) + {b:.4f}), j=1")
    for ell in range(3, 58, 2):
        i, j = p75_leading_ij(ell)
        print(f"  ell={ell:>3}: (i,j)=({i},{j})")
