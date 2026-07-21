"""Persistent, append-only store of verified Legendre pairs.

Successful search runs call ``record_pair`` to save the pair they found to
``results/found_pairs.csv``.  The store is append-only -- it never overwrites the
file -- and it keeps at most ONE row per ``(method, ell)``: the first pair found
for a given method at a given length is kept, and later finds for that same
``(method, ell)`` are skipped (never duplicated).  This matches the per-method
layout of ``results/found_pairs.md`` while making the CSV the live, safe-to-append
source of truth.  Every pair is verified with ``is_legendre_pair`` before it is
written, so a bug in a search can never poison the store.

Dedup key is ``(method, ell)`` on purpose: different methods (greedy, anneal,
basinhop, ...) may each keep their own entry for the same length, so the CSV
stays comparable across methods.  To make it one-pair-per-length regardless of
method, change ``_key`` to use ``ell`` alone.
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legendre import is_legendre_pair  # noqa: E402

_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
_CSV = os.path.join(_RESULTS, "found_pairs.csv")
_HEADER = ["ell", "method", "A", "B", "seconds", "params", "timestamp"]


def _fmt(seq) -> str:
    """+-1 sequence -> compact +/- string (matches found_pairs.md)."""
    return "".join("+" if int(x) == 1 else "-" for x in seq)


def _key(row) -> tuple:
    return (str(int(row["ell"])), str(row["method"]))


def _load(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def has_pair(ell: int, method: str, path: str | None = None) -> bool:
    """True if a pair for ``(method, ell)`` is already stored."""
    path = path or _CSV
    want = (str(int(ell)), str(method))
    return any(_key(r) == want for r in _load(path))


def record_pair(ell: int, method: str, A, B, seconds="", params: str = "",
                path: str | None = None) -> dict:
    """Verify ``(A, B)`` and append one row unless ``(method, ell)`` is stored.

    Returns ``{"written": bool, "reason": str, "path": str}``.  Raises
    ``ValueError`` if ``(A, B)`` is not a genuine Legendre pair -- the store only
    ever holds verified pairs."""
    path = path or _CSV
    ell = int(ell)
    ok, reason = is_legendre_pair(list(A), list(B))
    if not ok:
        raise ValueError(f"refusing to store non-Legendre pair (ell={ell}): {reason}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if has_pair(ell, method, path):
        return {"written": False, "reason": "already_recorded", "path": path}
    new_file = not os.path.exists(path)
    secs = f"{seconds:.4f}" if isinstance(seconds, (int, float)) else str(seconds)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(_HEADER)
        w.writerow([ell, method, _fmt(A), _fmt(B), secs, params,
                    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")])
    return {"written": True, "reason": "recorded", "path": path}
