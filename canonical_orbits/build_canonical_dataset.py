"""build_canonical_dataset.py — merge mine + friend into one canonical LP database.

Extends ``compare_databases.py`` (which only cross-checks the two sources
class-by-class) into a completed, deduplicated canonical dataset: for every
odd ell where EITHER "mine" (rle_experiment/results/lps/LP{ell}.csv) or
"friend" (Legendre_database/legendre_pair{ell}.csv) has data, take the union
of canonical classes and record, for each, which source(s) attest it plus the
exact size of its equivalence-class orbit (lp_group.orbit_size).

ell range: only mine/friend are used (NOT the character-construction pairs in
Legendre_construction/ or the root LP_*.txt files) — those come from one
algebraic family and are not an exhaustive census, so mixing them in would
bias any downstream distribution statistics. See conversation for the
discussion; revisit if that scope changes.

Output: canonical_orbits/results/canonical_dataset.csv, columns
    ell, canonical_u, canonical_v, orbit_size, mine, mine_index, friend, friend_index
"""
from __future__ import annotations

import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "..")
sys.path.insert(0, _REPO)

from compare_databases import (  # noqa: E402
    MINE_DIR, FRIEND_DIR, parse_mine, parse_friend, _canon_index,
)
from lp_group import orbit_size  # noqa: E402

OUT_DIR = os.path.join(_HERE, "results")
OUT_PATH = os.path.join(OUT_DIR, "canonical_dataset.csv")

FIELDS = ["ell", "canonical_u", "canonical_v", "orbit_size",
          "mine", "mine_index", "friend", "friend_index"]


def available_lengths(max_ell: int = 200) -> list[int]:
    """Odd ell for which EITHER database has a file."""
    out = []
    for ell in range(3, max_ell + 1, 2):
        has_mine = os.path.exists(os.path.join(MINE_DIR, f"LP{ell}.csv"))
        has_friend = os.path.exists(os.path.join(FRIEND_DIR, f"legendre_pair{ell}.csv"))
        if has_mine or has_friend:
            out.append(ell)
    return out


def build_ell(ell: int) -> list[dict]:
    mine_path = os.path.join(MINE_DIR, f"LP{ell}.csv")
    friend_path = os.path.join(FRIEND_DIR, f"legendre_pair{ell}.csv")

    mine_idx = _canon_index(parse_mine(mine_path)) if os.path.exists(mine_path) else {}
    friend_idx = _canon_index(parse_friend(friend_path)) if os.path.exists(friend_path) else {}

    keys = set(mine_idx) | set(friend_idx)
    rows = []
    for u_str, v_str in keys:
        rows.append({
            "ell": ell,
            "canonical_u": u_str,
            "canonical_v": v_str,
            "orbit_size": orbit_size(u_str, v_str),
            "mine": "X" if (u_str, v_str) in mine_idx else "",
            "mine_index": ";".join(mine_idx.get((u_str, v_str), [])),
            "friend": "X" if (u_str, v_str) in friend_idx else "",
            "friend_index": ";".join(friend_idx.get((u_str, v_str), [])),
        })
    rows.sort(key=lambda r: (r["canonical_u"], r["canonical_v"]))
    return rows


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows: list[dict] = []
    print(f"{'ell':>3} {'classes':>7} {'mine_only':>9} {'friend_only':>11} {'both':>5}")
    print("-" * 42)
    for ell in available_lengths():
        rows = build_ell(ell)
        all_rows.extend(rows)
        mine_only = sum(1 for r in rows if r["mine"] and not r["friend"])
        friend_only = sum(1 for r in rows if r["friend"] and not r["mine"])
        both = sum(1 for r in rows if r["mine"] and r["friend"])
        print(f"{ell:>3} {len(rows):>7} {mine_only:>9} {friend_only:>11} {both:>5}", flush=True)

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nwrote {len(all_rows)} canonical classes across "
          f"{len(available_lengths())} lengths to {OUT_PATH}")


if __name__ == "__main__":
    main()
