"""Exact k-swap endgame: crack a low-objective plateau by brute neighbourhood.

Local search reaches the ell=41 (and ell=29) plateau E=32 almost instantly but
cannot descend from it: a *coordinated* multi-bit move is required and random
2-swaps -- even CP-SAT window repair over a fixed set of positions -- miss it.
This module asks the direct question instead: *is a genuine Legendre pair within
k swaps of the plateau?*  A "swap" flips one +1 and one -1 in a sequence (so it
preserves the sum=+1 normalization); k swaps split any way across the two
sequences.  Unlike LNS the k swaps may land *anywhere*, so this is a strictly
richer neighbourhood than a fixed window.

The search is exact.  Enumerating every combination is a product of two large
sets, but we never form that product: a pair is a Legendre pair iff

    PAF_u(s) = -(PAF_v(s) + 2)   for all s,

so we hash one side's modifications by their PAF vector and probe with the
other side's *target* vector -- an O(#u-mods + #v-mods) meet-in-the-middle
instead of O(#u-mods * #v-mods).  Scanning total distance t = 0, 1, 2, ... and
returning the first hit yields the exact swap-distance from plateau to solution
(and a negative result at radius k is itself the measurement: the barrier is
wider than k swaps).
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from legendre import _fmt, is_legendre_pair  # noqa: E402


# --------------------------------------------------------------------------- #
# PAF helpers
# --------------------------------------------------------------------------- #
def _paf_vec(seq, half):
    """PAF vector (shifts 1..half) of a +-1 numpy sequence, as a Python tuple."""
    return tuple(int(np.dot(seq, np.roll(seq, -sh))) for sh in range(1, half + 1))


def _violation(a, b):
    """Sum_s |PAF_a(s) + PAF_b(s) + 2| over half shifts -- 0 iff a pair."""
    ell = len(a)
    half = (ell - 1) // 2
    pa = _paf_vec(np.asarray(a), half)
    pb = _paf_vec(np.asarray(b), half)
    return sum(abs(pa[i] + pb[i] + 2) for i in range(half))


def _energy(a, b):
    """Sum_s (PAF_a(s) + PAF_b(s) + 2)^2 over half shifts (local_search scale)."""
    ell = len(a)
    half = (ell - 1) // 2
    pa = _paf_vec(np.asarray(a), half)
    pb = _paf_vec(np.asarray(b), half)
    return sum((pa[i] + pb[i] + 2) ** 2 for i in range(half))


# --------------------------------------------------------------------------- #
# k-swap enumeration
# --------------------------------------------------------------------------- #
def _count_mods(nplus, nminus, m):
    """How many configs are exactly m swaps from a seq with these +1/-1 counts."""
    return comb(nplus, m) * comb(nminus, m)


def _iter_mods(seq, m, half):
    """Yield ``(paf_tuple, modified_seq)`` for every config exactly ``m`` swaps
    from ``seq`` (m plus-positions set to -1, m minus-positions set to +1)."""
    seq = np.asarray(seq, dtype=np.int64)
    if m == 0:
        yield _paf_vec(seq, half), seq.copy()
        return
    plus = np.where(seq == 1)[0].tolist()
    minus = np.where(seq == -1)[0].tolist()
    for pc in itertools.combinations(plus, m):
        pc = list(pc)
        for mc in itertools.combinations(minus, m):
            t = seq.copy()
            t[pc] = -1
            t[list(mc)] = 1
            yield _paf_vec(t, half), t


# --------------------------------------------------------------------------- #
# the endgame
# --------------------------------------------------------------------------- #
def endgame(a, b, max_swaps=3, verbose=True):
    """Exact search for a Legendre pair within ``max_swaps`` swaps of ``(a, b)``.

    Returns a dict: ``solved``, ``A``/``B`` (the pair, or None), ``distance`` (the
    minimal total swap count at which a solution was found, or None), ``split``
    (``(mu, mv)`` swaps used in each sequence), ``seconds``, and ``checked``
    (configs examined).
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    ell = len(a)
    half = (ell - 1) // 2
    napl, namin = int((a == 1).sum()), int((a == -1).sum())
    nbpl, nbmin = int((b == 1).sum()), int((b == -1).sum())

    t0 = time.perf_counter()
    checked = 0
    for t in range(0, max_swaps + 1):
        for mu in range(0, t + 1):
            mv = t - mu
            if mu > napl or mu > namin or mv > nbpl or mv > nbmin:
                continue
            cu = _count_mods(napl, namin, mu)
            cv = _count_mods(nbpl, nbmin, mv)
            checked += cu + cv
            # meet in the middle: hash the smaller side by matching key
            if cu <= cv:
                umap = {}
                for pu, ua in _iter_mods(a, mu, half):
                    umap.setdefault(pu, ua)          # PAF_u vector -> config
                for pv, vb in _iter_mods(b, mv, half):
                    target = tuple(-(x + 2) for x in pv)   # need PAF_u == target
                    hit = umap.get(target)
                    if hit is not None:
                        ok, _ = is_legendre_pair(hit.tolist(), vb.tolist())
                        if ok:
                            return _hit(hit, vb, t, (mu, mv), t0, checked)
            else:
                vmap = {}
                for pv, vb in _iter_mods(b, mv, half):
                    key = tuple(-(x + 2) for x in pv)      # target PAF_u
                    vmap.setdefault(key, vb)
                for pu, ua in _iter_mods(a, mu, half):
                    hit = vmap.get(pu)                     # PAF_u matches target?
                    if hit is not None:
                        ok, _ = is_legendre_pair(ua.tolist(), hit.tolist())
                        if ok:
                            return _hit(ua, hit, t, (mu, mv), t0, checked)
        if verbose:
            print(f"  distance {t}: no solution "
                  f"({time.perf_counter() - t0:.2f}s, {checked:,} configs)")
    return {"solved": False, "A": None, "B": None, "distance": None,
            "split": None, "seconds": time.perf_counter() - t0,
            "checked": checked}


def _hit(a, b, dist, split, t0, checked):
    return {"solved": True, "A": a.tolist(), "B": b.tolist(), "distance": dist,
            "split": split, "seconds": time.perf_counter() - t0,
            "checked": checked}


# --------------------------------------------------------------------------- #
# driver / CLI
# --------------------------------------------------------------------------- #
def _incumbent(ell, restarts, steps, seconds, seed):
    """Run basinhop; return (solved, a, b, E) using best config even if unsolved."""
    from local_search import search  # noqa: E402  (sys.path already set)

    res = search(ell, strategy="basinhop", restarts=restarts, steps=steps,
                 max_seconds=seconds, seed=seed)
    if res["solved"]:
        return True, res["A"], res["B"], 0
    a, b = res["best_A"], res["best_B"]
    return False, a, b, _energy(a, b)


def main() -> int:
    p = argparse.ArgumentParser(description="Exact k-swap endgame from a "
                                "low-objective plateau (radius measurement).")
    p.add_argument("ell", type=int, help="odd length of the pair")
    p.add_argument("--max-swaps", type=int, default=3,
                   help="largest total swap distance to search (default 3)")
    p.add_argument("--trials", type=int, default=1,
                   help="number of independent plateau incumbents to test")
    p.add_argument("--basin-restarts", type=int, default=1)
    p.add_argument("--basin-steps", type=int, default=200000)
    p.add_argument("--basin-seconds", type=float, default=3.0,
                   help="wall-clock cap for the phase-1 basinhop (default 3s)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.ell <= 0 or args.ell % 2 == 0:
        p.error(f"ell must be a positive odd integer, got {args.ell}")

    print(f"[endgame] ell={args.ell}  max_swaps={args.max_swaps}  "
          f"trials={args.trials}")
    distances = []
    for trial in range(args.trials):
        seed = args.seed + trial
        solved, a, b, E = _incumbent(args.ell, args.basin_restarts,
                                     args.basin_steps, args.basin_seconds, seed)
        if solved:
            print(f"trial {trial} (seed {seed}): basinhop solved directly "
                  f"-> distance 0")
            distances.append(0)
            continue
        v = _violation(a, b)
        print(f"trial {trial} (seed {seed}): incumbent E={E} violation={v}")
        res = endgame(a, b, max_swaps=args.max_swaps, verbose=True)
        if res["solved"]:
            print(f"  -> SOLVED at swap-distance {res['distance']} "
                  f"split={res['split']}  ({res['seconds']:.2f}s)")
            ok, reason = is_legendre_pair(res["A"], res["B"])
            print(f"     verified: {ok}{'' if ok else ' -- ' + reason}")
            print(f"     A = {_fmt(res['A'])}")
            print(f"     B = {_fmt(res['B'])}")
            distances.append(res["distance"])
        else:
            print(f"  -> no solution within {args.max_swaps} swaps "
                  f"({res['checked']:,} configs, {res['seconds']:.2f}s)")
            distances.append(None)

    found = [d for d in distances if d is not None]
    print(f"\nsummary: {len(found)}/{args.trials} incumbents cracked within "
          f"{args.max_swaps} swaps", end="")
    if found:
        print(f"; distances = {distances}")
    else:
        print(f"; the barrier is wider than {args.max_swaps} swaps for all "
              f"tested incumbents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
