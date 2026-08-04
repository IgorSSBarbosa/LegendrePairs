"""restricted_local_search.py — restricted tabu/SA/basinhop via rle_experiment/lp_rle.

rle_experiment/lp_rle/walk.py already builds JointWalker/BasinHopper as
representation-agnostic: they only touch the state through `random_seq(L,rng)`
(init) and `moves.propose(v, rng)` (a MoveSet). To add restriction support
with minimal new code, we don't reimplement the walkers: we (a) subclass
JointWalker/BasinHopper to override _reset() with a restricted initializer,
and (b) wrap any MoveSet in a gate that rejects proposals touching the forced
zone. Meta(kind="tabu"/"sa") and BasinHopper's descent/kick cycle then work
unmodified -- restriction is entirely a matter of what states are reachable.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "rle_experiment"))
from lp_rle.walk import (  # noqa: E402
    Move, MoveSet, BinaryMoveSet, RLEMoveSet, Meta, Stats, JointWalker, BasinHopper,
)
from lp_rle.conventions import P_of  # noqa: E402


def restricted_random_seq(L: int, i: int, j: int, rng: np.random.Generator) -> np.ndarray:
    """Row-sum-+1 sequence with the first i entries forced +1, last j forced -1,
    the interior [i, L-j) filled with a uniformly random arrangement of the
    remaining +1/-1 budget."""
    v = np.empty(L, dtype=np.int8)
    v[:i] = 1
    v[L - j:] = -1  # no-op (empty slice) when j==0
    interior = L - i - j
    p_free = P_of(L) - i
    if interior < 0 or p_free < 0 or p_free > interior:
        raise ValueError(f"infeasible restriction: L={L}, i={i}, j={j}")
    body = -np.ones(interior, dtype=np.int8)
    ones = rng.choice(interior, size=p_free, replace=False)
    body[ones] = 1
    v[i:L - j] = body
    return v


class RestrictionGate(MoveSet):
    """Wrap any MoveSet so proposals touching the forced [0,i) / [L-j,L) zone
    are rejected (returns None -> caller re-proposes). i=j=0 is a pure
    passthrough (the unrestricted baseline uses the same code path)."""

    def __init__(self, inner: MoveSet, L: int, i: int, j: int):
        self.inner = inner
        self.name = f"restricted[{inner.name}]"
        self.L = L
        self.i = i
        self.j = j

    def _in_zone(self, mv) -> bool:
        if mv is None:
            return False
        lo, hi = self.i, self.L - self.j
        return bool(np.any(mv.J < lo) or np.any(mv.J >= hi))

    def propose(self, v, rng):
        mv = self.inner.propose(v, rng)
        return None if self._in_zone(mv) else mv

    def propose_local(self, v, rng):
        mv = self.inner.propose_local(v, rng)
        return None if self._in_zone(mv) else mv

    def propose_kick(self, v, rng):
        mv = self.inner.propose_kick(v, rng)
        return None if self._in_zone(mv) else mv


class RestrictedJointWalker(JointWalker):
    def __init__(self, L, i, j, moves, meta, rng):
        self.i, self.j = i, j
        super().__init__(L, moves, meta, rng)

    def _reset(self):
        self.u = restricted_random_seq(self.L, self.i, self.j, self.rng)
        self.v = restricted_random_seq(self.L, self.i, self.j, self.rng)
        from lp_rle.paf import paf_naive
        self.pu = paf_naive(self.u)
        self.pv = paf_naive(self.v)
        self.e = self.pu + self.pv + 2
        self.f = int(np.dot(self.e, self.e))
        self.Eu = int(np.dot(self.pu, self.pu))
        self.Ev = int(np.dot(self.pv, self.pv))


class RestrictedBasinHopper(BasinHopper):
    def __init__(self, L, i, j, moves, meta, rng, n_kick=3, descent_patience=None):
        self.i, self.j = i, j
        super().__init__(L, moves, meta, rng, n_kick=n_kick, descent_patience=descent_patience)

    def _reset(self):
        self.u = restricted_random_seq(self.L, self.i, self.j, self.rng)
        self.v = restricted_random_seq(self.L, self.i, self.j, self.rng)
        from lp_rle.paf import paf_naive
        self.pu = paf_naive(self.u)
        self.pv = paf_naive(self.v)
        self.e = self.pu + self.pv + 2
        self.f = int(np.dot(self.e, self.e))
        self.Eu = int(np.dot(self.pu, self.pu))
        self.Ev = int(np.dot(self.pv, self.pv))


REP_MOVESETS = {"binary": BinaryMoveSet, "rle": RLEMoveSet}


def run_trial(algo: str, rep: str, ell: int, i: int, j: int, seed: int,
              max_steps: int = 200_000, time_budget: float | None = None) -> dict:
    """One (algo, rep, ell, i, j, seed) trial. algo in {sa, tabu, basinhop}.

    i=j=0 is the unrestricted baseline (RestrictionGate becomes a passthrough,
    restricted_random_seq degenerates to a plain random draw).
    """
    rng = np.random.default_rng(seed)
    inner = REP_MOVESETS[rep]()
    moves = RestrictionGate(inner, ell, i, j)
    if algo == "basinhop":
        walker = RestrictedBasinHopper(ell, i, j, moves, Meta(kind="sa"), rng)
    else:
        walker = RestrictedJointWalker(ell, i, j, moves, Meta(kind=algo), rng)
    st = walker.run(max_steps, time_budget=time_budget)
    return {"found": st.found, "seconds": st.time_to_first, "steps": st.steps,
            "best_f": st.best_f}


if __name__ == "__main__":
    for algo in ("sa", "tabu", "basinhop"):
        for rep in ("binary", "rle"):
            r = run_trial(algo, rep, 21, 1, 3, seed=0, max_steps=50_000)
            print(f"{algo:>8} {rep:>6}: found={r['found']} steps={r['steps']} "
                  f"seconds={r['seconds']}")
