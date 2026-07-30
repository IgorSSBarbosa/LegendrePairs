"""walk.py — stochastic search: residual state, move sets, metaheuristics.

Two formulations (spec §4):
  * JointWalker      -- minimizes f = sum_s e[s]^2 with e = PAF_u+PAF_v+2,
                        f = 0 <=> (u, v) is a Legendre pair. Exact integers.
  * TwoStageSearch   -- stage 1 walks single sequences minimizing psd_excess,
                        harvesting a pool of PSD-feasible sequences; stage 2
                        hash-matches the pool exactly as in the exhaustive
                        driver.

Move sets are swappable and instrumented. The RLE set logs micro (2-flip-
equivalent transfers) vs macro (swap_runs, long-range/band transfer, relocate-
singleton split/merge) proposal/acceptance separately: if macro moves are never
accepted, that is the experimental answer (spec §4.3). The BinaryMoveSet is the
plain pm1 2-flip baseline (baseline.py wraps it).

Objective deltas go through the exact integer kernels in delta.py; nothing in
the joint inner loop touches floating point (spec §4.2).
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .conventions import half_len, P_of, M_of, check_odd
from .paf import paf_naive, psd_from_paf, psd_excess
from .runs import seq_to_runs, canonical, runs_to_seq
from .delta import delta_paf_general, delta_f, assert_delta_valid
from .filter import paf_key
from .match import find_pairs, inequivalent_classes


# --------------------------------------------------------------------------- #
# random row-sum-+1 sequence
# --------------------------------------------------------------------------- #
def random_seq(L: int, rng: np.random.Generator) -> np.ndarray:
    """A uniform random pm1 sequence with exactly P ones (row sum +1)."""
    check_odd(L)
    v = -np.ones(L, dtype=np.int8)
    ones = rng.choice(L, size=P_of(L), replace=False)
    v[ones] = 1
    return v


# --------------------------------------------------------------------------- #
# run helpers (linear run decomposition; anchoring is irrelevant to PAF)
# --------------------------------------------------------------------------- #
def _linear_runs(v: np.ndarray) -> List[Tuple[int, int, int]]:
    """Maximal constant segments of v[0..L-1] as (start, length, sign)."""
    L = v.shape[0]
    runs = []
    i = 0
    while i < L:
        j = i
        while j < L and v[j] == v[i]:
            j += 1
        runs.append((i, j - i, int(v[i])))
        i = j
    return runs


def _rebuild(runs: List[Tuple[int, int, int]], L: int) -> np.ndarray:
    v = np.empty(L, dtype=np.int8)
    pos = 0
    for _, length, sign in runs:
        v[pos : pos + length] = sign
        pos += length
    return v


# --------------------------------------------------------------------------- #
# move sets
# --------------------------------------------------------------------------- #
@dataclass
class Move:
    v_new: np.ndarray
    J: np.ndarray
    delta: np.ndarray
    macro: bool
    descriptor: tuple


class MoveSet:
    """Abstract move set; propose a P-preserving neighbor of v."""

    name = "abstract"

    def propose(self, v: np.ndarray, rng: np.random.Generator) -> Optional[Move]:
        raise NotImplementedError


class BinaryMoveSet(MoveSet):
    """Baseline: uniform pm1 2-flip (swap a random +1 with a random -1)."""

    name = "binary"

    def propose(self, v, rng):
        pos = np.where(v == 1)[0]
        neg = np.where(v == -1)[0]
        a = int(pos[rng.integers(pos.shape[0])])
        b = int(neg[rng.integers(neg.shape[0])])
        v_new = v.copy()
        v_new[a] = -1
        v_new[b] = 1
        J = np.array([a, b], dtype=np.int64)
        delta = np.array([-2, 2], dtype=np.int64)
        return Move(v_new, J, delta, macro=False, descriptor=("flip", a, b))


class RLEMoveSet(MoveSet):
    """Run-length move set: micro transfer + macro swap/band/split-merge.

    Weights control the macro vs micro proposal mix.
    """

    name = "rle"

    def __init__(self, w_micro=0.6, w_swap=0.2, w_band=0.15, w_splitmerge=0.05):
        tot = w_micro + w_swap + w_band + w_splitmerge
        self.weights = np.array([w_micro, w_swap, w_band, w_splitmerge]) / tot

    def propose(self, v, rng):
        kind = rng.choice(4, p=self.weights)
        if kind == 0:
            return self._micro(v, rng)
        if kind == 1:
            return self._swap(v, rng)
        if kind == 2:
            return self._band(v, rng)
        return self._split_merge(v, rng)

    @staticmethod
    def _diff_move(v, v_new, macro, descriptor):
        J = np.where(v != v_new)[0].astype(np.int64)
        if J.shape[0] == 0:
            return None
        delta = (v_new[J].astype(np.int64) - v[J].astype(np.int64))
        return Move(v_new, J, delta, macro, descriptor)

    def _micro(self, v, rng):
        # boundary-restricted 2-flip: a +1 at a run edge, a -1 at a run edge.
        L = v.shape[0]
        edge_plus = [i for i in range(L) if v[i] == 1 and (v[(i - 1) % L] == -1 or v[(i + 1) % L] == -1)]
        edge_minus = [i for i in range(L) if v[i] == -1 and (v[(i - 1) % L] == 1 or v[(i + 1) % L] == 1)]
        if not edge_plus or not edge_minus:
            return None
        a = edge_plus[rng.integers(len(edge_plus))]
        b = edge_minus[rng.integers(len(edge_minus))]
        v_new = v.copy()
        v_new[a] = -1
        v_new[b] = 1
        J = np.array([a, b], dtype=np.int64)
        delta = np.array([-2, 2], dtype=np.int64)
        return Move(v_new, J, delta, macro=False, descriptor=("micro", int(a), int(b)))

    def _swap(self, v, rng):
        # swap the LENGTHS of two same-sign runs (preserves the length multiset).
        runs = _linear_runs(v)
        for _ in range(8):
            sign = 1 if rng.random() < 0.5 else -1
            same = [i for i, r in enumerate(runs) if r[2] == sign]
            if len(same) < 2:
                continue
            i, j = rng.choice(same, size=2, replace=False)
            if runs[i][1] == runs[j][1]:
                continue
            new_runs = list(runs)
            ri, rj = new_runs[i], new_runs[j]
            new_runs[i] = (ri[0], rj[1], ri[2])
            new_runs[j] = (rj[0], ri[1], rj[2])
            v_new = _rebuild(new_runs, v.shape[0])
            return self._diff_move(v, v_new, True, ("swap", int(i), int(j)))
        return None

    def _band(self, v, rng):
        # long-range transfer: move one unit between two same-sign runs.
        runs = _linear_runs(v)
        for _ in range(8):
            sign = 1 if rng.random() < 0.5 else -1
            same = [i for i, r in enumerate(runs) if r[2] == sign and r[1] > 1]
            targets = [i for i, r in enumerate(runs) if r[2] == sign]
            if not same or len(targets) < 2:
                continue
            i = int(rng.choice(same))
            j = int(rng.choice([t for t in targets if t != i]))
            new_runs = list(runs)
            new_runs[i] = (new_runs[i][0], new_runs[i][1] - 1, sign)
            new_runs[j] = (new_runs[j][0], new_runs[j][1] + 1, sign)
            v_new = _rebuild(new_runs, v.shape[0])
            return self._diff_move(v, v_new, True, ("band", i, j))
        return None

    def _split_merge(self, v, rng):
        # relocate a singleton: merge an isolated cell, split a long run.
        L = v.shape[0]
        singles = [i for i in range(L)
                   if v[(i - 1) % L] == -v[i] and v[(i + 1) % L] == -v[i]]
        if not singles:
            return None
        p = int(rng.choice(singles))
        target_sign = int(v[p])  # this cell will flip to the opposite (merge)
        # need a long run of the OPPOSITE sign to split (interior cell)
        runs = _linear_runs(v)
        longs = [r for r in runs if r[2] == -target_sign and r[1] >= 3]
        if not longs:
            return None
        r0 = longs[rng.integers(len(longs))]
        q = r0[0] + 1 + int(rng.integers(r0[1] - 2))  # interior cell
        if q == p:
            return None
        v_new = v.copy()
        v_new[p] = -v_new[p]  # merge
        v_new[q] = -v_new[q]  # split
        return self._diff_move(v, v_new, True, ("splitmerge", p, q))


# --------------------------------------------------------------------------- #
# metaheuristic acceptance
# --------------------------------------------------------------------------- #
@dataclass
class Meta:
    kind: str = "sa"           # 'sa' | 'tabu' | 'restart'
    T0: float = 4.0
    cooling: float = 0.9995
    tabu_tenure: int = 20
    plateau_restart: int = 2000  # steps without improvement before restart


# --------------------------------------------------------------------------- #
# instrumentation
# --------------------------------------------------------------------------- #
@dataclass
class Stats:
    steps: int = 0
    accepts: int = 0
    micro_prop: int = 0
    micro_acc: int = 0
    macro_prop: int = 0
    macro_acc: int = 0
    df_micro_sum: float = 0.0
    df_macro_sum: float = 0.0
    best_f: float = math.inf
    time_to_first: Optional[float] = None
    found: bool = False

    def as_dict(self) -> Dict:
        d = dict(self.__dict__)
        d["micro_acc_rate"] = self.micro_acc / self.micro_prop if self.micro_prop else 0.0
        d["macro_acc_rate"] = self.macro_acc / self.macro_prop if self.macro_prop else 0.0
        d["mean_df_micro"] = self.df_micro_sum / self.micro_prop if self.micro_prop else 0.0
        d["mean_df_macro"] = self.df_macro_sum / self.macro_prop if self.macro_prop else 0.0
        return d


# --------------------------------------------------------------------------- #
# joint walker
# --------------------------------------------------------------------------- #
class JointWalker:
    """Joint formulation: search over (u, v), objective f = sum_s e[s]^2."""

    def __init__(self, L: int, moves: MoveSet, meta: Meta, rng: np.random.Generator):
        check_odd(L)
        self.L = L
        self.h = half_len(L)
        self.moves = moves
        self.meta = meta
        self.rng = rng
        self._reset()

    def _reset(self):
        self.u = random_seq(self.L, self.rng)
        self.v = random_seq(self.L, self.rng)
        self.pu = paf_naive(self.u)
        self.pv = paf_naive(self.v)
        self.e = self.pu + self.pv + 2
        self.f = int(np.dot(self.e, self.e))

    def run(self, max_steps: int, time_budget: Optional[float] = None) -> Stats:
        import time

        st = Stats()
        t0 = time.perf_counter()
        T = self.meta.T0
        tabu = deque(maxlen=self.meta.tabu_tenure)
        last_improve = 0
        for step in range(max_steps):
            st.steps += 1
            # choose which sequence to perturb
            side = "u" if self.rng.random() < 0.5 else "v"
            v_cur = self.u if side == "u" else self.v
            paf_cur = self.pu if side == "u" else self.pv
            mv = self.moves.propose(v_cur, self.rng)
            if mv is None:
                continue
            dpaf = delta_paf_general(v_cur, mv.J, mv.delta, self.L)
            df = int(delta_f(self.e, dpaf))
            desc = (side,) + mv.descriptor

            if mv.macro:
                st.macro_prop += 1
                st.df_macro_sum += df
            else:
                st.micro_prop += 1
                st.df_micro_sum += df

            if self.meta.kind == "tabu" and desc in tabu and (self.f + df) >= st.best_f:
                continue
            accept = self._accept(df, T)
            if accept:
                # commit
                if side == "u":
                    self.u = mv.v_new
                    self.pu = paf_cur + dpaf
                else:
                    self.v = mv.v_new
                    self.pv = paf_cur + dpaf
                self.e = self.e + dpaf
                self.f += df
                st.accepts += 1
                if mv.macro:
                    st.macro_acc += 1
                else:
                    st.micro_acc += 1
                if self.meta.kind == "tabu":
                    tabu.append(desc)
                if self.f < st.best_f:
                    st.best_f = self.f
                    last_improve = step

            if self.f == 0 and not st.found:
                st.found = True
                st.time_to_first = time.perf_counter() - t0
                st.best_f = 0
                break

            # cooling / restart
            if self.meta.kind == "sa" or self.meta.kind == "tabu":
                T *= self.meta.cooling
            if self.meta.kind == "restart" and step - last_improve > self.meta.plateau_restart:
                self._reset()
                last_improve = step
            if time_budget is not None and (time.perf_counter() - t0) > time_budget:
                break
        st.best_f = min(st.best_f, self.f)
        return st

    def _accept(self, df: int, T: float) -> bool:
        if df <= 0:
            return True
        if T <= 1e-12:
            return False
        return self.rng.random() < math.exp(-df / T)

    def solution(self):
        return (self.u.copy(), self.v.copy()) if self.f == 0 else None


# --------------------------------------------------------------------------- #
# two-stage search
# --------------------------------------------------------------------------- #
class TwoStageSearch:
    """Stage 1: harvest PSD-feasible single sequences; Stage 2: hash-match."""

    def __init__(self, L: int, moves: MoveSet, meta: Meta, rng: np.random.Generator, tol: float = 1e-6):
        check_odd(L)
        self.L = L
        self.h = half_len(L)
        self.moves = moves
        self.meta = meta
        self.rng = rng
        self.tol = tol
        self.pool: Dict[tuple, List[np.ndarray]] = defaultdict(list)
        self._pool_canon = set()

    def _harvest(self, v: np.ndarray, pv: np.ndarray):
        key = canonical(seq_to_runs(v)).tobytes()
        if key in self._pool_canon:
            return
        self._pool_canon.add(key)
        self.pool[paf_key(pv)].append(v.copy())

    def stage1(self, max_steps: int, time_budget: Optional[float] = None) -> Stats:
        import time

        st = Stats()
        t0 = time.perf_counter()
        T = self.meta.T0
        v = random_seq(self.L, self.rng)
        pv = paf_naive(v)
        ex = psd_excess(psd_from_paf(pv, self.L), self.L)
        last_improve = 0
        for step in range(max_steps):
            st.steps += 1
            mv = self.moves.propose(v, self.rng)
            if mv is None:
                continue
            dpaf = delta_paf_general(v, mv.J, mv.delta, self.L)
            pv_new = pv + dpaf
            ex_new = psd_excess(psd_from_paf(pv_new, self.L), self.L)
            df = ex_new - ex
            if mv.macro:
                st.macro_prop += 1
                st.df_macro_sum += df
            else:
                st.micro_prop += 1
                st.df_micro_sum += df
            if self._accept(df, T):
                v = mv.v_new
                pv = pv_new
                ex = ex_new
                st.accepts += 1
                if mv.macro:
                    st.macro_acc += 1
                else:
                    st.micro_acc += 1
                if ex < st.best_f:
                    st.best_f = ex
                    last_improve = step
            if ex <= self.tol:
                self._harvest(v, pv)
                if st.time_to_first is None:
                    st.time_to_first = time.perf_counter() - t0
                    st.found = True
            T *= self.meta.cooling
            if self.meta.kind == "restart" and step - last_improve > self.meta.plateau_restart:
                v = random_seq(self.L, self.rng)
                pv = paf_naive(v)
                ex = psd_excess(psd_from_paf(pv, self.L), self.L)
                last_improve = step
            if time_budget is not None and (time.perf_counter() - t0) > time_budget:
                break
        return st

    def stage2(self):
        """Hash-match the harvested pool into inequivalent Legendre pairs."""
        pairs = find_pairs(self.pool)
        return inequivalent_classes(pairs)

    def _accept(self, df: float, T: float) -> bool:
        if df <= 0:
            return True
        if T <= 1e-12:
            return False
        return self.rng.random() < math.exp(-df / T)
