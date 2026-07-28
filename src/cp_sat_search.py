"""Constraint-programming search for a Legendre pair via OR-Tools CP-SAT.

Where the local-search modules (``local_search``, ``anneal_search``) wander the
PAF landscape and can stall at a plateau they cannot climb out of (empirically
E=32 for ell=41, regardless of restart budget), this is a *complete* solver: it
either returns a Legendre pair or proves none exists in the symmetry-reduced
space.  The trade is the usual one -- no exponential is defeated, but a real SAT
engine crosses the coordinated multi-bit barriers that random 2-swaps miss.

The encoding (derived from Perera-Kotsireas 2025 and the group paper of
Bulutoglu-Baczkowski-Yauney)
------------------------------------------------------------------------------
Booleans ``bx[i], by[i]`` with the sign map ``x_i = 1 - 2*bx[i]`` (so bit 1 means
-1).  For a pair of +-1 values ``x_i x_j = 1 - 2*(bx[i] XOR bx[j])``.  Introduce
``zx[i,s] = bx[i] XOR bx[(i+s) mod ell]`` ("the two bits at distance s differ").
Then

    PAF_x(s) = sum_i x_i x_{i+s} = ell - 2 * sum_i zx[i,s],

and the Legendre condition ``PAF_x(s) + PAF_y(s) = -2`` collapses to a single
*linear* equality per shift:

    sum_i zx[i,s] + sum_i zy[i,s] = ell + 1,   s = 1, ..., (ell-1)/2.        (*)

Normalization ``sum(x) = sum(y) = 1`` becomes a cardinality constraint: exactly
(ell-1)/2 of the bits in each sequence are 1 (i.e. that many -1 entries).

Symmetry breaking (orbit reduction under the equivalence group)
---------------------------------------------------------------
The group sends a pair to an equivalent pair by: independent cyclic shifts of x
and of y, a shared decimation ``i -> i*k^{-1}`` (k a unit mod ell), and the switch
(x,y)->(y,x).  We break the cheap, sound part:

* ``bx[0] = by[0] = 0`` -- rotate each sequence so position 0 is +1 (there is
  always a +1 since #(+1) = (ell+1)/2).  Shifts of x and y are independent, so
  fixing both is valid.  (Full rotational canonicalization is a necklace
  constraint; skipped -- it costs clauses and this partial break already helps.)
* ``x <=_lex y`` -- break the switch.

Decimation (a phi(ell) factor) is left unbroken; encoding it statically is
awkward and the payoff is small.

Modes
-----
* ``solve``     -- build (*) as hard constraints and let CP-SAT decide.  For
  ell around 41 the model is tiny (~2*ell bits, ell*(ell-1) reified XORs, (ell-1)/2
  equalities); worth trying directly.
* ``solve_lns`` -- large-neighbourhood search: hold an incumbent, unfix a window
  of positions, and let CP-SAT *optimally repair* that window (minimize total PAF
  violation).  The exact repair performs the coordinated move local search cannot,
  which is the whole point when the plain solve gets slow at larger ell.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from legendre import _fmt, is_legendre_pair  # noqa: E402


# --------------------------------------------------------------------------- #
# model construction
# --------------------------------------------------------------------------- #
def _xor(model, z, a, b) -> None:
    """Constrain boolean ``z == a XOR b`` with the standard 4 linear clauses."""
    model.Add(z <= a + b)
    model.Add(z >= a - b)
    model.Add(z >= b - a)
    model.Add(z <= 2 - a - b)


def _decode(solver, bits) -> list[int]:
    """Turn solved 0/1 bits into a +-1 sequence (bit 1 -> -1)."""
    return [1 - 2 * int(solver.Value(v)) for v in bits]


def _add_lex_le(model, A, B) -> None:
    """Constrain the bit vector ``A`` to be lexicographically <= ``B``.

    Standard reified chain: ``pre[i]`` is true iff A and B agree on all bits
    before i; where they first differ we require A_i <= B_i.
    """
    n = len(A)
    pre = model.NewBoolVar("lex_pre_0")
    model.Add(pre == 1)
    for i in range(n):
        # if all-equal so far, then A[i] <= B[i]
        model.Add(A[i] <= B[i]).OnlyEnforceIf(pre)
        nxt = model.NewBoolVar(f"lex_pre_{i + 1}")
        # nxt <=> pre AND (A[i] == B[i]); encode A[i]==B[i] via eq bool
        eq = model.NewBoolVar(f"lex_eq_{i}")
        model.Add(A[i] == B[i]).OnlyEnforceIf(eq)
        model.Add(A[i] != B[i]).OnlyEnforceIf(eq.Not())
        model.AddBoolAnd([pre, eq]).OnlyEnforceIf(nxt)
        model.AddBoolOr([pre.Not(), eq.Not()]).OnlyEnforceIf(nxt.Not())
        pre = nxt


def build_model(ell, *, break_symmetry=True, fixed=None, soft=False):
    """Build the CP-SAT model for a Legendre pair of length ``ell``.

    Parameters
    ----------
    break_symmetry : add the sound orbit-reduction constraints.
    fixed : optional dict ``{('x', i): 0/1, ('y', i): 0/1}`` pinning bits (used by
        LNS to freeze everything outside the repair window).
    soft : if True, replace each hard equality (*) by a deviation variable and
        return the list of deviations for the caller to Minimize (LNS repair).
        If False, (*) is hard and the model is a pure feasibility problem.

    Returns ``(model, bx, by, devs)`` where ``devs`` is the list of per-shift
    deviation IntVars when ``soft`` else ``None``.
    """
    if ell <= 0 or ell % 2 == 0:
        raise ValueError(f"ell must be a positive odd integer, got {ell}")
    half = (ell - 1) // 2
    fixed = fixed or {}

    model = cp_model.CpModel()
    bx = [model.NewBoolVar(f"x{i}") for i in range(ell)]
    by = [model.NewBoolVar(f"y{i}") for i in range(ell)]

    # normalization: exactly (ell-1)/2 minus-ones (bits set to 1) in each sequence
    model.Add(sum(bx) == half)
    model.Add(sum(by) == half)

    devs = [] if soft else None
    for s in range(1, half + 1):
        zx = [model.NewBoolVar(f"zx_{i}_{s}") for i in range(ell)]
        zy = [model.NewBoolVar(f"zy_{i}_{s}") for i in range(ell)]
        for i in range(ell):
            _xor(model, zx[i], bx[i], bx[(i + s) % ell])
            _xor(model, zy[i], by[i], by[(i + s) % ell])
        total = sum(zx) + sum(zy)                       # = ell + 1 for a pair
        if soft:
            t = model.NewIntVar(-2 * ell, 2 * ell, f"t_{s}")
            model.Add(t == total - (ell + 1))
            d = model.NewIntVar(0, 2 * ell, f"dev_{s}")
            model.AddAbsEquality(d, t)
            devs.append(d)
        else:
            model.Add(total == ell + 1)

    if break_symmetry:
        model.Add(bx[0] == 0)                           # x_0 = +1
        model.Add(by[0] == 0)                           # y_0 = +1
        _add_lex_le(model, bx, by)                      # break the switch

    for (which, i), val in fixed.items():
        vars_ = bx if which == "x" else by
        model.Add(vars_[i] == val)

    return model, bx, by, devs


# --------------------------------------------------------------------------- #
# complete solve
# --------------------------------------------------------------------------- #
def solve(ell, *, max_seconds=60.0, workers=8, break_symmetry=True, seed=0,
          log=False):
    """Complete CP-SAT search for a Legendre pair of length ``ell``.

    Returns a dict: ``solved`` (bool), ``A``/``B`` (lists or None), ``status``
    (CP-SAT status name), ``proved_infeasible`` (bool), ``seconds``.
    A True ``status`` of INFEASIBLE means: no Legendre pair exists *in the
    symmetry-reduced space*, which (the breaks being sound) means none exists.
    """
    model, bx, by, _ = build_model(ell, break_symmetry=break_symmetry)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max_seconds)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = int(seed)
    solver.parameters.log_search_progress = bool(log)

    t0 = time.perf_counter()
    status = solver.Solve(model)
    seconds = time.perf_counter() - t0
    name = solver.StatusName(status)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        a, b = _decode(solver, bx), _decode(solver, by)
        return {"solved": True, "A": a, "B": b, "status": name,
                "proved_infeasible": False, "seconds": seconds}
    return {"solved": False, "A": None, "B": None, "status": name,
            "proved_infeasible": status == cp_model.INFEASIBLE,
            "seconds": seconds}


# --------------------------------------------------------------------------- #
# large-neighbourhood search (CP-SAT repair of a window)
# --------------------------------------------------------------------------- #
def _violation(a, b) -> int:
    """Sum_s |PAF_a(s) + PAF_b(s) + 2| over half shifts -- 0 iff a pair."""
    ell = len(a)
    half = (ell - 1) // 2
    tot = 0
    for s in range(1, half + 1):
        pa = sum(a[k] * a[(k + s) % ell] for k in range(ell))
        pb = sum(b[k] * b[(k + s) % ell] for k in range(ell))
        tot += abs(pa + pb + 2)
    return tot


def _residuals(a, b):
    """Per-shift residual r_s = PAF_a(s) + PAF_b(s) + 2 (all zero iff a pair)."""
    ell = len(a)
    half = (ell - 1) // 2
    r = []
    for s in range(1, half + 1):
        pa = sum(a[k] * a[(k + s) % ell] for k in range(ell))
        pb = sum(b[k] * b[(k + s) % ell] for k in range(ell))
        r.append(pa + pb + 2)
    return r


def _defect_scores(a, b):
    """Per-position defect score = best single-flip violation reduction.

    Flipping ``a_i`` shifts PAF_a(s) by ``-2*a_i*(a_{i+s} + a_{i-s})``; summing
    ``|r_s| - |r_s + delta_s|`` over shifts scores how much a repair *localized at
    position i* could help.  High score = position sits in a bad neighbourhood,
    so it belongs in the repair window.
    """
    ell = len(a)
    half = (ell - 1) // 2
    r = _residuals(a, b)
    score = [0.0] * ell
    for i in range(ell):
        ga = gb = 0
        for idx, s in enumerate(range(1, half + 1)):
            da = -2 * a[i] * (a[(i + s) % ell] + a[(i - s) % ell])
            ga += abs(r[idx]) - abs(r[idx] + da)
            db = -2 * b[i] * (b[(i + s) % ell] + b[(i - s) % ell])
            gb += abs(r[idx]) - abs(r[idx] + db)
        score[i] = max(ga, gb)
    return score


def _targeted_window(a, b, window, ell, rng):
    """Window of positions: half the highest-defect ones, half random.

    Concentrating the repair on defective positions gives CP-SAT the coordinated
    multi-swap that matters; the random half keeps the search from getting stuck
    repairing the same neighbourhood every round.
    """
    w = min(window, ell)
    scores = _defect_scores(a, b)
    order = sorted(range(ell), key=lambda i: scores[i], reverse=True)
    n_top = max(1, w // 2)
    top = order[:n_top]
    top_set = set(top)
    remaining = [i for i in range(ell) if i not in top_set]
    n_rand = w - len(top)
    if n_rand > 0 and remaining:
        extra = rng.choice(len(remaining), size=min(n_rand, len(remaining)),
                           replace=False)
        top = top + [remaining[int(j)] for j in extra]
    return list(top)


def _random_start(ell, rng):
    """A random normalized (+sum=1) pair as (a, b) lists of +-1."""
    import numpy as np  # local: numpy is only needed for the LNS driver

    half = (ell - 1) // 2

    def one():
        seq = np.ones(ell, dtype=int)
        seq[rng.choice(ell, size=half, replace=False)] = -1
        return seq.tolist()

    return one(), one()


def _perturb(a, b, rng, kick):
    """Cardinality-preserving kick: swap ``kick`` +1/-1 pairs in each sequence.

    A single window repair can leave the incumbent on a plateau no window can
    strictly improve; kicking to a nearby (equal-count) config moves the search
    into a fresh basin, which is what lets the LNS escape it -- basin-hopping at
    the LNS level rather than the 2-swap level.
    """
    import numpy as np

    a, b = list(a), list(b)
    for seq in (a, b):
        plus = [i for i in range(len(seq)) if seq[i] == 1]
        minus = [i for i in range(len(seq)) if seq[i] == -1]
        for _ in range(kick):
            if not plus or not minus:
                break
            i = plus[int(rng.integers(len(plus)))]
            j = minus[int(rng.integers(len(minus)))]
            seq[i], seq[j] = -1, 1
            plus.remove(i)
            minus.remove(j)
            plus.append(j)
            minus.append(i)
    return a, b


def solve_lns(ell, *, window=16, iters=400, per_solve_seconds=2.0, workers=8,
              seed=0, incumbent=None, targeted=True, stall_patience=12, kick=2,
              log=False):
    """Iterated large-neighbourhood search with exact CP-SAT window repair.

    Hold an incumbent pair; each round freeze all bits outside a window of
    ``window`` positions and ask CP-SAT to set the window to *minimize* total PAF
    violation (an exact repair over that neighbourhood).  Accept if it does not
    worsen; stop on the first violation-0 (a genuine Legendre pair).  After
    ``stall_patience`` rounds with no strict improvement, apply a ``kick`` to jump
    basins (see ``_perturb``) so the search does not freeze on a plateau.

    Returns the same dict shape as ``solve`` plus ``iters_used`` and ``best_v``.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    if incumbent is None:
        a, b = _random_start(ell, rng)
    else:
        a, b = list(incumbent[0]), list(incumbent[1])
    cur_v = _violation(a, b)
    gbest_v, gbest_a, gbest_b = cur_v, list(a), list(b)   # best ever seen
    since_improve = 0

    t0 = time.perf_counter()
    it = -1
    for it in range(iters):
        if gbest_v == 0:
            break
        # choose a window: defect-focused if targeted, else uniform random
        if targeted:
            cols = _targeted_window(a, b, window, ell, rng)
        else:
            cols = rng.choice(ell, size=min(window, ell), replace=False).tolist()
        fixed = {}
        for i in range(ell):
            if i not in cols:
                fixed[("x", i)] = 0 if a[i] == 1 else 1
                fixed[("y", i)] = 0 if b[i] == 1 else 1
        # symmetry breaking off: the incumbent already fixes the frame, and the
        # bx[0]/by[0]/lex constraints could conflict with frozen bits.
        model, bx, by, devs = build_model(ell, break_symmetry=False,
                                           fixed=fixed, soft=True)
        model.Minimize(sum(devs))
        # warm start from the incumbent
        for i in range(ell):
            model.AddHint(bx[i], 0 if a[i] == 1 else 1)
            model.AddHint(by[i], 0 if b[i] == 1 else 1)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(per_solve_seconds)
        solver.parameters.num_search_workers = int(workers)
        solver.parameters.random_seed = int(seed + it)
        solver.parameters.log_search_progress = bool(log)
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            na, nb = _decode(solver, bx), _decode(solver, by)
            nv = _violation(na, nb)
            if nv <= cur_v:                       # accept non-worsening basin
                if nv < cur_v:
                    since_improve = 0
                a, b, cur_v = na, nb, nv
                if nv < gbest_v:
                    gbest_v, gbest_a, gbest_b = nv, list(na), list(nb)

        since_improve += 1
        if since_improve >= stall_patience and gbest_v != 0:
            a, b = _perturb(gbest_a, gbest_b, rng, kick)   # kick from global best
            cur_v = _violation(a, b)
            since_improve = 0

    seconds = time.perf_counter() - t0
    a, b, best_v = gbest_a, gbest_b, gbest_v
    solved = best_v == 0
    return {"solved": solved, "A": a if solved else None,
            "B": b if solved else None,
            "status": "SOLVED" if solved else "STALLED",
            "proved_infeasible": False, "seconds": seconds,
            "iters_used": it + 1, "best_v": best_v}


# --------------------------------------------------------------------------- #
# hybrid: basinhop to a near-solution, then CP-SAT LNS repair from there
# --------------------------------------------------------------------------- #
def solve_hybrid(ell, *, basin_restarts=64, basin_steps=1_000_000,
                 basin_seconds=None, basin_seed=0, window=16, iters=400,
                 per_solve_seconds=2.0, workers=8, seed=0, targeted=True,
                 stall_patience=12, kick=2, log=False, verbose=True):
    """Two-phase search: local basinhop -> CP-SAT LNS warm-started from it.

    Phase 1 runs the local-search basin-hopper, which drops quickly onto a deep
    plateau (empirically E=32 for ell=41) it cannot climb out of with random
    2-swaps.  Phase 2 hands that near-solution to CP-SAT LNS, which repairs a
    window *exactly* -- the coordinated multi-bit move the local search misses.

    Returns the ``solve_lns`` dict shape, plus ``phase`` (which stage solved) and
    ``incumbent_v`` (the violation of the basinhop hand-off).
    """
    from local_search import search as ls_search  # noqa: E402  (sys.path set)

    t0 = time.perf_counter()
    if verbose:
        cap = f"{basin_seconds}s" if basin_seconds else "no cap"
        print(f"[hybrid] phase 1: basinhop ell={ell}  restarts={basin_restarts} "
              f"steps={basin_steps} ({cap})")
    bres = ls_search(ell, strategy="basinhop", restarts=basin_restarts,
                     steps=basin_steps, max_seconds=basin_seconds,
                     seed=basin_seed)
    if bres["solved"]:
        if verbose:
            print(f"  basinhop already solved it in {bres['seconds']:.1f}s")
        return {"solved": True, "A": bres["A"], "B": bres["B"],
                "status": "BASINHOP", "proved_infeasible": False,
                "seconds": time.perf_counter() - t0, "phase": "basinhop",
                "incumbent_v": 0, "iters_used": 0, "best_v": 0}

    if bres.get("best_A") is None:
        if verbose:
            print("  basinhop produced no incumbent; nothing to repair")
        return {"solved": False, "A": None, "B": None, "status": "NO_INCUMBENT",
                "proved_infeasible": False, "seconds": time.perf_counter() - t0,
                "phase": "basinhop", "incumbent_v": None, "iters_used": 0,
                "best_v": None}

    incumbent = (bres["best_A"], bres["best_B"])
    inc_v = _violation(*incumbent)
    if verbose:
        print(f"  basinhop incumbent: best_E={bres['best_E']} violation={inc_v} "
              f"({bres['seconds']:.1f}s, {bres['restarts_used']} restarts)")
        print(f"[hybrid] phase 2: CP-SAT LNS  window={window} iters={iters} "
              f"targeted={targeted}")

    lres = solve_lns(ell, window=window, iters=iters,
                     per_solve_seconds=per_solve_seconds, workers=workers,
                     seed=seed, incumbent=incumbent, targeted=targeted,
                     stall_patience=stall_patience, kick=kick, log=log)
    lres["seconds"] = time.perf_counter() - t0
    lres["phase"] = "lns"
    lres["incumbent_v"] = inc_v
    return lres


# --------------------------------------------------------------------------- #
# persistence + CLI
# --------------------------------------------------------------------------- #
def _save_pair(ell, method, res, params) -> None:
    try:
        from pairs_store import record_pair
    except Exception as exc:                     # pragma: no cover - optional
        print(f"  (note: could not import pairs_store: {exc})")
        return
    rec = record_pair(ell, method, res["A"], res["B"],
                      seconds=round(res["seconds"], 4), params=params)
    rel = os.path.relpath(rec["path"])
    print(f"  {'saved to' if rec['written'] else 'already recorded in'} {rel}")


def main() -> int:
    p = argparse.ArgumentParser(description="Complete/CP-SAT search for a "
                                "Legendre pair (OR-Tools).")
    p.add_argument("ell", type=int, help="odd length of the pair")
    p.add_argument("--lns", action="store_true",
                   help="use large-neighbourhood search (CP-SAT window repair) "
                        "instead of the full complete solve")
    p.add_argument("--hybrid", action="store_true",
                   help="basinhop to a near-solution, then CP-SAT LNS from there")
    p.add_argument("--max-seconds", type=float, default=60.0,
                   help="time budget for the complete solve (default 60)")
    p.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1,
                   help="CP-SAT search workers (default: all cores)")
    p.add_argument("--no-symmetry", action="store_true",
                   help="disable orbit-reduction symmetry breaking")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=16, help="LNS repair window")
    p.add_argument("--iters", type=int, default=400, help="LNS rounds")
    p.add_argument("--per-solve-seconds", type=float, default=2.0,
                   help="time cap per LNS window repair")
    p.add_argument("--random-windows", action="store_true",
                   help="LNS/hybrid: uniform random windows (default: "
                        "defect-focused targeted windows)")
    p.add_argument("--stall-patience", type=int, default=12,
                   help="LNS/hybrid: kick after this many non-improving rounds")
    p.add_argument("--kick", type=int, default=2,
                   help="LNS/hybrid: +1/-1 swaps per sequence on a stall kick")
    p.add_argument("--basin-restarts", type=int, default=64,
                   help="hybrid phase-1 basinhop restarts")
    p.add_argument("--basin-steps", type=int, default=1_000_000,
                   help="hybrid phase-1 basinhop steps per restart")
    p.add_argument("--basin-seconds", type=float, default=None,
                   help="hybrid phase-1 basinhop wall-clock cap (default: none)")
    p.add_argument("--basin-seed", type=int, default=0,
                   help="hybrid phase-1 basinhop seed")
    p.add_argument("--log", action="store_true", help="CP-SAT search log")
    args = p.parse_args()

    if args.ell <= 0 or args.ell % 2 == 0:
        p.error(f"ell must be a positive odd integer, got {args.ell}")

    targeted = not args.random_windows
    if args.hybrid:
        print(f"[CP-SAT hybrid] ell={args.ell}  window={args.window}  "
              f"iters={args.iters}  {args.jobs} workers")
        res = solve_hybrid(args.ell, basin_restarts=args.basin_restarts,
                           basin_steps=args.basin_steps,
                           basin_seconds=args.basin_seconds,
                           basin_seed=args.basin_seed, window=args.window,
                           iters=args.iters,
                           per_solve_seconds=args.per_solve_seconds,
                           workers=args.jobs, seed=args.seed, targeted=targeted,
                           stall_patience=args.stall_patience, kick=args.kick,
                           log=args.log)
        method = "cpsat-hybrid"
        params = (f"basin_restarts={args.basin_restarts}, "
                  f"basin_steps={args.basin_steps}, window={args.window}, "
                  f"iters~{res.get('iters_used', 0)}, targeted={targeted}, "
                  f"seed={args.seed}, workers={args.jobs}")
    elif args.lns:
        print(f"[CP-SAT LNS] ell={args.ell}  window={args.window}  "
              f"iters={args.iters}  {args.jobs} workers  targeted={targeted}")
        res = solve_lns(args.ell, window=args.window, iters=args.iters,
                        per_solve_seconds=args.per_solve_seconds,
                        workers=args.jobs, seed=args.seed, targeted=targeted,
                        stall_patience=args.stall_patience, kick=args.kick,
                        log=args.log)
        method = "cpsat-lns"
        params = (f"window={args.window}, iters~{res['iters_used']}, "
                  f"per_solve={args.per_solve_seconds}, targeted={targeted}, "
                  f"seed={args.seed}, workers={args.jobs}")
    else:
        print(f"[CP-SAT solve] ell={args.ell}  cap={args.max_seconds}s  "
              f"{args.jobs} workers  symmetry={not args.no_symmetry}")
        res = solve(args.ell, max_seconds=args.max_seconds, workers=args.jobs,
                    break_symmetry=not args.no_symmetry, seed=args.seed,
                    log=args.log)
        method = "cpsat"
        params = (f"max_seconds={args.max_seconds}, seed={args.seed}, "
                  f"workers={args.jobs}, symmetry={not args.no_symmetry}")

    if res["solved"]:
        a, b = res["A"], res["B"]
        ok, reason = is_legendre_pair(a, b)
        print(f"SOLVED ell={args.ell}  (verified: {ok}"
              f"{'' if ok else ' -- ' + reason})  [{res['status']}]")
        print(f"  A = {_fmt(a)}   {a}")
        print(f"  B = {_fmt(b)}   {b}")
        print(f"  time = {res['seconds']:.3f}s")
        if ok:
            _save_pair(args.ell, method, res, params)
        return 0

    if res.get("proved_infeasible"):
        print(f"NO PAIR EXISTS for ell={args.ell} (CP-SAT proved infeasible in "
              f"the symmetry-reduced space, {res['seconds']:.3f}s)")
        return 2
    tail = (f"best violation = {res['best_v']}" if (args.lns or args.hybrid)
            else f"status = {res['status']}")
    print(f"NOT SOLVED ell={args.ell}  ({tail}, time = {res['seconds']:.3f}s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
