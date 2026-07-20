"""Find Legendre pairs by local search on a PAF objective function.

Idea
----
Brute force is exponential and hopeless past ell ~ 25. Instead, turn the
defining property into an objective to minimise. For sequences A, B of odd
length ell over {-1, +1}, with shifts s = 1, ..., (ell-1)/2:

    E(A, B) = sum_s ( PAF_A(s) + PAF_B(s) + 2 )^2   >= 0,

and E(A, B) = 0  iff  (A, B) is a Legendre pair.

Move
----
Swap one +1 with one -1 inside A or B. This keeps each row sum fixed, so if we
start both sequences at sum = 1 (the proven feasible sum-space) every candidate
stays feasible. Swaps connect all sum=1 sequences, so the search can reach any
of them.

Acceptance strategies
---------------------
* greedy   : accept only if E strictly decreases (the rule you described).
             Fast but stalls in local minima -> relies on many restarts.
* sideways : accept if E does not increase (also walks equal-E plateaus).
* anneal   : Metropolis / simulated annealing -- accept a worse move with
             probability exp(-dE / T), T cooled geometrically. Greedy is the
             T -> 0 special case. This is what reliably reaches E = 0.

PAF vectors are computed with an FFT (periodic autocorrelation), so each move
costs O(ell log ell) and the method scales to much larger ell than brute force.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from legendre import is_legendre_pair


class _Reporter:
    """Live one-line progress on stderr: current attempt, step, and objective.

    Refreshed only every ``every`` steps so it stays off the hot path. Tracks the
    best E seen across restarts so the line shows both the current and best E."""

    def __init__(self, enabled: bool, every: int, restarts: int):
        self.enabled = enabled
        self.every = max(1, every)
        self.restarts = restarts
        self.overall_best = None

    def update(self, r_idx, step, steps, E, run_best):
        if not self.enabled or step % self.every:
            return
        best = run_best if self.overall_best is None else min(self.overall_best, run_best)
        sys.stderr.write(
            f"\rattempt {r_idx:>4}/{self.restarts}  step {step:>7}/{steps}  "
            f"E={E:>7}  best={best:>7}    "
        )
        sys.stderr.flush()

    def note_best(self, best):
        self.overall_best = best if self.overall_best is None else min(self.overall_best, best)

    def close(self):
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()


def paf_vector(seq: np.ndarray, half: int) -> np.ndarray:
    """[PAF(1), ..., PAF(half)] via FFT periodic autocorrelation (integer)."""
    f = np.fft.rfft(seq.astype(float))
    ac = np.fft.irfft(f * np.conj(f), n=seq.size)
    return np.rint(ac[1 : half + 1]).astype(np.int64)


def objective(pa: np.ndarray, pb: np.ndarray) -> int:
    """E = sum_s (PAF_A(s) + PAF_B(s) + 2)^2."""
    d = pa + pb + 2
    return int(np.dot(d, d))


def random_sum_one(ell: int, rng: np.random.Generator) -> np.ndarray:
    """Random +-1 sequence of length ell with sum == 1."""
    seq = np.ones(ell, dtype=np.int64)
    minus = rng.choice(ell, (ell - 1) // 2, replace=False)
    seq[minus] = -1
    return seq


def _accept(dE: int, T: float, strategy: str, rng: np.random.Generator) -> bool:
    if dE < 0:
        return True
    if strategy == "greedy":
        return False
    if strategy == "sideways":
        return dE == 0
    # anneal
    if dE == 0:
        return True
    return rng.random() < np.exp(-dE / max(T, 1e-12))


def _one_run(ell, strategy, steps, t0, t_end, rng, reporter=None, r_idx=1):
    """Single random restart. Returns (solved, A, B, steps_used, best_E)."""
    half = (ell - 1) // 2
    a = random_sum_one(ell, rng)
    b = random_sum_one(ell, rng)
    pa = paf_vector(a, half)
    pb = paf_vector(b, half)
    E = objective(pa, pb)
    best_E = E

    cool = (t_end / t0) ** (1.0 / max(steps, 1))
    T = t0
    for step in range(steps):
        if reporter is not None:
            reporter.update(r_idx, step, steps, E, best_E)
        if E == 0:
            return True, a, b, step, 0

        # pick a sequence and a +1/-1 pair to swap
        seq, pself, pother = (a, pa, pb) if rng.random() < 0.5 else (b, pb, pa)
        plus = np.flatnonzero(seq == 1)
        minus = np.flatnonzero(seq == -1)
        i = plus[rng.integers(plus.size)]
        j = minus[rng.integers(minus.size)]

        seq[i], seq[j] = -1, 1
        pnew = paf_vector(seq, half)
        E_new = objective(pnew, pother)
        dE = E_new - E

        if _accept(dE, T, strategy, rng):
            pself[:] = pnew
            E = E_new
            if E < best_E:
                best_E = E
        else:
            seq[i], seq[j] = 1, -1  # revert
        T *= cool

    return False, a, b, steps, best_E


def _descend(a, b, pa, pb, E, half, rng, patience, budget):
    """Stochastic greedy descent to an (approximate) local minimum.

    Repeatedly try random swaps, accept the improving ones, and declare a local
    minimum after ``patience`` consecutive rejections. Mutates a, b, pa, pb in
    place. Returns (E_at_local_min, swap_evaluations_used)."""
    fails = 0
    evals = 0
    while E > 0 and fails < patience and evals < budget:
        seq, pself, pother = (a, pa, pb) if rng.random() < 0.5 else (b, pb, pa)
        plus = np.flatnonzero(seq == 1)
        minus = np.flatnonzero(seq == -1)
        i = plus[rng.integers(plus.size)]
        j = minus[rng.integers(minus.size)]
        seq[i], seq[j] = -1, 1
        pnew = paf_vector(seq, half)
        E_new = objective(pnew, pother)
        evals += 1
        if E_new < E:
            pself[:] = pnew
            E = E_new
            fails = 0
        else:
            seq[i], seq[j] = 1, -1
            fails += 1
    return E, evals


def _apply_kick(a, b, pa, pb, half, k, rng):
    """Perturb the state with k unconditional random swaps (a barrier-crossing
    move of size ~k*ell in E). Mutates in place; returns the new E."""
    for _ in range(k):
        seq, pself = (a, pa) if rng.random() < 0.5 else (b, pb)
        plus = np.flatnonzero(seq == 1)
        minus = np.flatnonzero(seq == -1)
        i = plus[rng.integers(plus.size)]
        j = minus[rng.integers(minus.size)]
        seq[i], seq[j] = -1, 1
        pself[:] = paf_vector(seq, half)
    return objective(pa, pb)


def _basin_hop_run(ell, steps, rng, patience=None, kick=3, reporter=None, r_idx=1):
    """Iterated local search / basin-hopping.

    Descend to a local min; then repeatedly {snapshot, kick by a few swaps,
    re-descend}, accepting the new basin if it is no worse. On a run of
    rejections the kick strength grows to escape a wide funnel. Reuses structure
    across basins instead of rerolling the whole configuration."""
    half = (ell - 1) // 2
    if patience is None:
        patience = max(4 * ell, 60)
    a = random_sum_one(ell, rng)
    b = random_sum_one(ell, rng)
    pa = paf_vector(a, half)
    pb = paf_vector(b, half)
    E = objective(pa, pb)
    used = 0
    E, e = _descend(a, b, pa, pb, E, half, rng, patience, steps - used)
    used += e
    if E == 0:
        return True, a, b, used, 0
    best_E = E
    stale = 0
    k = kick
    while used < steps:
        if reporter is not None:
            reporter.update(r_idx, used, steps, E, best_E)
        a2, b2, pa2, pb2 = a.copy(), b.copy(), pa.copy(), pb.copy()
        E2 = _apply_kick(a2, b2, pa2, pb2, half, k, rng)
        used += k
        E2, e = _descend(a2, b2, pa2, pb2, E2, half, rng, patience, steps - used)
        used += e
        if E2 == 0:
            return True, a2, b2, used, 0
        if E2 <= E:                       # accept the new basin
            a, b, pa, pb, E = a2, b2, pa2, pb2, E2
            best_E = min(best_E, E)
            stale = 0
            k = kick
        else:
            stale += 1
            if stale % 20 == 0:           # escalate kick to jump a wide funnel
                k = min(k + kick, max(1, ell // 2))
    return False, a, b, used, best_E


def _threshold_run(ell, steps, rng, c=20.0, patience=None, reporter=None, r_idx=1):
    """The magnitude-threshold heuristic: reroll a fresh random config whenever
    E > c*ell, otherwise take greedy descent steps. (Charitable version: if
    descent stalls at a local min with 0 < E <= c*ell we also reroll, since the
    literal rule would loop there forever.)"""
    half = (ell - 1) // 2
    if patience is None:
        patience = max(4 * ell, 60)
    thresh = c * ell
    a = random_sum_one(ell, rng)
    b = random_sum_one(ell, rng)
    pa = paf_vector(a, half)
    pb = paf_vector(b, half)
    E = objective(pa, pb)
    best_E = E
    used = 0
    fails = 0
    while used < steps:
        if reporter is not None:
            reporter.update(r_idx, used, steps, E, best_E)
        if E > thresh or (fails >= patience and E > 0):
            a = random_sum_one(ell, rng)
            b = random_sum_one(ell, rng)
            pa = paf_vector(a, half)
            pb = paf_vector(b, half)
            E = objective(pa, pb)
            used += 1
            fails = 0
            best_E = min(best_E, E)
            continue
        seq, pself, pother = (a, pa, pb) if rng.random() < 0.5 else (b, pb, pa)
        plus = np.flatnonzero(seq == 1)
        minus = np.flatnonzero(seq == -1)
        i = plus[rng.integers(plus.size)]
        j = minus[rng.integers(minus.size)]
        seq[i], seq[j] = -1, 1
        pnew = paf_vector(seq, half)
        E_new = objective(pnew, pother)
        used += 1
        if E_new < E:
            pself[:] = pnew
            E = E_new
            fails = 0
            best_E = min(best_E, E)
            if E == 0:
                return True, a, b, used, 0
        else:
            seq[i], seq[j] = 1, -1
            fails += 1
    return False, a, b, used, best_E


def search(
    ell: int,
    strategy: str = "anneal",
    restarts: int = 200,
    steps: int = 20000,
    t0: float = 3.0,
    t_end: float = 0.05,
    seed: int | None = None,
    max_seconds: float | None = None,
    progress: bool = False,
    progress_every: int = 1000,
    on_restart=None,
):
    """Search for a Legendre pair of length ell by local search.

    Returns a dict with keys: solved, A, B, restarts_used, steps_used,
    best_E, seconds. If ``max_seconds`` is set, stop (unsolved) once that wall
    time is exceeded between restarts. If ``progress`` is set, a one-line
    stderr bar shows the current attempt, step, objective E and best E,
    refreshed every ``progress_every`` steps.

    ``on_restart``, if given, is called ``on_restart(r, overall_best)`` after
    each completed restart -- used by ``parallel_search`` to report progress
    from a worker process back to the parent.
    """
    if ell <= 0 or ell % 2 == 0:
        raise ValueError(f"ell must be a positive odd integer, got {ell}")

    rng = np.random.default_rng(seed)
    reporter = _Reporter(progress, progress_every, restarts)
    t_start = time.perf_counter()
    overall_best = None
    for r in range(1, restarts + 1):
        if strategy == "basinhop":
            solved, a, b, used, best_E = _basin_hop_run(ell, steps, rng,
                                                        reporter=reporter, r_idx=r)
        elif strategy == "threshold":
            solved, a, b, used, best_E = _threshold_run(ell, steps, rng,
                                                        reporter=reporter, r_idx=r)
        else:
            solved, a, b, used, best_E = _one_run(ell, strategy, steps, t0, t_end,
                                                  rng, reporter=reporter, r_idx=r)
        if overall_best is None or best_E < overall_best:
            overall_best = best_E
        reporter.note_best(best_E)
        if on_restart is not None:
            on_restart(r, overall_best)
        if solved:
            reporter.close()
            return {
                "solved": True,
                "A": a.tolist(),
                "B": b.tolist(),
                "restarts_used": r,
                "steps_used": used,
                "best_E": 0,
                "seconds": time.perf_counter() - t_start,
            }
        if max_seconds is not None and time.perf_counter() - t_start >= max_seconds:
            reporter.close()
            return {
                "solved": False,
                "A": None,
                "B": None,
                "restarts_used": r,
                "steps_used": steps,
                "best_E": overall_best,
                "seconds": time.perf_counter() - t_start,
            }
    reporter.close()
    return {
        "solved": False,
        "A": None,
        "B": None,
        "restarts_used": restarts,
        "steps_used": steps,
        "best_E": overall_best,
        "seconds": time.perf_counter() - t_start,
    }


def _fmt(seq):
    return "".join("+" if x == 1 else "-" for x in seq)


def main() -> int:
    p = argparse.ArgumentParser(description="Find a Legendre pair by local search "
                                "on the PAF objective E = sum (PAF_A+PAF_B+2)^2.")
    p.add_argument("ell", type=int, help="odd length of the pair")
    p.add_argument("-s", "--strategy",
                   choices=["greedy", "sideways", "anneal", "basinhop", "threshold"],
                   default="anneal", help="acceptance rule (default: anneal)")
    p.add_argument("-r", "--restarts", type=int, default=200)
    p.add_argument("-n", "--steps", type=int, default=20000, help="steps per restart")
    p.add_argument("--t0", type=float, default=3.0, help="initial temperature (anneal)")
    p.add_argument("--t-end", type=float, default=0.05, help="final temperature (anneal)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for reproducibility (default 0)")
    p.add_argument("-P", "--progress", action="store_true",
                   help="show a live stderr bar: attempt, step, current E, best E")
    p.add_argument("--progress-every", type=int, default=1000,
                   help="refresh the progress bar every N steps (default 1000)")
    args = p.parse_args()

    if args.ell <= 0 or args.ell % 2 == 0:
        p.error(f"ell must be a positive odd integer, got {args.ell}")

    res = search(args.ell, args.strategy, args.restarts, args.steps,
                 args.t0, args.t_end, args.seed,
                 progress=args.progress, progress_every=args.progress_every)

    if res["solved"]:
        a, b = res["A"], res["B"]
        ok, reason = is_legendre_pair(a, b)
        print(f"SOLVED ell={args.ell}  (verified: {ok}{'' if ok else ' -- ' + reason})")
        print(f"  A = {_fmt(a)}   {a}")
        print(f"  B = {_fmt(b)}   {b}")
        print(f"  restarts used = {res['restarts_used']}, "
              f"steps in final run = {res['steps_used']}, "
              f"time = {res['seconds']:.3f}s")
        return 0

    print(f"NOT SOLVED ell={args.ell} after {args.restarts} restarts x "
          f"{args.steps} steps  (best E = {res['best_E']}, "
          f"time = {res['seconds']:.3f}s)")
    print("  try: more --restarts/--steps, or --strategy anneal with tuned --t0/--t-end")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
