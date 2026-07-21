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
             T -> 0 special case. This is what reliably reaches E = 0. Pass
             ``t0=None`` (CLI ``--auto-t0``) to auto-calibrate the start
             temperature so a typical uphill move is accepted with prob ~0.8.
* basinhop : iterated local search -- greedy-descend to a local min, then kick
             by a few random swaps and re-descend, escalating the kick when
             stuck (accept-worse at the *basin* level).
* threshold: reroll a fresh config whenever E exceeds c*ell, else greedy-descend.

Incremental objective
----------------------
A 2-swap is a rank-2 change to one PAF, so the exact integer change in the
half-shift PAF vector -- and hence in E -- is computed in O(ell) per move by
``_dpaf`` (no FFT in the inner loop).  ``paf_vector`` (an FFT) is used only to
seed the initial PAF of each restart.  The residual r = PAF_A + PAF_B + 2 is
carried and updated in place; E = <r, r>.  This is behaviour-identical to (and a
constant-factor faster than) recomputing the FFT every step -- the RNG stream and
every acceptance decision are unchanged, so runs stay reproducible.
"""

from __future__ import annotations

import argparse
import os
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


def _dpaf(seq: np.ndarray, p: int, q: int, half: int) -> np.ndarray:
    """Exact change in [PAF(1..half)] from swapping seq[p] (a +1) with seq[q] (-1).

    A 2-swap flips the sign at p and q, so seq' = seq + d with d nonzero only at
    p, q (d_p = -2 seq_p, d_q = -2 seq_q).  Then, for each shift s,
        dPAF(s) = sum_i [ seq_i d_{i+s} + d_i seq_{i+s} + d_i d_{i+s} ] ,
    evaluated over s = 1..half in O(ell).  The quadratic term d_i d_{i+s} is
    nonzero only at the shift |p-q| (and its mirror), handled explicitly."""
    ell = seq.shape[0]
    dp = -2 * int(seq[p])                    # = -2  (seq[p] = +1)
    dq = -2 * int(seq[q])                    # = +2  (seq[q] = -1)
    x = seq.astype(np.int64)
    s = np.arange(1, half + 1)
    A = dp * x[(p - s) % ell] + dq * x[(q - s) % ell]
    B = dp * x[(p + s) % ell] + dq * x[(q + s) % ell]
    d = A + B
    for shift in ((q - p) % ell, (p - q) % ell):   # quadratic (self) term
        if 1 <= shift <= half:
            d[shift - 1] += dp * dq
    return d


def _pick_swap(seq, rng):
    """A (+1 index, -1 index) pair to swap in ``seq``."""
    plus = np.flatnonzero(seq == 1)
    minus = np.flatnonzero(seq == -1)
    return int(plus[rng.integers(plus.size)]), int(minus[rng.integers(minus.size)])


def _auto_t0(a, b, r, half, rng, p0: float = 0.8, samples: int = 200) -> float:
    """Start temperature so a typical uphill move is accepted with prob ~ p0.

    Samples random candidate swaps (without committing), averages the positive
    dE, and inverts Metropolis: T0 = mean(dE>0) / -ln(p0)."""
    ups = []
    for _ in range(samples):
        seq = a if rng.random() < 0.5 else b
        i, j = _pick_swap(seq, rng)
        d = _dpaf(seq, i, j, half)
        dE = int(2 * np.dot(r, d) + np.dot(d, d))
        if dE > 0:
            ups.append(dE)
    if not ups:
        return 3.0
    return max(1e-6, float(np.mean(ups)) / -np.log(p0))


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
    r = paf_vector(a, half) + paf_vector(b, half) + 2    # residual, carried
    E = int(np.dot(r, r))
    best_E = E

    if t0 is None:                                       # auto-calibrate T0
        t0 = _auto_t0(a, b, r, half, rng)
    cool = (t_end / t0) ** (1.0 / max(steps, 1))
    T = t0
    for step in range(steps):
        if reporter is not None:
            reporter.update(r_idx, step, steps, E, best_E)
        if E == 0:
            return True, a, b, step, 0

        seq = a if rng.random() < 0.5 else b
        i, j = _pick_swap(seq, rng)
        d = _dpaf(seq, i, j, half)
        dE = int(2 * np.dot(r, d) + np.dot(d, d))
        if _accept(dE, T, strategy, rng):
            seq[i], seq[j] = -1, 1
            r += d
            E += dE
            if E < best_E:
                best_E = E
        T *= cool

    return False, a, b, steps, best_E


def _descend(a, b, r, half, rng, patience, budget):
    """Stochastic greedy descent to an (approximate) local minimum.

    Accept only strictly-improving swaps; declare a local min after ``patience``
    consecutive rejections. Mutates a, b, r in place. Returns (E, evals)."""
    E = int(np.dot(r, r))
    fails = 0
    evals = 0
    while E > 0 and fails < patience and evals < budget:
        seq = a if rng.random() < 0.5 else b
        i, j = _pick_swap(seq, rng)
        d = _dpaf(seq, i, j, half)
        dE = int(2 * np.dot(r, d) + np.dot(d, d))
        evals += 1
        if dE < 0:
            seq[i], seq[j] = -1, 1
            r += d
            E += dE
            fails = 0
        else:
            fails += 1
    return E, evals


def _apply_kick(a, b, r, half, k, rng):
    """Perturb with k unconditional random swaps (a barrier-crossing move of size
    ~k*ell in E). Mutates a, b, r in place; returns the new E."""
    for _ in range(k):
        seq = a if rng.random() < 0.5 else b
        i, j = _pick_swap(seq, rng)
        d = _dpaf(seq, i, j, half)
        seq[i], seq[j] = -1, 1
        r += d
    return int(np.dot(r, r))


def _basin_hop_run(ell, steps, rng, patience=None, kick=3, reporter=None, r_idx=1):
    """Iterated local search / basin-hopping.

    Descend to a local min; then repeatedly {snapshot, kick by a few swaps,
    re-descend}, accepting the new basin if it is no worse. On a run of
    rejections the kick strength grows to escape a wide funnel."""
    half = (ell - 1) // 2
    if patience is None:
        patience = max(4 * ell, 60)
    a = random_sum_one(ell, rng)
    b = random_sum_one(ell, rng)
    r = paf_vector(a, half) + paf_vector(b, half) + 2
    used = 0
    E, e = _descend(a, b, r, half, rng, patience, steps - used)
    used += e
    if E == 0:
        return True, a, b, used, 0
    best_E = E
    stale = 0
    k = kick
    while used < steps:
        if reporter is not None:
            reporter.update(r_idx, used, steps, E, best_E)
        a2, b2, r2 = a.copy(), b.copy(), r.copy()
        E2 = _apply_kick(a2, b2, r2, half, k, rng)
        used += k
        E2, e = _descend(a2, b2, r2, half, rng, patience, steps - used)
        used += e
        if E2 == 0:
            return True, a2, b2, used, 0
        if E2 <= E:                       # accept the new basin
            a, b, r, E = a2, b2, r2, E2
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
    r = paf_vector(a, half) + paf_vector(b, half) + 2
    E = int(np.dot(r, r))
    best_E = E
    used = 0
    fails = 0
    while used < steps:
        if reporter is not None:
            reporter.update(r_idx, used, steps, E, best_E)
        if E > thresh or (fails >= patience and E > 0):
            a = random_sum_one(ell, rng)
            b = random_sum_one(ell, rng)
            r = paf_vector(a, half) + paf_vector(b, half) + 2
            E = int(np.dot(r, r))
            used += 1
            fails = 0
            best_E = min(best_E, E)
            continue
        seq = a if rng.random() < 0.5 else b
        i, j = _pick_swap(seq, rng)
        d = _dpaf(seq, i, j, half)
        dE = int(2 * np.dot(r, d) + np.dot(d, d))
        used += 1
        if dE < 0:
            seq[i], seq[j] = -1, 1
            r += d
            E += dE
            fails = 0
            best_E = min(best_E, E)
            if E == 0:
                return True, a, b, used, 0
        else:
            fails += 1
    return False, a, b, used, best_E


def search(
    ell: int,
    strategy: str = "anneal",
    restarts: int = 200,
    steps: int = 20000,
    t0: float | None = 3.0,
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
    refreshed every ``progress_every`` steps.  Pass ``t0=None`` to auto-calibrate
    the annealing start temperature per restart.

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
    p.add_argument("--auto-t0", action="store_true",
                   help="auto-calibrate the anneal start temperature (ignores --t0)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for reproducibility (default 0)")
    p.add_argument("-P", "--progress", action="store_true",
                   help="show a live stderr bar: attempt, step, current E, best E")
    p.add_argument("--progress-every", type=int, default=1000,
                   help="refresh the progress bar every N steps (default 1000)")
    args = p.parse_args()

    if args.ell <= 0 or args.ell % 2 == 0:
        p.error(f"ell must be a positive odd integer, got {args.ell}")

    t0 = None if args.auto_t0 else args.t0
    res = search(args.ell, args.strategy, args.restarts, args.steps,
                 t0, args.t_end, args.seed,
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
        _save_pair(args, res)
        return 0

    print(f"NOT SOLVED ell={args.ell} after {args.restarts} restarts x "
          f"{args.steps} steps  (best E = {res['best_E']}, "
          f"time = {res['seconds']:.3f}s)")
    print("  try: more --restarts/--steps, or --strategy anneal with tuned --t0/--t-end")
    return 1


def _save_pair(args, res) -> None:
    """Persist a solved pair to results/found_pairs.csv (dedup per method+ell)."""
    try:
        from pairs_store import record_pair
    except Exception as exc:                 # pragma: no cover - store is optional
        print(f"  (note: could not import pairs_store: {exc})")
        return
    t0str = "auto" if args.auto_t0 else args.t0
    params = (f"restarts_used={res['restarts_used']}, steps={args.steps}, "
              f"seed={args.seed}")
    if args.strategy == "anneal":
        params += f", t0={t0str}, t_end={args.t_end}"
    rec = record_pair(args.ell, args.strategy, res["A"], res["B"],
                      seconds=res["seconds"], params=params)
    rel = os.path.relpath(rec["path"])
    print(f"  {'saved to' if rec['written'] else 'already recorded in'} {rel}")


if __name__ == "__main__":
    raise SystemExit(main())
