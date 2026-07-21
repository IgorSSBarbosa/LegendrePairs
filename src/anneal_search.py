"""Simulated-annealing search for Legendre pairs (accept-worse to escape plateaus).

Motivation
----------
Greedy methods top out on this landscape.  The continuous 1/(2*ell) descent and
the (r, n) restart+kick search both stall at a plateau local minimum (E ~ 128 for
ell = 29) and cannot descend to E = 0, because every move they accept is
non-increasing -- there is no way to climb out of a basin.  E has quantized local
minima on long plateaus, so an *accept-worse* rule is exactly what is missing.

This module runs Metropolis simulated annealing on the same state space:

  * State.  A normalized pair (u, v) in {+-1}^ell, each with (ell-1)/2 entries
    equal to -1 (so 1^T u = 1^T v = +1).
  * Move.   A single 2-swap on u OR v (exchange a +1 with a -1), which keeps the
    normalization -- swap-distance 1.
  * Objective.  E(u, v) = sum_{s=1}^{ell-1} (PAF_u(s) + PAF_v(s) + 2)^2, the same
    E as rn_search / continuous_descent.  E >= 0, a multiple of 4, and E == 0 iff
    (u, v) is a Legendre pair (given both stay normalized, which every move does).
  * Acceptance.  Propose a swap, compute dE.  Accept if dE <= 0, else accept with
    probability exp(-dE / T).  T cools geometrically from T0 to Tmin over the run;
    a fresh random restart is taken when the schedule bottoms out without solving.

Why it can work where greedy cannot: the accept-worse step lets the chain climb
the O(ell) barrier separating plateau basins, so it is not trapped at the E-floor.

Efficiency
----------
A 2-swap is a rank-2 change to ONE of the two PAFs, so the exact integer change
dPAF (and hence dE) is computed in O(ell) per step -- no FFT in the inner loop.
``_dpaf_swap`` derives dPAF in closed form; ``_paf_int`` seeds the initial PAF by
FFT once.  A unit test checks the incremental PAF/E against the FFT objective.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legendre_pairs import is_legendre_pair  # noqa: E402


# --------------------------------------------------------------------------- #
# PAF bookkeeping (integer, incremental)
# --------------------------------------------------------------------------- #
def _paf_int(x: np.ndarray) -> np.ndarray:
    """Periodic autocorrelation of a +-1 vector as an int64 array (via one FFT).

    ``PAF(s) = sum_i x_i x_{(i+s) mod ell}``; ``PAF(0) = ell``."""
    f = np.fft.fft(x.astype(float))
    paf = np.fft.ifft(np.abs(f) ** 2).real
    return np.rint(paf).astype(np.int64)


def _energy(pafu: np.ndarray, pafv: np.ndarray) -> int:
    """E = sum_{s=1}^{ell-1} (pafu[s] + pafv[s] + 2)^2 (excludes s = 0)."""
    r = pafu + pafv + 2
    return int((r ** 2).sum() - r[0] ** 2)


def _dpaf_swap(x: np.ndarray, p: int, q: int, s: np.ndarray) -> np.ndarray:
    """Exact change in PAF_x from swapping x[p] (a +1) with x[q] (a -1).

    A 2-swap flips the sign at p and q, so x' = x + d with d nonzero only at p, q
    (d_p = -2 x_p, d_q = -2 x_q).  Then
        dPAF(s) = sum_i [ x_i d_{i+s} + d_i x_{i+s} + d_i d_{i+s} ] .
    Each term has O(1) support (|{p, q}| = 2), evaluated over all shifts s in
    O(ell).  ``s`` is a precomputed ``arange(ell)``.  Returns an int64 array."""
    ell = x.shape[0]
    dp = -2 * int(x[p])                      # = -2  (x[p] = +1)
    dq = -2 * int(x[q])                      # = +2  (x[q] = -1)
    xi = x.astype(np.int64)
    # A[s] = sum_i x_i d_{i+s} : i+s in {p, q}  ->  i = p-s or q-s
    A = dp * xi[(p - s) % ell] + dq * xi[(q - s) % ell]
    # B[s] = sum_i d_i x_{i+s} : i in {p, q}
    B = dp * xi[(p + s) % ell] + dq * xi[(q + s) % ell]
    dpaf = A + B
    # C[s] = sum_i d_i d_{i+s} : both i and i+s in {p, q}
    dpaf[0] += dp * dp + dq * dq
    dpaf[(q - p) % ell] += dp * dq
    dpaf[(p - q) % ell] += dp * dq
    return dpaf


# --------------------------------------------------------------------------- #
# annealing schedule / acceptance
# --------------------------------------------------------------------------- #
def _accept(dE: float, T: float, rng: np.random.Generator) -> bool:
    """Metropolis rule: always accept downhill/flat, else accept with p=exp(-dE/T)."""
    if dE <= 0:
        return True
    if T <= 0.0:
        return False
    return rng.random() < math.exp(-dE / T)


def _random_pair(ell: int, rng: np.random.Generator):
    """A normalized (u, v): each has exactly (ell-1)/2 entries equal to -1."""
    neg = (ell - 1) // 2
    out = []
    for _ in range(2):
        x = np.ones(ell, dtype=np.int8)
        x[rng.permutation(ell)[:neg]] = -1
        out.append(x)
    return out


def _auto_T0(u, v, pafu, pafv, s, rng, p0=0.8, samples=400):
    """Pick T0 so a typical uphill move is accepted with probability ~ p0.

    Samples random candidate swaps (without committing), averages the positive
    dE, and inverts the Metropolis probability: T0 = mean(dE>0) / -ln(p0)."""
    ell = u.shape[0]
    r = pafu + pafv + 2
    ups = []
    for _ in range(samples):
        if rng.random() < 0.5:
            x, pr = u, r
        else:
            x, pr = v, r
        plus = np.nonzero(x == 1)[0]
        minus = np.nonzero(x == -1)[0]
        p = int(plus[rng.integers(plus.size)])
        q = int(minus[rng.integers(minus.size)])
        dpaf = _dpaf_swap(x, p, q, s)
        dE = int(2 * (pr * dpaf).sum() + (dpaf * dpaf).sum())
        if dE > 0:
            ups.append(dE)
    if not ups:
        return 1.0
    return max(1e-9, float(np.mean(ups)) / -math.log(p0))


# --------------------------------------------------------------------------- #
# the search
# --------------------------------------------------------------------------- #
def anneal_once(ell: int, steps: int, rng: np.random.Generator,
                T0: float | None = None, Tmin: float | None = None,
                p0: float = 0.8, record: bool = False):
    """One annealing run from a random start.  Returns a result dict.

    Cools T geometrically T0 -> Tmin across ``steps`` Metropolis moves, tracking
    the best (lowest-E) pair seen.  Stops early if E hits 0."""
    s = np.arange(ell)
    u, v = _random_pair(ell, rng)
    pafu, pafv = _paf_int(u), _paf_int(v)
    r = pafu + pafv + 2
    E = int((r ** 2).sum() - r[0] ** 2)

    if T0 is None:
        T0 = _auto_T0(u, v, pafu, pafv, s, rng, p0=p0)
    if Tmin is None:
        Tmin = max(0.05, T0 * 1e-3)
    ratio = (Tmin / T0) ** (1.0 / max(1, steps - 1))

    best_E = E
    best_u, best_v = u.copy(), v.copy()
    traj = []
    T = T0
    evals = 0
    for _ in range(steps):
        if E == 0:
            break
        # propose a single 2-swap on u or v
        on_u = rng.random() < 0.5
        x = u if on_u else v
        plus = np.nonzero(x == 1)[0]
        minus = np.nonzero(x == -1)[0]
        p = int(plus[rng.integers(plus.size)])
        q = int(minus[rng.integers(minus.size)])
        dpaf = _dpaf_swap(x, p, q, s)
        dE = int(2 * (r * dpaf).sum() + (dpaf * dpaf).sum())
        evals += 1
        if _accept(dE, T, rng):
            x[p], x[q] = x[q], x[p]           # commit the swap in place
            if on_u:
                pafu += dpaf
            else:
                pafv += dpaf
            r += dpaf
            E += dE
            if E < best_E:
                best_E = E
                best_u, best_v = u.copy(), v.copy()
        if record:
            traj.append(E)
        T *= ratio

    solved = best_E == 0
    return {"solved": solved, "u": best_u, "v": best_v, "best_E": best_E,
            "evals": evals, "T0": T0, "Tmin": Tmin, "traj": traj}


def search_anneal(ell: int, restarts: int = 20, steps: int = 200000,
                  seed: int = 0, T0: float | None = None,
                  Tmin: float | None = None, p0: float = 0.8,
                  verbose: bool = False) -> dict:
    """Restarted simulated annealing.  Returns a result dict.

    Keys: ``solved`` (bool), ``u``/``v`` (arrays, best seen), ``best_E``,
    ``evals`` (total moves proposed), ``restarts_used``, ``reason``."""
    if ell <= 0 or ell % 2 == 0:
        raise ValueError(f"ell must be a positive odd integer, got {ell}")
    rng = np.random.default_rng(seed)
    total_evals = 0
    best = {"best_E": None, "u": None, "v": None}
    for k in range(1, restarts + 1):
        res = anneal_once(ell, steps, rng, T0=T0, Tmin=Tmin, p0=p0)
        total_evals += res["evals"]
        if best["best_E"] is None or res["best_E"] < best["best_E"]:
            best = {"best_E": res["best_E"], "u": res["u"], "v": res["v"]}
        if verbose:
            print(f"[anneal] restart {k}/{restarts}: best_E={res['best_E']} "
                  f"(run T0={res['T0']:.2f}) global_best={best['best_E']}")
        if res["solved"]:
            u, v = np.asarray(res["u"], np.int8), np.asarray(res["v"], np.int8)
            assert is_legendre_pair(u, v, ell), "E==0 but is_legendre_pair failed"
            return {"solved": True, "u": u, "v": v, "best_E": 0,
                    "evals": total_evals, "restarts_used": k, "reason": "solved"}
    return {"solved": False, "u": best["u"], "v": best["v"],
            "best_E": best["best_E"], "evals": total_evals,
            "restarts_used": restarts, "reason": "restarts_exhausted"}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(
        description="Simulated-annealing (accept-worse) Legendre-pair search.")
    p.add_argument("ell", type=int)
    p.add_argument("--restarts", type=int, default=20)
    p.add_argument("--steps", type=int, default=200000, help="Metropolis moves/run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t0", type=float, default=None, help="initial temperature "
                   "(default: auto-calibrated for ~80%% uphill acceptance)")
    p.add_argument("--tmin", type=float, default=None, help="final temperature")
    p.add_argument("--p0", type=float, default=0.8, help="target uphill-accept "
                   "prob used to auto-calibrate T0")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    if args.ell <= 0 or args.ell % 2 == 0:
        p.error(f"ell must be a positive odd integer, got {args.ell}")

    res = search_anneal(args.ell, restarts=args.restarts, steps=args.steps,
                        seed=args.seed, T0=args.t0, Tmin=args.tmin, p0=args.p0,
                        verbose=args.verbose)
    if res["solved"]:
        print(f"SOLVED ell={args.ell} in {res['evals']} moves "
              f"[{res['restarts_used']} restart(s)]")
        print(f"  u = {list(map(int, res['u']))}")
        print(f"  v = {list(map(int, res['v']))}")
        return 0
    print(f"NOT FOUND ell={args.ell}: {res['reason']}, best_E={res['best_E']}, "
          f"evals={res['evals']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
