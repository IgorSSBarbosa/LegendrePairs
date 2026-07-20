"""Test whether CONTINUOUS gradient descent can find Legendre pairs.

Idea (proposed experiment)
--------------------------
Relax the +-1 constraint: let A, B be real vectors in R^ell. The PAF is a smooth
quadratic form, so

    E(A, B) = sum_{s=1}^{(ell-1)/2} ( PAF_A(s) + PAF_B(s) + 2 )^2

is a smooth degree-4 function of (A, B). Start at a random real point and take
gradient-descent steps

    x  <-  x - eta * grad E(x),      eta = 1/(2*ell)   (the proposed step size),

optionally normalising the step to a fixed length. After each step we ROUND to
signs, sign(x) in {-1,+1}, and check whether the rounded pair is a Legendre pair
(integer objective == 0).

Analytic expectation
--------------------
The zero set E=0 is reachable only at very special points; the relaxed objective
has bad critical points (e.g. x=0, where every PAF=0, r(s)=2, so E=4*half>0 and
grad=0 because every gradient term carries a factor of x). Continuous descent
tends to slide toward such non-binary critical points, so ROUNDING rarely lands
on a true pair. This script measures how often it does.

Gradient
--------
    d PAF_A(s) / d a_m = a_{(m+s) mod ell} + a_{(m-s) mod ell},
    d E / d a_m        = sum_s 2 r(s) ( a_{(m+s)} + a_{(m-s)} ),   r(s)=PAF_A(s)+PAF_B(s)+2,
and symmetrically for b (PAF_B is independent of a).

Usage
-----
    python3 continuous_descent.py                       # ell=21, default settings
    python3 continuous_descent.py --ell 21 --restarts 50 --steps 4000
    python3 continuous_descent.py --normalize           # fixed step length eta
    python3 continuous_descent.py --project sphere       # keep ||x||=sqrt(ell)
    python3 continuous_descent.py --seed 7
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(_RESULTS, exist_ok=True)

from legendre import is_legendre_pair
from local_search import objective, paf_vector


def cpaf(x: np.ndarray, half: int) -> np.ndarray:
    """Continuous PAF vector [PAF(1..half)] for a real vector (no rounding)."""
    f = np.fft.rfft(x)
    ac = np.fft.irfft(f * np.conj(f), n=x.size)
    return ac[1 : half + 1]


def grad_and_E(a: np.ndarray, b: np.ndarray, half: int):
    """Return (grad_a, grad_b, E) of the continuous objective at (a, b)."""
    pa = cpaf(a, half)
    pb = cpaf(b, half)
    r = pa + pb + 2.0
    E = float(r @ r)
    ga = np.zeros_like(a)
    gb = np.zeros_like(b)
    for s in range(1, half + 1):
        rs = r[s - 1]
        ga += 2.0 * rs * (np.roll(a, -s) + np.roll(a, s))
        gb += 2.0 * rs * (np.roll(b, -s) + np.roll(b, s))
    return ga, gb, E


def rounded_E(a: np.ndarray, b: np.ndarray, half: int) -> tuple[int, np.ndarray, np.ndarray]:
    """Sign-round (a, b) and return (integer objective, sa, sb)."""
    sa = np.where(a >= 0, 1, -1).astype(np.int64)
    sb = np.where(b >= 0, 1, -1).astype(np.int64)
    return objective(paf_vector(sa, half), paf_vector(sb, half)), sa, sb


def _project(x: np.ndarray, mode: str, ell: int) -> np.ndarray:
    if mode == "clip":
        return np.clip(x, -1.0, 1.0)
    if mode == "sphere":
        n = np.linalg.norm(x)
        return x * (np.sqrt(ell) / n) if n > 0 else x
    return x


def one_run(ell, steps, eta, normalize, project, rng, record=False):
    """One descent from a random real start. Returns dict with best rounded E,
    whether a pair was found, and (if record) the E-trajectories."""
    half = (ell - 1) // 2
    a = rng.uniform(-1.0, 1.0, ell)
    b = rng.uniform(-1.0, 1.0, ell)
    best_round = None
    found = None
    diverged = False
    cont_traj, round_traj = [], []
    for _ in range(steps):
        with np.errstate(over="ignore", invalid="ignore"):
            ga, gb, E = grad_and_E(a, b, half)
        if not np.isfinite(E):      # quartic gradient blew the step up
            diverged = True
            break
        rE, sa, sb = rounded_E(a, b, half)
        if record:
            cont_traj.append(E)
            round_traj.append(rE)
        if best_round is None or rE < best_round:
            best_round = rE
        if rE == 0 and found is None:
            ok, _ = is_legendre_pair(sa.tolist(), sb.tolist())
            if ok:
                found = (sa.copy(), sb.copy())
                break
        if normalize:
            g = np.concatenate([ga, gb])
            n = np.linalg.norm(g)
            if n > 0:
                ga, gb = ga / n, gb / n
        a = a - eta * ga
        b = b - eta * gb
        if project != "none":
            a = _project(a, project, ell)
            b = _project(b, project, ell)
    return {
        "best_round": best_round,
        "found": found,
        "diverged": diverged,
        "cont_traj": cont_traj,
        "round_traj": round_traj,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ell", type=int, default=21)
    p.add_argument("--restarts", type=int, default=50)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--eta", type=float, default=None,
                   help="step size (default: 1/(2*ell), the proposed rule)")
    p.add_argument("--normalize", action="store_true",
                   help="normalise the gradient so each step has fixed length eta")
    p.add_argument("--project", choices=["none", "clip", "sphere"], default="none",
                   help="keep iterates bounded: clip to [-1,1] or rescale to ||x||=sqrt(ell)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    p.add_argument("--png", default=os.path.join(_RESULTS, "continuous_descent.png"))
    args = p.parse_args()

    if args.ell <= 0 or args.ell % 2 == 0:
        p.error(f"ell must be a positive odd integer, got {args.ell}")

    eta = args.eta if args.eta is not None else 1.0 / (2 * args.ell)
    rng = np.random.default_rng(args.seed)

    print(f"continuous gradient descent, ell={args.ell}, eta={eta:.5g}, "
          f"normalize={args.normalize}, project={args.project}, seed={args.seed}")
    print(f"{'restarts':>8} {'solved':>8} {'best rounded E':>16} {'median rounded E':>18}")

    best_rounds = []
    solved = 0
    diverged = 0
    trajs = []
    for k in range(args.restarts):
        res = one_run(args.ell, args.steps, eta, args.normalize, args.project,
                      rng, record=(k < 4))
        best_rounds.append(res["best_round"])
        if res["found"] is not None:
            solved += 1
        if res["diverged"]:
            diverged += 1
        if res["cont_traj"]:
            trajs.append((res["cont_traj"], res["round_traj"]))

    best_rounds = np.array(best_rounds)
    print(f"{args.restarts:>8} {solved:>8} {int(best_rounds.min()):>16} "
          f"{int(np.median(best_rounds)):>18}")
    if diverged:
        print(f"  ({diverged}/{args.restarts} runs DIVERGED: the raw step overflowed "
              f"-- the quartic gradient ~x^3 makes a fixed eta unstable; "
              f"use --normalize or --project.)")
    print(f"\nVERDICT: continuous GD (round-to-sign) found a Legendre pair in "
          f"{solved}/{args.restarts} restarts.")
    if solved == 0:
        print("  -> as predicted, plain continuous descent does not reach the "
              "binary solutions: it settles at non-binary critical points whose "
              "sign-rounding leaves E > 0.")

    make_plot(args, trajs, eta)
    return 0


def make_plot(args, trajs, eta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_c, ax_r) = plt.subplots(1, 2, figsize=(13, 5.2))
    ax_c.set_title(f"Continuous E (relaxed), ell={args.ell}")
    ax_r.set_title("Rounded E (sign(x)) -- what actually matters")
    for c, r in trajs:
        ax_c.plot(c, alpha=0.7, lw=1.1)
        ax_r.plot(r, alpha=0.7, lw=1.1)
    ax_r.axhline(0, color="gray", ls=":", lw=1, label="E=0 (Legendre pair)")
    for ax in (ax_c, ax_r):
        ax.set_xlabel("gradient step")
        ax.set_ylabel("objective E")
        ax.set_yscale("symlog")
        ax.grid(True, which="both", alpha=0.3)
    ax_r.legend(fontsize=8)
    fig.suptitle(f"Continuous gradient descent  (eta={eta:.4g}, "
                 f"normalize={args.normalize}, project={args.project})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.png, dpi=130)
    print(f"saved plot -> {args.png}")


if __name__ == "__main__":
    raise SystemExit(main())
