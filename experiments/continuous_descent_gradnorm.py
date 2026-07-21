"""Continuous gradient descent with the gradient normalized by 1/(2*ell).

Variant requested by the user, kept as a SEPARATE function from
``continuous_descent.py`` (which sweeps step sizes / projections).

Idea
----
Relax +-1 to real vectors and minimize the smooth degree-4 objective

    E(a, b) = sum_{s=1}^{(ell-1)/2} ( PAF_a(s) + PAF_b(s) + 2 )^2 .

Instead of tuning a step size, treat ``1/(2*ell)`` as a fixed *preconditioner*
and take the update

    x  <-  x  -  grad E(x) / (2*ell),

then sign-round each iterate and test for a Legendre pair. The 1/(2*ell) factor
is motivated as a diagonal (Jacobi/Newton) scaling: the diagonal curvature of E
near the binary scale ||x|| ~ sqrt(ell) is O(ell), so dividing by ~2*ell
approximates one Newton step per coordinate.

Honest caveat (measured, not assumed): the gradient of a degree-4 form at that
scale is O(ell^2), so a 1/(2*ell) factor still leaves an O(ell) step -- large.
This script reports how often it diverges and how often the rounded iterate is
actually a Legendre pair, with an optional ``--project`` safeguard to keep the
iterates bounded so the rounded-E behavior is visible even when the raw step is
unstable.

Verdict (measured, seed=0)
--------------------------
Raw (no projection) diverges more the larger ell gets: solved/100 =
52, 32, 7, 1 for ell = 7, 9, 11, 13 (diverged the rest). With ``--project clip``
divergence vanishes but the per-restart solve *probability* still decays roughly
geometrically: solved = 68/100 (ell=11), 53/100 (13), 61/100 (15), then per 200
restarts 68 (17), 19 (19), 4 (21), 2 (25). ``--project sphere`` is far worse
(3/100 at ell=11, 0 beyond) -- it sticks at non-binary critical points.

Conclusion: continuous descent does NOT scale for this problem. Each restart is
cheap but gets trapped in a local minimum; the success probability falls off
exponentially in ell, so it is the same exponential-effort regime as random
restart. The 1/(2*ell) preconditioner buys a well-behaved trajectory (with clip),
not an escape from the plateau structure. Kept for the record, not recommended.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

from continuous_descent import cpaf, grad_and_E, rounded_E, _project  # noqa: E402
from legendre import is_legendre_pair  # noqa: E402


def descent_gradnorm(ell, steps, rng, project="none", record=False):
    """One descent from a random real start using x <- x - grad/(2*ell).

    Returns a dict with the best rounded objective seen, the found pair (if any),
    a divergence flag, and (if ``record``) the continuous/rounded E-trajectories.
    """
    half = (ell - 1) // 2
    scale = 1.0 / (2 * ell)             # the requested 1/(2*ell) gradient norm
    a = rng.uniform(-1.0, 1.0, ell)
    b = rng.uniform(-1.0, 1.0, ell)
    best_round = None
    found = None
    diverged = False
    cont_traj, round_traj = [], []
    for _ in range(steps):
        with np.errstate(over="ignore", invalid="ignore"):
            ga, gb, E = grad_and_E(a, b, half)
        if not np.isfinite(E):
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
        a = a - scale * ga
        b = b - scale * gb
        if project != "none":
            a = _project(a, project, ell)
            b = _project(b, project, ell)
    return {"best_round": best_round, "found": found, "diverged": diverged,
            "cont_traj": cont_traj, "round_traj": round_traj}


def run(ell, restarts, steps, project, seed, verbose=True):
    rng = np.random.default_rng(seed)
    best_rounds, solved, diverged = [], 0, 0
    pair = None
    for _ in range(restarts):
        res = descent_gradnorm(ell, steps, rng, project=project)
        best_rounds.append(res["best_round"])
        if res["found"] is not None:
            solved += 1
            pair = res["found"]
        if res["diverged"]:
            diverged += 1
    best_rounds = np.array(best_rounds)
    if verbose:
        print(f"ell={ell}  grad/(2*ell) descent  project={project}  "
              f"restarts={restarts} steps={steps} seed={seed}")
        print(f"  solved={solved}/{restarts}  diverged={diverged}/{restarts}  "
              f"best_rounded_E={int(best_rounds.min())}  "
              f"median_rounded_E={int(np.median(best_rounds))}")
        if solved == 0:
            print("  -> no Legendre pair reached: the rounded iterate never hit "
                  "E=0 (settles at a non-binary critical point).")
    return {"solved": solved, "restarts": restarts, "diverged": diverged,
            "best_rounded_E": int(best_rounds.min()), "pair": pair}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ell", type=int, default=13)
    p.add_argument("--restarts", type=int, default=100)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--project", choices=["none", "clip", "sphere"], default="none",
                   help="keep iterates bounded (safeguard against the O(ell) step)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.ell <= 0 or args.ell % 2 == 0:
        p.error(f"ell must be a positive odd integer, got {args.ell}")
    res = run(args.ell, args.restarts, args.steps, args.project, args.seed)
    return 0 if res["solved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
