"""(r, n) random-restart + kick search for Legendre pairs (GPU-friendly layout).

The method has two knobs, ``r`` (population / batch size) and ``n`` (kick
radius), and runs in two phases:

  Step 1 (restart).  Draw a batch of ``r`` random normalized pairs, evaluate the
  objective E on all of them.  If the batch minimum is <= n * ell / 2, keep that
  best pair and go to Step 2; otherwise draw a fresh batch.  (A ``max_batches``
  budget bounds this loop so it always terminates.)

  Step 2 (kick).  Copy the current best pair ``r`` times; each copy gets ``n``
  independent 2-swaps (each swaps a +1 with a -1 in u OR v, so normalization is
  preserved -- swap-distance n = Hamming-distance 2n).  Evaluate all ``r`` kicks;
  if one is an LP, stop.  Otherwise accept the best *improving* kick as the new
  best and repeat, for at most ``n`` rounds.  If no round solves, return
  "not found".

Why this shape (see the project's landscape analysis): E has quantized local
minima on long plateaus, so a single-swap neighborhood stalls.  The gate ties
the kick radius to the objective -- E ~ n*ell means you are ~ n swaps from a
solution -- so a batch of n-swap kicks is aimed squarely at plateau escape.

Objective.  E(u, v) = sum_{s != 0} (PAF_u(s) + PAF_v(s) + 2)^2, computed by FFT
and evaluated for a whole batch at once (vectorized over the r-axis -- the exact
layout a cupy/torch GPU port would use).  E is a nonnegative integer, a multiple
of 4, and E == 0 iff (u, v) is a Legendre pair (given both are normalized, which
every move here preserves).

The exhaustive-neighborhood tools (``min_swaps_to_lp``, ``sample_E_vs_distance``)
are calibration only: on the enumerated lengths ell <= 19 they measure the true
swap-distance from a seed to the nearest Legendre pair, giving a LOWER BOUND on
the kick radius n that any such method needs.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legendre_pairs import is_legendre_pair  # noqa: E402

try:                                    # optional progress bar
    from tqdm import tqdm
except Exception:                       # pragma: no cover - tqdm is optional
    def tqdm(it=None, **kw):
        return it if it is not None else iter(())


# --------------------------------------------------------------------------- #
# objective (batched)
# --------------------------------------------------------------------------- #
def objective_batch(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """E for a batch of pairs. ``U``, ``V`` are ``(B, ell)`` arrays of +-1.

    Returns an int64 array of length ``B``. Vectorized over the batch axis via
    one FFT per array -- this is the GPU-friendly hot path."""
    U = U.astype(float)
    V = V.astype(float)
    pu = np.fft.ifft(np.abs(np.fft.fft(U, axis=1)) ** 2, axis=1).real
    pv = np.fft.ifft(np.abs(np.fft.fft(V, axis=1)) ** 2, axis=1).real
    resid = pu + pv + 2.0
    resid[:, 0] = 0.0                       # exclude s = 0 (there E-term is 2ell+2)
    return np.rint((resid ** 2).sum(axis=1)).astype(np.int64)


def objective(u: np.ndarray, v: np.ndarray) -> int:
    """E for a single pair."""
    return int(objective_batch(np.asarray(u)[None, :], np.asarray(v)[None, :])[0])


# --------------------------------------------------------------------------- #
# random normalized sequences + n-swap kicks (batched)
# --------------------------------------------------------------------------- #
def random_normalized(B: int, ell: int, rng: np.random.Generator) -> np.ndarray:
    """``(B, ell)`` batch of +-1 rows, each with exactly ``(ell-1)/2`` entries
    equal to -1 (so every row sums to +1)."""
    neg = (ell - 1) // 2
    A = np.ones((B, ell), dtype=np.int8)
    idx = np.argsort(rng.random((B, ell)), axis=1)[:, :neg]
    np.put_along_axis(A, idx, np.int8(-1), axis=1)
    return A


def _pick_true(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One uniformly-random ``True`` column index per row of a boolean matrix."""
    keys = rng.random(mask.shape)
    keys[~mask] = np.inf
    return keys.argmin(axis=1)


def _swap_rows(M: np.ndarray, rowmask: np.ndarray, rng: np.random.Generator) -> None:
    """In place: for each selected row, swap a random +1 with a random -1."""
    rows = np.nonzero(rowmask)[0]
    if rows.size == 0:
        return
    sub = M[rows]
    plus = _pick_true(sub == 1, rng)
    minus = _pick_true(sub == -1, rng)
    ri = np.arange(rows.size)
    sub[ri, plus] = -1
    sub[ri, minus] = 1
    M[rows] = sub


def kick_batch(u: np.ndarray, v: np.ndarray, B: int, n: int,
               rng: np.random.Generator):
    """``B`` copies of ``(u, v)``, each perturbed by ``n`` independent 2-swaps.

    Each swap acts on u or v (chosen per copy per step) and exchanges a +1 with a
    -1, so every copy stays normalized. Returns ``(U, V)`` of shape ``(B, ell)``.
    """
    U = np.tile(np.asarray(u, dtype=np.int8), (B, 1))
    V = np.tile(np.asarray(v, dtype=np.int8), (B, 1))
    for _ in range(n):
        act_u = rng.random(B) < 0.5
        _swap_rows(U, act_u, rng)
        _swap_rows(V, ~act_u, rng)
    return U, V


# --------------------------------------------------------------------------- #
# the search
# --------------------------------------------------------------------------- #
def search_rn(ell: int, r: int, n: int, seed: int = 0,
              max_batches: int = 1000, verbose: bool = False) -> dict:
    """Run the (r, n) restart+kick search. Returns a result dict.

    Keys: ``solved`` (bool), ``u``/``v`` (arrays or None), ``best_E``,
    ``evals`` (total pairs scored), ``batches`` (Step-1 batches drawn),
    ``rounds`` (Step-2 kick rounds used), and ``reason`` (why it stopped).
    """
    if ell <= 0 or ell % 2 == 0:
        raise ValueError(f"ell must be a positive odd integer, got {ell}")
    rng = np.random.default_rng(seed)
    tau = n * ell / 2.0
    evals = 0

    # --- Step 1: restart until a batch minimum falls under the gate ---------- #
    best_u = best_v = None
    best_E = None
    batches = 0
    for batches in range(1, max_batches + 1):
        U = random_normalized(r, ell, rng)
        V = random_normalized(r, ell, rng)
        E = objective_batch(U, V)
        evals += r
        k = int(E.argmin())
        if E[k] == 0:                       # struck a pair while sampling
            return _solved(ell, U[k], V[k], evals, batches, 0, entered_step2=False)
        if E[k] <= tau:
            best_u, best_v, best_E = U[k].copy(), V[k].copy(), int(E[k])
            if verbose:
                print(f"[rn] gate passed at batch {batches}: E={best_E} <= tau={tau}")
            break
    else:
        if verbose:
            print(f"[rn] gate never passed in {max_batches} batches (tau={tau})")
        return {"solved": False, "u": None, "v": None, "best_E": None,
                "evals": evals, "batches": batches, "rounds": 0,
                "entered_step2": False, "reason": "step1_budget_exhausted"}

    # --- Step 2: up to n rounds of n-swap kicks, accept best improving ------- #
    rounds = 0
    for rounds in range(1, n + 1):
        U, V = kick_batch(best_u, best_v, r, n, rng)
        E = objective_batch(U, V)
        evals += r
        j = int(E.argmin())
        if E[j] == 0:
            return _solved(ell, U[j], V[j], evals, batches, rounds,
                           entered_step2=True)
        if E[j] < best_E:                   # accept-the-best-improving-kick
            best_u, best_v, best_E = U[j].copy(), V[j].copy(), int(E[j])

    return {"solved": False, "u": None, "v": None, "best_E": best_E,
            "evals": evals, "batches": batches, "rounds": rounds,
            "entered_step2": True, "reason": "step2_rounds_exhausted"}


def _solved(ell, u, v, evals, batches, rounds, entered_step2) -> dict:
    u = np.asarray(u, dtype=np.int8)
    v = np.asarray(v, dtype=np.int8)
    assert is_legendre_pair(u, v, ell), "objective==0 but is_legendre_pair failed"
    return {"solved": True, "u": u, "v": v, "best_E": 0, "evals": evals,
            "batches": batches, "rounds": rounds, "entered_step2": entered_step2,
            "reason": "solved"}


# --------------------------------------------------------------------------- #
# exhaustive calibration: true swap-distance from a seed to the nearest LP
# --------------------------------------------------------------------------- #
_POP16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int16)


def _popcount(a: np.ndarray) -> np.ndarray:
    """Vectorized popcount for nonnegative ints < 2**32 (ell <= 19 fits easily)."""
    a = a.astype(np.int64)
    return _POP16[a & 0xFFFF] + _POP16[(a >> 16) & 0xFFFF]


def load_lp_pairs(ell: int):
    """All ordered Legendre pairs of length ``ell`` as bitmask arrays.

    Returns ``(bu, bv)`` int arrays where each mask has its ``-1`` positions as
    set bits. Uses the census checkpoint if present, else runs the (cheap)
    signature census. Only meaningful for the enumerated lengths ell <= ~19."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "census"))
    import census
    loaded = census._load_ckpt(ell)
    if loaded is not None:
        count, sigmap, _ = loaded
    else:
        count, sigmap = census.census_signatures(ell, progress=False)
    bu, bv = [], []
    for a, b in census.ordered_lp_pairs(count, sigmap):
        bu.append(a)
        bv.append(b)
    return np.array(bu, dtype=np.int64), np.array(bv, dtype=np.int64)


def _seq_to_mask(seq: np.ndarray) -> int:
    m = 0
    for p, x in enumerate(seq):
        if x == -1:
            m |= 1 << p
    return m


def min_swaps_to_lp(u: np.ndarray, v: np.ndarray, bu: np.ndarray,
                    bv: np.ndarray) -> int:
    """Fewest 2-swaps turning ``(u, v)`` into some Legendre pair in the set.

    swap-distance = (Hamming(u,u*) + Hamming(v,v*)) / 2, minimized over all LPs.
    (Both Hamming terms are even, equal weight -> the sum is even.) The ``switch``
    symmetry is already inside the ordered LP set, so we needn't swap u<->v."""
    su, sv = _seq_to_mask(u), _seq_to_mask(v)
    d = (_popcount(su ^ bu) + _popcount(sv ^ bv)) // 2
    return int(d.min())


def sample_E_vs_distance(ell: int, n_seeds: int = 300, seed: int = 0,
                         progress: bool = True):
    """For random normalized seeds, pair (objective E, true swap-distance to the
    nearest LP). The distance is a per-seed LOWER BOUND on the kick radius needed
    to solve from that seed. Returns ``(E_array, dist_array)``.

    The per-seed nearest-LP scan is the slow part (linear in the LP-set size), so
    it shows a progress bar unless ``progress=False``."""
    bu, bv = load_lp_pairs(ell)
    rng = np.random.default_rng(seed)
    U = random_normalized(n_seeds, ell, rng)
    V = random_normalized(n_seeds, ell, rng)
    E = objective_batch(U, V)
    dist = np.fromiter(
        (min_swaps_to_lp(U[i], V[i], bu, bv)
         for i in tqdm(range(n_seeds), desc=f"ell={ell} dist",
                       disable=not progress)),
        dtype=np.int64, count=n_seeds)
    return E, dist


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description="(r,n) restart+kick Legendre search.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("search", help="run one (r,n) search")
    ps.add_argument("ell", type=int)
    ps.add_argument("-r", type=int, default=256, help="batch size")
    ps.add_argument("-n", type=int, default=2, help="kick radius (and gate/round knob)")
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--max-batches", type=int, default=1000)
    ps.add_argument("-v", "--verbose", action="store_true")

    pc = sub.add_parser("calibrate", help="E vs true swap-distance to nearest LP")
    pc.add_argument("ell", type=int)
    pc.add_argument("--seeds", type=int, default=300)
    pc.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    if args.cmd == "search":
        res = search_rn(args.ell, args.r, args.n, seed=args.seed,
                        max_batches=args.max_batches, verbose=args.verbose)
        if res["solved"]:
            print(f"SOLVED ell={args.ell} (r={args.r}, n={args.n}) in "
                  f"{res['evals']} evals [{res['batches']} batches, "
                  f"{res['rounds']} kick rounds]")
            print(f"  u = {list(map(int, res['u']))}")
            print(f"  v = {list(map(int, res['v']))}")
            return 0
        print(f"NOT FOUND ell={args.ell} (r={args.r}, n={args.n}): "
              f"{res['reason']}, best_E={res['best_E']}, evals={res['evals']}")
        return 1

    E, dist = sample_E_vs_distance(args.ell, n_seeds=args.seeds, seed=args.seed)
    print(f"ell={args.ell}  seeds={args.seeds}")
    print(f"  E:     min={E.min()} median={int(np.median(E))} max={E.max()}")
    print(f"  dist:  min={dist.min()} median={int(np.median(dist))} max={dist.max()}")
    print(f"  corr(E, dist) = {np.corrcoef(E, dist)[0, 1]:.3f}")
    # median true distance among seeds that would pass the gate E <= n*ell/2
    for n in (1, 2, 3, 4, 5):
        tau = n * args.ell / 2.0
        sel = dist[E <= tau]
        if sel.size:
            print(f"  gate n={n} (tau={tau:5.1f}): {sel.size:4d} seeds pass, "
                  f"true dist min={sel.min()} median={int(np.median(sel))}")
        else:
            print(f"  gate n={n} (tau={tau:5.1f}): no seed passes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
