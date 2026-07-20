"""Test the conjecture: a single swap changes the objective E by ~ sqrt(ell).

For random sum=1 configurations (A, B) we sample many swap moves and record the
exact change dE. We report how the typical magnitude of dE scales with ell, and
decompose it into

    dE = 2 * <r, dP>   (gradient . step, random sign)  +  ||dP||^2  (curvature),

where r(s) = PAF_A(s)+PAF_B(s)+2 and dP is the PAF change of the mutated
sequence. Fitting log(rms dE) vs log(ell) gives the scaling exponent, which we
compare against 0.5 (sqrt) and 1.0 (linear).

Usage:
    python3 measure_flip_scaling.py
    python3 measure_flip_scaling.py --configs 200 --swaps 200
"""

from __future__ import annotations

import argparse

import numpy as np

from local_search import objective, paf_vector, random_sum_one


def sample_deltas(ell, n_configs, n_swaps, rng):
    """Return arrays of dE, cross-term T1, curvature T2 over random swaps."""
    half = (ell - 1) // 2
    dE, T1, T2 = [], [], []
    for _ in range(n_configs):
        a = random_sum_one(ell, rng)
        b = random_sum_one(ell, rng)
        pa = paf_vector(a, half)
        pb = paf_vector(b, half)
        E = objective(pa, pb)
        for _ in range(n_swaps):
            seq, pself, pother = (a, pa, pb) if rng.random() < 0.5 else (b, pb, pa)
            plus = np.flatnonzero(seq == 1)
            minus = np.flatnonzero(seq == -1)
            i = plus[rng.integers(plus.size)]
            j = minus[rng.integers(minus.size)]
            seq[i], seq[j] = -1, 1
            pnew = paf_vector(seq, half)
            seq[i], seq[j] = 1, -1  # revert (we only measure the delta)

            dP = pnew - pself
            r = pself + pother + 2
            t1 = 2 * int(np.dot(r, dP))
            t2 = int(np.dot(dP, dP))
            dE.append(objective(pnew, pother) - E)
            T1.append(t1)
            T2.append(t2)
    return np.array(dE), np.array(T1), np.array(T2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--configs", type=int, default=120)
    p.add_argument("--swaps", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--png", default="flip_scaling.png")
    args = p.parse_args()

    lengths = [11, 15, 21, 31, 41, 61, 81, 101, 151, 201]
    rng = np.random.default_rng(args.seed)

    ells, rms_dE, mean_abs, rms_T1, mean_T2 = [], [], [], [], []
    print(f"{'ell':>4} {'rms|dE|':>10} {'mean|dE|':>10} "
          f"{'rms|dE|/sqrt(l)':>16} {'rms|dE|/l':>10} "
          f"{'rms T1(grad)':>13} {'mean T2(curv)':>14}")
    for ell in lengths:
        dE, T1, T2 = sample_deltas(ell, args.configs, args.swaps, rng)
        r = float(np.sqrt(np.mean(dE.astype(float) ** 2)))
        ma = float(np.mean(np.abs(dE)))
        ells.append(ell)
        rms_dE.append(r)
        mean_abs.append(ma)
        rms_T1.append(float(np.sqrt(np.mean(T1.astype(float) ** 2))))
        mean_T2.append(float(np.mean(T2)))
        print(f"{ell:>4} {r:>10.2f} {ma:>10.2f} {r/np.sqrt(ell):>16.3f} "
              f"{r/ell:>10.3f} {rms_T1[-1]:>13.2f} {mean_T2[-1]:>14.2f}")

    ell_arr = np.array(ells, float)
    slope_rms = float(np.polyfit(np.log(ell_arr), np.log(rms_dE), 1)[0])
    slope_abs = float(np.polyfit(np.log(ell_arr), np.log(mean_abs), 1)[0])
    slope_T1 = float(np.polyfit(np.log(ell_arr), np.log(rms_T1), 1)[0])
    slope_T2 = float(np.polyfit(np.log(ell_arr), np.log(mean_T2), 1)[0])

    print("\n--- fitted scaling exponents  (t ~ ell^p) ---")
    print(f"rms|dE|      : p = {slope_rms:.3f}")
    print(f"mean|dE|     : p = {slope_abs:.3f}")
    print(f"rms T1 (grad): p = {slope_T1:.3f}")
    print(f"mean T2 (curv): p = {slope_T2:.3f}")
    print("reference: sqrt(ell) -> p=0.5,  linear -> p=1.0")
    verdict = ("~ sqrt(ell)" if abs(slope_rms - 0.5) < 0.15
               else "~ linear in ell" if abs(slope_rms - 1.0) < 0.15
               else f"~ ell^{slope_rms:.2f}")
    print(f"\nVERDICT: a swap changes E by {verdict} (rms), NOT sqrt unless p~0.5.")

    make_plot(ell_arr, rms_dE, mean_abs, slope_rms, args.png)
    return 0


def make_plot(ell, rms_dE, mean_abs, slope, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.loglog(ell, rms_dE, "o-", color="tab:blue", label="rms |dE| (measured)")
    ax.loglog(ell, mean_abs, "s--", color="tab:cyan", label="mean |dE| (measured)")
    # reference slopes anchored at the first point
    c = rms_dE[0]
    ax.loglog(ell, c * (ell / ell[0]) ** 0.5, ":", color="tab:green",
              label=r"$\propto \sqrt{\ell}$  (slope 0.5)")
    ax.loglog(ell, c * (ell / ell[0]) ** 1.0, ":", color="tab:red",
              label=r"$\propto \ell$  (slope 1.0)")
    ax.set_title(f"Objective change per swap vs ell   (fitted slope = {slope:.2f})")
    ax.set_xlabel(r"$\ell$ (log scale)")
    ax.set_ylabel("change in objective |dE| (log scale)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"saved plot -> {out_png}")


if __name__ == "__main__":
    raise SystemExit(main())
