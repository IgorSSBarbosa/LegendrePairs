"""restriction_matrix.py — does forcing leading '+'/trailing '-' runs pay off?

For a given ell, restrict candidate sequences to have >=i forced leading '+'
and >=j forced trailing '-' (both u and v get the SAME i,j). This shrinks the
per-sequence search space from C(ell,(ell-1)/2) down to C(ell-i-j, (ell-1)/2-j)
sum-normalized candidates (closed form, see `denominator`). The question is
how the *yield* -- fraction of raw (u,v) candidate pairs in that restricted,
normalized space that are actual Legendre pairs -- behaves as (i,j) grows,
and whether the best restriction level scales like log(ell) (the user's
hypothesis) or something else.

Key implementation point: rather than re-running a restricted search once per
(i,j) cell (2^ell-ish candidates each), we run the standard normalized
exhaustive search ONCE per ell (src/legendre.find_legendre_pairs_parallel,
the same PAF-bucket algorithm the rest of this project already uses), record
each found raw pair's (lead_pair, trail_pair) = min(lead_u,lead_v),
min(trail_u,trail_v), and 2D-suffix-sum that histogram. Cell (i,j)'s numerator
is exactly the count of raw pairs with lead_pair>=i and trail_pair>=j -- which
is exactly what a restricted search rooted at (i,j) would have found, since
every candidate with a longer run trivially also satisfies a shorter
restriction. This turns an O((ell/2)^2) x (expensive search) job into one
expensive search + a cheap O((ell/2)^2) aggregation.

ell <= MAX_EXACT_ELL=27 is computed exactly this way. Beyond that, exhaustive
search is infeasible, so for each fixed (i,j) cell we fit
    log(fraction) = a(i,j) + b(i,j) * ell
by least squares over the exact points where fraction > 0, and extrapolate
the cell value up to MAX_ELL=57 (matching the range of the other two scripts).

Outputs (canonical_orbits/results/):
  restriction_exact/ell{ell}.csv, ell{ell}_heatmap.png        for ell<=27
  restriction_extrapolated/ell{ell}.csv, ell{ell}_heatmap.png for 27<ell<=57
  restriction_fits.csv         -- per (i,j) cell: fitted slope/intercept + how
                                   many exact points went into the fit
  restriction_ridge.png        -- argmax(i,j) of fraction vs ell, against
                                   log(ell) and sqrt(ell) reference curves
"""
from __future__ import annotations

import csv
import os
import sys
import time
from math import comb, log, sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
from legendre import find_legendre_pairs_parallel  # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "results")
EXACT_DIR = os.path.join(RESULTS_DIR, "restriction_exact")
EXTRAP_DIR = os.path.join(RESULTS_DIR, "restriction_extrapolated")
FITS_PATH = os.path.join(RESULTS_DIR, "restriction_fits.csv")
RIDGE_PLOT = os.path.join(RESULTS_DIR, "restriction_ridge.png")

MAX_EXACT_ELL = 27
MAX_ELL = 57


# --------------------------------------------------------------------------- #
# candidate-space bookkeeping
# --------------------------------------------------------------------------- #
def lead_run(seq) -> int:
    n = 0
    for x in seq:
        if x != 1:
            break
        n += 1
    return n


def trail_run(seq) -> int:
    n = 0
    for x in reversed(seq):
        if x != -1:
            break
        n += 1
    return n


def denominator(ell: int, i: int, j: int) -> int:
    """# sum=+1 candidates of length ell with the first i forced +1, last j
    forced -1 (0 if i,j leave no room to hit the required minus-count)."""
    m = ell - i - j
    if m < 0:
        return 0
    n_minus = (ell - 1) // 2 - j     # required # of -1 among the m free slots
    p = (ell + 1) // 2 - i           # required # of +1 among the m free slots
    if n_minus < 0 or p < 0 or n_minus + p != m:
        return 0
    return comb(m, n_minus)


def grid_shape(ell: int) -> tuple[int, int]:
    return (ell + 1) // 2, (ell - 1) // 2  # (imax, jmax), each inclusive


def denom_grid(ell: int) -> np.ndarray:
    """Pair search-space size per cell: D(ell,i,j)^2, since u and v are each
    drawn independently from the same restricted, sum-normalized candidate set.

    dtype=object (arbitrary-precision Python int): by ell~50 this routinely
    exceeds int64 (C(ell,~ell/2) squared blows past 9.2e18 well before ell=57).
    """
    imax, jmax = grid_shape(ell)
    D = np.empty((imax + 1, jmax + 1), dtype=object)
    for i in range(imax + 1):
        for j in range(jmax + 1):
            d = denominator(ell, i, j)
            D[i, j] = d * d
    return D


def fraction_grid(N: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Elementwise N/D as float64, safe for D held as arbitrary-precision ints."""
    imax, jmax = D.shape[0] - 1, D.shape[1] - 1
    frac = np.full((imax + 1, jmax + 1), np.nan)
    for i in range(imax + 1):
        for j in range(jmax + 1):
            d = D[i, j]
            if d and d > 0:
                frac[i, j] = float(N[i, j]) / float(d)
    return frac


# --------------------------------------------------------------------------- #
# exact pass: one exhaustive search per ell, aggregated into the (i,j) grid
# --------------------------------------------------------------------------- #
def exact_ell(ell: int, workers: int | None = None) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    imax, jmax = grid_shape(ell)
    hist = np.zeros((imax + 1, jmax + 1), dtype=np.int64)

    n_pairs = 0
    for a, b in find_legendre_pairs_parallel(ell, normalized=True, workers=workers):
        lp = min(lead_run(a), lead_run(b))
        tp = min(trail_run(a), trail_run(b))
        hist[lp, tp] += 1
        n_pairs += 1

    # suffix sum: N[i,j] = sum_{i'>=i, j'>=j} hist[i',j']  (flip, cumsum, flip back)
    N = hist[::-1, ::-1].cumsum(0).cumsum(1)[::-1, ::-1]
    D = denom_grid(ell)
    frac = fraction_grid(N, D)
    return n_pairs, N, D, frac


def write_grid_csv(path: str, N: np.ndarray, D: np.ndarray, frac: np.ndarray) -> None:
    imax, jmax = N.shape[0] - 1, N.shape[1] - 1
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "j", "n_valid", "denominator", "fraction"])
        for i in range(imax + 1):
            for j in range(jmax + 1):
                d = int(D[i, j])
                fr = frac[i, j]
                w.writerow([i, j, int(N[i, j]), d, "" if np.isnan(fr) else fr])


def plot_heatmap(path: str, frac: np.ndarray, ell: int, extrapolated: bool) -> None:
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="0.85")
    vmax = np.nanmax(frac) if np.any(~np.isnan(frac)) else 1.0

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(frac.T, origin="lower", cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xlabel("i (leading '+' restriction)")
    ax.set_ylabel("j (trailing '-' restriction)")
    tag = " (extrapolated)" if extrapolated else ""
    ax.set_title(rf"$\ell$={ell}{tag}: fraction valid / restricted space")
    fig.colorbar(im, ax=ax, label="fraction (normalized to this ell's max)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_exact(ells: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    os.makedirs(EXACT_DIR, exist_ok=True)
    out = {}
    for ell in ells:
        t0 = time.time()
        n_pairs, N, D, frac = exact_ell(ell)
        dt = time.time() - t0
        write_grid_csv(os.path.join(EXACT_DIR, f"ell{ell}.csv"), N, D, frac)
        plot_heatmap(os.path.join(EXACT_DIR, f"ell{ell}_heatmap.png"), frac, ell, extrapolated=False)
        out[ell] = (N, D, frac)
        print(f"ell={ell:>3}: {n_pairs:>10,} raw pairs, grid {frac.shape}, {dt:6.1f}s", flush=True)
    return out


# --------------------------------------------------------------------------- #
# semilog extrapolation: fit log(fraction) vs ell per fixed (i,j) cell
# --------------------------------------------------------------------------- #
def fit_cells(exact: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict[tuple[int, int], tuple[float, float, int]]:
    """Return {(i,j): (intercept, slope, n_points)} for log(fraction) = a + b*ell,
    using only (ell, fraction) points with fraction > 0. Cells with < 2 such
    points are omitted (no extrapolation possible)."""
    points: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for ell, (N, D, frac) in exact.items():
        imax, jmax = frac.shape[0] - 1, frac.shape[1] - 1
        for i in range(imax + 1):
            for j in range(jmax + 1):
                fr = frac[i, j]
                if not np.isnan(fr) and fr > 0:
                    points.setdefault((i, j), []).append((ell, fr))

    fits = {}
    for key, pts in points.items():
        if len(pts) < 2:
            continue
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.log(np.array([p[1] for p in pts], dtype=float))
        b, a = np.polyfit(xs, ys, 1)  # ys ~ a + b*xs
        fits[key] = (float(a), float(b), len(pts))
    return fits


def write_fits_csv(fits: dict[tuple[int, int], tuple[float, float, int]]) -> None:
    with open(FITS_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "j", "intercept", "slope", "n_exact_points"])
        for (i, j), (a, b, n) in sorted(fits.items()):
            w.writerow([i, j, a, b, n])


def extrapolate_ell(ell: int, fits: dict[tuple[int, int], tuple[float, float, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Predicted fraction grid for `ell` from the per-cell semilog fits."""
    imax, jmax = grid_shape(ell)
    D = denom_grid(ell)
    frac = np.full((imax + 1, jmax + 1), np.nan)
    for (i, j), (a, b, n) in fits.items():
        if i <= imax and j <= jmax and D[i, j] > 0:
            frac[i, j] = np.exp(a + b * ell)
    return D, frac


def run_extrapolation(fits: dict[tuple[int, int], tuple[float, float, int]],
                       ells: list[int]) -> dict[int, np.ndarray]:
    os.makedirs(EXTRAP_DIR, exist_ok=True)
    out = {}
    for ell in ells:
        D, frac = extrapolate_ell(ell, fits)
        # best-effort expected-count reconstruction for the CSV; D can be
        # astronomically large here, so keep this as Python floats, not int64.
        imax, jmax = frac.shape[0] - 1, frac.shape[1] - 1
        N_placeholder = np.zeros((imax + 1, jmax + 1), dtype=object)
        for i in range(imax + 1):
            for j in range(jmax + 1):
                if not np.isnan(frac[i, j]):
                    N_placeholder[i, j] = frac[i, j] * D[i, j]
        write_grid_csv(os.path.join(EXTRAP_DIR, f"ell{ell}.csv"), N_placeholder, D, frac)
        plot_heatmap(os.path.join(EXTRAP_DIR, f"ell{ell}_heatmap.png"), frac, ell, extrapolated=True)
        out[ell] = frac
        n_cells = int(np.sum(~np.isnan(frac)))
        print(f"ell={ell:>3}: extrapolated {n_cells} cells", flush=True)
    return out


# --------------------------------------------------------------------------- #
# ridge (argmax restriction) vs ell, tested against log/sqrt scaling
# --------------------------------------------------------------------------- #
def ridge_points(exact: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
                  extrap: dict[int, np.ndarray]) -> list[tuple[int, int, int, float]]:
    all_frac = {ell: frac for ell, (N, D, frac) in exact.items()}
    all_frac.update(extrap)
    pts = []
    for ell in sorted(all_frac):
        frac = all_frac[ell]
        if np.all(np.isnan(frac)):
            continue
        flat = np.nan_to_num(frac, nan=-1.0)
        i_star, j_star = np.unravel_index(np.argmax(flat), flat.shape)
        pts.append((ell, int(i_star), int(j_star), float(frac[i_star, j_star])))
    return pts


def plot_ridge(pts: list[tuple[int, int, int, float]]) -> None:
    ells = np.array([p[0] for p in pts])
    i_star = np.array([p[1] for p in pts])
    j_star = np.array([p[2] for p in pts])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(ells, i_star, "o-", label="i* (leading)")
    ax.plot(ells, j_star, "s-", label="j* (trailing)")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel("optimal restriction")
    ax.set_title("linear axis")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(np.log(ells), i_star, "o-", label="i*")
    ax.plot(np.log(ells), j_star, "s-", label="j*")
    ax.set_xlabel(r"$\log(\ell)$")
    ax.set_title("vs log(ell)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(np.sqrt(ells), i_star, "o-", label="i*")
    ax.plot(np.sqrt(ells), j_star, "s-", label="j*")
    ax.set_xlabel(r"$\sqrt{\ell}$")
    ax.set_title("vs sqrt(ell)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("optimal (i*, j*) restriction vs " + r"$\ell$" +
                 f"  (ell<={MAX_EXACT_ELL} exact, beyond extrapolated)")
    fig.tight_layout()
    fig.savefig(RIDGE_PLOT, dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    exact_ells = list(range(3, MAX_EXACT_ELL + 1, 2))
    extrap_ells = list(range(MAX_EXACT_ELL + 2, MAX_ELL + 1, 2))

    print(f"=== exact pass (ell<={MAX_EXACT_ELL}) ===")
    exact = run_exact(exact_ells)

    print(f"\n=== semilog fit + extrapolation (ell up to {MAX_ELL}) ===")
    fits = fit_cells(exact)
    write_fits_csv(fits)
    print(f"fitted {len(fits)} (i,j) cells with >=2 exact points")
    extrap = run_extrapolation(fits, extrap_ells)

    print("\n=== ridge (argmax restriction) vs ell ===")
    pts = ridge_points(exact, extrap)
    for ell, i_star, j_star, fr in pts:
        print(f"ell={ell:>3}: (i*,j*)=({i_star},{j_star})  fraction={fr:.3e}")
    plot_ridge(pts)

    print(f"\nwrote exact grids to {EXACT_DIR}")
    print(f"wrote extrapolated grids to {EXTRAP_DIR}")
    print(f"wrote fits to {FITS_PATH}")
    print(f"wrote ridge plot to {RIDGE_PLOT}")


if __name__ == "__main__":
    main()
