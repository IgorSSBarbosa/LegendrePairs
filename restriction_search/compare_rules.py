"""compare_rules.py -- head-to-head: argmax-yield vs p75-envelope vs
max-envelope restriction rules.

Context: sweep_methods.py's restricted-space comparison used
(i*, j*) = argmax cell of restriction_exact -- the best-YIELD cell for that
specific ell. Re-examining that rule found it degenerates once extrapolated
past ell=27: it freezes at a FIXED (2,1) for every ell up to 57, so its space
reduction is a flat ~57-60x constant that never grows with ell -- no real
asymptotic win, which is why sweep_methods_24seeds.png shows the restricted
exhaustive lines tracking the same growth rate as everything else.

Two alternative rules, both derived from canonical_orbits/results/
run_length_rows.csv (actual leading-'+'/trailing-'-' run lengths of real
canonical LP classes, not a yield-argmax):

  p75_leading   -- i(ell) = round(a*sqrt(ell)+b) fit to the 75th percentile of
                   lead_pair = min(lead_u, lead_v) across canonical classes
                   (refit here from run_length_rows.csv, not hardcoded);
                   j(ell) = 1 fixed, since trailing-run percentiles are too
                   noisy (R^2 ~ 0.2-0.6) to support a growing rule.
  max_envelope  -- i(ell), j(ell) both sqrt(ell) fits of lead_pair_max /
                   trail_pair_max (canonical_orbits/results/
                   pair_restriction_scaling_fits.csv, R^2 ~ 0.94-0.95) -- the
                   largest restriction for which at least one known canonical
                   pair survives. More aggressive, riskier (rests on a single
                   extremal observation per ell rather than a robust quantile).

For ell<=27, where canonical_orbits/restriction_exact has exact ground truth
(built from a real exhaustive search), this script:
  1. computes each rule's (i,j) for that ell,
  2. looks up the ground-truth n_valid/fraction at that exact cell -- does the
     box really contain a real pair, not just heuristic belief,
  3. runs a live restricted binary-exhaustive search (24 seeds, adaptive
     per-ell time ceiling off the current rule's own baseline, mirroring
     sweep_methods.py) and records median time-to-first-pair + success rate.

It also plots the CLOSED-FORM space-reduction factor D(ell,0,0)/D(ell,i,j) for
all three rules across the full ell=3..57 range (cheap, no search needed) --
this is the part that actually answers "does this generalize to larger ell."

Output: results/compare_rules.csv, results/compare_rules.png
"""
from __future__ import annotations

import csv
import os
import sys
import time
from math import sqrt

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "canonical_orbits"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from restriction_matrix import denominator, grid_shape  # noqa: E402
from restricted_binary import find_first_binary_exhaustive_multi_seed  # noqa: E402
from restriction_rule import fit_lead_p75  # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "results")
CANON_RESULTS = os.path.join(_HERE, "..", "canonical_orbits", "results")
EXACT_DIR = os.path.join(CANON_RESULTS, "restriction_exact")
EXTRAP_DIR = os.path.join(CANON_RESULTS, "restriction_extrapolated")
PAIR_FITS = os.path.join(CANON_RESULTS, "pair_restriction_scaling_fits.csv")

CSV_PATH = os.path.join(RESULTS_DIR, "compare_rules.csv")
PLOT_PATH = os.path.join(RESULTS_DIR, "compare_rules.png")

LIVE_LADDER = list(range(3, 28, 2))     # ell with restriction_exact ground truth
FULL_LADDER = list(range(3, 58, 2))     # full range for the closed-form reduction panel
BASELINE_SEEDS = list(range(9))
N_SEEDS = 24
PRUNE_FACTOR = 3.0
MIN_CEILING = 0.5
BASELINE_HARD_CAP = 30.0

COLORS = {"current_argmax": "black", "p75_leading": "tab:blue", "max_envelope": "tab:red"}
MARKERS = {"current_argmax": "o", "p75_leading": "s", "max_envelope": "^"}
LABELS = {"current_argmax": "current (argmax yield)",
          "p75_leading": "p75-leading envelope",
          "max_envelope": "max envelope"}


# --------------------------------------------------------------------------- #
# rule definitions
# --------------------------------------------------------------------------- #
def _argmax_ij(path: str) -> tuple[int, int]:
    best = (0, 0, -1.0)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            fr = row["fraction"]
            if fr == "":
                continue
            fr = float(fr)
            if fr > best[2]:
                best = (int(row["i"]), int(row["j"]), fr)
    return best[0], best[1]


def _clip(ell: int, i: int, j: int) -> tuple[int, int]:
    imax, jmax = grid_shape(ell)
    return max(0, min(i, imax)), max(0, min(j, jmax))


def rule_current(ell: int) -> tuple[int, int]:
    path = os.path.join(EXACT_DIR if ell <= 27 else EXTRAP_DIR, f"ell{ell}.csv")
    return _argmax_ij(path)


def fit_max_envelope() -> tuple[float, float, float, float]:
    """sqrt(ell) fits of lead_pair_max, trail_pair_max from the saved fits csv."""
    rows = list(csv.DictReader(open(PAIR_FITS)))
    lead = next(r for r in rows if r["series"] == "lead_pair_max" and r["model"] == "sqrt")
    trail = next(r for r in rows if r["series"] == "trail_pair_max" and r["model"] == "sqrt")
    return float(lead["a"]), float(lead["b"]), float(trail["a"]), float(trail["b"])


def make_rules():
    a_p75, b_p75 = fit_lead_p75()
    a_lm, b_lm, a_tm, b_tm = fit_max_envelope()
    print(f"p75_leading fit:  i(ell) = round({a_p75:.4f}*sqrt(ell) + {b_p75:.4f}), j=1")
    print(f"max_envelope fit: i(ell) = round({a_lm:.4f}*sqrt(ell) + {b_lm:.4f}), "
          f"j(ell) = round({a_tm:.4f}*sqrt(ell) + {b_tm:.4f})")

    def rule_p75(ell: int) -> tuple[int, int]:
        i = round(a_p75 * sqrt(ell) + b_p75)
        return _clip(ell, i, 1)

    def rule_max_envelope(ell: int) -> tuple[int, int]:
        i = round(a_lm * sqrt(ell) + b_lm)
        j = round(a_tm * sqrt(ell) + b_tm)
        return _clip(ell, i, j)

    return {
        "current_argmax": rule_current,
        "p75_leading": rule_p75,
        "max_envelope": rule_max_envelope,
    }


# --------------------------------------------------------------------------- #
# live benchmark (ell<=27, ground truth available)
# --------------------------------------------------------------------------- #
def load_exact_grid(ell: int) -> dict[tuple[int, int], tuple[int, int, float]]:
    path = os.path.join(EXACT_DIR, f"ell{ell}.csv")
    grid = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            i, j = int(row["i"]), int(row["j"])
            fr = row["fraction"]
            grid[(i, j)] = (int(row["n_valid"]), int(row["denominator"]),
                             float(fr) if fr != "" else 0.0)
    return grid


def bench_live(rules: dict) -> list[dict]:
    rows = []
    for ell in LIVE_LADDER:
        t0 = time.time()
        exact_grid = load_exact_grid(ell)

        i0, j0 = rules["current_argmax"](ell)
        baseline_recs = find_first_binary_exhaustive_multi_seed(
            ell, i0, j0, BASELINE_SEEDS, time_budget=BASELINE_HARD_CAP)
        baseline_times = [r["seconds"] for r in baseline_recs if r.get("found")]
        baseline_time = float(np.median(baseline_times)) if baseline_times else BASELINE_HARD_CAP
        ceiling = max(PRUNE_FACTOR * baseline_time, MIN_CEILING)

        line = [f"ell={ell:>3} ceiling={ceiling:.4g}s"]
        for name, fn in rules.items():
            i, j = fn(ell)
            n_valid, denom, frac = exact_grid.get((i, j), (0, denominator(ell, i, j), 0.0))
            recs = find_first_binary_exhaustive_multi_seed(
                ell, i, j, list(range(N_SEEDS)), time_budget=ceiling)
            times = [r["seconds"] for r in recs if r.get("found")]
            success = float(np.mean([bool(r.get("found")) for r in recs]))
            median_ttf = float(np.median(times)) if times else None
            rows.append({
                "ell": ell, "rule": name, "i": i, "j": j,
                "n_valid_ground_truth": n_valid, "denominator": denom, "fraction": frac,
                "seeds": N_SEEDS, "success_frac": success, "median_ttf": median_ttf,
                "ceiling": ceiling,
            })
            ttf_s = f"{median_ttf:.4g}s" if median_ttf is not None else "None"
            line.append(f"{name}(i={i},j={j},gt_valid={n_valid}): "
                        f"succ={success:.2f} med_ttf={ttf_s}")
        print("  ".join(line) + f"  [{time.time()-t0:.1f}s wall]", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# closed-form reduction factor, full range
# --------------------------------------------------------------------------- #
def reduction_factors(rules: dict) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = {name: [] for name in rules}
    for ell in FULL_LADDER:
        d0 = denominator(ell, 0, 0) ** 2
        for name, fn in rules.items():
            i, j = fn(ell)
            d = denominator(ell, i, j) ** 2
            out[name].append((ell, d0 / d if d > 0 else float("nan")))
    return out


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
FIELDS = ["ell", "rule", "i", "j", "n_valid_ground_truth", "denominator",
          "fraction", "seeds", "success_frac", "median_ttf", "ceiling"]


def write_csv(rows: list[dict]) -> None:
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def make_plot(rows: list[dict], reductions: dict[str, list[tuple[int, float]]]) -> None:
    by_rule = {name: {r["ell"]: r for r in rows if r["rule"] == name} for name in COLORS}

    fig, (ax_t, ax_s, ax_r) = plt.subplots(1, 3, figsize=(18, 5.2))

    ax_t.set_title("Time to first Legendre pair (live, ell<=27)")
    for name in COLORS:
        xs = [e for e in LIVE_LADDER if by_rule[name].get(e, {}).get("median_ttf") is not None]
        ys = [by_rule[name][e]["median_ttf"] for e in xs]
        if xs:
            ax_t.semilogy(xs, ys, MARKERS[name] + "-", color=COLORS[name],
                          label=LABELS[name])
    ax_t.set_xlabel(r"$\ell$")
    ax_t.set_ylabel("seconds (log scale)")
    ax_t.grid(True, which="both", alpha=0.3)
    ax_t.legend(fontsize=8)

    ax_s.set_title(f"Success rate within ceiling ({N_SEEDS} seeds)")
    for name in COLORS:
        xs = [e for e in LIVE_LADDER if e in by_rule[name]]
        ys = [by_rule[name][e]["success_frac"] * 100 for e in xs]
        ax_s.plot(xs, ys, MARKERS[name] + "-", color=COLORS[name], label=LABELS[name])
    ax_s.set_xlabel(r"$\ell$")
    ax_s.set_ylabel("success rate [%]")
    ax_s.set_ylim(-5, 105)
    ax_s.grid(alpha=0.3)
    ax_s.legend(fontsize=8)

    ax_r.set_title("Space-reduction factor D(0,0)/D(i,j), closed-form")
    for name in COLORS:
        xs = [e for e, _ in reductions[name]]
        ys = [v for _, v in reductions[name]]
        ax_r.semilogy(xs, ys, MARKERS[name] + "-", color=COLORS[name],
                      label=LABELS[name], markersize=4)
    ax_r.axvline(27, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax_r.text(27.5, ax_r.get_ylim()[0], "exact | extrapolated", fontsize=7, color="gray")
    ax_r.set_xlabel(r"$\ell$")
    ax_r.set_ylabel("reduction factor (log scale)")
    ax_r.grid(True, which="both", alpha=0.3)
    ax_r.legend(fontsize=8)

    fig.suptitle("Restriction-rule comparison: argmax-yield vs p75-envelope vs max-envelope",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PLOT_PATH, dpi=130)
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rules = make_rules()

    print("\n=== ground-truth coverage + live search, ell=3..27 ===")
    rows = bench_live(rules)

    print("\n=== coverage check: does each rule's box contain a REAL pair? ===")
    bad = [r for r in rows if r["n_valid_ground_truth"] == 0]
    if bad:
        for r in bad:
            print(f"  EMPTY BOX: ell={r['ell']} rule={r['rule']} (i={r['i']},j={r['j']})")
    else:
        print("  all rule/ell cells contain at least one real pair (ground truth).")

    print("\n=== closed-form space-reduction factor, ell=3..57 ===")
    reductions = reduction_factors(rules)
    for ell in FULL_LADDER:
        vals = "  ".join(f"{name}={dict(reductions[name])[ell]:.3e}" for name in COLORS)
        print(f"  ell={ell:>3}: {vals}")

    write_csv(rows)
    make_plot(rows, reductions)
    print(f"\nwrote {CSV_PATH}\nwrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
