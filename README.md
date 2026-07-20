# Legendre pairs — code overview and log of ideas tested

A **Legendre pair** is a pair of $\pm 1$ sequences $A, B$ of the same odd length
$\ell$ whose periodic autocorrelations cancel to a constant:

$$\mathrm{PAF}_A(s) + \mathrm{PAF}_B(s) = -2 \qquad \text{for all shifts } s = 1, \dots, \tfrac{\ell-1}{2},$$

where $\mathrm{PAF}_L(s) = \sum_{k=1}^{\ell} a_k\,a_{k+s}$ (indices mod $\ell$).
Legendre pairs are used to build Hadamard matrices; the **smallest open order is
$\ell = 115$**. This folder explores how far one can push a search for them,
starting from exhaustive enumeration and moving to local-search optimization.

Reference: S. M. Perera, I. S. Kotsireas, *A low-complexity algorithm to search
for Legendre pairs*, Linear Algebra and its Applications **721** (2025) 149–171.

Everything is Python 3 + NumPy. Run with `python3` (not `python`).

---

## Repository layout

```
legendre_pairs/
├── src/          # importable core: checker + search engines
│   ├── legendre.py          # checker + exhaustive PAF-bucket search
│   ├── local_search.py      # objective + local-search strategies
│   ├── parallel_search.py   # multiprocessing over restarts (NEW)
│   └── incremental_paf.py   # O(ell) rank-2 PAF update, batched (NEW)
├── experiments/  # analyses & benchmarks (each writes into results/)
│   ├── benchmark_legendre.py  benchmark_methods.py  analyze_landscape.py
│   ├── measure_flip_scaling.py  continuous_descent.py  collect_pairs.py
├── tests/        # test_legendre.py
└── results/      # generated .png / .csv / found_pairs.md
```

Scripts in `experiments/` and `tests/` add `../src` to `sys.path` at import time,
so run them by path (e.g. `python3 experiments/benchmark_methods.py`) from the
repo root — no install or `PYTHONPATH` needed.

---

## The Python files

### `legendre.py` — checker + exhaustive search
The foundational module.

- **`is_legendre_pair(a, b)`** — verifies the defining property directly
  (returns `(bool, reason)`); rejects even lengths, entries outside $\{-1,+1\}$,
  length mismatches, and optionally checks the sum-$=1$ normalization.
- **`paf`, `paf_profile`** — the reference $O(\ell^2)$ autocorrelation.
- **Exhaustive search by PAF-vector bucketing.** Instead of the naive
  $O(4^\ell)$ double loop over pairs, it enumerates only sequences whose row-sum
  is $\pm 1$ (see *Idea 1*), buckets each by its PAF vector $v$, and matches each
  bucket against its complement $-2 - v$. Three entry points:
  - `find_legendre_pairs` (serial, builds all buckets then matches),
  - `find_legendre_pairs_parallel` (splits bucket-building across processes with
    `ProcessPoolExecutor`, striping the candidate index space — bypasses the GIL),
  - `find_legendre_pairs_incremental` (checks each candidate against already-seen
    complements, so with `limit=1` it **early-stops** the instant a pair exists).
- **CLI:** `python3 src/legendre.py ELL [-n N] [-a] [-j JOBS] [-P] [-1]`
  (`-1` = incremental early-stop, `-P` = progress bar on stderr).
- `_Progress` is a dependency-free stderr progress bar throttled to ~10 fps.

### `test_legendre.py` — correctness tests
Brute-force cross-checks: the bucketing search must return exactly the same set
of pairs as a naive scan; the parallel path must match the serial path; small
odd $\ell$ each yield a known pair; `limit` is respected; malformed inputs are
rejected. All pass.

### `benchmark_legendre.py` — cost of brute force, extrapolated to $\ell=115$
Times the exhaustive PAF scan for $\ell = 5, 7, \dots, 25$ and extrapolates.
The key pedagogical point: the candidate count is $\binom{\ell}{(\ell-1)/2}
\sim 2^\ell/\sqrt{\ell}$ — **exponential**.

- On a **semilog-y** plot an exponential is a straight line (correct model).
- On a **log-log** plot it looks curved, so a power-law line fit there
  **systematically under-estimates** large $\ell$.

Both are drawn (`benchmark_legendre_pairs.png`). The principled "complexity
model" $t \approx k \cdot \binom{\ell}{(\ell-1)/2}\cdot\ell\cdot\tfrac{\ell-1}{2}$
predicts $\ell = 115$ takes on the order of $10^{22}$ years — hopelessly
infeasible. **This motivates abandoning brute force.**

### `local_search.py` — the optimization approach
Turns the defining property into an objective to minimize:

$$E(A,B) = \sum_{s=1}^{(\ell-1)/2}\big(\mathrm{PAF}_A(s)+\mathrm{PAF}_B(s)+2\big)^2 \;\ge\; 0,
\qquad E = 0 \iff (A,B)\text{ is a Legendre pair}.$$

- **Move:** swap one $+1$ with one $-1$ inside $A$ or $B$. This preserves each
  row sum, so starting both at sum $=1$ keeps every candidate feasible; swaps
  connect all sum-$1$ sequences.
- PAF vectors are recomputed by **FFT** (`paf_vector`), so each move costs
  $O(\ell\log\ell)$.
- **Acceptance strategies** (the ideas tested, in order of sophistication):
  - `greedy` — accept only if $E$ strictly decreases. Stalls in local minima.
  - `sideways` — also accept equal-$E$ moves (walks plateaus).
  - `anneal` — Metropolis: accept a worse move with probability
    $e^{-\Delta E / T}$, $T$ cooled geometrically. Reliably reaches $E=0$.
  - `threshold` — *the magnitude-threshold heuristic:* reroll a fresh random
    configuration whenever $E > 20\ell$, otherwise do greedy descent (with a
    stall-reroll safeguard). Restart-heavy.
  - `basinhop` — *iterated local search / basin-hopping:* descend to a local
    min, then repeatedly {snapshot, kick by a few random swaps, re-descend},
    accepting the new basin if no worse; the kick grows on a run of rejections.
    Reuses structure across basins instead of rerolling everything.
- **CLI:** `python3 src/local_search.py ELL -s STRATEGY [-r RESTARTS] [-n STEPS]
  [--t0 T0] [--t-end TEND] [--seed S]`.

### `parallel_search.py` — the restarts, across CPU cores
The restarts of `local_search.search` are independent random walks that all stop
the instant one hits $E=0$, so they are **embarrassingly parallel**. This module
splits the `restarts` budget into chunks, farms them to a `multiprocessing.Pool`
(one interpreter per core, bypassing the GIL), and returns the first solution —
`terminate()`-ing the still-running workers so the process exits immediately
instead of blocking on a straggler chunk. Task $i$ is seeded `seed + 7919*i`
for reproducibility. This
is a **constant-factor** win (≈ number of physical cores); it does not touch the
exponential scaling.

- **CLI:** `python3 src/parallel_search.py ELL -s STRATEGY [-r RESTARTS]
  [-n STEPS] [-j JOBS] [--seed S]`.

### `incremental_paf.py` — O(ell) PAF update (prototype for GPU populations)
Recomputing $\mathrm{PAF}(1..\tfrac{\ell-1}{2})$ from scratch after a flip costs
$O(\ell\log\ell)$ (FFT). But a single-entry flip $a_p\to-a_p$ changes PAF by a
closed-form rank-2 update

$$\mathrm{PAF}(s)\;\mathrel{+}=\;-2\,a_p\big(a_{(p+s)\bmod\ell}+a_{(p-s)\bmod\ell}\big),$$

costing $O(\ell)$; a swap is two such flips applied in sequence (so neighbour
interactions stay exact). The module provides a single-sequence version, a
**batched `(W, ell)` version over $W$ walkers**, a verification that the
incremental PAF matches the FFT PAF exactly, and a throughput micro-benchmark.

- **Finding:** on CPU/NumPy the batched incremental update is actually *slower*
  than an FFT recompute (≈ 0.3–0.6×) — NumPy's per-op overhead and a very
  optimized `rfft` dominate at these sizes. Its purpose is the **GPU population**
  route: the update is a few gathers + a multiply-add, so thousands of walkers
  fuse into one kernel while the tiny per-walker FFT is latency-bound. That is
  what enables parallel tempering / population annealing at scale.
- **CLI:** `python3 src/incremental_paf.py [--verify] [--bench]
  [--ells ...] [--walkers ...]`.

### `benchmark_methods.py` — exhaustive vs. local search
For each $\ell$ it reports the exhaustive scan time (one number) against each
stochastic method's **success rate** and **median time-to-solve** over many
trials, plus a two-panel plot (`benchmark_methods.png`). This is where the
methods are compared head-to-head.

### `analyze_landscape.py` — why plain descent fails
Diagnoses the objective landscape at a fixed $\ell$:

1. **Steepest-descent census** — from many random starts, follow the single
   most-improving swap to a local minimum, and histogram where they land.
2. **Trajectory + plateau stats** — records $E$ vs. iteration for descent
   (stalls) and annealing (escapes), measuring plateau lengths.

Findings at $\ell = 23$: steepest descent reaches $E=0$ only **~2%** of the
time; local minima are **quantized** (clustered at $E = 32, 64$) with **very
long plateaus**. So the bottleneck is *escaping* plateaus, not descending —
which is exactly why annealing / restarts beat pure gradient descent.

### `measure_flip_scaling.py` — how big is one flip?
Tests the conjecture "each flip changes $E$ by about $\sqrt{\ell}$." It samples
many random swaps, records the exact $\Delta E$, and decomposes it as

$$\Delta E = \underbrace{2\langle r,\Delta P\rangle}_{\text{gradient step}}
          + \underbrace{\lVert\Delta P\rVert^2}_{\text{curvature}},
\qquad r(s) = \mathrm{PAF}_A(s)+\mathrm{PAF}_B(s)+2,$$

then fits $\mathrm{rms}\,|\Delta E| \sim \ell^{\,p}$ across
$\ell \in [11, 201]$ (`flip_scaling.png`).

### `continuous_descent.py` — does continuous gradient descent work?
Relaxes the $\pm 1$ constraint to real vectors and runs true gradient descent on
the smooth quartic $E$, with the proposed step $x \leftarrow x - \frac{1}{2\ell}
\nabla E$ (analytic gradient $\partial E/\partial a_m = \sum_s 2r(s)\,(a_{m+s} +
a_{m-s})$), then **rounds to signs** and checks. Options: `--normalize` (fixed
step length), `--project clip|sphere` (keep iterates bounded), `--seed`.

### `collect_pairs.py` — save every found pair to Markdown
Runs each method over a range of $\ell$ and writes `found_pairs.md`: **one table
per method** with columns $\ell$, the pair (A / B as $\pm$ strings), search time,
and the algorithm parameters. Every tabulated pair is re-verified by
`is_legendre_pair`. `--seed` makes the whole sweep reproducible.

---

## Overview of ideas tested

| # | Idea | Where | Verdict |
|---|------|-------|---------|
| 1 | **Reduce to row-sum $=1$.** Every Legendre pair has $(\sum a)^2+(\sum b)^2 = 2$, so both sums are $\pm 1$; work with sum-$1$ canonical reps. | `legendre.py` | ✅ Shrinks the space and keeps swap moves feasible. |
| 2 | **Meet-in-the-middle bucketing.** Bucket candidates by PAF vector $v$, match against complement $-2-v$, instead of an $O(4^\ell)$ pair loop. | `legendre.py` | ✅ Correct + far faster than naive, but still exponential. |
| 3 | **Parallelize + incremental early-stop.** Stripe bucket-building across processes; stop the moment one pair is found. | `legendre.py` | ✅ Linear speed-up; early-stop is the fast way to "just find one." |
| 4 | **Is brute force viable for $\ell=115$?** Time small $\ell$, extrapolate with the *right* (exponential) model. | `benchmark_legendre.py` | ❌ ~$10^{22}$ years. Infeasible — pivot to optimization. |
| 5 | **Objective + local search.** Minimize $E=\sum(\mathrm{PAF}_A+\mathrm{PAF}_B+2)^2$ via swap moves. | `local_search.py` | ✅ Solves $\ell\le 21$ instantly; annealing $\gg$ greedy. |
| 6 | **Can classical gradient / quadratic methods help?** $E$ is a degree-4 polynomial over the boolean cube. | `analyze_landscape.py` | ❌ Descent is trivial; the hard part is escaping quantized plateaus. Continuous/QP relaxations don't address that. |
| 7 | **Flip-size conjecture: $\Delta E \sim \sqrt{\ell}$?** | `measure_flip_scaling.py` | ❌ Measured exponent $p \approx 1.06$ — **linear in $\ell$**, not $\sqrt\ell$. Both the gradient and curvature terms are $O(\ell)$. |
| 8 | **Magnitude-threshold heuristic** (reroll if $E>20\ell$, else descend) and **basin-hopping** (kick + re-descend). | `local_search.py`, `benchmark_methods.py` | ⏳ Implemented; competitive with annealing on $\ell\le 21$ (all 100% success). Discriminating them needs the harder $\ell\ge 23$ regime. |
| 9 | **Continuous gradient descent** on the relaxed quartic, step $x\!\leftarrow\!x-\frac{1}{2\ell}\nabla E$, then round to signs. | `continuous_descent.py` | ❌ Raw step **diverges** (gradient $\sim x^3$); stable variants converge to non-binary critical points → rounding gives $E>0$. Found 0 pairs (bar a trivial $\ell=5$ fluke). |
| 10 | **Parallelize the restarts** across CPU cores; return the first $E=0$ and cancel the rest. | `parallel_search.py` | ✅ Near-linear (core-count) speed-up, reproducible per seed. Constant factor only — scaling unchanged. |
| 11 | **Incremental $O(\ell)$ PAF update** (rank-2 per flip), batched over $W$ walkers. | `incremental_paf.py` | ⚖️ Verified exact vs. FFT. Loses to `rfft` on CPU/NumPy; the point is GPU population-parallelism (tempering / population annealing). |

### Selected quantitative results

- **Flip scaling** ($\ell\in[11,201]$): $\mathrm{rms}\,|\Delta E|/\ell$ stays
  flat at $\approx 8$–$10$ while $\mathrm{rms}\,|\Delta E|/\sqrt{\ell}$ climbs
  from 29 to 139 → the true scaling is **linear**, fitted exponent $1.06$.
- **Method benchmark** ($\ell = 5$–$21$, 12 trials): greedy dips below 100%
  from $\ell=19$; `sideways`, `anneal`, `threshold`, `basinhop` all stay at
  100%, with basin-hopping and threshold marginally fastest at $\ell=21$
  (median ≈ 0.18–0.21 s vs. exhaustive scan 5.7 s).
- **Landscape** ($\ell=23$): ~2% steepest-descent success; minima quantized at
  $E=32, 64$; plateaus up to thousands of steps.

### Where this points
The productive direction is **not** continuous optimization but better
plateau-escape: simulated annealing with restarts, and the newer
`threshold` / `basinhop` iterated-local-search variants, run as a **parallel
population**. The cheap, certain win is `parallel_search.py` (restarts across
cores); the ambitious route is a GPU population using the `incremental_paf.py`
update, enabling parallel tempering / population annealing. Neither changes the
exponential scaling — the next discriminating experiment is still a stress test
at $\ell = 23$–$31$, where success rates finally separate the methods.

---

## Quick start

Run everything from the repo root.

```bash
# core search (src/)
python3 src/legendre.py 13 -1                 # find one pair of length 13, fast
python3 src/legendre.py 13 -a -j4 -P          # all ordered pairs, 4 procs, progress
python3 src/local_search.py 21 -s anneal      # simulated annealing
python3 src/local_search.py 21 -s basinhop    # iterated local search
python3 src/parallel_search.py 21 -s anneal -j4   # restarts across CPU cores
python3 src/incremental_paf.py --verify --bench   # O(ell) PAF update: check + time

# experiments (each writes into results/)
python3 experiments/benchmark_legendre.py --max 21   # brute-force cost + ell=115 extrap.
python3 experiments/benchmark_methods.py  --max 21   # exhaustive vs local search
python3 experiments/analyze_landscape.py  --ell 23   # ruggedness / plateaus
python3 experiments/measure_flip_scaling.py          # flip-size scaling test
python3 experiments/continuous_descent.py --normalize # continuous GD (does it work? no)
python3 experiments/collect_pairs.py --min 5 --max 21 # write results/found_pairs.md

# tests
python3 tests/test_legendre.py                # (or: python3 -m pytest tests/)
```

All randomised scripts take `--seed` (default 0), so every run above is
reproducible.
