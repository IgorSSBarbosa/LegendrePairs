# Legendre Pair search via a run-length (RLE) representation — implementation spec (v2)

Spec for Claude Code. Scope decisions now fixed:

- **Regime: validation only, `L <= 25`.** No parallelism, no compiled extensions beyond Numba.
  The search space is tiny (`C(25,13)/25 = 208012` rotation classes), so correctness and
  measurement quality matter far more than throughput.
- **Stack: Python 3 + NumPy + Numba** (`@njit` on the hot kernels). (This workspace uses plain
  `python3`; Poetry is used in a *different* project. Deps are declared in pyproject/requirements.)
- **Stochastic search: implement BOTH formulations** (two-stage and joint), and benchmark both
  against a baseline that uses the **standard binary (`±1`) representation** with the same
  metaheuristic and the same budget accounting.
- **Objective evaluated by exact integer deltas, never by full recomputation** in the inner
  loop. This is the structural core of the design; see §4.

The scientific goal is to **measure whether the run-length reparametrization helps**, not to
find a new Legendre pair. Build the measurement apparatus first.

---

## 0. Mathematical setting (use these exact conventions)

`L` odd, `v ∈ {−1,+1}^L`, indices cyclic mod `L`.

    PAF_v(s) = sum_{i} v[i] * v[(i+s) mod L]
    vhat[k]  = sum_j v[j] * exp(2*pi*I*j*k/L)
    PSD_v(k) = |vhat[k]|^2

**(u, v) is a Legendre pair of length L** iff

    PAF_u(s) + PAF_v(s) = -2     for all s = 1..L-1                        (LP-PAF)

Consequences encoded as invariants and unit tests:

- `PAF_v(0) = L`; `sum_{s=0}^{L-1} PAF_v(s) = (sum_j v[j])^2`.
- (LP-PAF) forces `(sum u)^2 = (sum v)^2 = 1`. Normalize both sequences to row sum `+1`.
- With row sum `+1`: `P = #{+1} = (L+1)/2`, `M = #{-1} = (L-1)/2`.
- Fourier form: `PSD_u(k) + PSD_v(k) = 2L + 2` for `k = 1..L-1`.
- **PSD test** (one-sequence filter): `PSD_v(k) <= 2L + 2` for all `k != 0`.
- `PAF_v(s) = PAF_v(L-s)`, `PSD_v(k) = PSD_v(L-k)`: store only `s, k = 1..(L-1)/2`.
- **`PAF_v(s) ≡ L (mod 4)`.**

**Exactness rule.** Use PSD (float, tolerance) **only** as a filter. Use PAF (exact integers)
for the pairing test, for hashing, and for the whole stochastic objective. Never hash floats.

---

## 1. The run-length representation

Anchor at the start of a positive run; the number of runs is even, `2k`:

    r = (r[0], ..., r[2k-1]),  r[i] >= 1,  even = +1 runs, odd = -1 runs
    sum_{i even} r[i] = P,  sum_{i odd} r[i] = M

Pair up `w = ((r[0],r[1]), ...)`, length `k`. Rotations of `w` = the `L` cyclic shifts of `v`,
collapsing to `k` distinct arrays. Odd rotations of `r` = shift∘negation (excluded by the +1
normalization). **Canonical form = lexicographically least rotation of `w`, via Booth, O(k).**
Every orbit has size exactly `L` (aperiodicity: `L` odd ⇒ `L/d >= 3`).

    N(L, k) = (1/k) * C(P-1, k-1) * C(M-1, k-1)
    sum_k N(L, k) = C(L, P) / L

**Decimation** has no clean action on runs; do decimation/reversal/swap dedup as
post-processing on `±1` arrays. `reverse_runs` is implemented via round-trip through `±1`.

---

## 2. PAF/PSD — three cross-checked implementations

`paf_naive` (O(L^2), production at L<=25), `paf_fft` (cross-check / larger L), `paf_runs`
(run-merge reference + benchmark subject; report the crossover in `k`). Plus `psd_from_paf`,
`psd_test`, `psd_excess`.

---

## 3. Exhaustive driver + validation

Enumerate canonical runs → PSD filter → hash-match by exact integer PAF tuple → dedup modulo
the full group. **Self-validation:** an independent `±1` brute force (no RLE code) must agree
on survivors, LP count, and LP class set for every odd `L <= 25`. **Literature cross-check:**
match the published inequivalent-LP counts (Fletcher–Gysin–Seberry 2001).

---

## 4. Stochastic search — delta evaluation is the architecture

Joint residual `e[s] = PAF_u(s) + PAF_v(s) + 2`, objective `f = 2*sum e[s]^2`, `f=0 ⇔ LP`.
`sum_s e[s] = 0` and `e[s] ≡ 0 (mod 4)` (store `E = e/4`). 2-flip delta kernel is O(L) exact
integer; on acceptance `e += DeltaPAF`. No FFT in the inner loop.

**RLE move set:** `transfer` (micro, = 2-flip), `swap_runs`, long-range/band transfer,
`split`/`merge` (macro). The macro-moves are the hypothesis under test; instrument micro vs
macro proposal/acceptance/mean-Δf. **Baseline:** plain `±1` 2-flip with the identical kernel,
state, metaheuristic, and budget. Two-stage: stage-1 minimizes `psd_excess`, harvests a pool,
stage-2 hash-matches. Metaheuristics: SA / tabu / random restart, move-set-agnostic.

---

## 5. Benchmark protocol

Six configs = {joint, two-stage} × {RLE, binary}. Metrics over ≥50 seeds: time-to-first-LP,
success fraction, distinct LP classes/CPU-sec, F distribution, moves/sec, micro vs macro
acceptance, PSD pass rate per L/k. Ladder `L = 13..25`. Headline: fraction of total LP classes
recovered per unit time (possible because exhaustive ground truth exists).

---

## 6. Non-goals

No search-space claim beyond the rotation quotient (factor `L`); no decimation in the RLE
layer; no float hashing; run-merge PAF is not a production path; never recompute the objective
inside the walk; assert `L` odd everywhere. Deferred: CAT/prenecklace generation, parallelism,
compiled extensions, Djoković–Kotsireas compression, `L > 25`.

---

## 7. Milestones

1. `runs.py` + property tests + `N(L,k)`.  2. `paf.py` three-way + crossover.  3. exhaustive +
independent brute force + literature.  4. `delta.py` integer kernels vs full recompute.
5. `walk.py` (both formulations) + `baseline.py`.  6. `bench.py` head-to-head report.

## 8. Reading list

Fletcher–Gysin–Seberry (2001); Chiarandini–Kotsireas–Koukouvinos–Paquete (2008); Djoković–
Kotsireas (2015); Djoković–Kotsireas–Recoskie–Sawada (charm bracelets); "Determining the group
that sends each Legendre pair to an equivalent Legendre pair".
