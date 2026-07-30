# Legendre Pair search via a run-length (RLE) reparametrization

A **measurement apparatus** for one question: *does representing a ±1 sequence by
its run-length array help a stochastic search for Legendre pairs, versus the
standard binary (±1) representation?* The goal is not to find a new pair — it is
to measure the reparametrization, on a regime (`L ≤ 25`) where exhaustive ground
truth is available.

See [`SPEC.md`](SPEC.md) for the full design spec this implements.

## Mathematical setting

`L` odd, `v ∈ {−1,+1}^L`, indices cyclic mod `L`. `(u, v)` is a **Legendre pair** iff

$$\mathrm{PAF}_u(s) + \mathrm{PAF}_v(s) = -2 \quad\text{for all } s = 1,\dots,L-1.$$

Both sequences are normalized to **row sum +1** (`P = (L+1)/2` ones, `M = (L−1)/2`
minus-ones); complementation is used up as a symmetry. PAF values are exact
integers and are the object we hash, pair, and optimize on. PSD is float and is
used **only** as a filter — never hashed.

## Packaging

Poetry is mandated for a *different* project in this workspace; here we follow
the repo convention of plain `python3`. Dependencies are declared in
[`pyproject.toml`](pyproject.toml) (PEP 621) and [`requirements.txt`](requirements.txt),
but nothing needs to be installed to run the code — just have NumPy, Numba, and
(for plots) Matplotlib. `pip install -e .` works if you want an editable install.

```
python3 -m pytest tests            # fast suite (L <= 19 exhaustive validation)
python3 -m pytest -m slow          # add the L in {21,23,25} brute-force checks
python3 lp_rle/exhaust.py          # validate RLE vs brute force + literature, L=3..25
python3 lp_rle/bench.py            # PAF crossover + six-config head-to-head
python3 lp_rle/bench.py --paf      # only the PAF crossover
```

## Module layout

| module | role |
|---|---|
| `conventions.py` | `P`, `M`, `(L−1)/2`, the magic constants, `check_odd` |
| `runs.py` | RLE ↔ ±1 bijection, Booth canonicalization, `rotate_runs`, `reverse_runs` |
| `paf.py` | three cross-checked PAF impls (`naive`/`fft`/`runs`) + PSD helpers |
| `enumerate.py` | exhaustive canonical run arrays; the `N(L,k)` orbit-count identity |
| `symmetry.py` | full LP equivalence group (shift/reversal/decimation/swap) on ±1 |
| `filter.py` | PSD test, exact-integer PAF keying, survivor collection |
| `match.py` | hash-based pairing of survivors into inequivalent LP classes |
| `bruteforce.py` | **independent** ±1 enumeration — the ground-truth cross-check |
| `exhaust.py` | RLE exhaustive driver + `validate(L)` (RLE vs brute force) |
| `delta.py` | exact integer delta kernels (the stochastic-search hot path) |
| `walk.py` | joint & two-stage walkers, RLE/binary move sets, SA/tabu/restart |
| `baseline.py` | thin factories building the six benchmark configurations |
| `bench.py` | PAF crossover + head-to-head report and headline plot |
| `litdata.py` | published inequivalent-LP counts (Fletcher–Gysin–Seberry 2001) |

## Correctness backbone

Two independent checks gate everything (spec §3):

1. **Self-validation.** `bruteforce.py` enumerates *all* `C(L,P)` sign sequences
   directly in ±1 space — sharing **no** code with the RLE layer — and must agree
   with the RLE driver on the set of feasible rotation classes, the exact
   rotation-multiplicity of survivor counts, and the set of LP equivalence
   classes, for every odd `L` from 3 to 25.
2. **Literature cross-check.** The inequivalent-LP count must match the published
   NGL-pairs table. Verified: `L=3..19 → 1,1,1,1,2,4,8,7,9` (and `21,23,25 →
   22,28,46` via `-m slow`).

## The hypothesis under test

The RLE does **not** create a new micro-neighbourhood: `transfer(i,i+2)` is a
2-flip in two adjacent positions, and a long-range transfer is a chain of
2-flips. What the RLE *does* supply is (i) a structured **bias** over the 2-flip
neighbourhood and (ii) genuinely non-local **macro-moves** — `swap_runs`,
band transfer, and singleton relocation (`split`/`merge`) — that have no natural
counterpart in ±1 space. **Those macro-moves are the hypothesis.** The walker
logs proposal counts, acceptance rates, and mean `Δf` separately for micro and
macro moves: if macro-moves are never accepted, that is the answer.

The baseline (`representation="binary"`) is the *same* search — same residual
state, same delta kernel, same metaheuristic, same budget accounting — driven by
the plain ±1 2-flip. Without it the experiment says nothing.

## Non-goals (spec §6)

No decimation in the RLE layer; no float hashing; the `O(k·L)` run-merge PAF is a
reference/benchmark subject, never a production path; the objective is never
recomputed from scratch inside the walk. Deferred: CAT/prenecklace generation,
parallelism, compiled extensions, Djoković–Kotsireas compression, `L > 25`.
