# RLE reparametrization experiment (`rle_experiment/`)

A correctness-first study of one question: **does a run-length (RLE)
reparametrization of $\pm1$ sequences help stochastic search for Legendre
pairs, compared with the plain binary encoding?** Everything here is built so
the answer can be *measured* rather than asserted — validated at lengths
$\ell\le 25$(+) where an exhaustive ground truth exists.

## What a Legendre pair is (recap)

A Legendre pair (LP) of odd length $\ell$ is $(u,v)\in\{-1,+1\}^\ell\times\{-1,+1\}^\ell$,
normalized to row sum $+1$ (so $P=(\ell+1)/2$ ones, $M=(\ell-1)/2$ minus-ones),
whose periodic autocorrelations cancel:

$$\mathrm{PAF}(u,s)+\mathrm{PAF}(v,s)=-2\qquad\text{for all }s\not\equiv 0\pmod\ell.$$

The joint objective driving every solver is

$$f(u,v)=\sum_{s=1}^{\ell-1}\big(\mathrm{PAF}(u,s)+\mathrm{PAF}(v,s)+2\big)^2,
\qquad f=0\iff (u,v)\text{ is an LP}.$$

---

## Strategies used

### 1. Two representations (the thing under test)

- **`binary`** — the control. A single uniform $\pm1$ **2-flip** move (swap a
  random $+1$ with a random $-1$), which preserves the row-sum constraint. This
  is standard binary local search.
- **`rle`** — the treatment. Moves act on the *run-length encoding* of the
  sequence, split into two flavors:
  - **micro** — a boundary 2-flip that nudges a single run edge (local, fine);
  - **macro** — run-level `swap` / `band` / `split-merge` moves that restructure
    many coordinates at once (large, structural).

  The hypothesis is that macro moves tunnel between basins that binary 2-flips
  can only cross one coordinate at a time.

### 2. Two stochastic solvers

- **`sa` — simulated annealing.** The `JointWalker` Metropolis walk on $f$ with
  geometric cooling $T \leftarrow \texttt{cooling}\cdot T$ from $T_0$.
- **`basinhop` — basin-hopping.** Each outer *hop*: greedy **descent** to a local
  minimum using `propose_local` moves, a **kick** of `n_kick` `propose_kick`
  moves accepted blindly, a second descent, then a **Metropolis accept** of the
  new minimum over the old one. For `rle`, descent = micro, kick = macro — the
  natural pairing the representation was designed for. For `binary`, both flavors
  fall back to the single 2-flip, which is exactly textbook binary basin-hopping.

This gives a clean $2\times2$ grid — `{sa, basinhop} × {rle, binary}` — with the
binary column as a control for each solver.

### 3. Exact incremental bookkeeping

Both solvers maintain the residual vector $e = \mathrm{PAF}(u)+\mathrm{PAF}(v)+2$
and scalar $f=\langle e,e\rangle$ **incrementally** as moves are applied, never
recomputing from scratch inside the hot loop. A regression test
(`test_basinhop_bookkeeping_matches_recompute`) asserts the incremental $(e,f)$
equals a from-scratch recompute after a full run, for both representations — so
speed never buys silent drift.

### 4. Exhaustive ground truth + parallelization

`exhaust.py` enumerates all inequivalent LP classes at a given $\ell$ by walking
canonical run-compositions and PSD-filtering survivors before exact PAF
verification. `parallel.py` partitions that enumeration into $(k,a)$ tasks in
**serial order** and runs them through `multiprocessing.Pool.map` (order-
preserving), so the merged result is **byte-identical** to the serial run — the
parallelism is a pure speedup with zero semantic risk. Tests
(`test_parallel.py`) freeze both survivor buckets and class dicts to bytes and
assert serial $=$ parallel at $\ell=3\ldots17$, including the `workers=1`
fast-path.

### 5. A complete invariant for dataset comparison

The hardest measurement problem: **an inequivalent class has no canonical
representative on paper.** The same class appears under independent cyclic
shifts, independent reversal, a common decimation $d\in(\mathbb{Z}/\ell)^*$, and
the swap $u\leftrightarrow v$. So two datasets cannot be compared by string-
matching stored sequences. `symmetry.canonical_pair(u,v)` solves this by
returning **the same bytes key for every pair in an orbit** and different keys
for different orbits. Each dataset collapses to a *set of keys*; then

$$\text{equivalent as collections}\iff\text{equal key sets},$$

and set differences pinpoint exactly which classes are unique to each side.
`compare.py` wraps this with convention-robust parsing (0/1 or $\pm$), row-sum
normalization, and a PAF re-verification of every stored pair.

### 6. Triangulated adjudication of count discrepancies

When our exhaustive counts disagreed with the literature (FGS-2001) and a
friend's database, the discrepancy was attacked three independent ways so no
single tool could be the point of failure:

- **canonical invariant** (Strategy 5) — key-set differences;
- **pairwise brute force** — re-derive equivalence *without* `canonical_pair`,
  explicitly looping common-decimation × independent-shift × independent-reversal
  × swap × negation, testing each "ours-only" pair against every friend pair;
- **full orbit-universe membership** — materialize the *entire* friend orbit
  universe ($2.2\times10^{7}$ keys at $\ell=31$, $2.5\times10^{7}$ at $\ell=33$)
  and test direct byte-membership.

All three agree, which is the whole point of running all three.

---

## An essay: what the comparisons showed

Three independent methods — the canonical invariant, the pairwise brute force,
and full orbit-universe membership — plus an *exact* 829-way agreement with
everyone at $\ell=35$, unanimously establish that our `canonical_pair` invariant
is genuine and the extra classes we find are **real, not artifacts of an
incomplete symmetry group**. That agreement matters more than any single count:
when a from-scratch equivalence test with no shared code reproduces the same
verdict as the invariant, the invariant is vindicated.

The discrepancies then decompose into two structurally different causes.

At **$\ell=31$** (ours 201 / FGS 200 / friend 199) the gap is a *counting
convention*. Length $31=2^5-1$ is a Mersenne length, so $m$-sequences exist:
their PAF is $\equiv-1$ at every nonzero shift, which means **any** pair of
$m$-sequences — including the diagonal $u=v$ — is automatically a valid LP. All
four of our $m$-sequence-pair classes are legitimate. The friend database
excludes *all* diagonal self-pairs (0 vs our 2), while the two non-diagonal
$m$-sequence classes match exactly. So the entire $\ell=31$ gap is the question
of whether self-paired LPs count — and these can only occur at Mersenne lengths,
which is precisely why $\ell=33,35$ show none.

At **$\ell=33$** (ours 287 / FGS 284 / friend 284) the three extras are *ordinary
generic* LPs: $u\neq v$, not $m$-sequences, not self-paired. Both FGS-2001 and
the friend lack them. The most economical explanation is an FGS-2001 undercount
— unsurprising for 2001-era compute — reinforced by our exact agreement with
both other sources at the larger $\ell=35$. The friend database additionally
looks independently incomplete: it misses one ordinary LP at $\ell=13$ and sits
one below FGS at $\ell=31$ beyond the diagonal issue.

**Net verdict:** of the three datasets, our exhaustive search is the most
complete, and none of our extras is spurious. The discrepancy that first looked
like a bug turned out to be one convention boundary ($\ell=31$ self-pairs) and
one historical undercount ($\ell=33$).

A note on the literature adjudication: `kotsireas2021.pdf` ("Legendre pairs of
lengths $\ell\equiv 0\pmod 3$") turned out **not** to cover $31\equiv1$ or
$35\equiv2\pmod3$; its contribution is existence at large lengths plus a list of
open lengths $<200$, not a small-$\ell$ enumeration table. So it neither
confirms nor refutes our $\ell=31/33$ counts — it is simply out of scope, which
is itself a useful thing to have pinned down.

---

## Head-to-head results: `{sa,basinhop} × {rle,binary}`

The measurement the project exists for, over $\ell\in\{13,15,17,19,21\}$, 40
seeds, an equal 20 000-proposal budget (`bench.py --solvers`). The trustworthy
metrics are **success rate** and **classes recovered**, both fixed on the *step*
budget — so they are unaffected by CPU contention. (Wall-time `cls/s` was noisy
here because an $\ell=37$ exhaustive run was using every core, so it is omitted.)

**Success rate** (fraction of 40 seeds that found an LP):

| $\ell$ | sa/rle | sa/binary | basinhop/rle | basinhop/binary |
|---|---|---|---|---|
| 13 | 0.97 | 1.00 | 1.00 | 1.00 |
| 15 | 0.82 | 1.00 | 1.00 | 1.00 |
| 17 | 0.62 | 1.00 | 1.00 | 1.00 |
| 19 | 0.17 | 0.95 | 0.80 | **0.97** |
| 21 | 0.07 | 0.40 | 0.50 | **0.70** |

**Distinct LP classes recovered** (of the total at that $\ell$):

| $\ell$ | total | sa/rle | sa/binary | basinhop/rle | basinhop/binary |
|---|---|---|---|---|---|
| 13 | 4 | 4 | 4 | 3 | 3 |
| 15 | 8 | 7 | 7 | 7 | 7 |
| 17 | 7 | 6 | 6 | 5 | 6 |
| 19 | 9 | 5 | 7 | **8** | 7 |
| 21 | 22 | 2 | 9 | 14 | **17** |

### Verdict (an essay)

The result is largely **negative for RLE**, and that is a clean finding. Three
things fall out, in decreasing order of evidential strength.

1. **RLE actively *hurts* under plain simulated annealing.** `sa/rle` decays
   monotonically ($0.97\to0.82\to0.62\to0.17\to0.07$) while `sa/binary` stays
   near-perfect until $\ell=21$. The mechanism is visible in the macro-move
   acceptance rate, which falls $0.37\to0.04$ as $\ell$ grows: under a
   Metropolis-only walk the big structural moves are almost always uphill in
   $f$, so as $T$ cools they are frozen out, leaving only micro boundary-flips —
   a strictly *worse* explorer than the binary 2-flip. The reparametrization's
   whole selling point (macro moves that tunnel between basins) is unusable in
   the SA formulation.

2. **Basin-hopping rescues RLE — because it is the only formulation that can
   accept a macro move.** The kick→descend→Metropolis-on-the-minimum structure
   accepts a structural jump *blindly* and only then judges the resulting basin,
   so macro moves finally do work. The rescue is dramatic at $\ell=21$:
   `sa/rle` $0.07\to$ `basinhop/rle` $0.50$. This says the *solver* matters far
   more than the *representation* — switching sa→basinhop buys more than
   switching binary→rle ever does.

3. **But RLE still does not beat binary.** At the only genuinely discriminating
   length, $\ell=21$, the ranking is `basinhop/binary` (0.70, 17/22) $>$
   `basinhop/rle` (0.50, 14/22) $>$ `sa/binary` (0.40) $\gg$ `sa/rle` (0.07). The
   single best configuration across the whole ladder is **plain binary under
   basin-hopping**. RLE's one flicker of an edge is *diversity* at $\ell=19$
   (8/9 classes vs binary's 7/9) — it sometimes reaches different basins — but
   that does not translate into a higher solve rate.

**Bottom line:** RLE reparametrization is not worth it here. Binary 2-flips
under basin-hopping dominate, and RLE only stops being harmful once you adopt the
one solver whose structure tolerates its moves. The honest headline is
**solver ≫ representation**, with RLE at best neutral and at worst (under SA) a
real regression.

**Caveats.** 40 seeds gives $\approx\pm0.08$ standard error near $p=0.5$, so the
$\ell=21$ gap $0.50$ vs $0.70$ is $\sim2\sigma$ — suggestive, reinforced by the
monotone trend across $\ell$, but a larger rerun would firm it up. `macroAcc`
means different things per solver (hop-acceptance for basinhop vs move-acceptance
for sa), so it is not comparable across the solver columns. Raw rows are in
`results/head_to_head_solvers.csv`; the recovered-fraction plot is
`results/headline_recovery_solvers.png`.

---

## How to run it

```bash
# stochastic search: pick a solver and a representation
python3 -m lp_rle.search --solver sa       --rep rle    --L 19 --seeds 10
python3 -m lp_rle.search --solver basinhop --rep binary --L 17 --seeds 10
python3 -m lp_rle.search --solver basinhop --rep rle    --L 21 --time 5

# exhaustive ground truth, parallel; --write persists results/lps/LP{L}.csv
python3 -m lp_rle.parallel 27 29 31 --workers 12 --write

# dataset comparison vs the friend database (canonical-invariant key sets)
python3 -m lp_rle.compare 31 33 35 --diff

# the {sa,basinhop}x{rle,binary} head-to-head -> results/head_to_head_solvers.csv
python3 lp_rle/bench.py --solvers --Ls 13 15 17 19 21 --seeds 40 --steps 20000
```

Saving is **opt-in**: without `--write` the exhaustive classes are computed and
discarded; with it, each length is written incrementally (interrupt-safe for
lengths already finished).

## Future ideas to implement

- **Convention flag.** Add `--exclude-diagonal` to the exhaust/compare pipeline
  so counts can be reported under *both* conventions (with vs without $u=v$
  self-pairs), reconciling ours ↔ friend ↔ FGS automatically instead of by hand.
- **Dual-provenance literature table.** Extend `litdata.py` with a second column
  of modern/complete counts alongside the historical FGS-2001 values, with
  per-length provenance notes (verified $\le25$; $\ell=31$ convention; $\ell=33$
  likely FGS undercount).
- **Drift-guard regression test.** Assert that `run_exhaustive(\ell)` class counts
  equal the distinct-class counts stored in `LP{\ell}.csv`, so future pipeline
  changes cannot silently shift the ground truth.
- **Confirm the head-to-head at higher seed count.** The $2\times2$ sweep is now
  implemented (`bench.py --solvers`) and reported above at 40 seeds; rerun at
  ~200 seeds once cores are free (so wall-time `cls/s` also becomes meaningful),
  to tighten the $\ell=21$ margin from $\sim2\sigma$ to a firm interval.
- **Push exhaustive to $\ell\ge33$** as persisted ground truth using the parallel
  runner with `--write` and background execution, extending the validated range.
- **Sharper PSD prune.** Wire in the compressed-spectrum feasibility bound
  (currently a deliberate stub in the FFT toolkit) to shrink the survivor pool
  before pairwise verification at large $\ell$.
