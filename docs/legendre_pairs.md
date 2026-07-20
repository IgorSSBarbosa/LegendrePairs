# FFT-based Legendre-pair toolkit (`src/legendre_pairs.py`)

A self-contained numpy module for searching and verifying **Legendre pairs**
using the frequency domain. Sits alongside the older `src/legendre.py` checker;
this one is organized around the FFT / power-spectral-density (PSD) view.

## What a Legendre pair is

A Legendre pair (LP) of odd length $\ell$ is $(u,v)\in\{-1,+1\}^\ell\times\{-1,+1\}^\ell$
with $\sum u_i=\sum v_i=1$ (normalized) whose periodic autocorrelations cancel:

$$\mathrm{PAF}(u,j)+\mathrm{PAF}(v,j)=-2\qquad\text{for all }j\not\equiv 0\pmod\ell,$$

where $\mathrm{PAF}(u,j)=\sum_i u_i\,u_{(i-j)\bmod\ell}$. LPs build Hadamard
matrices of order $2\ell+2$; the smallest open case is $\ell=115$.

## The spectral identity (why FFT)

By Wiener–Khinchin the PAF condition is equivalent to a pointwise constraint on
the power spectrum. With $\hat u(k)=\sum_i u_i\,\omega^{-ik}$, $\omega=e^{2\pi i/\ell}$
(exactly numpy's `fft`):

$$|\hat u(k)|^2+|\hat v(k)|^2 = 2\ell+2\qquad\text{for all }k\neq 0.$$

Two consequences the module leans on:

- **Necessary single-sequence prune.** Since $|\hat v(k)|^2\ge 0$, any LP member
  satisfies $|\hat u(k)|^2\le 2\ell+2$ for $k\neq 0$. This is the cheap
  `passes_psd_test` filter applied *before* we ever look for a matching $v$.
- **$k=0$ is special.** There $\hat u(0)=\sum u_i$, so the sum is
  $(\sum u)^2+(\sum v)^2 = 1+1 = 2$, **not** $2\ell+2$. The redundant spectral
  cross-check in `is_legendre_pair` skips $k=0$ for this reason.

## API

| function | what it does | cost |
|---|---|---|
| `paf(u)` | PAF via `ifft(|fft(u)|²).real`; index 0 is $\sum u_i^2=\ell$ | $O(\ell\log\ell)$ |
| `paf_naive(u)` | reference $O(\ell^2)$ double sum (tests only) | $O(\ell^2)$ |
| `psd_values(u)` | full $|\hat u(k)|^2$ from one `rfft` + Hermitian mirror | $O(\ell\log\ell)$ |
| `passes_psd_test(u, ell)` | normalized **and** $|\hat u(k)|^2\le 2\ell+2$ for $k\neq0$ | $O(\ell\log\ell)$ |
| `is_legendre_pair(u, v, ell)` | PAF check + redundant spectral cross-check | $O(\ell\log\ell)$ |
| `cyclotomic_cosets(ell, H)` | orbits of $\mathbb{Z}_\ell$ under $\langle H\rangle\le\mathbb{Z}_\ell^*$ | — |
| `candidates_from_cosets(cosets, w)` | $\pm1$ vectors constant on each coset, weight $w$ | $2^{\#\text{cosets}}$ |
| `compress(u, m)` | fold onto $\mathbb{Z}_m$ (sum over residue classes) | $O(\ell)$ |
| `compressed_psd_bound(ell, m)` | **stub** — raises `NotImplementedError` | — |
| `search(ell, ...)` | candidates → PSD filter → pairwise verify | see below |

`search` supports multiplier-subgroup restriction (`H=`), a candidate cap
(`max_candidates=`), multiprocessing (`workers=`), and atomic JSON checkpointing
(`checkpoint=`, resumable). It returns every unordered pair $i\le j$ over the PSD
survivor pool (so $u=v$ solutions appear once).

## Compression identity (provable)

For $\ell=m\cdot n$ and $c=\texttt{compress}(u,m)$, the length-$m$ DFT of $c$
subsamples the length-$\ell$ DFT of $u$:

$$\hat c(\kappa)=\hat u(\kappa n),\qquad\text{so}\qquad |\hat c(\kappa)|^2=|\hat u(\kappa n)|^2.$$

This is verified in the test suite. It gives the *naive* compressed bound
$|\hat c(\kappa)|^2\le 2\ell+2$ for $\kappa\neq0$ for free.

## Known limitations

- **Multiplier-subgroup search is heuristic, not exhaustive.**
  `candidates_from_cosets` only enumerates sequences invariant under the chosen
  $H$. An empty result is **not** a proof of non-existence — LPs with a trivial
  multiplier group are missed unless $H$ is trivial.
- **`compressed_psd_bound` is a deliberate stub.** The *sharper* compressed-
  spectrum feasibility test from Kotsireas & Koutschan, "Legendre pairs of
  lengths …" (Section 2), ties $|\hat c(\kappa)|^2$ to sums over each fiber of
  size $n=\ell/m$ and is tighter than the naive subsample bound above. It is not
  transcribed here — a guessed bound could silently prune valid candidates. The
  provable subsample relation is available now (see `compress`).
- **CPU / numpy only**, intended for $\ell<500$. The Galois / cyclotomic-integer
  refinement of the PSD test and any GPU population route are out of scope.

## Running the tests

```
python3 tests/test_legendre_pairs.py     # standalone runner, no pytest needed
pytest tests/test_legendre_pairs.py      # also works under pytest
```

17 tests cover PAF vs naive, Parseval $\sum_k|\hat u(k)|^2=\ell^2$, PSD vs
`|fft|²`, known LPs at $\ell=3,7$, a negative case, coset structure at $\ell=7$
with $H=\{1,2,4\}$, the compression identity, the stub, and the search driver
(serial/parallel/checkpoint).
