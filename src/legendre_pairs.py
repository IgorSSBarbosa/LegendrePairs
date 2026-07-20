"""FFT-based Legendre-pair search / verification toolkit.

A **Legendre pair** (LP) of odd length ``ell`` is a pair of sequences
``(u, v)`` in ``{-1, +1}^ell`` with ``1.u == 1.v`` (normalized to ``+1``) whose
periodic autocorrelations cancel to a constant:

    PAF(u, j) + PAF(v, j) = -2      for every shift j != 0 (mod ell),

where ``PAF(u, j) = sum_i u_i * u_{i-j}`` (indices mod ell). By Wiener-Khinchin
this is equivalent to the **spectral / PSD** form

    |u_hat(k)|^2 + |v_hat(k)|^2 = 2*ell + 2     for every k != 0,

with ``u_hat(k) = sum_i u_i * w^{-ik}``, ``w = exp(2*pi*i/ell)``. numpy's ``fft``
computes exactly this ``u_hat``. LPs build Hadamard matrices; the smallest open
order is ``ell = 115``.

Design goals (see ``docs/legendre_pairs.md``):
* every autocorrelation is done by FFT in ``O(ell log ell)`` -- never the
  ``O(ell^2)`` double sum;
* the single-sequence PSD test ``|u_hat(k)|^2 <= 2*ell + 2`` is a cheap
  *necessary* condition applied to prune candidates before we ever look for a
  matching ``v``;
* candidate sequences may be restricted to those constant on the orbits of a
  multiplier subgroup ``H <= Z_ell^*`` (a heuristic -- see the caveat on
  ``candidates_from_cosets``).

Pure numpy, ``ell < 500``. No GPU. The Galois / cyclotomic-integer refinement of
the PSD test and the exact compressed-spectrum bound are left as documented
follow-ups (see ``compressed_psd_bound``).
"""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import os
import time
from typing import Iterator, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Periodic autocorrelation (PAF)
# --------------------------------------------------------------------------- #
def paf(u: np.ndarray) -> np.ndarray:
    """Periodic autocorrelation of ``u`` via FFT, in ``O(ell log ell)``.

    Returns the length-``ell`` real vector ``r`` with
    ``r[j] = sum_i u_i * u_{(i-j) mod ell}``. By Wiener-Khinchin
    ``r = ifft(|fft(u)|^2)``; index 0 is ``sum_i u_i^2`` (``= ell`` for +-1).

    This is the whole point of the toolkit: NOT the O(ell^2) double sum.
    """
    u = np.asarray(u, dtype=float)
    f = np.fft.fft(u)
    return np.fft.ifft(np.abs(f) ** 2).real


def paf_naive(u: Sequence[int]) -> np.ndarray:
    """Reference ``O(ell^2)`` PAF -- for tests / cross-validation only."""
    u = np.asarray(u, dtype=float)
    ell = u.size
    r = np.empty(ell)
    for j in range(ell):
        r[j] = float(np.sum(u * np.roll(u, j)))
    return r


# --------------------------------------------------------------------------- #
# 2. Single-sequence PSD test
# --------------------------------------------------------------------------- #
def psd_values(u: np.ndarray) -> np.ndarray:
    """Power spectrum ``|u_hat(k)|^2`` for ``k = 0 .. ell-1``, one rfft call.

    ``u`` is real, so ``rfft`` gives ``k = 0 .. ell//2`` and the rest follow by
    Hermitian symmetry ``|u_hat(k)|^2 = |u_hat(ell-k)|^2``; we mirror them back
    to return the full length-``ell`` spectrum. ``O(ell log ell)``.
    """
    u = np.asarray(u, dtype=float)
    ell = u.size
    half = np.abs(np.fft.rfft(u)) ** 2          # k = 0 .. ell//2
    full = np.empty(ell)
    full[: half.size] = half
    if ell % 2 == 0:                            # k=0 and Nyquist are unmirrored
        full[half.size :] = half[1:-1][::-1]
    else:                                       # odd ell (our case): only k=0
        full[half.size :] = half[1:][::-1]
    return full


def weight(u: np.ndarray) -> int:
    """Number of ``-1`` entries. For +-1 vectors, ``sum(u) == 1`` iff this is
    ``(ell-1)/2``."""
    u = np.asarray(u)
    return int(np.count_nonzero(u == -1))


def passes_psd_test(u: np.ndarray, ell: int, tol: float = 1e-6) -> bool:
    """Necessary condition on a single sequence before matching a ``v``.

    True iff (a) ``u`` is normalized, ``sum(u) == 1`` (equivalently
    ``weight(u) == (ell-1)/2``), and (b) ``|u_hat(k)|^2 <= 2*ell + 2`` for every
    ``k != 0``. ``O(ell log ell)`` -- no O(ell^2) fallback.

    Rationale: in an LP, ``|v_hat(k)|^2 = 2*ell + 2 - |u_hat(k)|^2 >= 0`` forces
    ``|u_hat(k)|^2 <= 2*ell + 2``; a ``u`` violating this can be in NO pair.
    """
    u = np.asarray(u)
    if u.size != ell:
        return False
    if int(np.rint(u.sum())) != 1:
        return False
    p = psd_values(u)
    return bool(np.all(p[1:] <= 2 * ell + 2 + tol))


# --------------------------------------------------------------------------- #
# 3. Pair verification
# --------------------------------------------------------------------------- #
def is_legendre_pair(u: np.ndarray, v: np.ndarray, ell: int,
                     tol: float = 1e-6) -> bool:
    """True iff ``(u, v)`` is a Legendre pair of length ``ell`` (FFT-based).

    Primary check: ``paf(u) + paf(v)`` equals ``2*ell`` at index 0 and ``-2``
    everywhere else. A redundant *spectral* assertion
    ``|u_hat(k)|^2 + |v_hat(k)|^2 == 2*ell + 2`` for ``k != 0`` cross-validates
    it -- the two are equivalent by Wiener-Khinchin, so a disagreement would
    signal an FFT normalization bug rather than a non-pair.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    if u.size != ell or v.size != ell:
        return False

    target = np.full(ell, -2.0)
    target[0] = 2 * ell
    combined = paf(u) + paf(v)
    paf_ok = bool(np.allclose(combined, target, atol=1e-6))

    # Redundant spectral cross-check (skip k=0: there the sum is s_u^2 + s_v^2).
    spec = psd_values(u) + psd_values(v)
    spec_ok = bool(np.allclose(spec[1:], 2 * ell + 2, atol=1e-6))

    assert paf_ok == spec_ok, (
        "PAF-based and PSD-based Legendre checks disagree -- likely an FFT "
        "normalization bug in paf() or psd_values()."
    )
    return paf_ok


# --------------------------------------------------------------------------- #
# 4. Multiplier-subgroup-restricted candidate generation
# --------------------------------------------------------------------------- #
def _subgroup_closure(ell: int, H: Sequence[int]) -> list[int]:
    """Multiplicative closure of ``H`` in ``Z_ell^*`` (so callers may pass
    generators, not the full subgroup)."""
    gens = {int(h) % ell for h in H}
    gens.discard(0)
    if not gens:
        gens = {1}
    group = {1}
    frontier = set(gens)
    while frontier:
        nxt = set()
        for a in frontier:
            for g in gens:
                x = (a * g) % ell
                if x not in group:
                    group.add(x)
                    nxt.add(x)
        frontier = nxt
    return sorted(group)


def cyclotomic_cosets(ell: int, H: Sequence[int]) -> list[list[int]]:
    """Partition ``Z_ell = {0, .., ell-1}`` into orbits under multiplication by
    the subgroup generated by ``H <= Z_ell^*``.

    ``0`` is always its own singleton orbit. Each nonzero ``a`` gives the orbit
    ``{h*a mod ell : h in <H>}``. Returns a list of sorted cosets.
    """
    group = _subgroup_closure(ell, H)
    seen = [False] * ell
    cosets: list[list[int]] = []
    for a in range(ell):
        if seen[a]:
            continue
        orbit = sorted({(a * h) % ell for h in group})
        for x in orbit:
            seen[x] = True
        cosets.append(orbit)
    return cosets


def candidates_from_cosets(cosets: Sequence[Sequence[int]],
                           target_weight: int) -> Iterator[np.ndarray]:
    """Yield +-1 vectors that are **constant on each coset** and have exactly
    ``target_weight`` entries equal to ``-1`` (i.e. Hamming weight, so
    ``sum == ell - 2*target_weight``; use ``target_weight = (ell-1)//2`` for the
    normalized ``sum == 1``).

    Iterates the ``2^{#cosets}`` sign assignments (which cosets are ``-1``),
    which is far smaller than ``C(ell, (ell-1)/2)``.

    CAVEAT -- this is a *heuristic*, NOT exhaustive. It only finds LPs whose
    sequences happen to be invariant under the chosen multiplier subgroup ``H``.
    Solutions with a trivial multiplier group (constant only on singleton
    cosets) are missed unless ``H`` is trivial. Do not treat an empty result as
    a proof of non-existence.
    """
    ell = sum(len(c) for c in cosets)
    idx = [np.asarray(c, dtype=int) for c in cosets]
    ncos = len(cosets)
    for mask in range(1 << ncos):
        u = np.ones(ell, dtype=np.int8)
        for b in range(ncos):
            if (mask >> b) & 1:
                u[idx[b]] = -1
        if np.count_nonzero(u == -1) == target_weight:
            yield u


# --------------------------------------------------------------------------- #
# 5. Compression for composite ell = m * n
# --------------------------------------------------------------------------- #
def compress(u: np.ndarray, m: int) -> np.ndarray:
    """Fold ``u`` onto ``Z_m`` by summing over residue classes mod ``m``.

    Returns the length-``m`` integer vector ``c`` with
    ``c[r] = sum_{i == r (mod m)} u_i``. Requires ``m | len(u)``.

    Spectral fact (provable, used in tests): writing ``ell = m*n``, the length-m
    DFT of ``c`` subsamples the length-ell DFT of ``u``:
        ``c_hat(kappa) = u_hat(kappa * n)``  for ``kappa in Z_m``,
    because ``exp(-2*pi*i*kappa*(i mod m)/m) = exp(-2*pi*i*kappa*i/m)`` and
    ``n/ell = 1/m``. Hence ``|c_hat(kappa)|^2 = |u_hat(kappa*n)|^2``.
    """
    u = np.asarray(u)
    ell = u.size
    if ell % m != 0:
        raise ValueError(f"m={m} must divide len(u)={ell}")
    return np.bincount(np.arange(ell) % m, weights=u, minlength=m).astype(np.int64)


def compressed_psd_bound(ell: int, m: int):
    """[STUB] Exact PSD bound on the compressed vector for ell = m*n.

    What is provable and available now (see ``compress``): the compressed
    spectrum is a subsample of the original, ``|c_hat(kappa)|^2 =
    |u_hat(kappa*n)|^2``, so the single-sequence bound ``|u_hat(k)|^2 <=
    2*ell+2`` immediately gives ``|c_hat(kappa)|^2 <= 2*ell+2`` for ``kappa != 0``.

    What is NOT implemented: the *sharper* compressed-spectrum constraint from
    Kotsireas & Koutschan, "Legendre pairs of lengths ..." (Section 2), which
    ties ``|c_hat(kappa)|^2`` to sums over each fiber of size ``n = ell/m`` and
    yields a tighter feasibility test than the naive subsample bound above.

    TODO: transcribe the exact identity from Kotsireas-Koutschan Section 2 and
    replace this stub. Deliberately NOT guessed -- a wrong bound would silently
    prune valid candidates. See docs/legendre_pairs.md ("Known limitations").
    """
    raise NotImplementedError(
        "compressed_psd_bound: exact Kotsireas-Koutschan Section 2 identity not "
        "yet transcribed; see the provable subsample relation in compress()'s "
        "docstring and the TODO here."
    )


# --------------------------------------------------------------------------- #
# 6. Search driver
# --------------------------------------------------------------------------- #
def _brute_candidates(ell: int) -> Iterator[np.ndarray]:
    """All normalized +-1 vectors (``sum == 1``), i.e. every choice of which
    ``(ell-1)/2`` positions are ``-1``. Exponential -- the fallback when no
    multiplier subgroup is supplied."""
    neg = (ell - 1) // 2
    for combo in itertools.combinations(range(ell), neg):
        u = np.ones(ell, dtype=np.int8)
        u[list(combo)] = -1
        yield u


def _psd_filter_batch(args) -> list[list[int]]:
    """Worker: keep the PSD-surviving candidates of one batch. Top-level and
    picklable for ``multiprocessing.Pool``. Batches are lists of int lists so
    they pickle cheaply."""
    batch, ell = args
    out = []
    for row in batch:
        u = np.asarray(row, dtype=np.int8)
        if passes_psd_test(u, ell):
            out.append(row)
    return out


def _chunked(it: Iterator, size: int) -> Iterator[list]:
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            return
        yield chunk


def _save_checkpoint(path, ell, tested, survivors):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({"ell": ell, "tested": tested,
                   "survivors": [list(map(int, s)) for s in survivors]}, f)
    os.replace(tmp, path)   # atomic: never leave a half-written checkpoint


def _load_checkpoint(path, ell):
    with open(path) as f:
        data = json.load(f)
    if data.get("ell") != ell:
        raise ValueError(f"checkpoint ell={data.get('ell')} != requested {ell}")
    return int(data["tested"]), [np.asarray(s, dtype=np.int8)
                                 for s in data["survivors"]]


def search(
    ell: int,
    H: Sequence[int] | None = None,
    max_candidates: int | None = None,
    workers: int | None = None,
    checkpoint: str | None = None,
    batch_size: int = 2000,
    log_every: int = 50000,
    verbose: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Search for Legendre pairs of length ``ell``.

    Pipeline: enumerate ``u``-candidates (constant-on-multiplier-cosets if ``H``
    is given, else brute force over normalized +-1 vectors) -> keep those that
    pass the ``O(ell log ell)`` PSD test -> among the survivors, verify every
    pair with ``is_legendre_pair``. Both members of an LP must pass the PSD test,
    so the survivor pool serves as candidates for *both* ``u`` and ``v``.

    * ``max_candidates`` caps how many ``u``-candidates are tested.
    * ``workers`` > 1 shards the PSD filtering across a ``multiprocessing.Pool``
      (each worker needs only ``ell``); pairwise verification runs in the parent.
    * ``checkpoint`` is a JSON file: written atomically as the scan progresses
      and, if it already exists, resumed from (skips already-tested candidates).

    Returns a list of ``(u, v)`` pairs (``i <= j`` over the survivor pool, so
    each unordered pair -- including ``u == v`` -- appears once).
    """
    if ell <= 0 or ell % 2 == 0:
        raise ValueError(f"ell must be a positive odd integer, got {ell}")

    target_weight = (ell - 1) // 2
    if H is not None:
        cosets = cyclotomic_cosets(ell, H)
        cand_iter: Iterator[np.ndarray] = candidates_from_cosets(cosets, target_weight)
        if verbose:
            print(f"[search] ell={ell}  {len(cosets)} cosets from H={list(H)} "
                  f"-> <= 2^{len(cosets)} coset-constant candidates")
    else:
        cand_iter = _brute_candidates(ell)
        if verbose:
            print(f"[search] ell={ell}  brute force over normalized +-1 vectors")

    tested = 0
    survivors: list[np.ndarray] = []
    if checkpoint and os.path.exists(checkpoint):
        tested, survivors = _load_checkpoint(checkpoint, ell)
        cand_iter = itertools.islice(cand_iter, tested, None)
        if verbose:
            print(f"[search] resumed from {checkpoint}: tested={tested}, "
                  f"survivors={len(survivors)}")

    if max_candidates is not None:
        remaining = max_candidates - tested
        if remaining <= 0:
            cand_iter = iter(())
        else:
            cand_iter = itertools.islice(cand_iter, remaining)

    t0 = time.perf_counter()
    next_log = tested + log_every
    pool = None
    if workers and workers > 1:
        pool = mp.Pool(processes=workers)

    def _handle_batch(n_batch: int, surv_rows: list) -> None:
        """Fold one processed batch into the running state (advance ``tested``,
        collect survivors, checkpoint, log)."""
        nonlocal tested, next_log
        tested += n_batch
        survivors.extend(np.asarray(s, dtype=np.int8) for s in surv_rows)
        if checkpoint:
            _save_checkpoint(checkpoint, ell, tested, survivors)
        if verbose and tested >= next_log:
            rate = tested / max(time.perf_counter() - t0, 1e-9)
            print(f"[search] tested={tested}  psd_pass={len(survivors)}  "
                  f"({rate:,.0f} cand/s)")
            next_log += log_every

    try:
        batches = _chunked(cand_iter, batch_size)
        if pool is not None:
            # ``imap`` preserves input order, so the k-th survivor list matches
            # the k-th batch. We record each batch's length as the pool pulls it
            # (the batch is always fed before its result is produced, so
            # ``batch_lens[k]`` is set by the time we consume result k).
            batch_lens: list[int] = []

            def _feed():
                for b in batches:
                    batch_lens.append(len(b))
                    yield (b, ell)

            for k, surv in enumerate(pool.imap(_psd_filter_batch, _feed())):
                _handle_batch(batch_lens[k], surv)
        else:
            for b in batches:
                _handle_batch(len(b), _psd_filter_batch((b, ell)))
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # Verify pairs among PSD survivors (small pool, O(|S|^2) FFT checks).
    hits: list[tuple[np.ndarray, np.ndarray]] = []
    n = len(survivors)
    for i in range(n):
        for j in range(i, n):
            if is_legendre_pair(survivors[i], survivors[j], ell):
                hits.append((survivors[i], survivors[j]))
    if verbose:
        print(f"[search] done: tested={tested}  psd_pass={n}  "
              f"pairs_found={len(hits)}  ({time.perf_counter() - t0:.2f}s)")
    return hits
