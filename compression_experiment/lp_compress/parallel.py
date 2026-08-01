"""parallel.py — multiprocess Approach B (compression funnel) for larger ell.

Route B (``pipeline_B``) has two embarrassingly-parallel stages that dominate at
larger ``ell``; this module farms both across processes while keeping the
sequential ``cascade_pairs`` sieve intact:

  1. ORBIT REDUCTION — ``canonical_compressed_pair`` is computed for every sieve
     survivor in parallel (chunked, ORDER-PRESERVING), then de-duplicated in the
     parent keeping the first occurrence (identical result to the serial
     :func:`lp_compress.orbit.orbit_reduced_pairs`, just faster);
  2. LIFT + LP TEST — each orbit representative's fiber is lifted and PAF-tested
     independently. Tasks are one-rep-each and dispatched LONGEST-FIBER-FIRST via
     ``imap_unordered`` (LPT scheduling), so the makespan approaches
     ``max(largest_rep, total_lift / n_workers)``.

Correctness is UNCHANGED from the serial route: the class set is keyed by
``lp_rle.symmetry.canonical_pair`` (the same complete invariant Approach A uses),
so ``lp_classes`` here must equal the RLE ground truth / the stored database.

This is a SPEED experiment, not a new method: same funnel, same counts, more
cores. Results (found pairs + per-stage wall-clock) are written by
:mod:`lp_compress.run_parallel`.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lp_rle.symmetry import canonical_pair

from collections import defaultdict

from .compress import cascade_pairs, cascade_pairs_vec, _default_modulus, compress, compressed_psd
from .orbit import canonical_compressed_pair, orbit_reduce_arrays
from .lift import lift, fiber_size
from .core import paf_half

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _chunks(seq: Sequence, n: int) -> List[List]:
    """Split ``seq`` into up to ``n`` contiguous, order-preserving chunks."""
    k, r = divmod(len(seq), n)
    out, i = [], 0
    for j in range(n):
        size = k + (1 if j < r else 0)
        if size:
            out.append(seq[i:i + size])
            i += size
    return out


def _to_str(v: np.ndarray) -> str:
    return "".join("+" if x > 0 else "-" for x in v)


# --------------------------------------------------------------------------- #
# progress meter (dependency-free, tty-aware)
# --------------------------------------------------------------------------- #
class _Progress:
    """Live progress meter for a completed-task stream (no external deps).

    On a TTY it rewrites ONE line with a bar + rate + ETA; when stdout is piped
    (a log) it instead prints a line at each ~10% milestone so files stay clean.
    ``enabled=False`` (e.g. non-verbose runs) makes every call a no-op.
    """

    def __init__(self, total: int, label: str = "lift",
                 enabled: bool = True, width: int = 28):
        self.total = max(1, total)
        self.label = label
        self.enabled = enabled and total > 0
        self.width = width
        self.done = 0
        self.t0 = time.perf_counter()
        self.tty = sys.stdout.isatty()
        self._next = 0.1                       # next milestone for the piped path

    def update(self, k: int = 1, suffix: str = "") -> None:
        if not self.enabled:
            return
        self.done += k
        frac = self.done / self.total
        elapsed = time.perf_counter() - self.t0
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else 0.0
        tail = f" {suffix}" if suffix else ""
        if self.tty:
            fill = int(self.width * frac)
            bar = "#" * fill + "-" * (self.width - fill)
            sys.stdout.write(f"\r  {self.label} [{bar}] {self.done}/{self.total} "
                             f"({100 * frac:3.0f}%) {rate:5.1f}/s ETA {eta:4.0f}s{tail}")
            sys.stdout.flush()
        elif frac >= self._next or self.done == self.total:
            print(f"  {self.label} {self.done}/{self.total} ({100 * frac:.0f}%) "
                  f"{rate:.1f}/s ETA {eta:.0f}s{tail}", flush=True)
            while self._next <= frac:
                self._next += 0.1

    def close(self) -> None:
        if self.enabled and self.tty:
            sys.stdout.write("\n")
            sys.stdout.flush()


# --------------------------------------------------------------------------- #
# stage 1 worker: canonical compressed keys for a chunk of survivors
# --------------------------------------------------------------------------- #
def _canon_chunk(args) -> List[Tuple[bytes, bytes]]:
    chunk, m, use_mult = args
    return [canonical_compressed_pair(cA, cB, m, use_mult) for cA, cB in chunk]


def orbit_reduce_parallel(pairs: List[Tuple[np.ndarray, np.ndarray]], m: int,
                          pool: mp.pool.Pool, n_tasks: int,
                          use_mult: bool = False
                          ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Parallel, order-stable one-rep-per-orbit reduction (matches the serial fn)."""
    chunks = _chunks(pairs, n_tasks)
    keylists = pool.map(_canon_chunk, [(c, m, use_mult) for c in chunks])
    reps: Dict[Tuple[bytes, bytes], Tuple[np.ndarray, np.ndarray]] = {}
    for chunk, keys in zip(chunks, keylists):
        for (cA, cB), key in zip(chunk, keys):
            if key not in reps:
                reps[key] = (np.asarray(cA, np.int64), np.asarray(cB, np.int64))
    return list(reps.values())


# --------------------------------------------------------------------------- #
# Q1: secondary-modulus necessary-condition filter (prune fibers before PAF)
# --------------------------------------------------------------------------- #
def _passes_secondary(A: np.ndarray, ell: int, sec_mods: Tuple[int, ...],
                      tol: float = 1e-6) -> bool:
    """Necessary spectral condition from OTHER moduli (VERIFIED, sound as a filter).

    If ``(A, B)`` is an LP then ``PSD_A(k) + PSD_B(k) = 2*ell+2`` with
    ``PSD_B(k) >= 0``, so ``PSD_A(k) <= 2*ell+2`` at EVERY frequency ``k != 0``, and
    ``PSD_A(0) = (sum A)^2 = 1``. A compression mod ``m2`` exposes ``PSD_A`` at the
    frequencies ``m2`` sees (multiples of ``ell/m2``) via
    ``PSD^{m2}_{compress(A,m2)} = PSD_A(n2 * s)``. Checking those bounds rejects
    fiber elements that cannot lie in any LP and NEVER drops a true LP (the final
    decision stays the exact PAF join).

    HONEST CAVEAT (measured ell=33, m=11, m2=3): the filter is LOSSLESS (287 LP
    classes unchanged) and prunes ~25% of fiber candidates, but it is a NET
    SLOWDOWN on top of the PAF hash-join (lift 88s -> 254s). The hash-join is
    already ``O(|LA|+|LB|)`` — it computes one ``paf_half`` per element, not per
    product — so the per-element ``compress + compressed_psd`` costs more than the
    handful of PAF vectors it removes. This answers Q1 empirically: extra-divisor
    pruning "makes sense" combinatorially but does NOT beat the hash-join, which
    already delivers the asymptotic win. Kept OFF by default; worthwhile only with
    the brute ``O(|LA|*|LB|)`` join, where pruning each side compounds.
    """
    for m2 in sec_mods:
        p = compressed_psd(compress(A, m2))
        if abs(p[0] - 1.0) > tol:               # row-sum: PSD(0)=(sum A)^2 must be 1
            return False
        if np.any(p[1:] > 2 * ell + 2 + tol):   # PSD_A(k) <= 2*ell+2 at m2 frequencies
            return False
    return True


# --------------------------------------------------------------------------- #
# pre-flight RAM guard: refuse a lift that would OOM the box
# --------------------------------------------------------------------------- #
def _total_ram_gb() -> Optional[float]:
    """Physical RAM in GiB, or ``None`` if the platform won't report it."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3
    except (ValueError, AttributeError, OSError):
        return None


def _budget_bytes(max_lift_gb: Optional[float]) -> float:
    """RAM budget in bytes: explicit override, else ``0.6 *`` RAM (4 GiB fallback)."""
    if max_lift_gb is not None:
        return max_lift_gb * 1024 ** 3
    ram = _total_ram_gb()
    return 0.6 * ram * 1024 ** 3 if ram else 4 * 1024 ** 3


def _check_sieve_budget(ell: int, m: int, max_lift_gb: Optional[float],
                        verbose: bool) -> None:
    """Refuse the vectorized sieve if enumerating ``(n+1)**m`` rows would OOM.

    ``cascade_pairs_vec`` materializes a ``(n+1)**m`` x ``m`` int8 enumeration plus
    int16 PAF keys for BOTH halves (~``6 * m`` bytes/row). Larger ``m`` shrinks the
    lift but grows this sieve, so guard it too. Hint points the OTHER way (smaller
    modulus) — mind that it enlarges the lift.
    """
    n = ell // m
    n_enum = (n + 1) ** m
    est = n_enum * m * 6                        # int8 C + int16 keys, both halves
    budget = _budget_bytes(max_lift_gb)
    if verbose:
        print(f"  sieve budget: enum {n_enum:,} rows, peak ~{est / 1024**3:.2f} GiB "
              f"vs budget {budget / 1024**3:.2f} GiB", flush=True)
    if est > budget:
        divs = [d for d in range(2, ell) if ell % d == 0 and d < m]
        hint = (f" Try a smaller modulus (e.g. m={max(divs)}, n={ell // max(divs)}) "
                f"— but that enlarges the lift.") if divs else ""
        raise MemoryError(
            f"sieve for ell={ell}, m={m} would enumerate {n_enum:,} rows "
            f"(~{est / 1024**3:.1f} GiB), over the {budget / 1024**3:.1f} GiB "
            f"budget.{hint} Override with max_lift_gb=<GiB> if you have the RAM.")


def _bytes_per_stored_A(ell: int) -> int:
    """Conservative bytes held per A kept in ``by_paf`` (array + PAF key + slots).

    A stored A is an ``int8[ell]`` numpy array (~``ell`` data + ~112 object
    overhead), a list slot (8), plus its share of the PAF-key bytes/dict entry.
    We round UP to stay safe — the guard should trip BEFORE the OS OOM-killer.
    """
    return max(256, ell + 240)


def _estimate_peak_lift_bytes(reps, ell: int, m: int, n_workers: int) -> Tuple[int, int]:
    """(peak_bytes, biggest_fiber): RAM when the ``n_workers`` heaviest reps run.

    ``_lift_rep`` streams both fibers but holds the whole A-side (``by_paf``), so a
    rep costs ``|fiber(cA)| * _bytes_per_stored_A``. Workers run concurrently, and
    LPT dispatch front-loads the largest fibers, so the peak is the sum of the
    ``n_workers`` largest A-fibers. Uses the PRE-filter ``fiber_size`` (an upper
    bound; the secondary filter can only shrink it) — deliberately pessimistic.
    """
    fibers = sorted((fiber_size(cA, ell, m) for cA, _ in reps), reverse=True)
    if not fibers:
        return 0, 0
    concurrent = sum(fibers[:max(1, n_workers)])
    return concurrent * _bytes_per_stored_A(ell), fibers[0]


def _suggest_larger_modulus(ell: int, m: int) -> Optional[Tuple[int, int]]:
    """Largest proper divisor ``!= m`` (⇒ smallest ``n``); returns ``(m2, n2)``."""
    divs = [d for d in range(2, ell) if ell % d == 0 and d != m]
    if not divs:
        return None
    best = max(divs)
    return best, ell // best


def _check_lift_budget(reps, ell: int, m: int, n_workers: int,
                       max_lift_gb: Optional[float], verbose: bool) -> None:
    """Abort BEFORE spawning workers if the lift would exhaust RAM.

    Cost is exponential in ``n = ell/m``; a bad modulus (e.g. ``ell=39, m=3`` ⇒
    ``n=13``) makes a single fiber billions of sequences and silently OOMs the
    machine. This turns that crash into a fast, explanatory ``MemoryError`` with a
    better-modulus hint. Pass ``max_lift_gb`` to override the auto budget.
    """
    peak, biggest = _estimate_peak_lift_bytes(reps, ell, m, n_workers)
    budget = _budget_bytes(max_lift_gb)
    if verbose:
        print(f"  lift budget: peak ~{peak / 1024**3:.2f} GiB "
              f"(biggest fiber {biggest:,}) vs budget {budget / 1024**3:.2f} GiB",
              flush=True)
    if peak > budget:
        n = ell // m
        hint = ""
        sug = _suggest_larger_modulus(ell, m)
        if sug:
            hint = (f" The cost is exponential in n=ell/m={n}; retry with a larger "
                    f"modulus, e.g. m={sug[0]} (n={sug[1]}).")
        raise MemoryError(
            f"lift for ell={ell}, m={m} needs ~{peak / 1024**3:.1f} GiB "
            f"(largest single fiber {biggest:,} sequences), over the "
            f"{budget / 1024**3:.1f} GiB budget.{hint} "
            f"Override with max_lift_gb=<GiB> if you really have the RAM.")


# --------------------------------------------------------------------------- #
# stage 2 worker: lift one representative's fiber and PAF-test it
# --------------------------------------------------------------------------- #
def _lift_rep(args) -> Tuple[Dict[Tuple[bytes, bytes], Tuple[str, str]], int, int]:
    """Lift one rep's fibers and PAF-HASH-JOIN them (not the O(|LA||LB|) test).

    A pair (A, B) is a Legendre pair iff ``paf_half(A) + paf_half(B) == -2``
    componentwise. So we bucket the A-fiber by its PAF key and, for each B, look
    up the required complementary key ``-2 - paf_half(B)``. This computes only
    ``|LA| + |LB|`` PAF vectors per rep instead of testing every product pair.

    When ``sec_mods`` is non-empty each fiber is first pruned by the secondary-
    modulus necessary condition (:func:`_passes_secondary`) — lossless (cannot
    change the class set) but, with this hash-join, a net slowdown at tested ell
    (see that function's docstring). Off by default.

    MEMORY: both fibers are STREAMED (never materialized into lists). Only the
    A-side hash table ``by_paf`` is held — that is inherent to the join (you must
    keep one side to look the other up against), so peak scales with
    ``|fiber(cA)|``. The driver refuses to dispatch a rep whose fiber would blow
    the RAM budget (see :func:`_check_lift_budget`); this worker assumes the check
    already passed.
    """
    cA, cB, ell, m, sec_mods = args
    by_paf: "defaultdict[bytes, list]" = defaultdict(list)
    n_la = 0
    for A in lift(cA, ell, m):                       # STREAM the A-fiber
        if not _passes_secondary(A, ell, sec_mods):
            continue
        n_la += 1
        by_paf[paf_half(A).tobytes()].append(A)

    classes: Dict[Tuple[bytes, bytes], Tuple[str, str]] = {}
    n_lb = n_lps = 0
    for B in lift(cB, ell, m):                       # STREAM the B-fiber (not stored)
        if not _passes_secondary(B, ell, sec_mods):
            continue
        n_lb += 1
        target = (-2 - paf_half(B)).astype(np.int64).tobytes()
        for A in by_paf.get(target, ()):
            n_lps += 1
            nA = A if int(A.sum()) > 0 else -A
            nB = B if int(B.sum()) > 0 else -B
            key = canonical_pair(nA, nB)
            if key not in classes:
                classes[key] = (_to_str(nA), _to_str(nB))
    return classes, n_la * n_lb, n_lps


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def pipeline_B_parallel(ell: int, m: Optional[int] = None,
                        n_workers: Optional[int] = None,
                        use_multipliers: bool = False,
                        secondary_moduli: Optional[Sequence[int]] = None,
                        max_lift_gb: Optional[float] = None,
                        verbose: bool = True) -> Dict:
    """Multiprocess compression funnel; returns counts, per-stage timings, classes.

    Identical LP-class set to serial route B, just parallelized over ``n_workers``
    processes for the orbit-reduction and lift stages.

    A pre-flight RAM guard (:func:`_check_lift_budget`) runs after orbit reduction
    and raises ``MemoryError`` — with a larger-modulus hint — if the lift would
    exhaust memory, instead of letting a bad modulus (e.g. ``ell=39, m=3`` ⇒
    ``n=13``) silently OOM the machine. ``max_lift_gb`` overrides the auto budget
    (default ``0.6 *`` physical RAM).
    """
    m = _default_modulus(ell, m)
    n = ell // m
    n_workers = n_workers or os.cpu_count() or 1
    sec_mods = tuple(mm for mm in (secondary_moduli or ()) if ell % mm == 0 and mm != m)

    # Pre-flight RAM guard #1: the sieve enumerates (n+1)**m rows.
    _check_sieve_budget(ell, m, max_lift_gb, verbose)

    # Sieve: vectorized meet-in-the-middle (identical survivor set to cascade_pairs).
    t0 = time.perf_counter()
    CA_all, CB_all = cascade_pairs_vec(ell, m)
    n_pairs = CA_all.shape[0]
    t_sieve = time.perf_counter() - t0

    # Orbit reduction: the vectorized base-(2n+1) int64 path is default and far
    # faster than farming tiny per-pair canonicalizations across processes; keep
    # the process-parallel byte path only for the gated multiplier group.
    t1 = time.perf_counter()
    if use_multipliers:
        with mp.Pool(n_workers) as _pool:
            reps = orbit_reduce_parallel(list(zip(CA_all, CB_all)), m,
                                         _pool, n_workers * 8, True)
    else:
        reps = orbit_reduce_arrays(CA_all, CB_all, m, n)
    t_orbit = time.perf_counter() - t1

    # Pre-flight RAM guard: turn a would-be OOM into a fast, explanatory error.
    _check_lift_budget(reps, ell, m, n_workers, max_lift_gb, verbose)

    with mp.Pool(n_workers) as pool:
        # LPT scheduling: dispatch the largest fibers first to minimize makespan
        reps_sorted = sorted(
            reps, key=lambda p: fiber_size(p[0], ell, m) * fiber_size(p[1], ell, m),
            reverse=True)
        tasks = [(cA, cB, ell, m, sec_mods) for cA, cB in reps_sorted]

        t2 = time.perf_counter()
        classes: Dict[Tuple[bytes, bytes], Tuple[str, str]] = {}
        n_lift = n_lps = 0
        prog = _Progress(len(tasks), label="lift", enabled=verbose)
        for cls, nl, nlp in pool.imap_unordered(_lift_rep, tasks, chunksize=1):
            n_lift += nl
            n_lps += nlp
            for k, v in cls.items():
                classes.setdefault(k, v)
            prog.update()
        prog.close()
        t_lift = time.perf_counter() - t2

    total = time.perf_counter() - t0
    result = {
        "ell": ell, "m": m, "n": n, "n_workers": n_workers,
        "n_compressed_pairs": n_pairs, "n_orbit_reps": len(reps),
        "n_lift_candidates": n_lift, "n_lps": n_lps,
        "lp_classes": len(classes), "classes": classes,
        "secondary_moduli": sec_mods,
        "sieve_s": t_sieve, "orbit_s": t_orbit, "lift_s": t_lift, "total_s": total,
    }
    if verbose:
        print(f"ell={ell} m={m} workers={n_workers}: "
              f"survivors={n_pairs:,} reps={len(reps):,} "
              f"lift={n_lift:,} classes={len(classes)}  "
              f"[sieve {t_sieve:.1f}s | orbit {t_orbit:.1f}s | "
              f"lift {t_lift:.1f}s | total {total:.1f}s]", flush=True)
    return result
