# compression_experiment — LP sieve pipeline (RLE + compression)

Implements the PLAN.md sieve pipeline: an RLE exhaustive ground truth (Approach A)
and a **compression** sieve + lift pipeline (Approach B), cross-checked against an
independent brute force.

Core PAF/PSD/RLE/brute-force primitives are **reused by import** from the sibling
`../rle_experiment/lp_rle` package (added to `sys.path` by `lp_compress/__init__.py`
and `conftest.py`). Only the genuinely new pieces — compression, lifting, and the
compressed sieves — are implemented here.

## Environment
No Poetry. Plain numpy + numba (same as `rle_experiment`):

    python3 -m venv .venv && . .venv/bin/activate   # optional isolation
    pip install -r requirements.txt

## Layout
    lp_compress/
      core.py       # PAF (full int + fft), PSD, is_legendre_pair — thin reuse of lp_rle
      compress.py   # (Phase 3) compress(A,m), compressed PSD/PAF sieves, divisor lattice
      lift.py       # (Phase 4) fiber enumeration for a compressed survivor
      sieve.py      # (Phase 3) integer/Diophantine sieve STUB + cascade driver
      pipeline.py   # Approach A (RLE) and Approach B (compression) drivers
      validate.py   # independent brute-force cross-checks (reused from lp_rle)
    tests/          # test_core.py, ... (tests are the spec)
    results/        # run artifacts (git-ignored logs)

## Running tests
    python3 -m pytest compression_experiment/tests -q
    # or from inside the folder:
    cd compression_experiment && python3 -m pytest -q

## Status
- Phase 0 (core primitives) — implemented, reusing `lp_rle`.
- Phases 1–6 — see PLAN.md.
