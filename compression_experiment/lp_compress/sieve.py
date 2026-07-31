"""sieve.py — Phase 3: integer/Diophantine sieve in compressed space (STUB).

The compressed DFT values live in a small cyclotomic field over the integer
entries ``cA_j``, so their norms/traces are rational integers and admit a
sum-of-squares Diophantine filter that can cut the admissible compressed pairs
dramatically (the mechanism reported to reduce ell=129's admissible
``[Ahat_43, Bhat_43]`` to ~2 pairs).

This is implemented as a DOCUMENTED STUB. We do NOT guess the closed form.

    TODO(cite): Kotsireas-Koutschan compression section; constant-PAF
    integer-PSD admissible-value enumeration.
"""
from __future__ import annotations

import numpy as np


def compressed_integer_sieve(cA, cB, ell: int, m: int) -> bool:
    """STUB integer/Diophantine sieve in compressed coordinates.

    NON-EXHAUSTIVE by construction: any tightened form assumes a constant-PAF /
    trivial-multiplier compression and can discard genuine solutions whose
    compression is non-constant-PAF. It must therefore NEVER gate the exhaustive
    ground-truth path (pipeline_A). Currently a pass-through (returns True) until
    the admissible-value enumeration is pinned against a citation.

    Once implemented for m=3 it must reproduce the ell=9 anchor survivor count of
    72 exactly (see the xfail test in test_compress.py).
    """
    # TODO(cite): replace pass-through with the sum-of-squares admissible filter.
    return True
