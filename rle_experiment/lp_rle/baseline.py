"""baseline.py — the binary-representation baseline.

The baseline is the *same* search (same residual state, same delta.py kernel,
same metaheuristic, same budget accounting) driven by the plain pm1 2-flip move
set instead of the RLE move set. Without this control the experiment says
nothing (spec §4.3). These are thin factory helpers so bench.py can build all
six configurations uniformly.
"""
from __future__ import annotations

import numpy as np

from .walk import BinaryMoveSet, RLEMoveSet, JointWalker, TwoStageSearch, Meta


def make_moveset(representation: str):
    if representation == "rle":
        return RLEMoveSet()
    if representation == "binary":
        return BinaryMoveSet()
    raise ValueError(f"unknown representation {representation!r}")


def make_walker(formulation: str, representation: str, L: int, meta: Meta,
                rng: np.random.Generator):
    """Build one of the six (formulation x representation) configurations."""
    moves = make_moveset(representation)
    if formulation == "joint":
        return JointWalker(L, moves, meta, rng)
    if formulation == "two-stage":
        return TwoStageSearch(L, moves, meta, rng)
    raise ValueError(f"unknown formulation {formulation!r}")
