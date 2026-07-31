"""lp_compress — LP compression sieve + lift pipeline.

Reuses the exact-integer PAF/PSD, RLE enumeration, matching, and brute-force
ground truth from the sibling ``lp_rle`` package (``../rle_experiment``). This
``__init__`` puts that package on ``sys.path`` so ``import lp_rle`` works whether
the code is run as a script or imported as a package.
"""
from __future__ import annotations

import os
import sys

# ../../rle_experiment relative to this file (lp_compress/ -> compression_experiment/ -> repo root)
_HERE = os.path.dirname(os.path.abspath(__file__))
_RLE = os.path.abspath(os.path.join(_HERE, "..", "..", "rle_experiment"))
if _RLE not in sys.path:
    sys.path.insert(0, _RLE)
