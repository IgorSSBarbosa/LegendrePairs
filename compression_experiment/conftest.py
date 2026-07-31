"""pytest path shim: make both ``lp_compress`` (this folder) and ``lp_rle``
(sibling ``../rle_experiment``) importable during tests without installing.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RLE = os.path.abspath(os.path.join(_HERE, "..", "rle_experiment"))
for _p in (_HERE, _RLE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
