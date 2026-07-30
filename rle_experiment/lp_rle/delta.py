"""delta.py — exact integer delta kernels (the stochastic-search hot path).

The walk never needs an absolute objective, only the DIFFERENCE between the
current and proposed configuration. Done in exact integers this keeps the whole
inner loop free of FFT and floating point (spec §4).

Move v -> v' flips positions J with delta[j] = v'[j]-v[j] in {-2,+2}. Expanding
v' = v + delta:

  DeltaPAF(s) = sum_{j in J} delta[j] * ( v[(j-s)%L] + v[(j+s)%L] )
              + sum_{j,j' in J} delta[j]*delta[j'] * [ (j'-j)%L == s ]

using only the OLD v, so it is exact. Two facts, both asserted (spec §4.1):
  * sum over s=1..(L-1)/2 of DeltaPAF(s) == 0  (row-sum-preserving move),
  * DeltaPAF(s) ≡ 0 (mod 4)  (since PAF ≡ L (mod 4) for both sequences).
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def delta_paf_2flip(v, a, b, L):
    """DeltaPAF half-vector for the 2-flip v[a]:+1->-1, v[b]:-1->+1 (old v).

    Specialized from the general formula:
      DeltaPAF(s) = 2*( v[b-s]+v[b+s]-v[a-s]-v[a+s] )
                    - 4*( [s==(b-a)%L] + [s==(a-b)%L] ).
    """
    h = (L - 1) // 2
    out = np.zeros(h, dtype=np.int64)
    ba = (b - a) % L
    ab = (a - b) % L
    for s in range(1, h + 1):
        t = (
            np.int64(v[(b - s) % L])
            + np.int64(v[(b + s) % L])
            - np.int64(v[(a - s) % L])
            - np.int64(v[(a + s) % L])
        )
        val = 2 * t
        if s == ba:
            val -= 4
        if s == ab:
            val -= 4
        out[s - 1] = val
    return out


@njit(cache=True)
def delta_paf_general(v, J, delta, L):
    """DeltaPAF half-vector for an arbitrary flip set J with signed deltas."""
    h = (L - 1) // 2
    out = np.zeros(h, dtype=np.int64)
    m = J.shape[0]
    for s in range(1, h + 1):
        acc = np.int64(0)
        for t in range(m):
            j = J[t]
            d = delta[t]
            acc += d * (np.int64(v[(j - s) % L]) + np.int64(v[(j + s) % L]))
        for t1 in range(m):
            for t2 in range(m):
                if ((J[t2] - J[t1]) % L) == s:
                    acc += delta[t1] * delta[t2]
        out[s - 1] = acc
    return out


@njit(cache=True)
def delta_f(e_half, dpaf_half):
    """Change in f = sum_s e[s]^2 when the residual e gains dpaf on one side.

    Delta_f = sum_s ( 2*e[s]*dpaf[s] + dpaf[s]^2 ).  (F = f/16 with E = e/4.)
    """
    acc = np.int64(0)
    for s in range(e_half.shape[0]):
        acc += 2 * e_half[s] * dpaf_half[s] + dpaf_half[s] * dpaf_half[s]
    return acc


def assert_delta_valid(dpaf_half: np.ndarray) -> None:
    """The two free consistency checks of §4.1."""
    assert int(dpaf_half.sum()) == 0, "sum_s DeltaPAF(s) != 0 (row sum not preserved?)"
    assert np.all(dpaf_half % 4 == 0), "DeltaPAF(s) not divisible by 4"
