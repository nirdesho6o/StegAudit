"""Quantisation: float probabilities -> integer intervals that tile [0, 2**beta).

Idea in plain words
-------------------
Meteor picks a token with a beta-bit integer x (masked message bits). For
that we cut the number line [0, 2**beta) into one interval per token, with
interval width proportional to the token's probability:

    width_i = floor(p_i * 2**beta)        for every token
    the few units lost to flooring go to the most likely token (argmax),
    so the widths add up to exactly 2**beta.

    lo_0 = 0,  lo_{i+1} = lo_i + width_i,  interval_i = [lo_i, lo_i + width_i)

Tokens are kept in the FIXED order they arrive in (lm.py hands them over
sorted by token id). Sender and Receiver must use the same order, or the same
x points to different tokens.

If x is uniform on [0, 2**beta), then P(token i) = width_i / 2**beta, which is
p_i up to rounding: every token except the argmax is off by less than 2**-beta,
and the total variation distance to p is below k * 2**-beta (k = number of
tokens). At beta = 32, k = 50 that is below 1.2e-8 per token step.

Low-probability vanishing
-------------------------
floor(p_i * 2**beta) = 0 exactly when p_i < 2**-beta. Such a token gets an
EMPTY interval, so Meteor can never emit it, while ordinary sampling can
(rarely). This is the "low-probability vanishing" effect (see REFERENCES.md:
ACL Findings 2026 "Breaking the Provable Security"; citation details TODO).
`n_vanished` counts these tokens at each step so experiments can log the rate.
With top-k = 50 at beta = 32, vanishing should be extremely rare; at small beta
(8, 12, 16) it is common. That is why the beta sweep was pre-registered.

Why float64 (decision 4 in AI_USE.md)
-------------------------------------
fp32 holds about 24 significant bits, so at beta = 32 the low bits of
p * 2**32 would be rounding noise. Callers pass float64 probabilities
(lm.py casts the top-k logits to float64 before softmax). Multiplying by a
power of two is exact in floating point, so floor(p * 2**beta) is exact for
the given float64 p.

Security assumption
-------------------
Meteor's "the token is a fair sample" argument holds for the QUANTISED
distribution width_i / 2**beta, not for p itself. The gap is at most
k * 2**-beta in total variation per step, which is negligible at beta = 32 but
not at beta = 8.

What breaks if it fails
-----------------------
* Different order or rounding on the two sides: the Receiver's intervals
  differ from the Sender's and decoding returns wrong bits.
* Small beta: the quantised distribution differs measurably from the model
  (and vanished tokens never appear), which a detector with the model could
  in principle exploit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuantizedDist:
    beta: int
    widths: np.ndarray   # int64, one per token, sum == 2**beta
    lo: np.ndarray       # int64 interval starts
    n_vanished: int      # tokens with p > 0 but width 0 (p < 2**-beta)
    remainder: int       # units added to the argmax token after flooring

    @property
    def total(self) -> int:
        return 1 << self.beta

    @property
    def hi(self) -> np.ndarray:
        return self.lo + self.widths

    def interval(self, i: int) -> tuple[int, int]:
        """[lo, hi) of token position i, as Python ints."""
        return int(self.lo[i]), int(self.lo[i] + self.widths[i])

    def find(self, x: int) -> int:
        """Position of the token whose interval contains x. Never returns an empty interval.

        It is the first position with hi > x; an empty interval has hi == lo <= x,
        so it is skipped automatically.
        """
        if not 0 <= x < self.total:
            raise ValueError("x must be in [0, 2**beta)")
        return int(np.searchsorted(self.hi, x, side="right"))


def quantize(probs, beta: int) -> QuantizedDist:
    """Integer widths summing to 2**beta: floor each p_i * 2**beta, rest to the argmax."""
    if not 1 <= beta <= 62:
        raise ValueError("beta must be in [1, 62] (int64 arithmetic)")
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("probs must be a non-empty 1-D array")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("probs must be finite, non-negative and not all zero")
    p = p / p.sum()   # input should already sum to 1; this only removes float drift

    total = 1 << beta
    widths = np.floor(p * float(total)).astype(np.int64)
    top = int(np.argmax(p))            # ties: lowest position wins (np.argmax rule)
    remainder = total - int(widths.sum())
    widths[top] += remainder
    if widths[top] < 1 or np.any(widths < 0):
        raise AssertionError("quantisation produced an invalid width")  # should be impossible

    lo = np.concatenate(([0], np.cumsum(widths)[:-1])).astype(np.int64)
    n_vanished = int(np.sum((p > 0) & (widths == 0)))
    return QuantizedDist(beta, widths, lo, n_vanished, remainder)


def tv_distance(probs, qd: QuantizedDist) -> float:
    """Total variation distance between p and the quantised distribution widths / 2**beta."""
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum()
    return 0.5 * float(np.abs(p - qd.widths / float(qd.total)).sum())
