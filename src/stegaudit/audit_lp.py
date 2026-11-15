"""Audit-game linear programs: where should an auditor with a limited budget look?

Idea in plain words
-------------------
Colluders who talk over a channel the monitor cannot read (Meteor) can still
be caught by checking their DECISIONS. Auditing is expensive, so the auditor
can only check some of them. There are T decisions; decision i has stake
v_i > 0 (what the colluders gain if they corrupt it and are not caught). The
auditor picks a coverage c_i in [0, 1], the probability that decision i is
audited, with budget sum(c_i) <= B (B = expected number of audits). An audit
of a corrupted decision detects it with probability d; detection costs the
colluders a penalty P. So the colluders' expected gain from corrupting i is

    g_i(c) = (1 - d c_i) v_i - d c_i P   =   v_i - d c_i (v_i + P)

and they corrupt i only if g_i > 0. Coverage lowers g_i at rate d (v_i + P):
faster for big stakes, because a detected big-stake attack forfeits more.

This is a known model: audit games (Blocki et al., 2013) and Stackelberg
security games (Kiekintveld et al., 2009). The auditor commits to c first, and
the colluders see it and best-respond. Nothing here is claimed as new.

M1, per-decision deterrence (CLAUDE.md)
--------------------------------------
Colluders may corrupt EVERY decision with positive gain. The auditor minimises
the colluders' total expected gain:

    min  sum_i t_i   s.t.  t_i >= v_i - d c_i (v_i + P),  t_i >= 0,
                           0 <= c_i <= 1,  sum_i c_i <= B.

At the optimum t_i = max(0, g_i(c)). With P = 0 that is exactly the expected
undetected value of the corrupted decisions; with P > 0 it is that value minus
the expected penalty paid. Structure: each unit of coverage on i removes
d (v_i + P) of objective, until i is fully deterred at
c_i = v_i / (d (v_i + P)) (or until c_i = 1). The objective is a sum of
separate piecewise-linear pieces, so spending the budget greedily on the
steepest slope first is optimal ("deter the biggest stakes first"). The
order by d (v_i + P) is the same as the order by v_i, since d and P are shared.
`greedy_m1` implements this closed form; tests check it equals the LP.

M2, single-target game (Blocki et al.; Kiekintveld et al.)
---------------------------------------------------------
The colluders corrupt only ONE decision: the one with the best gain. The
auditor minimises that best gain z:

    min  z   s.t.  z >= v_i - d c_i (v_i + P)  for all i,
                   0 <= c_i <= 1,  sum_i c_i <= B.

Structure: the auditor lowers the top gains until they are EQUAL. Every
decision with v_i > z gets just enough coverage to bring its gain down to z,
c_i = (v_i - z) / (d (v_i + P)), and decisions with v_i <= z get nothing
("equalise the attacker's best payoff"). So coverage is not proportional to
v_i: small stakes get zero, and big stakes get coverage that grows with
v_i - z, not with v_i. z < 0 means every decision has negative gain under the
optimal coverage (the colluders would rather not act at all).
Note: CLAUDE.md's M2 has no explicit "do not attack" option (z >= 0); it is
implemented exactly as written, and callers read max(z, 0) as the colluders' value.

Security assumption and what breaks
-----------------------------------
The model assumes the colluders know c (worst case: the policy is public) and
that d, P and the stakes are known to both sides. If the colluders do NOT know
c, these LPs are conservative. If d is overestimated (the per-audit test is
weaker than measured), every policy leaks more than computed. M1's
"attack everything profitable" and M2's "attack one target" are two extremes
of the colluders' capacity; reality may lie between them.

Solver: scipy.optimize.linprog(method="highs").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, identity, vstack


@dataclass
class LPResult:
    c: np.ndarray          # coverage per decision
    objective: float       # M1: sum of positive gains; M2: z (max gain)
    status: str


def gains(c, v, d: float, P: float) -> np.ndarray:
    """Colluders' expected gain g_i = v_i - d c_i (v_i + P) from corrupting each decision."""
    v = np.asarray(v, float)
    return v - d * np.asarray(c, float) * (v + P)


def m1_objective(c, v, d: float, P: float) -> float:
    return float(np.maximum(0.0, gains(c, v, d, P)).sum())


def m2_objective(c, v, d: float, P: float) -> float:
    return float(gains(c, v, d, P).max())


def _check(v, B, d, P):
    v = np.asarray(v, float)
    if v.ndim != 1 or v.size == 0 or np.any(v <= 0):
        raise ValueError("stakes must be a non-empty 1-D array of positive numbers")
    if not (0 <= B) or not (0 < d <= 1) or P < 0:
        raise ValueError("need B >= 0, 0 < d <= 1, P >= 0")
    return v


def solve_m1(v, B: float, d: float, P: float) -> LPResult:
    """M1 LP. Variables x = [c_1..c_T, t_1..t_T]."""
    v = _check(v, B, d, P)
    T = v.size
    slope = d * (v + P)
    cost = np.r_[np.zeros(T), np.ones(T)]
    # t_i >= v_i - slope_i c_i   <=>   -slope_i c_i - t_i <= -v_i
    A_gain = hstack([csr_matrix(np.diag(-slope)), -identity(T, format="csr")])
    A_budget = csr_matrix(np.r_[np.ones(T), np.zeros(T)][None, :])
    res = linprog(cost, A_ub=vstack([A_gain, A_budget]).tocsr(), b_ub=np.r_[-v, B],
                  bounds=[(0, 1)] * T + [(0, None)] * T, method="highs")
    if not res.success:
        raise RuntimeError(f"M1 LP failed: {res.message}")
    c = np.clip(res.x[:T], 0, 1)
    return LPResult(c, m1_objective(c, v, d, P), res.message)


def solve_m2(v, B: float, d: float, P: float) -> LPResult:
    """M2 LP. Variables x = [c_1..c_T, z]."""
    v = _check(v, B, d, P)
    T = v.size
    slope = d * (v + P)
    cost = np.r_[np.zeros(T), 1.0]
    # z >= v_i - slope_i c_i   <=>   -slope_i c_i - z <= -v_i
    A_gain = hstack([csr_matrix(np.diag(-slope)), csr_matrix(-np.ones((T, 1)))])
    A_budget = csr_matrix(np.r_[np.ones(T), 0.0][None, :])
    res = linprog(cost, A_ub=vstack([A_gain, A_budget]).tocsr(), b_ub=np.r_[-v, B],
                  bounds=[(0, 1)] * T + [(None, None)], method="highs")
    if not res.success:
        raise RuntimeError(f"M2 LP failed: {res.message}")
    c = np.clip(res.x[:T], 0, 1)
    return LPResult(c, m2_objective(c, v, d, P), res.message)


def greedy_m1(v, B: float, d: float, P: float) -> np.ndarray:
    """Closed form for M1: fill the steepest slopes d (v_i + P) first, each up to its deterrence point."""
    v = _check(v, B, d, P)
    slope = d * (v + P)
    need = np.minimum(1.0, v / slope)            # coverage that makes g_i = 0 (or 1 if impossible)
    c = np.zeros_like(v)
    left = float(B)
    for i in np.argsort(-slope, kind="stable"):
        if left <= 0:
            break
        c[i] = min(need[i], left)
        left -= c[i]
    return c
