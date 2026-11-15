"""Colluders vs auditors on synthetic stakes: policies, closed-form metrics, Monte Carlo.

Idea in plain words
-------------------
audit_lp.py says how an auditor SHOULD spread a budget. This module compares
it with simpler policies, against different colluders, on synthetic stakes
with a chosen inequality (Gini coefficient).

Stakes: T decisions with stakes drawn from a lognormal or a Pareto
distribution, with the shape set from a target Gini G:
  * lognormal(0, s):  G = 2 Phi(s / sqrt 2) - 1   ->  s = sqrt 2 * Phi^-1((G + 1) / 2)
  * Pareto(alpha), x_min = 1:  G = 1 / (2 alpha - 1)   ->  alpha = (1 + G) / (2 G)
  (standard formulas; tests check the empirical Gini of large samples).

Auditor policies (coverage c_i = probability decision i is audited, sum c_i = B):
  uniform            c_i = B / T
  stake_proportional c_i = min(1, lambda v_i), lambda chosen so sum c_i = B ("water-filling")
  suspicion          audit B decisions chosen by transcript suspicion. Against
                     Meteor, transcript suspicion carries no information (that is the
                     point of a cryptographically secure channel; experiment 02 tests
                     it), so this is the same as auditing B decisions at RANDOM:
                     expected coverage B / T, like uniform. Monte Carlo draws exactly
                     B random decisions per round to show this.
  m1, m2             the LP optima from audit_lp.py

Colluder policies (a_i = probability decision i is corrupted):
  uniform(eps)       a_i = eps for every decision
  top_k(k)           corrupt the k largest stakes
  best_response      corrupt every decision with positive expected gain under the
                     announced coverage: g_i = v_i - d c_i (v_i + P) > 0

Metrics, closed form: expected undetected value = sum_i a_i (1 - d c_i) v_i,
expected detections = sum_i a_i d c_i, colluders' expected net gain =
sum_i a_i g_i. Monte Carlo draws who is corrupted, who is audited and whether
the audit detects; its mean must agree with the closed form within its CI.
In "bernoulli" mode each decision is audited independently with prob c_i,
so the budget holds in expectation (not in every round).

Security assumption and what breaks
-----------------------------------
Same as audit_lp.py: colluders know the coverage and the stakes; d and P are
fixed. "suspicion = random" assumes the channel really is undetectable from
transcripts. If a transcript detector works (AUROC > 0.5 in experiment 02),
suspicion-based auditing would do better than random, and this module
understates it.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from stegaudit.audit_lp import gains, solve_m1, solve_m2

# ---------------- stakes ----------------

def gini(x) -> float:
    """Gini coefficient of non-negative values (0 = all equal, -> 1 = one holds everything)."""
    x = np.sort(np.asarray(x, float))
    n = x.size
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n)


def stakes(dist: str, target_gini: float, T: int, rng: np.random.Generator) -> np.ndarray:
    if not 0 < target_gini < 1:
        raise ValueError("target_gini must be in (0, 1)")
    if dist == "lognormal":
        s = math.sqrt(2) * norm.ppf((target_gini + 1) / 2)
        return rng.lognormal(0.0, s, T)
    if dist == "pareto":
        alpha = (1 + target_gini) / (2 * target_gini)
        return 1.0 + rng.pareto(alpha, T)            # numpy's pareto is Lomax; +1 gives x_min = 1
    raise ValueError(f"unknown dist {dist!r}")


# ---------------- auditor policies ----------------

def coverage_uniform(T: int, B: float) -> np.ndarray:
    return np.full(T, min(1.0, B / T))


def coverage_stake_proportional(v, B: float, iters: int = 200) -> np.ndarray:
    """c_i = min(1, lam * v_i) with sum c_i = B (bisection on lam)."""
    v = np.asarray(v, float)
    if B >= v.size:
        return np.ones_like(v)
    lo, hi = 0.0, 1.0 / v.min()
    for _ in range(iters):
        lam = (lo + hi) / 2
        if np.minimum(1.0, lam * v).sum() < B:
            lo = lam
        else:
            hi = lam
    return np.minimum(1.0, hi * v)


def coverage(policy: str, v, B: float, d: float, P: float) -> np.ndarray:
    v = np.asarray(v, float)
    if policy in ("uniform", "suspicion"):          # suspicion = random B items: same expected coverage
        return coverage_uniform(v.size, B)
    if policy == "stake_proportional":
        return coverage_stake_proportional(v, B)
    if policy == "m1":
        return solve_m1(v, B, d, P).c
    if policy == "m2":
        return solve_m2(v, B, d, P).c
    raise ValueError(f"unknown policy {policy!r}")


# ---------------- colluder policies ----------------

def attack_uniform(T: int, eps: float) -> np.ndarray:
    return np.full(T, float(eps))


def attack_top_k(v, k: int) -> np.ndarray:
    a = np.zeros(len(v))
    a[np.argsort(-np.asarray(v, float), kind="stable")[:k]] = 1.0
    return a


def attack_best_response(c, v, d: float, P: float) -> np.ndarray:
    return (gains(c, v, d, P) > 0).astype(float)


# ---------------- closed-form metrics ----------------

def expected_undetected(a, c, v, d: float) -> float:
    a, c, v = (np.asarray(x, float) for x in (a, c, v))
    return float(np.sum(a * (1 - d * c) * v))


def expected_detections(a, c, d: float) -> float:
    return float(np.sum(np.asarray(a, float) * d * np.asarray(c, float)))


def expected_colluder_gain(a, c, v, d: float, P: float) -> float:
    return float(np.sum(np.asarray(a, float) * gains(c, v, d, P)))


# ---------------- Monte Carlo ----------------

def sample_audits(c, n: int, rng: np.random.Generator, mode: str = "bernoulli", B: float | None = None):
    """Boolean (n, T) matrix of audited decisions."""
    c = np.asarray(c, float)
    if mode == "bernoulli":
        return rng.random((n, c.size)) < c
    if mode == "subset":                            # exactly round(B) uniformly random decisions
        k = int(round(B))
        out = np.zeros((n, c.size), dtype=bool)
        if k > 0:
            idx = np.argpartition(rng.random((n, c.size)), k - 1, axis=1)[:, :k]
            np.put_along_axis(out, idx, True, axis=1)
        return out
    raise ValueError(mode)


def monte_carlo_undetected(a, c, v, d: float, n_draws: int, rng: np.random.Generator,
                           mode: str = "bernoulli", B: float | None = None, chunk: int = 1000) -> dict:
    """Mean undetected value over n_draws rounds, with a normal-approximation 95 % CI."""
    a, v = np.asarray(a, float), np.asarray(v, float)
    vals = []
    for start in range(0, n_draws, chunk):
        n = min(chunk, n_draws - start)
        attacked = rng.random((n, v.size)) < a
        audited = sample_audits(c, n, rng, mode, B)
        detected = attacked & audited & (rng.random((n, v.size)) < d)
        vals.append(((attacked & ~detected) * v).sum(axis=1))
    vals = np.concatenate(vals)
    se = vals.std(ddof=1) / math.sqrt(n_draws)
    return {"mean": float(vals.mean()), "se": float(se),
            "ci95": [float(vals.mean() - 1.96 * se), float(vals.mean() + 1.96 * se)]}


# ---------------- P3a: audits needed against a uniform-eps colluder ----------------

def audits_to_detect(eps: float, d: float, delta: float = 0.05, fpr: float = 0.0,
                     power: float = 0.8) -> float:
    """Audits needed to catch a uniform-eps colluder. Which definition applies is a design choice:

    fpr == 0 (an audit never flags an honest decision): smallest n with
        P(at least one detection) = 1 - (1 - eps d)^n >= 1 - delta
        -> n = ln(delta) / ln(1 - eps d)  ~  ln(1/delta) / (eps d): scales as 1/eps.
    fpr > 0 (honest decisions are flagged at rate fpr): one-sided test of flag rate
        p0 = fpr vs p1 = fpr + (1 - fpr) eps d at level delta with the given power
        (normal approximation) -> n ~ (z-terms / (p1 - p0))^2: scales as 1/eps^2.
    """
    if fpr == 0:
        return math.log(delta) / math.log(1 - eps * d)
    p0, p1 = fpr, fpr + (1 - fpr) * eps * d
    za, zb = norm.ppf(1 - delta), norm.ppf(power)
    return ((za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))) / (p1 - p0)) ** 2
