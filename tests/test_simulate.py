"""simulate.py: stakes, policies, metrics, Monte Carlo.

Outside judge (CLAUDE.md): Monte Carlo and closed form must agree within the
Monte Carlo CI, for every auditor x colluder combination on random instances.
"""

import zlib

import numpy as np
import pytest

from stegaudit.audit_lp import gains
from stegaudit.simulate import (attack_best_response, attack_top_k, attack_uniform, audits_to_detect, coverage,
                                coverage_stake_proportional, expected_colluder_gain, expected_detections,
                                expected_undetected, gini, monte_carlo_undetected, sample_audits, stakes)

POLICIES = ["uniform", "suspicion", "stake_proportional", "m1", "m2"]


def test_gini_known_values():
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini([0, 0, 1]) == pytest.approx(2 / 3)        # (n - 1) / n when one holds everything
    assert gini([1, 2, 3, 4]) == pytest.approx(0.25)


@pytest.mark.parametrize("dist", ["lognormal", "pareto"])
@pytest.mark.parametrize("g", [0.2, 0.4, 0.6, 0.8])
def test_stakes_hit_target_gini(dist, g):
    v = stakes(dist, g, 400_000, np.random.default_rng(0))
    assert np.all(v > 0)
    # Pareto with G = 0.8 has alpha = 1.125: infinite variance, slow convergence -> looser tolerance
    tol = 0.06 if (dist == "pareto" and g == 0.8) else 0.02
    assert abs(gini(v) - g) < tol


def test_stake_proportional_waterfilling():
    v = np.array([100.0, 10.0, 5.0, 1.0, 1.0])
    c = coverage_stake_proportional(v, 2.0)
    assert c.sum() == pytest.approx(2.0, abs=1e-9)
    assert c[0] == pytest.approx(1.0)                      # capped
    free = c < 1 - 1e-9
    assert np.allclose(c[free] / v[free], (c[free] / v[free])[0])   # proportional where not capped
    assert np.all(coverage_stake_proportional(v, 5.0) == 1.0)


@pytest.mark.parametrize("policy", POLICIES)
def test_policies_respect_budget(policy):
    v = stakes("lognormal", 0.6, 300, np.random.default_rng(1))
    c = coverage(policy, v, 15.0, 0.8, float(v.mean()))
    assert np.all((c >= -1e-9) & (c <= 1 + 1e-9)) and c.sum() <= 15.0 + 1e-6


def test_attackers():
    v = np.array([3.0, 1.0, 2.0])
    assert list(attack_top_k(v, 2)) == [1.0, 0.0, 1.0]
    assert list(attack_uniform(3, 0.1)) == [0.1, 0.1, 0.1]
    c = np.array([1.0, 0.0, 0.5])
    a = attack_best_response(c, v, 1.0, 0.0)
    assert list(a) == list((gains(c, v, 1.0, 0.0) > 0).astype(float)) == [0.0, 1.0, 1.0]


def test_closed_form_metrics_by_hand():
    a, c, v = np.array([1, 0, 1.0]), np.array([0.5, 1, 0]), np.array([10, 5, 2.0])
    assert expected_undetected(a, c, v, 0.8) == pytest.approx(10 * 0.6 + 2)
    assert expected_detections(a, c, 0.8) == pytest.approx(0.4)
    assert expected_colluder_gain(a, c, v, 0.8, 5.0) == pytest.approx((10 - 0.4 * 15) + 2)


def test_subset_audits_exactly_B():
    m = sample_audits(np.full(50, 0.2), 200, np.random.default_rng(0), "subset", B=10)
    assert np.all(m.sum(axis=1) == 10)
    assert abs(m.mean() - 0.2) < 0.01                    # expected coverage B / T


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("colluder", ["uniform", "top_k", "best_response"])
def test_monte_carlo_agrees_with_closed_form(policy, colluder):
    rng = np.random.default_rng(zlib.crc32(f"{policy}/{colluder}".encode()))   # hash() is salted per process
    v = stakes("pareto", 0.6, 200, rng)
    B, d, P = 10.0, 0.8, float(v.mean())
    c = coverage(policy, v, B, d, P)
    a = {"uniform": attack_uniform(200, 0.1), "top_k": attack_top_k(v, 20),
         "best_response": attack_best_response(c, v, d, P)}[colluder]
    mode = "subset" if policy == "suspicion" else "bernoulli"
    mc = monte_carlo_undetected(a, c, v, d, 20_000, rng, mode=mode, B=B)
    exact = expected_undetected(a, c, v, d)
    assert abs(mc["mean"] - exact) <= 4 * mc["se"] + 1e-9, (mc, exact)


def test_audits_to_detect_scaling():
    """fpr = 0 -> ~1/eps (halving eps doubles n); fpr > 0 -> ~1/eps^2 (halving eps ~4x n)."""
    r0 = audits_to_detect(0.01, 0.8) / audits_to_detect(0.02, 0.8)
    r1 = audits_to_detect(0.01, 0.8, fpr=0.1) / audits_to_detect(0.02, 0.8, fpr=0.1)
    assert 1.9 < r0 < 2.1
    assert 3.5 < r1 < 4.1
    # fpr = 0 formula: P(at least one detection in n audits) = 1 - delta
    n = audits_to_detect(0.05, 0.8, delta=0.05)
    assert 1 - (1 - 0.05 * 0.8) ** n == pytest.approx(0.95)
