"""audit_lp.py: M1 and M2 audit LPs.

Outside judges (CLAUDE.md):
* Brute force: for T <= 8 and a coarse coverage grid, exhaustive search must
  match the LP optimum within the provable rounding error of the grid
  (rounding each c_i down to the grid costs at most step * slope_i).
* Closed form: M1 equals greedy fractional allocation in d (v_i + P) order.
* Sanity: B = T gives M1 objective = sum max(0, v_i (1 - d) - d P).
"""

import itertools

import numpy as np
import pytest

from stegaudit.audit_lp import gains, greedy_m1, m1_objective, m2_objective, solve_m1, solve_m2

rng = np.random.default_rng(0)


def instances(n=40, T_range=(2, 30)):
    out = []
    for _ in range(n):
        T = int(rng.integers(*T_range))
        v = rng.lognormal(0, 1, T)
        out.append((v, float(rng.uniform(0, T)), float(rng.choice([0.5, 0.8, 0.95, 1.0])),
                    float(rng.choice([0.0, v.mean(), 10 * v.mean()]))))
    return out


@pytest.mark.parametrize("v,B,d,P", instances())
def test_feasible_and_bounded(v, B, d, P):
    for res in (solve_m1(v, B, d, P), solve_m2(v, B, d, P)):
        assert np.all(res.c >= -1e-9) and np.all(res.c <= 1 + 1e-9)
        assert res.c.sum() <= B + 1e-7


@pytest.mark.parametrize("v,B,d,P", instances())
def test_m1_equals_greedy_closed_form(v, B, d, P):
    lp = solve_m1(v, B, d, P)
    g = greedy_m1(v, B, d, P)
    assert g.sum() <= B + 1e-12
    assert lp.objective == pytest.approx(m1_objective(g, v, d, P), rel=1e-7, abs=1e-7)


def grid_search(v, B, d, P, step, objective):
    levels = np.round(np.arange(0, 1 + 1e-9, step), 10)
    best = np.inf
    for c in itertools.product(levels, repeat=len(v)):
        c = np.array(c)
        if c.sum() <= B + 1e-9:
            best = min(best, objective(c, v, d, P))
    return best


@pytest.mark.parametrize("T,step", [(5, 0.1), (8, 0.25)])
@pytest.mark.parametrize("seed", range(3))
def test_brute_force_matches_lp(T, step, seed):
    r = np.random.default_rng(100 + seed)
    v = r.lognormal(0, 1, T)
    d, P = 0.8, float(v.mean())
    B = float(np.round(r.uniform(0.5, T / 2) / step) * step)
    slope = d * (v + P)
    m1 = solve_m1(v, B, d, P).objective
    b1 = grid_search(v, B, d, P, step, m1_objective)
    assert m1 <= b1 + 1e-7                                  # LP is the true optimum
    assert b1 - m1 <= step * slope.sum() + 1e-7             # grid rounding bound
    m2 = solve_m2(v, B, d, P).objective
    b2 = grid_search(v, B, d, P, step, m2_objective)
    assert m2 <= b2 + 1e-7
    assert b2 - m2 <= step * slope.max() + 1e-7


def test_brute_force_exact_when_optimum_on_grid():
    """d = 1, P = 0: deterring i needs c_i = 1, so the M1 optimum (integer B) lies on the grid."""
    v = np.array([5.0, 3.0, 2.0, 1.0, 0.5])
    lp = solve_m1(v, 2.0, 1.0, 0.0).objective
    assert lp == pytest.approx(3.5)                         # deter 5 and 3, leave 2 + 1 + 0.5
    assert grid_search(v, 2.0, 1.0, 0.0, 0.25, m1_objective) == pytest.approx(lp)


@pytest.mark.parametrize("d,P", [(0.5, 0.0), (0.8, 1.0), (0.95, 10.0), (1.0, 0.0)])
def test_audit_everything_sanity(d, P):
    v = rng.lognormal(0, 1, 50)
    expected = np.maximum(0, v * (1 - d) - d * P).sum()
    assert solve_m1(v, 50.0, d, P).objective == pytest.approx(expected, abs=1e-7)


def test_zero_budget():
    v = np.array([3.0, 1.0, 2.0])
    assert solve_m1(v, 0.0, 0.9, 1.0).objective == pytest.approx(6.0)
    assert solve_m2(v, 0.0, 0.9, 1.0).objective == pytest.approx(3.0)


def test_m1_deters_biggest_stakes_first():
    v = np.array([1.0, 10.0, 3.0, 7.0])
    c = greedy_m1(v, 1.5, 1.0, 0.0)                         # d = 1, P = 0: each needs c = 1
    assert list(c) == [0.0, 1.0, 0.0, 0.5]


def test_m2_equalises_top_gains_and_skips_small_stakes():
    v = np.array([10.0, 8.0, 6.0, 1.0, 0.5])
    d, P = 0.9, 2.0
    res = solve_m2(v, 1.0, d, P)
    g = gains(res.c, v, d, P)
    covered = res.c > 1e-6
    assert np.allclose(g[covered], res.objective, atol=1e-6)        # equalised at z
    assert np.all(v[~covered] <= res.objective + 1e-6)              # uncovered stakes are below z
    assert not covered[3] and not covered[4]
    # coverage is not proportional to stakes
    ratio = res.c[covered] / v[covered]
    assert ratio.max() - ratio.min() > 1e-3


@pytest.mark.parametrize("bad", [dict(v=[1, -1], B=1, d=0.5, P=0), dict(v=[], B=1, d=0.5, P=0),
                                 dict(v=[1, 2], B=-1, d=0.5, P=0), dict(v=[1, 2], B=1, d=0, P=0),
                                 dict(v=[1, 2], B=1, d=0.5, P=-1)])
def test_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        solve_m1(**bad)
