"""appcore helpers + a startup smoke test of the Streamlit app (no Qwen, no API)."""

import json

import numpy as np
import pytest

from stegaudit.appcore import (EPS_VALUES, POLICIES, audit_game, audits_needed, colluder_labels,
                               deterrence_boundary, load_d_estimate)
from stegaudit.audit_lp import gains
from stegaudit.provenance import REPO_ROOT
from stegaudit.simulate import audits_to_detect, expected_undetected


@pytest.fixture(scope="module")
def game():
    return audit_game("lognormal", 0.6, 0.02, 0.8, 10, T=300, seed=1)


def test_audit_game_shapes_and_budget(game):
    assert set(game["policies"]) == set(POLICIES)
    assert game["P"] == pytest.approx(10 * game["v"].mean())
    for r in game["policies"].values():
        assert r["c"].shape == (300,) and r["budget_used"] <= game["B"] + 1e-6
        assert set(r["undetected"]) == set(colluder_labels())
        assert 0 <= r["attacked"] <= 300 and r["ties"] >= 0


def test_audit_game_matches_simulate(game):
    """The app's numbers are the experiment's numbers (recomputed independently here)."""
    v, P, d = game["v"], game["P"], 0.8
    for r in game["policies"].values():
        attacked = gains(r["c"], v, d, P) > 1e-9 * (v + P)
        assert r["attacked"] == attacked.sum()
        assert r["undetected"]["best response"] == pytest.approx(expected_undetected(attacked, r["c"], v, d))
        u = np.full(300, EPS_VALUES[0])
        assert r["undetected"][f"uniform ε = {EPS_VALUES[0]:g}"] == pytest.approx(expected_undetected(u, r["c"], v, d))


def test_m1_beats_uniform_against_best_response(game):
    p = game["policies"]
    assert p["m1"]["undetected"]["best response"] <= p["uniform"]["undetected"]["best response"] + 1e-6


def test_deterrence_boundary():
    v = np.array([1.0, 3.0])
    assert np.allclose(deterrence_boundary(v, 0.8, 1.0), [1 / 1.6, 0.9375])
    assert np.all(deterrence_boundary(v, 0.5, 0.0) == 1.0)          # P = 0, d < 1: cannot deter


def test_audits_needed_uses_p3a_definition():
    n = audits_needed(0.8, 0.05)
    assert n[0.01] == pytest.approx(audits_to_detect(0.01, 0.8, fpr=0.05))


def test_load_d_estimate(tmp_path):
    assert load_d_estimate(tmp_path / "missing.json") == []
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"results": {"pairs": [{"sentinel": "x", "d": 0.5, "f": 0.1}]}}))
    assert load_d_estimate(p)[0]["d"] == 0.5


def test_app_starts_and_tab4_renders():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    assert any("Best response attacks" in m.label for m in at.metric)
    at.slider[0].set_value(0.8).run()                               # Gini slider -> recompute
    assert not at.exception
    at.selectbox[0].set_value("m2").run()                           # policy selector
    assert not at.exception and any("M2" in str(df.value) for df in at.dataframe)
