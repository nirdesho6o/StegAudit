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


# ---------------- tab 1 helpers (no model) ----------------

@pytest.mark.parametrize("secret", ["C", "Hi", "ab!?", "é"])
def test_secret_bits_round_trip(secret):
    from stegaudit.appcore import bits_to_secret, secret_to_bits
    bits = secret_to_bits(secret)
    assert len(bits) == 8 * len(secret.encode()) and bits_to_secret(bits) == secret


def test_key_exchange_view_sender_receiver_agree_eve_differs():
    from stegaudit.appcore import key_exchange_view
    kx = key_exchange_view(5)
    assert kx["sender_fingerprint"] == kx["receiver_fingerprint"] != kx["eve_fingerprint"]
    ks, kr, ke = kx["keys"]
    assert ks.mask_key == kr.mask_key != ke.mask_key


def test_step_detail_matches_worked_example():
    """meteor.py docstring example: beta 4, widths [9, 4, 3], chunk 1011, mask 0110 -> x 13 in C, n = 2."""
    from stegaudit.appcore import step_detail
    step = {"pos": 2, "lo": [0, 9, 13], "widths": [9, 4, 3], "chunk": 0b1011, "mask": 0b0110, "x": 13,
            "n_bits": 2, "probs": [0.5, 0.3, 0.2]}
    d = step_detail(step, 4)
    assert (d["x"], d["lo"], d["hi_minus_1"], d["bits_sent"]) == ("1101", "1101", "1111", "10")


# ---------------- tab 2 helpers (saved results) ----------------

def test_detector_rows_from_saved_02():
    from stegaudit.appcore import detector_rows
    rs = detector_rows()
    assert len(rs) >= 8 and all(r["ci95"][0] <= r["auroc"] <= r["ci95"][1] for r in rs)
    whole_lp = next(r for r in rs if r["view"] == "whole" and r["detector"] == "mean token log-prob")
    assert whole_lp["auroc"] == pytest.approx(0.522, abs=1e-3) and whole_lp["events_meteor"] is None


def test_text_pair_is_paired_and_quiz_side_fixed():
    from stegaudit.appcore import text_pair
    a, b = text_pair(7), text_pair(7)
    assert a == b and a["meteor_text"] != a["normal_text"]
    assert {text_pair(i)["meteor_on_left"] for i in range(20)} == {True, False}


@pytest.mark.parametrize("tab", ["1", "2", "3", "4"])
def test_every_tab_opens_in_demo_mode(tab):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.query_params["tab"] = tab
    at.run()
    assert not at.exception


@pytest.mark.slow
def test_tab1_live_encodes_and_decodes_with_qwen():
    """Demo mode off: tab 1 loads Qwen, encodes the typed secret live, right key decodes it, Eve's is rejected."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=600)
    at.query_params["tab"] = "1"
    at.run()
    at.sidebar.toggle[0].set_value(False).run()
    at.text_input(key="m_secret").set_value("Hi").run()
    assert not at.exception
    assert any("secret “Hi”" in s.value for s in at.success)
    assert any("tag_mismatch" in e.value for e in at.error)


def test_gap_helpers_report_missing_results(tmp_path):
    from stegaudit.appcore import gap_available
    assert not gap_available(tmp_path)
