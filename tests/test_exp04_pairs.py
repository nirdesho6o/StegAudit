"""Experiment 04 takes Phase 2's (d, f) pairs as registered (PREREG 188981d): each d with its own f."""

import importlib.util
import json

from stegaudit.provenance import REPO_ROOT

spec = importlib.util.spec_from_file_location("exp04", REPO_ROOT / "experiments" / "04_audit_lp.py")
exp04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp04)


def test_p3a_cells_pair_each_d_with_its_own_f(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"results": {"pairs": [{"sentinel": "q", "d": 0.7, "f": 0.18},
                                                   {"sentinel": "g", "d": 0.6, "f": 0.2}]}}))
    pairs = exp04.load_phase2_pairs(p)
    d_values = exp04.D_VALUES + [x["d"] for x in pairs]
    cells = exp04.p3a_cells(d_values, pairs)
    registered = [c for c in cells if c[2] is None]
    assert len(registered) == len(exp04.P3A_FPR) * 5                 # registered f x all five d values
    assert {c[0] for c in registered} == set(exp04.P3A_FPR)          # measured f never crossed with other d's
    assert [c for c in cells if c[2] is not None] == [(0.18, 0.7, "q"), (0.2, 0.6, "g")]
