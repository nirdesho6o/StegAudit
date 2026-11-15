"""Every result JSON must carry commit, seed, model + revision, versions, config."""

import json

import torch
import transformers

from stegaudit.config import MeteorConfig
from stegaudit.provenance import dirty_from_porcelain, run_metadata, write_result_json


def test_metadata_fields():
    cfg = MeteorConfig(beta=16, seed=7)
    m = run_metadata(cfg, n_messages=3)
    assert len(m["git_commit"]) == 40
    assert isinstance(m["git_dirty"], bool)
    assert m["seed"] == 7 and m["beta"] == 16
    assert m["model_name"] == cfg.model_name
    assert m["model_revision"] == cfg.model_revision
    assert m["config"]["model_revision"] == cfg.model_revision
    assert m["versions"]["torch"] == torch.__version__
    assert m["versions"]["transformers"] == transformers.__version__
    assert m["params"] == {"n_messages": 3}
    assert m["timestamp_utc"]


def test_write_result_json_roundtrip(tmp_path):
    cfg = MeteorConfig()
    p = write_result_json(tmp_path / "sub" / "r.json", {"acc": 1.0}, cfg, k=5)
    d = json.loads(p.read_text())
    assert d["results"] == {"acc": 1.0}
    assert d["meta"]["model_revision"] == cfg.model_revision
    assert d["meta"]["params"] == {"k": 5}


def test_dirty_ignores_results_dir_but_lists_it():
    assert dirty_from_porcelain("") == (False, [])
    out = "?? results/01_meteor_roundtrip.json\n?? results/figs/a.png\n"
    assert dirty_from_porcelain(out) == (False, ["results/01_meteor_roundtrip.json", "results/figs/a.png"])
    assert dirty_from_porcelain(" M src/stegaudit/meteor.py\n?? results/x.json\n")[0] is True
    assert dirty_from_porcelain("?? experiments/new.py\n")[0] is True     # untracked code counts
    assert dirty_from_porcelain("unknown") == (True, ["unknown"])


def test_metadata_lists_changed_paths():
    assert isinstance(run_metadata(MeteorConfig())["git_changed_paths"], list)


def test_git_keeps_leading_space_of_porcelain(monkeypatch):
    """Regression: ' M path' must keep its leading space or the path gets cut wrongly."""
    import subprocess
    from stegaudit import provenance
    fake = subprocess.CompletedProcess([], 0, stdout=" M src/stegaudit/meteor.py\n?? results/x.json\n")
    monkeypatch.setattr(provenance.subprocess, "run", lambda *a, **k: fake)
    dirty, paths = dirty_from_porcelain(provenance._git("status", "--porcelain"))
    assert paths == ["src/stegaudit/meteor.py", "results/x.json"] and dirty
