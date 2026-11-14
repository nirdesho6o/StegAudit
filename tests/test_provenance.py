"""Every result JSON must carry commit, seed, model + revision, versions, config."""

import json

import torch
import transformers

from stegaudit.config import MeteorConfig
from stegaudit.provenance import run_metadata, write_result_json


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
