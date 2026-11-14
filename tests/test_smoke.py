"""Phase 0 smoke tests: the package imports and the config behaves."""

import subprocess
from pathlib import Path

import pytest

import stegaudit
from stegaudit.config import MeteorConfig

ROOT = Path(__file__).resolve().parents[1]


def test_import():
    assert stegaudit.__version__


def test_config_defaults():
    cfg = MeteorConfig()
    assert cfg.beta == 32
    assert cfg.top_k == 50
    assert cfg.payload_bits == 16
    assert cfg.message_bits + cfg.tag_bits == cfg.payload_bits
    assert cfg.model_name == "Qwen/Qwen2.5-0.5B-Instruct"


def test_config_one_parameter_change():
    cfg = MeteorConfig(beta=16)
    assert cfg.beta == 16 and cfg.top_k == 50


@pytest.mark.parametrize("bad", [{"beta": 0}, {"top_k": 0}, {"tag_bits": 16}, {"temperature": 0}])
def test_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        MeteorConfig(**bad)


def test_env_file_is_gitignored():
    """Secrets must never be committed."""
    r = subprocess.run(["git", "check-ignore", ".env"], cwd=ROOT, capture_output=True)
    assert r.returncode == 0, ".env is not ignored by git"
