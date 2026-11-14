"""Provenance metadata stamped into every result JSON.

Idea in plain words
-------------------
A result is only useful if anyone can tell exactly what produced it. Every
JSON in results/ therefore records: the git commit and whether the working
tree had uncommitted changes, the seed, the model name AND its pinned
revision, the full config, the torch / transformers / tokenizers versions and
a UTC timestamp.

Why the library versions matter here
------------------------------------
Meteor decoding needs bit-identical logits (see lm.py). Those are only
guaranteed for the same software versions, so a JSON without versions could
not be reproduced.

What breaks if this is skipped
------------------------------
A result could not be matched to the code that made it. If git_dirty is
True, the result came from uncommitted code, so treat it as provisional.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import tokenizers
import torch
import transformers

from stegaudit.config import MeteorConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_metadata(cfg: MeteorConfig, **params) -> dict:
    """Everything needed to reproduce a run. `params` holds experiment-specific settings."""
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain") not in ("", "unknown"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": cfg.seed,
        "model_name": cfg.model_name,
        "model_revision": cfg.model_revision,
        "beta": cfg.beta,
        "config": cfg.to_dict(),
        "params": params,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
        },
        "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
    }


def write_result_json(path: str | Path, results: dict, cfg: MeteorConfig, **params) -> Path:
    """Write {'meta': run_metadata(...), 'results': results} to path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": run_metadata(cfg, **params), "results": results}
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return path
