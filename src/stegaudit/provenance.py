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

git_dirty ignores changes under results/: an experiment's own output files
(e.g. a checkpoint of its result JSON) are not code. Before this fix
(found 2026-11-15 in experiment 01) those files made every later stamp say
dirty. git_changed_paths lists every changed path, results/ included, so
nothing is hidden.
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
        return r.stdout.rstrip("\n")   # not strip(): porcelain lines can start with a space
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dirty_from_porcelain(status: str) -> tuple[bool, list[str]]:
    """(dirty, changed paths) from `git status --porcelain` output.

    dirty = some changed or untracked path OUTSIDE results/. "unknown" (git
    unavailable) counts as dirty, to be safe.
    """
    if status == "unknown":
        return True, ["unknown"]
    paths = [line[3:] for line in status.splitlines() if line.strip()]
    return any(not path.startswith("results/") for path in paths), paths


def run_metadata(cfg: MeteorConfig, **params) -> dict:
    """Everything needed to reproduce a run. `params` holds experiment-specific settings."""
    dirty, changed = dirty_from_porcelain(_git("status", "--porcelain"))
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": dirty,
        "git_changed_paths": changed,
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
