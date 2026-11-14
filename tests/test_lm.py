"""Phase 0: the local model loads on the CPU at the pinned revision and is deterministic.

Outside judge: the model is loaded twice (two independent instances), and also
in a fresh Python process. All three must give bit-identical next-token
logits. Comparison is torch.equal (exact), never allclose.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from stegaudit.config import MODEL_REVISIONS, MeteorConfig
from stegaudit.lm import LocalLM

PROMPT = "Write a short study tip about chemistry."


@pytest.fixture(scope="module")
def two_models():
    a = LocalLM()
    b = LocalLM()
    return a, b


def test_pinned_revision_is_used(two_models):
    a, _ = two_models
    assert a.cfg.model_revision == MODEL_REVISIONS[a.cfg.model_name]
    # weights and tokenizer were read from the snapshot folder named by the commit hash
    snap = Path(a.snapshot_dir)
    assert snap.name == a.cfg.model_revision and snap.parent.name == "snapshots"
    assert any(snap.glob("*.safetensors"))


def test_cpu_fp32_deterministic_settings(two_models):
    a, _ = two_models
    p = next(a.model.parameters())
    assert p.device.type == "cpu" and p.dtype == torch.float32
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.get_num_threads() == 1


def test_two_loads_bit_identical(two_models):
    a, b = two_models
    ids_a, ids_b = a.chat_prompt_ids(PROMPT), b.chat_prompt_ids(PROMPT)
    assert ids_a == ids_b
    la, lb = a.next_token_logits(ids_a), b.next_token_logits(ids_b)
    assert la.dtype == torch.float32
    assert torch.equal(la, lb)
    # distributions too (softmax of identical logits must be identical)
    assert torch.equal(torch.softmax(la, -1), torch.softmax(lb, -1))


def test_same_instance_repeat_bit_identical(two_models):
    a, _ = two_models
    ids = a.chat_prompt_ids(PROMPT)
    assert torch.equal(a.next_token_logits(ids), a.next_token_logits(ids))


def test_fresh_process_bit_identical(two_models, tmp_path):
    """A separate interpreter (no shared state) must reproduce the same logits."""
    a, _ = two_models
    ids = a.chat_prompt_ids(PROMPT)
    expected = a.next_token_logits(ids)
    out = tmp_path / "logits.pt"
    code = (
        "import torch, sys\n"
        "from stegaudit.lm import LocalLM\n"
        "m = LocalLM()\n"
        f"torch.save(m.next_token_logits(m.chat_prompt_ids({PROMPT!r})), {str(out)!r})\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    assert torch.equal(torch.load(out), expected)


def test_unpinned_model_rejected():
    with pytest.raises(ValueError):
        MeteorConfig(model_name="Qwen/some-unpinned-model")
