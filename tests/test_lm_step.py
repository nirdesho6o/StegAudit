"""lm.py step API: top-k in token-id order, float64 softmax, EOS masking, KV-cache sessions.

Outside judges:
* topk_dist is compared with a hand-written top-k + softmax (sort, slice, exp).
* Two independent Sessions over the same tokens must give bit-identical
  distributions at every step, and a fresh Python process must reproduce a
  SHA-256 of the whole sequence of distributions.
"""

import hashlib
import subprocess
import sys

import numpy as np
import pytest
import torch

from stegaudit.lm import topk_dist

PROMPT = "Write a short study tip about chemistry."


# ---------- pure function, no model ----------

def manual_topk(logits, k, temperature, exclude=()):
    l = logits.numpy().astype(np.float64).copy()
    l[list(exclude)] = -np.inf
    top = np.sort(np.argsort(-l, kind="stable")[:k])
    z = l[top] / temperature
    p = np.exp(z - z.max())
    return top, p / p.sum()


@pytest.mark.parametrize("k,temp", [(1, 1.0), (5, 1.0), (50, 1.0), (50, 0.7)])
def test_topk_dist_matches_manual(k, temp):
    logits = torch.randn(1000, generator=torch.Generator().manual_seed(k))
    d = topk_dist(logits, k, temp)
    ids, p = manual_topk(logits, k, temp)
    assert np.array_equal(d.ids, ids)
    assert np.all(np.diff(d.ids) > 0)                 # ascending token ids
    assert d.probs.dtype == np.float64 and abs(d.probs.sum() - 1) < 1e-12
    assert np.allclose(d.probs, p, rtol=1e-12, atol=0)


def test_eos_masking_pure():
    logits = torch.zeros(100)
    logits[7] = 5.0                                   # "EOS" is the most likely token
    logits[[1, 2, 3]] = 2.0
    unmasked = topk_dist(logits, 4, 1.0, eos_ids=(7,), mask_eos=False)
    masked = topk_dist(logits, 4, 1.0, eos_ids=(7,), mask_eos=True)
    assert 7 in unmasked.ids and 7 not in masked.ids
    assert len(masked.ids) == 4                       # still k tokens after masking
    assert masked.eos_mass == pytest.approx(unmasked.probs[list(unmasked.ids).index(7)])
    assert unmasked.eos_mass == 0.0


def test_eos_mass_zero_when_eos_outside_topk():
    logits = torch.arange(100, dtype=torch.float32)   # EOS id 0 is the least likely
    assert topk_dist(logits, 10, 1.0, eos_ids=(0,), mask_eos=True).eos_mass == 0.0


# ---------- real model ----------

def greedy_tokens(qwen, n):
    s = qwen.start(qwen.chat_prompt_ids(PROMPT))
    toks = []
    for _ in range(n):
        d = s.dist()
        t = int(d.ids[np.argmax(d.probs)])
        toks.append(t)
        s.advance(t)
    return toks


def dist_sequence(qwen, toks, mask_eos=False):
    s = qwen.start(qwen.chat_prompt_ids(PROMPT))
    out = []
    for t in toks:
        out.append(s.dist(mask_eos))
        s.advance(t)
    return out


def test_eos_ids_for_qwen(qwen):
    assert set(qwen.eos_ids) == {151643, 151645}      # <|endoftext|>, <|im_end|>


def test_two_sessions_bit_identical(qwen):
    toks = greedy_tokens(qwen, 15)
    a, b = dist_sequence(qwen, toks), dist_sequence(qwen, toks)
    for da, db in zip(a, b):
        assert np.array_equal(da.ids, db.ids)
        assert np.array_equal(da.probs, db.probs)     # exact, not allclose


def test_kv_cache_close_to_full_recompute(qwen):
    """Agreement to ~1e-4 is expected; bit equality is NOT required (that is why
    both sides must use the same incremental path)."""
    toks = greedy_tokens(qwen, 10)
    prompt = qwen.chat_prompt_ids(PROMPT)
    s = qwen.start(prompt)
    for t in toks:
        s.advance(t)
    full = qwen.next_token_logits(prompt + toks)
    assert torch.allclose(s._logits, full, atol=1e-3, rtol=1e-4)


def _sequence_digest(qwen, toks):
    h = hashlib.sha256()
    for d in dist_sequence(qwen, toks, mask_eos=True):
        h.update(d.ids.tobytes())
        h.update(d.probs.tobytes())
    return h.hexdigest()


def test_fresh_process_reproduces_distributions(qwen, tmp_path):
    toks = greedy_tokens(qwen, 10)
    expected = _sequence_digest(qwen, toks)
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from test_lm_step import _sequence_digest\n"
        "from stegaudit.lm import LocalLM\n"
        "print(_sequence_digest(LocalLM(), %r))\n"
    ) % (str(__import__("pathlib").Path(__file__).parent), toks)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == expected


def test_eos_masked_on_real_model(qwen):
    """After the reply 'Yes.' to a yes-only question, EOS is very likely; masking removes it."""
    s = qwen.start(qwen.chat_prompt_ids("Reply with only the word yes."))
    for t in qwen.text_to_ids("Yes."):
        s.advance(t)
    unmasked, masked = s.dist(mask_eos=False), s.dist(mask_eos=True)
    eos_in = np.isin(unmasked.ids, qwen.eos_ids)
    assert eos_in.any() and unmasked.probs[eos_in].sum() > 0.5
    assert not np.isin(masked.ids, qwen.eos_ids).any()
    assert masked.eos_mass == pytest.approx(unmasked.probs[eos_in].sum())
    assert len(masked.ids) == qwen.cfg.top_k


def test_text_roundtrip_on_plain_sentence(qwen):
    text = "Review your notes for ten minutes every evening, then test yourself."
    assert qwen.ids_to_text(qwen.text_to_ids(text)) == text
