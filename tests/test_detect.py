"""detect.py: AUROC/KS helpers, judge parsing, baseline sampler, per-token scores.

Outside judges: per-token log-probs are compared with an independent
computation (log_softmax of LocalLM.next_token_logits at each prefix).
The baseline sampler's token frequencies are compared with the distribution
it samples from (chi-square on the toy LM).
"""

import numpy as np
import pytest
import torch
from scipy.stats import chisquare

from stegaudit.config import MeteorConfig
from stegaudit.detect import (auroc_ci, eos_sum, ks, mean_logprob, outside_topk_count, parse_judge,
                              sample_plain, token_scores, vanishing_count)
from toy_lm import ToyLM


def test_auroc_separable_and_identical():
    rng = np.random.default_rng(0)
    sep = auroc_ci(rng.normal(5, 1, 100), rng.normal(0, 1, 100), n_boot=300)
    assert sep["auroc"] > 0.99 and not sep["covers_0.5"]
    same = auroc_ci(rng.normal(0, 1, 200), rng.normal(0, 1, 200), n_boot=300)
    assert 0.4 < same["auroc"] < 0.6 and same["covers_0.5"]
    rev = auroc_ci(rng.normal(-5, 1, 100), rng.normal(0, 1, 100), n_boot=300)
    assert rev["auroc"] < 0.01                     # reversed signal is reported as such, not hidden


def test_ks():
    rng = np.random.default_rng(1)
    assert ks(rng.normal(0, 1, 300), rng.normal(0, 1, 300))["p_value"] > 0.01
    assert ks(rng.normal(1, 1, 300), rng.normal(0, 1, 300))["p_value"] < 1e-6


def test_simple_statistics():
    s = {"logp_full": np.array([-1.0, -2.0, -3.0]), "p_topk": np.array([0.5, 1e-11, np.nan]),
         "eos_p": np.array([0.1, 0.2, 0.3])}
    assert mean_logprob(s) == -2.0 and mean_logprob(s, 2) == -1.5
    assert vanishing_count(s, 32) == 1 and vanishing_count(s, 32, 1) == 0
    assert outside_topk_count(s) == 1
    assert eos_sum(s, 2) == pytest.approx(0.3)


@pytest.mark.parametrize("text,status,expected", [
    ("Looks normal.\nPROBABILITY: 12", "completed", 12),
    ("PROBABILITY: 80\nwait\nPROBABILITY: 30", "completed", 30),     # last one counts
    ("PROBABILITY: 100", "completed", 100),
    ("PROBABILITY: 101", "completed", None),
    ("I think 40%", "completed", None),                                # wrong format = parse failure
    ("PROBABILITY: 20", "incomplete", None),                           # incomplete = parse failure
    ("", "completed", None),
])
def test_parse_judge(text, status, expected):
    assert parse_judge(text, status) == expected


def test_sample_plain_matches_distribution():
    """First-token frequencies over many samples match the top-k distribution."""
    lm = ToyLM(MeteorConfig(top_k=5), eos_bias=-10)
    d = lm.start(lm.chat_prompt_ids("p")).dist()
    rng = np.random.default_rng(0)
    firsts = [sample_plain(lm, "p", rng, 1)[0] for _ in range(4000)]
    counts = np.array([firsts.count(int(t)) for t in d.ids])
    assert counts.sum() == 4000
    assert chisquare(counts, d.probs * 4000).pvalue > 0.001


def test_sample_plain_stops_at_eos_and_cap():
    lm = ToyLM(MeteorConfig(top_k=20), eos_bias=6)
    rng = np.random.default_rng(0)
    lens = [len(sample_plain(lm, "p", rng, 50)) for _ in range(50)]
    assert max(lens) < 50                         # EOS ends most samples early
    lm2 = ToyLM(MeteorConfig(top_k=20), eos_bias=-50)
    assert len(sample_plain(lm2, "p", rng, 7)) == 7


def test_token_scores_match_independent_logprobs(qwen):
    prompt = "Write a short study tip about chemistry."
    ids = qwen.text_to_ids("Review your notes every evening.")
    s = token_scores(qwen, prompt, ids)
    p_ids = qwen.chat_prompt_ids(prompt)
    for t in range(len(ids)):
        ref = torch.log_softmax(qwen.next_token_logits(p_ids + ids[:t]).double(), -1)[ids[t]].item()
        assert s["logp_full"][t] == pytest.approx(ref, abs=1e-3)
    assert np.all((s["eos_p"] >= 0) & (s["eos_p"] <= 1))
    assert np.all(np.isnan(s["p_topk"]) | ((s["p_topk"] > 0) & (s["p_topk"] <= 1)))
