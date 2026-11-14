"""meteor.py on the toy LM (fast) and on hand-made distributions.

Outside judges:
* The docstring's worked example is checked number by number.
* Exhaustive check at beta = 8: for a fixed message chunk, letting the mask
  run over ALL 256 values picks each token exactly width_i times (so a uniform
  mask gives an exact sample from the quantised distribution, whatever the
  message), and every one of those 256 cases decodes to the right bits.
"""

import numpy as np
import pytest

from stegaudit.config import MeteorConfig
from stegaudit.keys import pair_from_seed
from stegaudit.meteor import (PayloadTooLong, bits_to_int, check_payload, common_prefix_len,
                              decode, decode_ids, decode_step, encode, encode_step, int_to_bits,
                              make_payload, read_intervals, unmask)
from stegaudit.quantize import quantize
from toy_lm import ToyLM

KEYS, _ = pair_from_seed(0)
OTHER_KEYS, _ = pair_from_seed(1)
rng = np.random.default_rng(0)


def toy(**cfg_kw):
    eos_bias = cfg_kw.pop("eos_bias", 0.0)
    return ToyLM(MeteorConfig(**({"top_k": 20} | cfg_kw)), eos_bias=eos_bias)


def rand_msg(cfg):
    return [int(b) for b in rng.integers(0, 2, cfg.message_bits)]


# ---------- the worked example in meteor.py's docstring ----------

def test_worked_example_beta4():
    qd = quantize([0.5, 0.3, 0.2], 4)
    assert [int(w) for w in qd.widths] == [9, 4, 3]
    assert [qd.interval(i) for i in range(3)] == [(0, 9), (9, 13), (13, 16)]
    # mask 0110: x = 1011 ^ 0110 = 1101 = 13 -> C, prefix "11", 2 bits
    assert 0b1011 ^ 0b0110 == 13
    assert encode_step(qd, 0b1011, 0b0110) == (2, 2)
    assert decode_step(qd, 2, 0b0110) == (2, 0b10)          # = first two bits of 1011
    # mask 0011: x = 1000 = 8 -> A = [0, 9), no common prefix, 0 bits
    assert encode_step(qd, 0b1011, 0b0011) == (0, 0)
    assert decode_step(qd, 0, 0b0011) == (0, 0)


def test_common_prefix_len():
    assert common_prefix_len(0b1101, 0b1111, 4) == 2
    assert common_prefix_len(0b0000, 0b1000, 4) == 0
    assert common_prefix_len(5, 5, 4) == 4                   # width-1 interval: all bits fixed


def test_bits_helpers():
    assert bits_to_int([1, 0, 1, 1]) == 11 and int_to_bits(11, 4) == [1, 0, 1, 1]
    assert int_to_bits(0, 0) == [] and bits_to_int([]) == 0


# ---------- exhaustive single-step check ----------

@pytest.mark.parametrize("seed", range(10))
def test_all_masks_sample_exactly_and_decode_correctly(seed):
    beta = 8
    p = np.random.default_rng(seed).dirichlet(np.full(6, 0.5))
    qd = quantize(p, beta)
    for chunk in (0, 0b10110011, 255):
        counts = np.zeros(len(p), dtype=int)
        for mask in range(2**beta):
            pos, n = encode_step(qd, chunk, mask)
            counts[pos] += 1
            n2, bits = decode_step(qd, pos, mask)
            assert n2 == n and bits == chunk >> (beta - n)    # first n message bits recovered
        assert np.array_equal(counts, qd.widths)              # exact quantised distribution


# ---------- payload / tag ----------

def test_payload_roundtrip_and_tag_binding():
    cfg = MeteorConfig()
    msg = [1, 0, 1, 1, 0, 0, 1, 0]
    pl = make_payload(msg, KEYS, 7, cfg)
    assert len(pl) == cfg.payload_bits and pl[:8] == msg
    assert check_payload(pl, KEYS, 7, cfg) == (True, msg)
    assert not check_payload(pl, KEYS, 8, cfg)[0]            # tag binds msg_id
    for i in range(len(pl)):                                 # any single flipped bit fails
        bad = pl.copy()
        bad[i] ^= 1
        assert not check_payload(bad, KEYS, 7, cfg)[0]


def test_payload_rejects_wrong_length():
    with pytest.raises(ValueError):
        make_payload([1, 0], KEYS, 0, MeteorConfig())


# ---------- whole messages on the toy LM ----------

@pytest.mark.parametrize("beta", [8, 16, 32])
def test_roundtrip_300_messages(beta):
    lm = toy(beta=beta, max_new_tokens=0)
    for i in range(300):
        msg = rand_msg(lm.cfg)
        enc = encode(lm, KEYS, f"prompt {i % 7}", msg, msg_id=i)
        dec = decode(lm, KEYS, f"prompt {i % 7}", enc.text, msg_id=i)
        assert dec.ok and dec.message_bits == msg, (i, dec.failure)


def test_wrong_key_rejected_300_trials():
    lm = toy(max_new_tokens=0)
    accepted = 0
    for i in range(300):
        enc = encode(lm, KEYS, "p", rand_msg(lm.cfg), msg_id=i)
        accepted += decode(lm, OTHER_KEYS, "p", enc.text, msg_id=i).ok
    assert accepted <= 3                                     # >= 99% rejected (expect ~1.2)


def test_wrong_msg_id_rejected():
    lm = toy(max_new_tokens=0)
    accepted = sum(decode(lm, KEYS, "p", encode(lm, KEYS, "p", rand_msg(lm.cfg), msg_id=i).text,
                          msg_id=i + 1).ok for i in range(100))
    assert accepted <= 2


def test_intervals_do_not_depend_on_key():
    """How many bits each token carries is key-free; only the bit VALUES need the key."""
    lm = toy(max_new_tokens=0)
    msg = rand_msg(lm.cfg)
    enc = encode(lm, KEYS, "p", msg, msg_id=3)
    iv, fail = read_intervals(lm, "p", enc.ids)
    assert fail is None and len(iv) == enc.n_payload_tokens
    assert unmask(iv, KEYS, 3, lm.cfg) == enc.payload
    assert unmask(iv, OTHER_KEYS, 3, lm.cfg) != enc.payload
    assert [common_prefix_len(lo, hi - 1, lm.cfg.beta) for lo, hi in iv] == enc.bits_per_token


def test_deterministic():
    lm = toy(max_new_tokens=40)
    msg = rand_msg(lm.cfg)
    a, b = (encode(lm, KEYS, "p", msg, msg_id=5) for _ in range(2))
    assert a.ids == b.ids and a.text == b.text


def test_eos_masked_during_payload_and_ends_cover():
    lm = toy(max_new_tokens=200, eos_bias=4.0)               # EOS likely at most steps
    masked = total = 0
    for i in range(30):
        msg = rand_msg(lm.cfg)
        enc = encode(lm, KEYS, "p", msg, msg_id=i)
        assert 0 not in enc.ids                              # EOS never emitted as a token
        assert 0 < enc.eos_masked_steps <= enc.n_payload_tokens
        assert enc.eos_mass_removed > 0
        assert enc.stop_reason == "eos"
        assert decode(lm, KEYS, "p", enc.text, msg_id=i).message_bits == msg
        masked, total = masked + enc.eos_masked_steps, total + enc.n_payload_tokens
    # logit N(0, 2) + 4 is usually, not always, in the top 20 of 30; so "most", not "all"
    assert masked / total > 0.8


def test_max_new_tokens_never_cuts_payload():
    lm = toy(max_new_tokens=1)
    enc = encode(lm, KEYS, "p", rand_msg(lm.cfg), msg_id=0)
    assert enc.n_payload_tokens > 1 and enc.n_cover_tokens == 0
    assert decode(lm, KEYS, "p", enc.text, msg_id=0).ok


def test_cover_respects_max_new_tokens():
    lm = toy(max_new_tokens=25)
    for i in range(20):
        enc = encode(lm, KEYS, "p", rand_msg(lm.cfg), msg_id=i)
        assert len(enc.ids) <= max(25, enc.n_payload_tokens)
        assert enc.stop_reason in ("eos", "max_new_tokens")
        assert decode(lm, KEYS, "p", enc.text, msg_id=i).ok   # cover tokens ignored by decoder


def test_payload_token_limit_raises():
    lm = toy(top_k=1, payload_token_limit=30)                # one token, interval = everything: 0 bits/step
    with pytest.raises(PayloadTooLong):
        encode(lm, KEYS, "p", rand_msg(lm.cfg), msg_id=0)


def test_truncated_text_reports_too_short():
    lm = toy(max_new_tokens=0)
    enc = encode(lm, KEYS, "p", rand_msg(lm.cfg), msg_id=0)
    dec = decode_ids(lm, KEYS, "p", enc.ids[:-1], msg_id=0)
    assert not dec.ok and dec.failure == "text_too_short"


def test_foreign_token_reports_not_in_support():
    lm = toy(top_k=3, max_new_tokens=0)
    enc = encode(lm, KEYS, "p", rand_msg(lm.cfg), msg_id=0)
    first = lm.start(lm.chat_prompt_ids("p")).dist(mask_eos=True)
    outside = next(t for t in range(1, lm.vocab) if t not in first.ids)
    dec = decode_ids(lm, KEYS, "p", [outside] + enc.ids[1:], msg_id=0)
    assert dec.failure == "token_not_in_support"


def test_cfg_must_match_model():
    lm = toy()
    with pytest.raises(ValueError):
        encode(lm, KEYS, "p", rand_msg(lm.cfg), 0, cfg=MeteorConfig(top_k=50))
    # beta / token limits may differ from the loaded model's cfg
    enc = encode(lm, KEYS, "p", rand_msg(lm.cfg), 0, cfg=MeteorConfig(top_k=20, beta=16, max_new_tokens=0))
    assert decode(lm, KEYS, "p", enc.text, 0, cfg=MeteorConfig(top_k=20, beta=16)).ok
