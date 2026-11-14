"""meteor.py on the real local model (Qwen2.5-0.5B-Instruct, CPU, fp32).

CLAUDE.md required tests:
* Round trip: 50 random payloads, decode == encode, 100 %.
* Wrong key: tag check fails >= 99 % (300 trials).
* Determinism: same key + msg_id -> identical text twice.

Most messages stop right after the payload (max_new_tokens = 0) to keep the
suite fast; separate tests cover cover-text continuation and EOS masking.

Wrong-key trials: the model work (read_intervals) does not depend on the key,
so each of 10 texts is replayed once and then unmasked with 30 different wrong
keys: 300 independent tag checks on real Qwen intervals. A few wrong keys
also go through the full decode path.
"""

import numpy as np
import pytest

from stegaudit.config import MeteorConfig
from stegaudit.keys import pair_from_seed
from stegaudit.meteor import check_payload, decode, decode_ids, encode, read_intervals, unmask

pytestmark = pytest.mark.slow

KEYS, RECEIVER_KEYS = pair_from_seed(0)
SUBJECTS = ["chemistry", "history", "algebra", "biology", "geography",
            "physics", "literature", "economics", "music", "programming"]


def prompt(i):
    return f"Write a short study tip about {SUBJECTS[i % len(SUBJECTS)]}."


@pytest.fixture(scope="module")
def cfg0(qwen):
    return MeteorConfig(max_new_tokens=0)          # same model/top_k/temperature as qwen.cfg


@pytest.fixture(scope="module")
def fifty(qwen, cfg0):
    rng = np.random.default_rng(1234)
    out = []
    for i in range(50):
        msg = [int(b) for b in rng.integers(0, 2, cfg0.message_bits)]
        out.append((i, msg, encode(qwen, KEYS, prompt(i), msg, msg_id=i, cfg=cfg0)))
    return out


def test_sender_and_receiver_derive_same_keys():
    assert KEYS == RECEIVER_KEYS


def test_roundtrip_50_messages_from_ids(qwen, cfg0, fifty):
    """Meteor correctness: decoding the Sender's token ids recovers every message (100 %)."""
    for i, msg, enc in fifty:
        dec = decode_ids(qwen, RECEIVER_KEYS, prompt(i), enc.ids, msg_id=i, cfg=cfg0)
        assert dec.ok and dec.message_bits == msg, (i, dec.failure)


def test_roundtrip_50_messages_from_text(qwen, cfg0, fifty):
    """The realistic Receiver only sees text and must retokenise it.

    BPE does not guarantee text_to_ids(ids_to_text(ids)) == ids. Observed
    2026-11-15 on these 50 messages: 1 mismatch (msg 29: the model generated
    'Program' + 'ming', the tokenizer reads back 'Programming'). So this test
    requires: (a) every message that retokenises cleanly decodes correctly,
    (b) a mismatch is DETECTED as a failure, never returned as a wrong message.
    The mismatch rate itself is measured in experiment 01, not asserted here.
    """
    mismatches = []
    for i, msg, enc in fifty:
        dec = decode(qwen, RECEIVER_KEYS, prompt(i), enc.text, msg_id=i, cfg=cfg0)
        assert not (dec.ok and dec.message_bits != msg), f"msg {i}: wrong message accepted"
        if qwen.text_to_ids(enc.text) == enc.ids:
            assert dec.ok and dec.message_bits == msg, (i, dec.failure)
        else:
            mismatches.append((i, dec.failure))
            assert not dec.ok
    print(f"\nretokenisation mismatches: {len(mismatches)}/50 {mismatches}")


def test_payload_stats_are_sane(fifty, cfg0):
    for _, _, enc in fifty:
        assert enc.n_cover_tokens == 0 and enc.stop_reason == "max_new_tokens"
        assert sum(enc.bits_per_token) >= cfg0.payload_bits
        assert len(enc.bits_per_token) == enc.n_payload_tokens == len(enc.ids)
        assert not set(enc.ids) & set((151643, 151645))


def test_wrong_key_rejected_300_trials(qwen, cfg0, fifty):
    accepted = 0
    for i, _, enc in fifty[:10]:
        intervals, failure = read_intervals(qwen, prompt(i), enc.ids, cfg0)
        assert failure is None
        assert check_payload(unmask(intervals, KEYS, i, cfg0), KEYS, i, cfg0)[0]   # right key passes
        for j in range(30):
            wrong, _ = pair_from_seed(10_000 + 30 * i + j)
            accepted += check_payload(unmask(intervals, wrong, i, cfg0), wrong, i, cfg0)[0]
    assert accepted <= 3, f"{accepted}/300 wrong keys accepted"


def test_wrong_key_full_decode_path(qwen, cfg0, fifty):
    for i, _, enc in fifty[:3]:
        wrong, _ = pair_from_seed(99 + i)
        dec = decode(qwen, wrong, prompt(i), enc.text, msg_id=i, cfg=cfg0)
        assert not dec.ok and dec.failure == "tag_mismatch"


def test_deterministic_text(qwen, cfg0, fifty):
    for i, msg, enc in fifty[:3]:
        again = encode(qwen, KEYS, prompt(i), msg, msg_id=i, cfg=cfg0)
        assert again.ids == enc.ids and again.text == enc.text


def test_cover_continuation_roundtrip(qwen):
    cfg = MeteorConfig(max_new_tokens=60)
    rng = np.random.default_rng(7)
    for i in range(3):
        msg = [int(b) for b in rng.integers(0, 2, cfg.message_bits)]
        enc = encode(qwen, KEYS, prompt(i), msg, msg_id=1000 + i, cfg=cfg)
        assert enc.n_cover_tokens > 0
        assert len(enc.ids) <= max(cfg.max_new_tokens, enc.n_payload_tokens)
        assert not set(enc.ids) & set(qwen.eos_ids)
        assert qwen.text_to_ids(enc.text) == enc.ids, "retokenisation mismatch"
        dec = decode(qwen, RECEIVER_KEYS, prompt(i), enc.text, msg_id=1000 + i, cfg=cfg)
        assert dec.ok and dec.message_bits == msg


def test_eos_masked_when_model_wants_to_stop(qwen):
    """A yes-only question: the model wants to end after one word, but the payload is not done."""
    cfg = MeteorConfig(max_new_tokens=0)
    msg = [1, 0, 1, 1, 0, 0, 1, 0]
    p = "Reply with only the word yes."
    enc = encode(qwen, KEYS, p, msg, msg_id=2000, cfg=cfg)
    assert enc.eos_masked_steps >= 1 and enc.eos_mass_removed > 0.5
    dec = decode(qwen, RECEIVER_KEYS, p, enc.text, msg_id=2000, cfg=cfg)
    assert dec.ok and dec.message_bits == msg
