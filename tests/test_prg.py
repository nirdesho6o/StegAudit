"""prg.py: keyed mask bits.

Outside judge: the expected HMAC input bytes are written out literally here
(not built with prg_input), and the value is recomputed with stdlib hmac.
"""

import hashlib
import hmac
import itertools

import numpy as np
import pytest

from stegaudit.config import MeteorConfig
from stegaudit.prg import get_prg, hmac_sha256_bits, prg_input

KEY = bytes(range(32))
KEY2 = bytes(range(1, 33))


def test_matches_literal_spec():
    msg = b"\x07payload" + (5).to_bytes(8, "big") + (9).to_bytes(8, "big")
    assert prg_input("payload", 5, 9) == msg
    d = hmac.new(KEY, msg, hashlib.sha256).digest()
    assert hmac_sha256_bits(KEY, "payload", 5, 9, 256) == int.from_bytes(d, "big")
    assert hmac_sha256_bits(KEY, "payload", 5, 9, 32) == int.from_bytes(d[:4], "big")


def test_deterministic():
    assert hmac_sha256_bits(KEY, "payload", 3, 7, 32) == hmac_sha256_bits(KEY, "payload", 3, 7, 32)


def test_different_msg_id_gives_different_masks():
    """CLAUDE.md required test: different msg_id -> different masks."""
    masks = {hmac_sha256_bits(KEY, "payload", m, 0, 64) for m in range(1000)}
    assert len(masks) == 1000
    # and the whole per-message stream differs, not just step 0
    s0 = [hmac_sha256_bits(KEY, "payload", 0, t, 32) for t in range(50)]
    s1 = [hmac_sha256_bits(KEY, "payload", 1, t, 32) for t in range(50)]
    assert all(a != b for a, b in zip(s0, s1))


def test_step_label_key_each_change_the_mask():
    base = hmac_sha256_bits(KEY, "payload", 0, 0, 64)
    assert hmac_sha256_bits(KEY, "payload", 0, 1, 64) != base
    assert hmac_sha256_bits(KEY, "cover", 0, 0, 64) != base
    assert hmac_sha256_bits(KEY2, "payload", 0, 0, 64) != base


def test_encoding_is_unambiguous():
    # the classic concatenation collision is impossible with fixed widths
    assert prg_input("payload", 1, 12) != prg_input("payload", 11, 2)
    inputs = [prg_input(l, m, s) for l, m, s in
              itertools.product(("payload", "cover"), range(40), range(40))]
    assert len(set(inputs)) == len(inputs)
    assert prg_input("payload", 2**64 - 1, 0) != prg_input("payload", 0, 2**64 - 1)


@pytest.mark.parametrize("n", [1, 8, 12, 16, 32, 62, 256])
def test_output_range(n):
    vals = [hmac_sha256_bits(KEY, "payload", 0, t, n) for t in range(200)]
    assert all(0 <= v < 2**n for v in vals)


def test_shorter_output_is_prefix_of_longer():
    long = hmac_sha256_bits(KEY, "payload", 2, 3, 32)
    assert hmac_sha256_bits(KEY, "payload", 2, 3, 8) == long >> 24


def test_bits_are_balanced():
    """Each of the 32 bit positions is 1 about half the time over 4000 steps.
    The keys are fixed, so this test is deterministic (cannot flake). Bound 0.45-0.55
    is about 6 standard deviations (sd = 0.5/sqrt(4000) = 0.008)."""
    vals = np.array([hmac_sha256_bits(KEY, "payload", 0, t, 32) for t in range(4000)], dtype=np.uint64)
    bits = (vals[:, None] >> np.arange(32, dtype=np.uint64)) & np.uint64(1)
    freq = bits.mean(axis=0)
    assert np.all((freq > 0.45) & (freq < 0.55)), freq


@pytest.mark.parametrize("kwargs", [
    {"label": "other"}, {"msg_id": -1}, {"msg_id": 2**64}, {"step": 2**64}, {"n_bits": 0}, {"n_bits": 257},
])
def test_rejects_bad_arguments(kwargs):
    args = {"key": KEY, "label": "payload", "msg_id": 0, "step": 0, "n_bits": 32} | kwargs
    with pytest.raises(ValueError):
        hmac_sha256_bits(**args)


def test_registry_matches_config():
    assert get_prg(MeteorConfig().prg) is hmac_sha256_bits
    with pytest.raises(ValueError):
        get_prg("rc4")
