"""keys.py: X25519 exchange + HKDF key derivation.

Outside judge: the HKDF output is recomputed independently with the stdlib
`hmac` module, straight from RFC 5869 (extract, then expand), and must match.
"""

import hashlib
import hmac

import pytest

from stegaudit.keys import INFO, Party, keys_from_shared_secret, pair_from_seed


def rfc5869_hkdf_sha256(ikm: bytes, info: bytes, salt: bytes = b"", length: int = 32) -> bytes:
    """Independent HKDF from RFC 5869 using only hashlib/hmac."""
    prk = hmac.new(salt or bytes(32), ikm, hashlib.sha256).digest()  # extract
    return rfc5869_expand(prk, info, length)


def rfc5869_expand(prk: bytes, info: bytes, length: int = 32) -> bytes:
    out, t, i = b"", b"", 1
    while len(out) < length:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        out += t
        i += 1
    return out[:length]


def test_both_sides_derive_same_keys():
    a, b = Party(), Party()
    ka, kb = a.derive_keys(b.public_bytes()), b.derive_keys(a.public_bytes())
    assert ka == kb
    assert len(ka.stego_key) == len(ka.mask_key) == len(ka.tag_key) == 32


def test_different_pairs_give_different_keys():
    a, b, c = Party(), Party(), Party()
    assert a.derive_keys(b.public_bytes()) != a.derive_keys(c.public_bytes())


def test_mask_and_tag_keys_are_separated():
    k = Party().derive_keys(Party().public_bytes())
    assert len({k.stego_key, k.mask_key, k.tag_key}) == 3


def test_matches_independent_rfc5869_hkdf():
    shared = bytes(range(1, 33))
    k = keys_from_shared_secret(shared)
    stego = rfc5869_hkdf_sha256(shared, INFO)
    assert k.stego_key == stego
    assert k.mask_key == rfc5869_expand(stego, INFO + b"/mask")
    assert k.tag_key == rfc5869_expand(stego, INFO + b"/tag")


def test_seeded_pair_is_reproducible_and_seed_dependent():
    s1, r1 = pair_from_seed(0)
    s2, r2 = pair_from_seed(0)
    s3, _ = pair_from_seed(1)
    assert s1 == r1 == s2 == r2
    assert s1 != s3


def test_keys_never_printed():
    k = Party().derive_keys(Party().public_bytes())
    text = repr(k)
    assert k.stego_key.hex() not in text and k.mask_key.hex() not in text and k.tag_key.hex() not in text
    assert len(k.fingerprint()) == 8


@pytest.mark.parametrize("bad_pub", [bytes(32), (1).to_bytes(32, "little")])
def test_low_order_public_key_rejected(bad_pub):
    with pytest.raises(ValueError):
        Party().derive_keys(bad_pub)


def test_all_zero_shared_secret_rejected_explicitly():
    with pytest.raises(ValueError):
        keys_from_shared_secret(bytes(32))
