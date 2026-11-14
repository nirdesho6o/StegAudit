"""Keyed pseudorandom generator: the mask bits that hide the payload.

Idea in plain words
-------------------
Meteor needs, at every generated token, a fresh block of beta random-looking
bits that BOTH colluders can produce and nobody else can predict. We get them
by running HMAC-SHA256, keyed with mask_key, on a counter:

    mask(label, msg_id, step) = top n_bits of
        HMAC-SHA256(mask_key, len(label) || label || u64(msg_id) || u64(step))

Think of it as a one-time pad that both sides can regenerate on demand. The
Sender XORs the next message bits with the mask; the Receiver, who can
compute the same mask, XORs again to get the message back. To anyone without
the key the mask looks like fair coin flips, so the XORed value looks like
fair coin flips too, and the token Meteor picks with it looks like an ordinary
random sample from the model.

The input encoding is fixed-width so it is unambiguous
------------------------------------------------------
    1 byte   length of label
    n bytes  label, e.g. b"payload" or b"cover"
    8 bytes  msg_id, unsigned big-endian
    8 bytes  step,   unsigned big-endian
If we simply glued decimal strings together, (msg_id=1, step=12) and
(msg_id=11, step=2) would both become "112" and get the SAME mask. With fixed
widths every (label, msg_id, step) maps to a different input.

Labels separate streams: "payload" masks the hidden bits; "cover" drives the
ordinary sampling after the payload is sent. They can never collide, even at
the same msg_id and step.

msg_id is a message counter that Sender and Receiver both keep (message 0,
1, 2, ...). It is never derived from the prompt, because two messages with
the same prompt would then reuse the same masks.

Security assumption
-------------------
HMAC-SHA256 keyed with a secret uniform key is a pseudorandom function (PRF):
without the key its outputs cannot be told apart from uniform random bits,
even after seeing many of them. And each (label, msg_id, step) is used only
once per key.

What breaks if it fails
-----------------------
* Mask reuse (same key, same msg_id, same step, same label) is the two-time
  pad mistake. If the same payload is sent twice, the masked values are equal,
  so both texts start with the same tokens, which is a visible link. If the
  payloads differ, the leaked prefixes reveal the XOR of the two messages.
* A predictable mask (weak PRG or leaked key) means the token choices are no
  longer fair samples, so a detector could in principle tell stego text from
  normal text, and anyone with the key can read the payload.

Viva knob: `PRGS` maps the name in `MeteorConfig.prg` to a function. To swap
the PRG, add another function with the same signature and register it.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Callable

LABELS = ("payload", "cover")
_U64 = 1 << 64


def prg_input(label: str, msg_id: int, step: int) -> bytes:
    """The exact bytes fed to HMAC. Fixed-width, so different inputs never collide."""
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}; allowed: {LABELS}")
    if not (0 <= msg_id < _U64 and 0 <= step < _U64):
        raise ValueError("msg_id and step must be in [0, 2**64)")
    lab = label.encode()
    return bytes([len(lab)]) + lab + msg_id.to_bytes(8, "big") + step.to_bytes(8, "big")


def hmac_sha256_bits(key: bytes, label: str, msg_id: int, step: int, n_bits: int) -> int:
    """n_bits pseudorandom bits as an integer in [0, 2**n_bits): the TOP bits of the HMAC."""
    if not 1 <= n_bits <= 256:
        raise ValueError("n_bits must be in [1, 256] (one SHA-256 output)")
    digest = hmac.new(key, prg_input(label, msg_id, step), hashlib.sha256).digest()
    return int.from_bytes(digest, "big") >> (256 - n_bits)


PRGS: dict[str, Callable[[bytes, str, int, int, int], int]] = {
    "hmac-sha256": hmac_sha256_bits,
}


def get_prg(name: str) -> Callable[[bytes, str, int, int, int], int]:
    if name not in PRGS:
        raise ValueError(f"unknown PRG {name!r}; available: {list(PRGS)}")
    return PRGS[name]
