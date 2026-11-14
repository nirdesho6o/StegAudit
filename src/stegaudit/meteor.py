"""Meteor: hide a short payload in the token choices of a language model.

Idea in plain words
-------------------
A language model, at every step, gives a probability for each possible next
token. Normal sampling draws a uniform random number and picks the token
whose slice of [0, 1) contains it. Meteor does the same, except that the
"random number" is (message bits XOR secret mask bits). The mask comes from a
keyed PRG, so to anyone without the key it looks exactly like fair coin
flips, and so does the XOR. The chosen token is therefore an ordinary-looking
sample. The Receiver, who has the key and the same model, recomputes the
slices, sees which slice the token came from, and reads back the message bits
that the slice pins down.

Per generated token (beta = precision in bits, default 32):
 1. Get the model's top-k next-token distribution in a fixed order (lm.py).
 2. Quantise it into integer intervals that tile [0, 2**beta) (quantize.py).
 3. Sender: x = (next beta message bits) XOR (beta mask bits for this step).
    Emit the token whose interval [lo, hi) contains x.
 4. Every x in [lo, hi) shares the leading bits that lo and hi-1 share. Those
    leading bits of x are fixed by the token, so the Receiver can read them.
    That common prefix length n is the number of message bits sent this step.
    The message pointer moves forward by n (n can be 0).
 5. Receiver: rebuild the same interval for the observed token, take the
    first n bits of lo, XOR with the first n mask bits -> n message bits.

Worked example (beta = 4, three tokens A, B, C)
-----------------------------------------------
    p = [0.5, 0.3, 0.2]       -> floor(p * 16) = [8, 4, 3], sum 15
                                 1 unit left over goes to the argmax (A)
    widths = [9, 4, 3]           A = [0, 9)   B = [9, 13)   C = [13, 16)

    Sender, next message bits 1011, mask for this step 0110:
        x = 1011 XOR 0110 = 1101 = 13            -> 13 is in C = [13, 16)
        lo = 13 = 1101,  hi - 1 = 15 = 1111      -> common prefix "11", n = 2
        Every x in [13, 16) starts with 11, so the token C fixes the first 2
        bits of x and nothing more. 2 message bits are sent; pointer += 2.

    Receiver sees C, rebuilds C = [13, 16), same n = 2:
        prefix of lo = 11, first 2 mask bits = 01
        11 XOR 01 = 10  = the first 2 message bits of 1011.  Correct.

    Same step with mask 0011 instead:
        x = 1011 XOR 0011 = 1000 = 8             -> 8 is in A = [0, 9)
        lo = 0000, hi - 1 = 1000                 -> no common prefix, n = 0
        Nothing is sent. The next step uses a FRESH mask on the same message
        bits. High-probability tokens carry few bits, rare tokens carry many:
        on average the bits per token track the model's entropy.

(tests/test_meteor_toy.py checks these exact numbers.)

Payload format
--------------
payload (L = cfg.payload_bits, default 16) = message (L - t bits) || tag (t bits)
    tag = top t bits of HMAC-SHA256(tag_key,
              "stegaudit-v1/tag" || u64(msg_id) || u16(len(message)) || message bytes)
Without the tag the Receiver could not tell a real message from garbage: with
the wrong key (or a broken channel) decoding still returns L bits. With a
t-bit tag, garbage passes with probability about 2**-t (1/256 at t = 8). The
tag includes msg_id, so a payload cannot be replayed under another msg_id.

msg_id: a message counter that both sides keep (0, 1, 2, ...). Each
(key, msg_id) must be used for only one message (see prg.py).

When remaining bits < beta, the chunk is padded with zeros; x is still
uniform because the mask is. Bits beyond L that happen to be fixed are
discarded by the Receiver.

Stopping, EOS masking and cover text
------------------------------------
* While the payload is not fully sent, the EOS tokens are masked (lm.py), and
  max_new_tokens does not cut generation. DEVIATION FROM PURE METEOR (decision
  9 in AI_USE.md): without it, the model may end the reply before all L bits
  are out and the message is lost. Cost: during the payload phase the text
  cannot end, which can make it slightly detectable. Every encode logs
  `eos_masked_steps` (steps where EOS would have been in the top-k) and
  `eos_mass_removed` (the total probability taken from EOS). A hard limit
  cfg.payload_token_limit stops runaway encodes with an error.
* After the payload: ordinary sampling from the same quantised top-k
  distribution, driven by the separate PRG stream labelled "cover", until EOS
  or until max_new_tokens tokens in total. The Receiver ignores cover tokens.
  EOS is never included in the returned ids or text.

Two halves of decoding
----------------------
`read_intervals` replays the model over the observed tokens and records each
payload step's interval. It needs the MODEL but not the KEY: how many bits
each token carries does not depend on the key. `unmask` turns intervals into
bits. It needs the KEY but not the model. A Receiver with the wrong key gets
the right number of bits, but they are garbage, so the tag check fails.

Security assumption
-------------------
(1) The PRG is a secure PRF and each (key, msg_id, step, label) is used once,
so x is uniform at every step and the token is a sample from the quantised
distribution, whatever the message. (2) The key is secret. (3) Sender and
Receiver compute bit-identical distributions (lm.py), and decode sees the
same token ids the Sender produced.

What breaks if it fails
-----------------------
* (1) fails (mask reuse or weak PRG): token choices depend on the message, and
  repeated payloads produce repeated texts. A monitor could link or detect them.
* (2) fails: the monitor decodes exactly like the Receiver.
* (3) fails: an interval edge can move, so the Receiver reads wrong bits, or
  the observed token falls outside its top-k ("token_not_in_support").
  Retokenisation is a special case: text -> ids may not give back the
  Sender's ids (BPE can split a string differently). `decode` reports this as
  a decode failure, and experiments count retokenisation mismatches
  separately via `decode_ids`.
* Even with all three, Meteor is secure relative to the QUANTISED top-k
  distribution, not the full model; and EOS masking is a known, logged
  deviation.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field

import numpy as np

from stegaudit.config import MeteorConfig
from stegaudit.keys import StegoKeys
from stegaudit.prg import get_prg
from stegaudit.quantize import QuantizedDist, quantize

TAG_INFO = b"stegaudit-v1/tag"


class PayloadTooLong(RuntimeError):
    """The payload was not fully sent within cfg.payload_token_limit tokens."""


# ---------- bits ----------

def bits_to_int(bits) -> int:
    """[1, 0, 1, 1] -> 0b1011 (first bit = most significant)."""
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def int_to_bits(x: int, n: int) -> list[int]:
    """0b1011, 4 -> [1, 0, 1, 1]."""
    return [(x >> (n - 1 - i)) & 1 for i in range(n)]


# ---------- payload = message || tag ----------

def compute_tag(message_bits, keys: StegoKeys, msg_id: int, tag_bits: int) -> list[int]:
    m = len(message_bits)
    data = (TAG_INFO + msg_id.to_bytes(8, "big") + m.to_bytes(2, "big")
            + bits_to_int(message_bits).to_bytes(max(1, (m + 7) // 8), "big"))
    d = hmac.new(keys.tag_key, data, hashlib.sha256).digest()
    return int_to_bits(int.from_bytes(d, "big") >> (256 - tag_bits), tag_bits)


def make_payload(message_bits, keys: StegoKeys, msg_id: int, cfg: MeteorConfig) -> list[int]:
    message_bits = [int(b) for b in message_bits]
    if len(message_bits) != cfg.message_bits or any(b not in (0, 1) for b in message_bits):
        raise ValueError(f"message must be {cfg.message_bits} bits of 0/1")
    return message_bits + compute_tag(message_bits, keys, msg_id, cfg.tag_bits)


def check_payload(payload, keys: StegoKeys, msg_id: int, cfg: MeteorConfig):
    """(tag_ok, message_bits). Constant-time comparison of the tag."""
    msg, tag = list(payload[:cfg.message_bits]), list(payload[cfg.message_bits:])
    expected = compute_tag(msg, keys, msg_id, cfg.tag_bits)
    return hmac.compare_digest(bytes(tag), bytes(expected)), msg


# ---------- one token step (pure functions; the worked example) ----------

def common_prefix_len(a: int, b: int, beta: int) -> int:
    """Number of leading bits (out of beta) that a and b share."""
    return beta - (a ^ b).bit_length()


def encode_step(qd: QuantizedDist, chunk: int, mask: int) -> tuple[int, int]:
    """Sender: (position of the chosen token, number of message bits it carries)."""
    x = chunk ^ mask
    pos = qd.find(x)
    lo, hi = qd.interval(pos)
    return pos, common_prefix_len(lo, hi - 1, qd.beta)


def decode_step(qd: QuantizedDist, pos: int, mask: int) -> tuple[int, int]:
    """Receiver: (n, the n recovered message bits as an int) for the token at `pos`."""
    lo, hi = qd.interval(pos)
    n = common_prefix_len(lo, hi - 1, qd.beta)
    shift = qd.beta - n
    return n, (lo >> shift) ^ (mask >> shift)


# ---------- whole messages ----------

@dataclass
class EncodeResult:
    ids: list[int]
    text: str
    msg_id: int
    payload: list[int]
    n_payload_tokens: int
    bits_per_token: list[int]          # n at each payload step
    eos_masked_steps: int              # payload steps where EOS would have been in the top-k
    eos_mass_removed: float            # sum over payload steps of EOS probability removed
    n_vanished: int                    # vanished tokens summed over ALL steps (payload + cover)
    n_cover_tokens: int
    stop_reason: str                   # "eos" or "max_new_tokens"


@dataclass
class DecodeResult:
    ok: bool                           # tag verified
    message_bits: list[int]
    payload_bits: list[int]
    n_payload_tokens: int
    failure: str | None = None         # None | "tag_mismatch" | "token_not_in_support" | "text_too_short"
    intervals: list[tuple[int, int]] = field(default_factory=list, repr=False)


def _resolve_cfg(lm, cfg: MeteorConfig | None) -> MeteorConfig:
    """cfg may change beta / payload / token limits, but NOT what the model computes."""
    cfg = cfg or lm.cfg
    for f in ("model_name", "top_k", "temperature"):
        if getattr(cfg, f) != getattr(lm.cfg, f):
            raise ValueError(f"cfg.{f} differs from the loaded model's; load a LocalLM with that cfg")
    return cfg


def encode(lm, keys: StegoKeys, prompt: str, message_bits, msg_id: int,
           cfg: MeteorConfig | None = None) -> EncodeResult:
    cfg = _resolve_cfg(lm, cfg)
    prg = get_prg(cfg.prg)
    beta, L = cfg.beta, cfg.payload_bits
    payload = make_payload(message_bits, keys, msg_id, cfg)
    padded = payload + [0] * beta

    sess = lm.start(lm.chat_prompt_ids(prompt))
    ids: list[int] = []
    bits_per_token: list[int] = []
    eos_steps, eos_mass, vanished = 0, 0.0, 0

    # payload phase: EOS masked, max_new_tokens ignored
    ptr = 0
    while ptr < L:
        step = len(ids)
        if step >= cfg.payload_token_limit:
            raise PayloadTooLong(f"{ptr}/{L} bits after {step} tokens")
        d = sess.dist(mask_eos=True)
        qd = quantize(d.probs, beta)
        chunk = bits_to_int(padded[ptr:ptr + beta])
        mask = prg(keys.mask_key, "payload", msg_id, step, beta)
        pos, n = encode_step(qd, chunk, mask)
        tok = int(d.ids[pos])
        sess.advance(tok)
        ids.append(tok)
        bits_per_token.append(n)
        eos_steps += d.eos_mass > 0
        eos_mass += d.eos_mass
        vanished += qd.n_vanished
        ptr = min(L, ptr + n)
    n_payload = len(ids)

    # cover phase: ordinary keyed sampling until EOS or max_new_tokens
    stop_reason = "max_new_tokens"
    cstep = 0
    while len(ids) < cfg.max_new_tokens:
        d = sess.dist(mask_eos=False)
        qd = quantize(d.probs, beta)
        tok = int(d.ids[qd.find(prg(keys.mask_key, "cover", msg_id, cstep, beta))])
        vanished += qd.n_vanished
        if tok in lm.eos_ids:
            stop_reason = "eos"
            break
        sess.advance(tok)
        ids.append(tok)
        cstep += 1

    return EncodeResult(ids, lm.ids_to_text(ids), msg_id, payload, n_payload, bits_per_token,
                        int(eos_steps), eos_mass, vanished, len(ids) - n_payload, stop_reason)


def read_intervals(lm, prompt: str, ids, cfg: MeteorConfig | None = None):
    """Replay the model over `ids`; return (intervals of the payload steps, failure or None).

    Needs the model, NOT the key.
    """
    cfg = _resolve_cfg(lm, cfg)
    beta, L = cfg.beta, cfg.payload_bits
    sess = lm.start(lm.chat_prompt_ids(prompt))
    intervals: list[tuple[int, int]] = []
    ptr = 0
    for tok in ids:
        if ptr >= L:
            break
        d = sess.dist(mask_eos=True)           # same masking as the Sender's payload phase
        qd = quantize(d.probs, beta)
        where = np.flatnonzero(d.ids == int(tok))
        if where.size == 0 or qd.widths[where[0]] == 0:
            return intervals, "token_not_in_support"
        lo, hi = qd.interval(int(where[0]))
        intervals.append((lo, hi))
        ptr += common_prefix_len(lo, hi - 1, beta)
        sess.advance(int(tok))
    return intervals, (None if ptr >= L else "text_too_short")


def unmask(intervals, keys: StegoKeys, msg_id: int, cfg: MeteorConfig) -> list[int]:
    """Intervals -> payload bits. Needs the KEY, NOT the model."""
    prg = get_prg(cfg.prg)
    beta = cfg.beta
    bits: list[int] = []
    for step, (lo, hi) in enumerate(intervals):
        n = common_prefix_len(lo, hi - 1, beta)
        shift = beta - n
        mask = prg(keys.mask_key, "payload", msg_id, step, beta)
        bits += int_to_bits((lo >> shift) ^ (mask >> shift), n)
    return bits[:cfg.payload_bits]


def decode_ids(lm, keys: StegoKeys, prompt: str, ids, msg_id: int,
               cfg: MeteorConfig | None = None) -> DecodeResult:
    """Decode from token ids (skips retokenisation)."""
    cfg = _resolve_cfg(lm, cfg)
    intervals, failure = read_intervals(lm, prompt, ids, cfg)
    if failure:
        return DecodeResult(False, [], [], len(intervals), failure, intervals)
    payload = unmask(intervals, keys, msg_id, cfg)
    ok, msg = check_payload(payload, keys, msg_id, cfg)
    return DecodeResult(ok, msg if ok else [], payload, len(intervals),
                        None if ok else "tag_mismatch", intervals)


def decode(lm, keys: StegoKeys, prompt: str, text: str, msg_id: int,
           cfg: MeteorConfig | None = None) -> DecodeResult:
    """Decode from text, as a real Receiver would: retokenise, then decode_ids."""
    return decode_ids(lm, keys, prompt, lm.text_to_ids(text), msg_id, cfg)
