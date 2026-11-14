"""Key agreement: X25519 + HKDF-SHA256 -> the colluders' shared stego keys.

Idea in plain words
-------------------
Sender and Receiver each pick a random private number and publish a public
key derived from it (X25519, Diffie-Hellman on Curve25519). Each side
combines its own private key with the other side's public key and both get
the SAME 32-byte shared secret. An eavesdropper who sees only the two public
keys cannot compute it (this is the Diffie-Hellman assumption).

The raw shared secret is not used directly. It is run through HKDF-SHA256,
a key-derivation function that turns it into uniformly random keys:

    stego_key = HKDF-SHA256(shared_secret, salt=None, info="stegaudit-v1")  (32 bytes)
    mask_key  = HKDF-Expand(stego_key, info="stegaudit-v1/mask")            (32 bytes)
    tag_key   = HKDF-Expand(stego_key, info="stegaudit-v1/tag")             (32 bytes)

Why two sub-keys: key separation. mask_key only drives the PRG that hides the
payload bits; tag_key only authenticates the payload (HMAC tag). Using one key
for two jobs is a classic source of unexpected interactions; deriving
independent keys with different `info` labels rules that out cheaply.

salt=None: HKDF then uses an all-zero salt, which RFC 5869 allows. A salt adds
strength when the input secret is weak or reused; here the input is a fresh
DH secret, so a salt adds nothing essential. The `info` string binds the keys
to this protocol and version, so a future "stegaudit-v2" gets unrelated keys.

Threat model: the key is PRE-SHARED. The colluders run this exchange before
deployment, not over the monitored channel, because an X25519 exchange visible
to the monitor would itself look suspicious. Everything after this point
assumes the monitor does not know the key.

Security assumptions
--------------------
* Computational Diffie-Hellman is hard on Curve25519 (standard).
* Private keys come from a cryptographic RNG (os.urandom via `generate()`).
* HKDF-SHA256 behaves as a pseudorandom function.

What breaks if they fail
------------------------
* A predictable private key (e.g. `Party.from_seed`, which exists ONLY so that
  experiments are reproducible) lets anyone who knows the seed recompute all
  keys and decode every message. Fine for a lab pilot, never for real use.
* A low-order public key from a malicious peer would force an all-zero shared
  secret, which is known to everyone. The cryptography library rejects this,
  and `derive_keys` checks again explicitly.
* If the monitor learns stego_key, Meteor gives no secrecy at all: the monitor
  can decode exactly like the Receiver (this is control (ii) in Phase 2).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF, HKDFExpand
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

INFO = b"stegaudit-v1"
KEY_LEN = 32


@dataclass(frozen=True)
class StegoKeys:
    """The keys both colluders end up with. `repr` hides them so they never land in logs."""
    stego_key: bytes = field(repr=False)
    mask_key: bytes = field(repr=False)
    tag_key: bytes = field(repr=False)

    def fingerprint(self) -> str:
        """Short public identifier (first 8 hex chars of SHA-256 of stego_key), safe to log."""
        return hashlib.sha256(self.stego_key).hexdigest()[:8]


def _expand(key: bytes, label: bytes) -> bytes:
    return HKDFExpand(algorithm=hashes.SHA256(), length=KEY_LEN, info=INFO + b"/" + label).derive(key)


def keys_from_shared_secret(shared: bytes) -> StegoKeys:
    """HKDF-SHA256 the DH output into stego_key, then expand into mask_key and tag_key."""
    if shared == bytes(len(shared)):
        raise ValueError("all-zero shared secret (low-order public key); refusing")
    stego_key = HKDF(algorithm=hashes.SHA256(), length=KEY_LEN, salt=None, info=INFO).derive(shared)
    return StegoKeys(stego_key, _expand(stego_key, b"mask"), _expand(stego_key, b"tag"))


class Party:
    """One colluder (Sender or Receiver) holding an X25519 key pair."""

    def __init__(self, private_key: X25519PrivateKey | None = None):
        self._sk = private_key or X25519PrivateKey.generate()

    @classmethod
    def from_seed(cls, seed: int, name: str) -> "Party":
        """Deterministic key pair for REPRODUCIBLE EXPERIMENTS ONLY.

        The private key is SHA-256(f"stegaudit-test-key|{name}|{seed}"), so anyone
        who knows (name, seed) knows the key. Never use this outside the lab.
        """
        raw = hashlib.sha256(f"stegaudit-test-key|{name}|{seed}".encode()).digest()
        return cls(X25519PrivateKey.from_private_bytes(raw))

    def public_bytes(self) -> bytes:
        """32-byte public key; the only thing this party ever sends to the other."""
        return self._sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    def derive_keys(self, peer_public: bytes) -> StegoKeys:
        """Combine my private key with the peer's public key -> shared StegoKeys."""
        shared = self._sk.exchange(X25519PublicKey.from_public_bytes(peer_public))
        return keys_from_shared_secret(shared)


def pair_from_seed(seed: int) -> tuple[StegoKeys, StegoKeys]:
    """Convenience for experiments: Sender and Receiver run the exchange; returns both views."""
    sender, receiver = Party.from_seed(seed, "sender"), Party.from_seed(seed, "receiver")
    return sender.derive_keys(receiver.public_bytes()), receiver.derive_keys(sender.public_bytes())
