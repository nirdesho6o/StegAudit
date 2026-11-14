"""All tunable parameters in one place.

Idea in plain words
-------------------
The viva may ask for a live change ("set beta to 16", "use top-k 20", "switch
the model"). Every such knob lives here as one named field, so a change is a
one-line edit or a keyword argument, never a hunt through the code.

Security assumption
-------------------
Sender and Receiver MUST use the same config. Any mismatch in model_name,
beta, top_k, temperature or prg changes the distributions they compute, and
decoding silently returns garbage (the HMAC tag check then rejects it).

Usage
-----
    from stegaudit.config import MeteorConfig
    cfg = MeteorConfig()              # defaults below
    cfg16 = MeteorConfig(beta=16)     # one-parameter change
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Local channel models (Meteor runs ONLY on these, never on a remote API).
MODEL_SMALL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_LARGE = "Qwen/Qwen2.5-1.5B-Instruct"

# Pinned Hugging Face commit hashes (looked up 2026-11-15 with
# huggingface_hub.HfApi().model_info(name).sha). Loading by hash means a later
# upload to the model repo cannot silently change the weights or tokenizer,
# which would change the distributions and break reproducibility.
MODEL_REVISIONS = {
    MODEL_SMALL: "7ae557604adf67be50417f59c2c2f167def9a775",
    MODEL_LARGE: "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
}


@dataclass(frozen=True)
class MeteorConfig:
    # Local model used for the stego channel.
    model_name: str = MODEL_SMALL
    # Precision beta: probabilities are quantised to integers summing to 2**beta.
    beta: int = 32
    # Keep only the k most likely tokens, renormalise (applies to Meteor AND
    # to the normal-sampling baseline, so comparisons are fair).
    top_k: int = 50
    # Sampling temperature (same for Meteor and baseline).
    temperature: float = 1.0
    # Total payload length L in bits = message bits + truncated HMAC tag.
    payload_bits: int = 16
    # How many of the payload bits are the authentication tag.
    # Wrong-key false-accept probability is about 2**-tag_bits.
    tag_bits: int = 8
    # Name of the PRG construction (only one implemented; kept explicit so a
    # viva question "swap the PRG" has an obvious place to go).
    prg: str = "hmac-sha256"
    # Max new tokens for the WHOLE text. During the payload phase this cap
    # does not cut (EOS is also masked); see payload_token_limit.
    max_new_tokens: int = 120
    # Hard safety stop for the payload phase: if the payload is not fully
    # sent after this many tokens, encoding raises instead of looping forever.
    payload_token_limit: int = 256
    # Seed for anything not driven by the key (e.g. picking random payloads).
    seed: int = 0

    @property
    def message_bits(self) -> int:
        return self.payload_bits - self.tag_bits

    @property
    def model_revision(self) -> str:
        """Pinned HF commit hash for model_name (see MODEL_REVISIONS)."""
        return MODEL_REVISIONS[self.model_name]

    def __post_init__(self) -> None:
        if self.model_name not in MODEL_REVISIONS:
            raise ValueError(f"no pinned revision for {self.model_name}; add it to MODEL_REVISIONS")
        if not 1 <= self.beta <= 62:
            raise ValueError("beta must be in [1, 62]")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0 < self.tag_bits < self.payload_bits:
            raise ValueError("need 0 < tag_bits < payload_bits")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")

    def to_dict(self) -> dict:
        """For writing into every result JSON."""
        d = asdict(self)
        d["model_revision"] = self.model_revision
        return d
