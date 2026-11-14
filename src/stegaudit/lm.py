"""Local language model wrapper: the model that carries the stego channel.

Idea in plain words
-------------------
Meteor hides bits in the choice of each token. To do that, the Sender and
the Receiver must both compute the model's next-token probabilities, and
they must get exactly the same numbers, bit for bit. If one side's
probability for some token differs in the last bit, the token intervals can
shift and the Receiver reads the wrong bits.

This module loads a small Qwen model on the CPU in a way that makes those
numbers reproducible:
  * a pinned model revision (an HF commit hash), so the weights cannot change;
  * fp32 weights, CPU only, no GPU kernels;
  * torch.use_deterministic_algorithms(True), so torch refuses to run
    operations that have no deterministic implementation;
  * a single CPU thread, so floating-point sums are always added in the
    same order (with several threads the order, and so the last bits, can vary);
  * "eager" attention (plain matmul + softmax), the simplest code path.

Security assumption
-------------------
This module has nothing to do with secrecy. It is about correctness: the
assumption is that Sender and Receiver run the same model revision, the same
library versions and this same code path on the same kind of machine.

What breaks if the assumption fails
-----------------------------------
A different torch or transformers version, a different CPU instruction set,
or a different number of threads can change the low bits of the logits. Then
decoding returns wrong bits. The HMAC tag catches this (decode reports a
failure instead of garbage), but the message is lost. Determinism here is
only claimed for the same machine and software versions; every result JSON
records those versions (see provenance.py).

Phase 0 scope: load the model and return next-token logits for a prompt.
The incremental KV-cache step API used by Meteor is added in Phase 1.
"""

from __future__ import annotations

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from stegaudit.config import MeteorConfig


def setup_determinism(seed: int = 0) -> None:
    """Make torch on the CPU reproducible. Call this once before loading the model."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)


class LocalLM:
    """A Qwen chat model on the CPU, loaded at a pinned revision in fp32."""

    def __init__(self, cfg: MeteorConfig | None = None):
        self.cfg = cfg or MeteorConfig()
        setup_determinism(self.cfg.seed)
        # Download (or reuse from the HF cache) exactly the pinned commit, then load
        # from that local folder. The folder is named .../snapshots/<commit hash>, so
        # the files read are provably that revision.
        self.snapshot_dir = snapshot_download(
            self.cfg.model_name, revision=self.cfg.model_revision
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.snapshot_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.snapshot_dir,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        self.model.eval()

    def chat_prompt_ids(self, user_text: str) -> list[int]:
        """Token ids for a one-turn chat prompt, ending where the assistant's reply starts.

        Sender and Receiver must build this identically. It uses the model's own
        chat template (which includes Qwen's default system message).
        """
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    @torch.inference_mode()
    def next_token_logits(self, ids: list[int]) -> torch.Tensor:
        """fp32 logits over the full vocabulary for the token after `ids` (one full forward pass)."""
        out = self.model(torch.tensor([ids], dtype=torch.long))
        return out.logits[0, -1].clone()
