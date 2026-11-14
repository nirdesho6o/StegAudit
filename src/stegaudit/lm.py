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

The step API used by Meteor
---------------------------
    sess = lm.start(prompt_ids)          # one forward pass over the prompt ("prefill")
    d = sess.dist(mask_eos=True)         # top-k distribution for the next token
    sess.advance(token_id)               # feed ONE token, reuse the KV cache

KV cache: after the prefill, the model keeps the attention keys and values of
every earlier token, so each new token costs one small forward pass instead of
re-reading the whole text. The incremental path and a full recompute give
logits that are close but NOT bit-identical (different order of floating-point
operations). Measured 2026-11-15 after 10 tokens: 146,520 of 151,936 logits
differed, max |difference| 2.0e-5. Tiny, but enough to move an interval edge
at beta = 32. So Sender and Receiver must BOTH use exactly this path: prefill
the prompt, then one token at a time.

Fixed token order: `topk_dist` keeps the k most likely tokens and then sorts
them by TOKEN ID (ascending). Sorting by probability would also be
deterministic, but ties and near-ties could reorder; token ids never change.

EOS masking (decision 9 in AI_USE.md, a deviation from pure Meteor): while the
payload is being sent, the end-of-sequence tokens (`<|im_end|>`,
`<|endoftext|>` for Qwen2.5) get logit -inf BEFORE top-k is taken, so the text
cannot end before all payload bits are out. `StepDist.eos_mass` records how
much probability the EOS tokens would have had at that step (0 if they would
not have been in the top-k), so experiments can log how often and how strongly
the masking changed the distribution.

Float64: the k selected fp32 logits are cast to float64 before temperature
and softmax, so quantisation at beta = 32 works on meaningful bits
(decision 4).

Text <-> ids: `ids_to_text` and `text_to_ids` never add special tokens and
never "clean up" spaces, so text is exactly the concatenation of the token
strings. Even so, `text_to_ids(ids_to_text(ids)) == ids` is NOT guaranteed
(BPE can split a string differently from how it was generated); meteor.py
counts such retokenisation mismatches instead of assuming them away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from stegaudit.config import MeteorConfig


@dataclass(frozen=True)
class StepDist:
    """The next-token distribution as Meteor sees it, in the fixed order."""
    ids: np.ndarray       # int64 token ids, ascending
    probs: np.ndarray     # float64, same order, sums to 1
    eos_mass: float = 0.0 # mass EOS would have had in the unmasked top-k (0 if not masked)


def topk_dist(logits: torch.Tensor, k: int, temperature: float,
              eos_ids: tuple[int, ...] = (), mask_eos: bool = False) -> StepDist:
    """Top-k of fp32 logits, sorted by token id, softmax in float64 at `temperature`.

    With mask_eos=True the EOS ids are removed BEFORE taking the top-k (so k
    non-EOS tokens remain), and eos_mass reports what EOS would have had.
    """
    logits = logits.detach().to(torch.float32).flatten()
    eos_mass = 0.0
    if mask_eos and eos_ids:
        unmasked = topk_dist(logits, k, temperature)
        eos_mass = float(unmasked.probs[np.isin(unmasked.ids, eos_ids)].sum())
        logits = logits.clone()
        logits[list(eos_ids)] = float("-inf")
    idx = torch.topk(logits, k).indices
    ids = torch.sort(idx).values                      # fixed order: by token id
    z = logits[ids].numpy().astype(np.float64) / temperature
    p = np.exp(z - z.max())
    return StepDist(ids.numpy().astype(np.int64), p / p.sum(), eos_mass)


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

    # ---------- step API (Phase 1) ----------
    @property
    def eos_ids(self) -> tuple[int, ...]:
        """All token ids that end generation (generation_config + tokenizer)."""
        g = self.model.generation_config.eos_token_id
        ids = set(g if isinstance(g, (list, tuple)) else [g]) | {self.tokenizer.eos_token_id}
        return tuple(sorted(i for i in ids if i is not None))

    def start(self, prompt_ids: list[int]) -> "Session":
        return Session(self, prompt_ids)

    def ids_to_text(self, ids) -> str:
        return self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)

    def text_to_ids(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]


class Session:
    """One generation in progress: prefill once, then one token per step with the KV cache.

    Sender and Receiver each run their own Session over the same tokens; this
    single code path is what makes their distributions bit-identical.
    """

    def __init__(self, lm: LocalLM, prompt_ids: list[int]):
        self.lm = lm
        self._cache = None
        self._logits = self._forward(list(prompt_ids))   # prefill
        self.generated: list[int] = []

    @torch.inference_mode()
    def _forward(self, ids: list[int]) -> torch.Tensor:
        out = self.lm.model(torch.tensor([ids], dtype=torch.long),
                            past_key_values=self._cache, use_cache=True)
        self._cache = out.past_key_values
        return out.logits[0, -1].clone()

    def dist(self, mask_eos: bool = False) -> StepDist:
        cfg = self.lm.cfg
        return topk_dist(self._logits, cfg.top_k, cfg.temperature, self.lm.eos_ids, mask_eos)

    def advance(self, token_id: int) -> None:
        self._logits = self._forward([int(token_id)])
        self.generated.append(int(token_id))
