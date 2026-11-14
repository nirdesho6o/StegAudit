"""A tiny deterministic stand-in for LocalLM, so Meteor logic can be tested in milliseconds.

Same interface as LocalLM: cfg, eos_ids, chat_prompt_ids, start -> session with
dist(mask_eos) / advance, ids_to_text, text_to_ids. The "model" draws logits
from a RNG seeded by SHA-256 of the whole context, so the same context always
gives the same distribution. Token 0 is EOS (empty text); tokens 1..V-1 are
single distinct characters, so text <-> ids is always exact.
"""

import hashlib

import numpy as np
import torch

from stegaudit.config import MeteorConfig
from stegaudit.lm import topk_dist

CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP"


class ToyLM:
    def __init__(self, cfg=None, vocab=30, scale=2.0, eos_bias=0.0):
        self.cfg = cfg or MeteorConfig(top_k=20)
        self.vocab, self.scale, self.eos_bias = vocab, scale, eos_bias
        self.eos_ids = (0,)

    def chat_prompt_ids(self, text):
        return [1 + b % (self.vocab - 1) for b in hashlib.sha256(text.encode()).digest()[:8]]

    def start(self, prompt_ids):
        return ToySession(self, list(prompt_ids))

    def ids_to_text(self, ids):
        return "".join(CHARS[i - 1] for i in ids if i != 0)

    def text_to_ids(self, text):
        return [CHARS.index(c) + 1 for c in text]

    def logits(self, context):
        seed = int.from_bytes(hashlib.sha256(np.array(context, dtype=np.int64).tobytes()).digest()[:8], "big")
        z = np.random.default_rng(seed).normal(0, self.scale, self.vocab)
        z[0] += self.eos_bias
        return torch.tensor(z, dtype=torch.float32)


class ToySession:
    def __init__(self, lm, context):
        self.lm, self.context = lm, context

    def dist(self, mask_eos=False):
        c = self.lm.cfg
        return topk_dist(self.lm.logits(self.context), c.top_k, c.temperature, self.lm.eos_ids, mask_eos)

    def advance(self, tok):
        self.context = self.context + [int(tok)]
