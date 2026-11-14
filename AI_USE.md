# AI use log

Course rule: everything Claude Code generated is listed here, with what Nirdesh changed or verified.
The "Nirdesh verified / changed" column is filled in by Nirdesh, not by Claude.

## Log

| Date | Commit stage | What Claude Code generated | What Nirdesh changed / verified |
|---|---|---|---|
| 2026-11-15 | Phase 0, commit 1 | `.gitignore` (secrets, cache, weights, datasets, papers). CLAUDE.md was written by Nirdesh. | |
| 2026-11-15 | Phase 0, commit 2 | `pyproject.toml`, pinned `requirements.txt`, `src/stegaudit/__init__.py`, `config.py`, `tests/test_smoke.py`, stubs of README / NOTES / AI_USE / REFERENCES / `results/LOG.md`. Read (read-only) `apart-sprint/swarm-sprint/swarm/llm.py` to copy the Adaption SDK choice and version. | Answered the plan's open questions (Adaption SDK source, EOS policy, beta sweep, prefix detectors, git). |

## Design decisions (proposed by Claude Code, approved by Nirdesh on 2026-11-15)

Each one is a choice CLAUDE.md left open or that the plain spec would get wrong. One line on *why*, so it can be defended in the viva.

### Cryptography / Meteor
1. **msg_id is a message counter that both sides keep, not a hash of the prompt.**
   Why: the mask must never repeat under the same key. 200 messages over 20 prompts would reuse each prompt-derived mask 10 times, which is the two-time-pad mistake (XOR of two payloads leaks). Both sides know the counter, so the Receiver needs nothing extra.
2. **PRG input uses fixed-width fields and a domain label:** `HMAC-SHA256(k, label || u64(msg_id) || u64(step))`, with labels such as `payload` and `cover`.
   Why: plain concatenation is ambiguous ("1"‖"12" = "11"‖"2"), which could make two different (msg_id, step) pairs share a mask. The label keeps the payload stream and the cover stream from ever overlapping.
3. **Separate mask key and tag key from HKDF.**
   Why: key separation. One key should do one job, so a weakness in how the mask is used cannot leak into the authentication tag, and the reverse.
4. **Quantisation happens in float64, while the model runs in fp32.**
   Why: fp32 has about 24 bits of precision, so at beta = 32 the low bits of `p * 2^32` would be rounding noise. Casting the top-k logits to float64 before softmax and quantisation is deterministic and gives meaningful 32-bit intervals.
5. **Sender and Receiver use the exact same KV-cache code path** (prefill the prompt, then one token per step).
   Why: a full recompute and an incremental KV-cache step give slightly different fp32 logits. Meteor needs bit-identical distributions on both sides, so both must run the same arithmetic. Limitation: this is guaranteed only on the same machine with the same torch version.
6. **`decode_ids` path and a retokenisation mismatch count.**
   Why: BPE does not guarantee `tokenize(detokenize(ids)) == ids`. Decoding from text can fail for a tokeniser reason that has nothing to do with Meteor. Decoding from ids as well lets us tell the two causes apart, and the mismatch rate is reported, not hidden.
7. **Wrong-key test uses 300 trials.**
   Why: with an 8-bit tag a wrong key passes about 1/256 of the time. Expected ~1.2 false accepts in 300, so the ≥ 99 % threshold has a safe margin and the test is not flaky.
8. **Threat model: the key is pre-shared.** The X25519 exchange happens before deployment, not over the monitored channel.
   Why: a key exchange visible to the monitor would itself be evidence of collusion.
9. **EOS is masked, and max_tokens cannot cut, until the whole payload is sent.** *(Nirdesh's choice; deviation from pure Meteor.)*
   Why: otherwise the model can stop before all L bits are sent and decoding fails. Cost: the payload part of the text can never end early, which can make it slightly detectable. This is logged (how often EOS was masked) and pre-registered.

### Experiments / hygiene
10. **PREREG.md is committed before experiment 01**, not only before 02–04.
    Why: prediction P1a is about experiment 01.
11. **β sweep {8, 12, 16, 32} for the vanishing-token rate and detector** (32 stays the default).
    Why: with top-k = 50 at β = 32, no kept token realistically has p < 2^-32. The detector would give AUROC 0.5 by construction, which says nothing about Meteor.
12. **Detectors run on the whole text AND on the payload-carrying prefix only.**
    Why: 16 bits use only about 5–10 tokens out of 100+. Whole-text statistics would be diluted by cover text and pushed toward 0.5 for a trivial reason.
13. **src layout + `pyproject.toml`; the whole repo root is the `stegaudit/` folder from CLAUDE.md.**
    Why: `import stegaudit` from `src/` needs an installed package (`pip install -e .`); requirements.txt alone does not do it.
14. **Adaption client copies `apart-sprint/swarm-sprint`:** `adaption==0.19.0` SDK, `client.responses.create(...)`, only `ADAPTION_API_KEY`. It does not use the OpenAI client.
    Why: Nirdesh's instruction. This is the SDK and version already used successfully in that project.
