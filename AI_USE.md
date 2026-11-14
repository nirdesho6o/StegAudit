# AI use log

Course rule: everything Claude Code generated is listed here, with what Nirdesh changed or verified.
The "Nirdesh verified / changed" column is filled in by Nirdesh, not by Claude.

## Log

| Date | Commit stage | What Claude Code generated | What Nirdesh changed / verified |
|---|---|---|---|
| 2026-11-15 | Phase 0, commit 1 | `.gitignore` (secrets, cache, weights, datasets, papers). CLAUDE.md was written by Nirdesh. | |
| 2026-11-15 | Phase 0, commit 2 | `pyproject.toml`, pinned `requirements.txt`, `src/stegaudit/__init__.py`, `config.py`, `tests/test_smoke.py`, stubs of README / NOTES / AI_USE / REFERENCES / `results/LOG.md`. Read (read-only) `apart-sprint/swarm-sprint/swarm/llm.py` to copy the Adaption SDK choice and version. | Answered the plan's open questions (Adaption SDK source, EOS policy, beta sweep, prefix detectors, git). |
| 2026-11-15 | Phase 0, references | Reformatted REFERENCES.md with [VERIFIED] / [VERIFY] / TODO tags. Claude Code's first draft had author lists written from memory. | Checked Anwar et al. (2026) and Motwani et al. (2024) against the PDFs and supplied the corrected entries, including the real Anwar title (Claude's draft had only a placeholder marked TODO). Supplied the Kiekintveld et al. (2009) entry. Meteor, Audit Games and Kiekintveld are still to be checked against the PDFs. |
| 2026-11-15 | Phase 0, commit 3 | `lm.py` (deterministic CPU load of Qwen from the pinned snapshot, chat prompt ids, next-token logits), `provenance.py` (metadata for every result JSON), pinned `MODEL_REVISIONS` in `config.py`, `tests/test_lm.py`, `tests/test_provenance.py`. First attempt checked `config._commit_hash`, which transformers 5 no longer sets; changed to loading from the snapshot folder named by the hash. | Asked for: the revision pin, revision + torch/transformers versions in every result JSON, and a bit-for-bit two-load determinism test. |
| 2026-11-15 | Phase 0, commit 4 | `remote.py` (AsyncAdaption wrapper copying swarm-sprint's call pattern; response cache; credits log; cost gate), `tests/test_remote.py` (fake SDK, no network), `experiments/00_adaption_check.py`. Checked the `adaption` 0.19.0 SDK type definitions to see which metadata fields exist. | Asked for: a tiny live test call (≤ 16 output tokens) with its cost printed, server metadata saved, and the note that remote reproducibility comes from the cache. |

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
15. **Model revision pinned by HF commit hash; loaded from `snapshots/<hash>`; eager attention; 1 thread.** *(Revision pin requested by Nirdesh.)*
    Why: a later upload to the HF repo could otherwise change weights or tokenizer under the same name. Loading from the hash-named folder makes the revision checkable. Eager attention is the simplest, most transparent code path. One thread fixes the order of floating-point additions.
16. **Every result JSON gets `meta`:** git commit + dirty flag, seed, model + revision, full config, torch / transformers / tokenizers versions, platform, timestamp.
    Why: bit-identical logits are only guaranteed for the same software versions, so a result is only reproducible if those are recorded.

### Remote models (Adaption)
17. **Reproducibility of remote answers comes from the response cache, not from the model.** *(Nirdesh's instruction.)*
    Why: temperature 0 on a hosted model is not guaranteed deterministic. Batching, kernels and server-side updates can change the text, and the model behind an id can change without the id changing. Every answer is cached under `cache/` keyed by SHA-256 of (model, instructions, input, max_output_tokens, temperature), so reruns read the exact same text at zero cost. Deleting the cache means a new sample, which must be reported as such. No test assumes two live calls agree.
18. **Each cache entry and `results/adaption_models.json` store the server's own metadata:** model name as returned, response id, `created_at`; for listed models: id, created, owned_by, display_name, shutdown_date. *(Nirdesh's instruction.)*
    Why: this makes a silent model change behind the same id visible afterwards. The SDK has no explicit model-version field, so these are the strongest evidence available.
19. **The cache key covers all request fields, not only (model, prompt hash) as CLAUDE.md says.**
    Why: the same prompt at a different temperature or token limit is a different request and must not return a cached answer meant for the other one.
20. **An unpriced model's cost is `None` ("unknown"), never 0.** Prices are copied from swarm-sprint's table.
    Why: a silent 0 would make the cost gate pass when it shouldn't.
21. **Reasoning tokens are read from `usage.output_tokens_details` as a dict.**
    Why: in `adaption` 0.19.0 it is a dict. swarm-sprint read it with `getattr`, which would always give 0. This is the only deliberate difference from swarm-sprint's client.

### Phase 2 rules (set by Nirdesh on 2026-11-15, before any Phase 2 code)
22. **Cost estimates use billed token totals from cached pilot calls, including reasoning tokens, never the visible answer length.** *(Nirdesh's rule.)* Implemented as `remote.cached_token_stats` / `estimate_cost_from_cache`.
    Why: Phase 0 showed a 1-word answer ("ok", about 3 visible tokens) billed 24 output tokens, 21 of them reasoning. Estimating from visible text would under-count by about 8×. `usage.output_tokens` already includes reasoning (observed 24 ≥ 21 and 16 = 16), so billed output = `output_tokens`.
23. **Incomplete or unparseable answers are parse failures, logged and reported per model and per condition. If any condition's failure rate is above 5 %, stop and report to Nirdesh; do not drop the items.** *(Nirdesh's rule; also goes into PREREG.md.)*
    Why: if failures were dropped, and they happened more often for the Sentinel than the Receiver (or the reverse), the remaining items would differ between the two. The accuracy gap would then be biased by which items survived, not by who had the key.
24. **RQ1c Sentinels: same family = `qwen3.5-397b` (or `qwen3.8-27b` if cost matters); cross family = `glm-5.2` or `deepseek-v4-flash`.** *(Nirdesh's choice.)*
    Meaning of "same family": the same Qwen lineage as the local 0.5B channel model (Qwen2.5-0.5B-Instruct), NOT the same model and not the same size or version. A 397B Qwen 3.5 and a 0.5B Qwen 2.5 share a developer and likely overlapping training data and tokenizer lineage, which is the reason a same-family Sentinel might read the channel model's style better. It does not mean the Sentinel can compute the channel model's distributions, so it still cannot decode Meteor without the key.
