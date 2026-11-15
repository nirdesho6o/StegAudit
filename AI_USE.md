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
| 2026-11-15 | Phase 0, close | `--max-output-tokens` option for `00_adaption_check.py`; `cached_token_stats` / `estimate_cost_from_cache` in `remote.py` + test; Phase 0 viva questions in NOTES.md (questions only); README install/test/experiment sections; LOG entries for both live calls. | Approved the 256-token retry; set the Phase 2 rules (decisions 22–24). Viva answers: to be written by Nirdesh. |
| 2026-11-15 | Phase 1, keys.py | `keys.py` (X25519 `Party`, HKDF-SHA256 → stego_key → mask_key / tag_key, seeded parties for experiments, explicit all-zero secret check), `tests/test_keys.py` (incl. an independent RFC 5869 HKDF written with stdlib `hmac` as outside judge). | |
| 2026-11-15 | Phase 1, prg.py | `prg.py` (HMAC-SHA256 over a fixed-width (label, msg_id, step) encoding, top-n-bits output, `PRGS` registry), `tests/test_prg.py` (literal byte-layout spec checked with stdlib hmac, distinct masks per msg_id/step/label/key, bit balance, bad-argument rejection). | |
| 2026-11-15 | Phase 1, quantize.py | `quantize.py` (`quantize` → `QuantizedDist` with widths / lo / find / n_vanished / remainder; `tv_distance`), `tests/test_quantize.py` (exact tiling at beta 8–62, per-token error < 2^-β, TV < k·2^-β, exhaustive beta = 8 lookup vs linear scan, vanishing boundary at exactly 2^-β). | |
| 2026-11-15 | Phase 1, lm.py step API | `topk_dist` (top-k sorted by token id, float64 softmax, EOS masking before top-k with `eos_mass`), `Session` (prefill + one-token KV-cache steps), `eos_ids`, `ids_to_text` / `text_to_ids`; `tests/conftest.py` (one shared Qwen load), `tests/test_lm_step.py`. Measured the KV-vs-full-recompute difference and wrote it into the docstring. | |
| 2026-11-15 | Phase 1, meteor.py | `meteor.py` (payload = msg ‖ HMAC tag; `encode_step` / `decode_step`; `encode` with EOS masking + cover stream; `read_intervals` (model, no key) / `unmask` (key, no model); `decode_ids`, `decode`), worked β = 4 example in the docstring; `tests/toy_lm.py`, `tests/test_meteor_toy.py`, `tests/test_meteor_qwen.py`. First version of one toy test wrongly claimed EOS is in the top-k at *every* payload step; corrected to "most steps". First version of the Qwen round-trip test failed on a real retokenisation mismatch (msg 29); the test was split into ids-path (100 % required) and text-path (mismatch must be detected, never accepted), see decision 42. | |
| 2026-11-15 | Phase 1, docs | Edited CLAUDE.md's round-trip requirement (Phase 1 tests + P1a) to name the token-id path; added Nozaki & Murawaki (2022) and SyncPool (Qi et al. 2024) to REFERENCES.md as [VERIFY]. | Decided: keep the ids/text split (decision 44); chose the two candidate papers and the >5 % kill rule. |

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

### Phase 1: keys.py
25. **Key schedule:** `stego_key = HKDF-SHA256(shared, salt=None, info="stegaudit-v1")` (the 32-byte key in CLAUDE.md), then `mask_key` / `tag_key = HKDF-Expand(stego_key, info="stegaudit-v1/mask" | ".../tag")`.
    Why: keeps CLAUDE.md's 32-byte stego key as the single root and adds key separation (decision 3) with standard HKDF labels. salt=None (all-zero salt) is allowed by RFC 5869 and adds nothing essential for a fresh DH secret.
26. **`Party.from_seed` / `pair_from_seed` derive private keys from SHA-256 of (name, seed).**
    Why: experiments must be reproducible from a logged seed. Cost: anyone who knows the seed knows the key. Documented as lab-only; the security claim of the pilot is about what a monitor *without* the key sees, which this does not change.
27. **Explicit all-zero shared-secret check, and `StegoKeys.__repr__` hides key bytes; `fingerprint()` (8 hex chars of SHA-256) is what gets logged.**
    Why: the library already rejects low-order points, but the check makes the protection visible in our code. Hiding keys from repr prevents a key from leaking into result JSONs or logs by accident.

### Phase 1: prg.py
28. **Mask = the TOP n bits of one HMAC-SHA256 output; at most 256 bits per (label, msg_id, step).**
    Why: one HMAC call per token step is enough for any beta ≤ 62. Taking the top bits means a shorter mask is a prefix of a longer one, which is easy to check by hand.
29. **Label is length-prefixed (1 byte) and limited to {"payload", "cover"}; msg_id and step are u64 big-endian.**
    Why: the length prefix keeps the encoding unambiguous even if labels of different lengths are added later. Rejecting unknown labels stops a typo from silently creating a third stream.
30. **`PRGS` registry keyed by `MeteorConfig.prg`.**
    Why: "swap the PRG" is a viva modification point in CLAUDE.md; it becomes one new function plus one registry line.

### Phase 1: quantize.py
31. **Rounding: `width_i = floor(p_i · 2^β)`, the leftover units go to the argmax token (ties → lowest position).**
    Why: the simplest scheme that sums to exactly 2^β. It gives a clean bound (every non-argmax token off by < 2^-β; TV < k·2^-β), and it makes "vanishing" exactly `p_i < 2^-β`, matching CLAUDE.md. A minimum-width-1 scheme would remove vanishing but distort the distribution more and would change the phenomenon under study.
32. **`find(x)` = first position whose interval end exceeds x (`searchsorted(hi, x, "right")`).**
    Why: an empty interval has end = start ≤ x, so it is skipped automatically. Meteor can therefore never emit a vanished token, with no special case. Checked exhaustively at β = 8 against a linear scan.
33. **`n_vanished` counts only tokens with p > 0 and width 0; β is limited to [1, 62].**
    Why: a token with p = 0 never had any probability, so it does not count as vanished. The 62-bit cap keeps all sums inside int64. CLAUDE.md's β values (8–32) are well inside it.

### Phase 1: lm.py step API
34. **EOS is masked by setting its logits to -inf BEFORE top-k**, so k non-EOS tokens remain. `eos_mass` = probability EOS would have had in the unmasked top-k.
    Why: masking after top-k would leave fewer than k tokens and change the support size by step. `eos_mass` is what experiments log for "how often and how strongly EOS was masked" (Nirdesh's requirement on decision 9).
35. **EOS set = generation_config.eos_token_id ∪ tokenizer.eos_token_id** (Qwen2.5: 151643 `<|endoftext|>`, 151645 `<|im_end|>`).
    Why: either token ends a Qwen chat reply, so both must be masked.
36. **Fixed order = ascending token id after top-k.**
    Why: token ids never change between runs; sorting by probability could reorder near-ties.
37. **Measured, not assumed: KV-cache steps and full recompute differ in the low bits** (146,520 / 151,936 logits, max 2.0e-5 after 10 tokens).
    Why: this is the evidence for decision 5 (same code path on both sides). The test checks only `allclose` between the two paths, and exact equality between two runs of the same path.
38. **`ids_to_text` / `text_to_ids`: no special tokens, `clean_up_tokenization_spaces=False`.**
    Why: the text must be exactly the concatenation of the token strings; any clean-up would add retokenisation mismatches that are not BPE's fault.

### Phase 1: meteor.py
39. **Tag = top t bits of HMAC-SHA256(tag_key, "stegaudit-v1/tag" ‖ u64 msg_id ‖ u16 len ‖ message bytes); compared with `hmac.compare_digest`.**
    Why: binding msg_id stops a payload from being accepted under another message number. The length field makes the byte encoding unambiguous. Constant-time comparison is standard practice for tags.
40. **Decoding is split into `read_intervals` (needs the model, not the key) and `unmask` (needs the key, not the model).**
    Why: it shows the security idea directly. How many bits each token carries is public; only their values are hidden. It also lets the 300 wrong-key trials on Qwen reuse one model replay per text: 10 texts × 30 wrong keys, each a full, independent tag check on real Qwen intervals.
41. **The cover phase samples from the same quantised top-k distribution with the "cover" PRG stream; EOS is never included in ids/text; a `PayloadTooLong` error after `payload_token_limit` tokens.**
    Why: the cover phase keeps one sampling mechanism throughout, makes texts reproducible from (key, msg_id), and stops a degenerate distribution (e.g. top_k = 1, which carries 0 bits per token) from looping forever.
42. **Round-trip test split after a real failure (2026-11-15):** `decode_ids` must be 100 % (50/50 passed). `decode(text)` must decode every message that retokenises cleanly and must never accept a wrong message; mismatches are counted, not failed on.
    Why: on the first run, 1 of 50 Qwen messages (msg 29) had the model generate `Program` + `ming`, which the tokenizer reads back as `Programming`. The Receiver then saw one token too few and reported `text_too_short`. This is a BPE property, not a Meteor bug: the ids path decoded it correctly. CLAUDE.md's "decode == encode 100 %" holds for the ids path; for the text path it does not, and the test now says so instead of hiding it. The rate on 200 messages is measured in experiment 01. **Open for Nirdesh:** whether to keep this split or require a fix (e.g. a Sender-side retokenisation check); any fix must come from the literature, not be invented here.
43. **`slow` marker on the Qwen Meteor tests.** `pytest -q` runs everything (about 7 min on CPU); `pytest -q -m "not slow"` skips only `test_meteor_qwen.py` (about 1 min) for quick checks during live edits.

### Phase 1: round-trip definition (Nirdesh's decision, 2026-11-15)
44. **CLAUDE.md's "decode == encode, 100 %" now refers to the token-id path (`decode_ids`).** The text path must decode every cleanly retokenising message, and must never accept a mismatch as a wrong message. Mismatches are counted. *(Nirdesh's decision, confirming decision 42; CLAUDE.md edited accordingly.)*
    Why: the ids path tests Meteor itself, and the text path adds BPE tokenisation, which Meteor does not control. The pilot found 1/50 mismatches. The mismatch rate is pre-registered as P1d with a kill rule (> 5 % → implement a published disambiguation method before Phase 2). Candidates: Nozaki & Murawaki 2022, SyncPool (Qi et al. 2024); neither is implemented now, and both must be checked against the papers, not memory.
