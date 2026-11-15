# Results log

Plain-sentence outcome of every run vs its pre-registered prediction, including misses and failures.

## 2026-11-15 — 00_adaption_check (Phase 0 live check), code commit 5c020af

- **List models: worked.** The key sees 7 hosted models: glm-5.3-flash, glm-5.2, qwen3.5-397b, gemma-4-26b-a4b-it, qwen3.8-27b, deepseek-v4-flash, gemma-4-31b-it. All fields are saved in `results/adaption_models.json`. Families available: Qwen, GLM, DeepSeek, Gemma.
- **Test chat call: the API round trip worked, but there was no answer text.** deepseek-v4-flash (the cheapest priced model) got the prompt "Say ok." with max_output_tokens = 16. It returned `status=incomplete` and empty `output_text`: all 16 output tokens were reasoning tokens (usage: 92 in / 16 out / 16 reasoning). Cost 6.2e-06 credits.
  - Meaning: reasoning models spend hidden tokens before answering, and those tokens are billed. Phase 2 must give each model enough output tokens and log `incomplete` as a parse failure, never as a wrong answer. Not yet known: which of the 7 models reason, and whether reasoning can be switched off. Not guessed; to be checked.
- **Server metadata: weaker than hoped.** Every listed model has the same `created` = 1677610602 (2023-02-28), which looks like a placeholder, not a real release date. It cannot reveal a model change behind the same id. The response's `created_at` (2026-11-14 20:45:52 UTC) and `response_id` are real per-call values and are stored in the cache entry. The SDK has no model-version field. Conclusion: the response cache (decision 17 in AI_USE.md) is the only reliable record of what the remote models said.
- Total spend so far: 6.2e-06 credits (`results/credits.json`).

## 2026-11-15 — 00_adaption_check retry with max_output_tokens = 256 (approved by Nirdesh), code commit for the --max-output-tokens option

- **Test chat call: succeeded.** deepseek-v4-flash, same prompt, temperature 0, cap 256 → `status=completed`, `output_text='ok'`. Usage: 92 in / 24 out, of which 21 were reasoning tokens. Visible answer ≈ 3 tokens; billed output = 24 (8× the visible text). Cost 7.2e-06 credits. Saved to `results/adaption_check_mot256.json`.
- Reasoning length varies between calls: at least 16 tokens in the first run (cut off at the cap), 21 here, for the same prompt. Phase 2 cost estimates therefore use billed token totals from cached calls, not visible text length.
- **Phase 0 total spend:** 1.34e-05 credits over 2 calls (`results/credits.json`).
- **Phase 0 done criteria:** `import stegaudit` ✔, local model loads on CPU (bit-identical across loads and processes) ✔, one Adaption test call succeeds ✔ (on the second try), `pytest -q` ✔.

## 2026-11-15 — Experiment 01: Meteor round trip (code fdb3cda, PREREG ee63376)

Files: `results/01_meteor_roundtrip.json`, `results/01b_redecode.json`, `results/figs/01_tokens_for_payload.png`, `results/figs/01_vanishing_and_capacity.png`.
Setup as pre-registered: Qwen2.5-0.5B-Instruct @ 7ae5576, CPU, top-k 50, T = 1.0, 200 messages (20 prompts × 10), L = 16 bits (8 message + 8 tag). β = 32 with cover text up to 120 tokens; β = 8, 12, 16 payload only. Total run time 88 min (01) + 58 min (01b).

### Outcomes vs predictions
- **P1a (ids-path round trip 200/200 at every β): MET.** 200/200 at β = 32, 8, 12 and 16.
- **Independent check (fresh-process re-decode, 01b): MET.** All 800 messages re-decoded from disk with keys re-derived from the logged seed. Both paths agree with experiment 01 on every message (0 disagreements).
- **P1d (text-path retokenisation mismatch low but nonzero): MET, with a caveat.**
  - At β = 32: **5/200 = 2.5 %** of full texts do not retokenise to the Sender's ids. Exact 95 % CI **[0.8 %, 5.7 %]**.
  - **All 5 mismatches are in the cover text**, after the payload (first difference at tokens 18–89; the payload ended at tokens 10–23). Examples: `At`+`oms` → `Atoms`; `:`+`**` → `:**`; `-f`+`ace` → `-face`. So **text-path decoding still succeeded 200/200 at β = 32**.
  - At β = 8 / 12: 0/200. At β = 16: 1/200 (msg 91), inside the payload (`Program`+`ming` → `Programming`, the same split as the pilot). It was detected (`tag_mismatch`); text path 199/200.
  - **Wrong messages accepted on the text path: 0 of 800.**
- **Kill rule (> 5 % mismatches at β = 32): did NOT fire** (point estimate 2.5 %). Stated plainly: the CI's upper end (5.7 %) is above 5 %, so a rate above the threshold is not ruled out by 200 messages. The decode-relevant rate (mismatches inside the payload) was 0/200 at β = 32 and 1/800 over all runs. Proceeding to Phase 2 as the rule allows. Phase 2 counts any text-decode failure as a channel failure (PREREG).
- **P1c, vanishing (≈ 0 at β = 32; non-increasing as β grows; clearly nonzero at β = 8): PARTLY MISSED.**
  - Payload steps: 40.1 (β = 8) ≥ 22.7 (β = 12) ≥ 10.0 (β = 16) ≥ 0 (β = 32) vanished tokens per step, out of 50. Monotone and clearly nonzero at β = 8, as predicted.
  - **Miss:** PREREG said "we expect zero vanished tokens in all 200 texts" at β = 32. In the **cover text**, 84 vanished tokens occurred in 9 of 200 messages (0.0043 per cover step, 19 658 cover steps). Likely reason (not tested): some cover steps have extremely peaked distributions, where the 50th-best token has p < 2^-32. The payload phase (start of the reply) had none.
  - Consequence for 02: a vanished token is one Meteor can never emit and normal sampling emits with probability < 2^-32 per step, so the vanishing-token detector is still expected to see nothing at β = 32 in 200 texts. That prediction (P1c detector, AUROC ≈ 0.5) stands as registered.
- **EOS masking (small fraction of payload steps; no strong prior): MET.** β = 32: 0 of 3 427 payload steps had EOS in the top-k (0 messages). β = 8 / 12 / 16: 0.19 % / 0.06 % / 0.06 % of steps (4 / 2 / 2 messages); mean EOS mass removed per message ≤ 0.015. For these prompts the EOS-masking deviation barely ever acted.
- **Capacity (weak prior "roughly 5–12 payload tokens at β = 32"): MISSED.** Median **16** payload tokens at β = 32 (mean 17.1, range 3–42). That is **1.02 bits per payload token** (raw common-prefix bits) or 0.93 effective (16 bits ÷ tokens, not counting overshoot past L). The prior came from a handful of pilot tests and was too optimistic. β = 12 and 16 are the same as β = 32 (1.03 bits/token); β = 8 is lower (0.85 bits/token, median 19.5 tokens).
- **Wall time (CPU, 1 thread):** β = 32 encode 11.2 s per message (payload + ~98 cover tokens); text decode 2.1 s. Payload-only runs: about 2.0–2.4 s encode and decode each.

### Provenance note: `git_dirty: true` in both JSONs
Both result files record `git_dirty: true`. Cause: experiment 01 writes a checkpoint of its own result file after each β run. That new untracked file in `results/` made the next metadata stamp count the tree as dirty, and 01b likewise saw 01's outputs. Evidence that the code that ran is exactly commit fdb3cda: `git status` lists only the four untracked result files, and `git diff --stat fdb3cda` is empty. The result files are committed as produced, unedited. `provenance.py` is fixed in the next code commit to ignore `results/` and to record the list of changed paths.

## Exploratory (not pre-registered) — 01c: per-step entropy and efficiency (code 6bb27ab)

Requested by Nirdesh after experiment 01 had started; see PREREG.md → Deviations. No prediction was made. Files: `results/01c_entropy_explore.json`, `results/figs/01c_bits_vs_entropy.png`.
Method: the first 30 messages of each β run in 01. The model was replayed over 01's saved token ids (no new encoding); every saved token was found in the replayed top-k. H = entropy (bits) of the renormalised top-k (k = 50) distribution Meteor sampled from at each payload step, with EOS masked. That is the entropy of the top-k distribution, not of the full-vocabulary model. "Efficiency" is this project's own ratio, sum(bits) / sum(H); it is not claimed to match any definition in the Meteor paper (not checked against the PDF).

| β | payload steps | mean H (bits) | mean bits/step | efficiency raw | efficiency capped at 16 bits | corr(bits, H) |
|---|---|---|---|---|---|---|
| 32 | 527 | 1.68 | 0.98 | **0.58** | 0.54 | 0.37 |
| 8 | 582 | 1.66 | 0.88 | 0.53 | 0.50 | 0.44 |
| 12 | 529 | 1.66 | 1.03 | 0.62 | 0.55 | 0.47 |
| 16 | 625 | 1.69 | 0.84 | 0.50 | 0.45 | 0.46 |

What was seen, in plain sentences:
- At β = 32, Meteor carried about **0.58 bits per bit of top-k entropy** (0.54 counting only the 16 payload bits).
- **About 60 % of payload steps carried 0 bits** (61.7 % in these 30 messages; 60.4 % over all 200 at β = 32 from 01's saved data). On those steps the chosen interval straddled a high-bit boundary, so no prefix was fixed. The bits come in bursts: up to 11–12 bits on one step when a low-probability token was chosen.
- Bits carried rise with entropy (binned means go from ~0.2 bits at H < 0.5 to ~2 bits at H > 2.5; Pearson 0.37), but most low-entropy steps carry nothing.
- Mean top-k entropy at the start of a study-tip reply is low (~1.7 bits/step). That, together with the ~0.58 ratio, explains most of the capacity miss in experiment 01: ~1 bit per token, so ~16 tokens for 16 bits.
- Differences between β values here (0.50–0.62) come from 30 messages and different texts; no claim is made about a β effect on efficiency.
