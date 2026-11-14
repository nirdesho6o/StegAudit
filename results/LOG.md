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
