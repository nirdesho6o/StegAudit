"""Phase 0 live check of the Adaption account: list models + ONE tiny chat call.

What it does
------------
1. Lists the hosted models this API key can use and saves every field the SDK
   returns (id, created, owned_by, display_name, shutdown_date).
2. Makes one chat call with a trivial prompt (default max_output_tokens = 16,
   override with --max-output-tokens), at temperature 0, and prints its credit
   cost from COST_PER_M.
3. Writes the JSON (with provenance meta) and adds the token use to
   results/credits.json. Default output: results/adaption_models.json; with a
   non-default token cap: results/adaption_check_mot<N>.json, so earlier runs
   are never overwritten.

Usage:
    python experiments/00_adaption_check.py                       # 16-token call
    python experiments/00_adaption_check.py --max-output-tokens 256

The model for the test call is the cheapest model in COST_PER_M that the key
can see (falls back to the first listed model, whose cost is then "unknown").

The call is cached in cache/. Running this script again reads the cache and
spends nothing; the output says "cached: True" in that case.
"""

from __future__ import annotations

import argparse
import asyncio

from stegaudit.config import MeteorConfig
from stegaudit.provenance import REPO_ROOT, write_result_json
from stegaudit.remote import COST_PER_M, AdaptionClient, call_cost

DEFAULT_MAX_OUTPUT_TOKENS = 16
INSTRUCTIONS = "Reply with exactly one word."
PROMPT = "Say ok."


def out_path(max_output_tokens: int):
    if max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS:
        return REPO_ROOT / "results" / "adaption_models.json"
    return REPO_ROOT / "results" / f"adaption_check_mot{max_output_tokens}.json"


def pick_test_model(model_ids: list[str]) -> str:
    priced = [m for m in model_ids if m in COST_PER_M]
    if priced:
        return min(priced, key=lambda m: sum(COST_PER_M[m]))
    return model_ids[0]


async def main(max_output_tokens: int) -> None:
    out = out_path(max_output_tokens)
    client = AdaptionClient("00_adaption_check")
    try:
        models = await client.list_models()
        ids = [m["id"] for m in models]
        print(f"{len(ids)} hosted models visible:")
        for m in models:
            print(f"  {m['id']:<32} owned_by={m.get('owned_by')!s:<16} "
                  f"created={m.get('created')} shutdown={m.get('shutdown_date')}")

        model = pick_test_model(ids)
        r = await client.ask(model, INSTRUCTIONS, PROMPT,
                             max_output_tokens=max_output_tokens, temperature=0.0)
        cost = call_cost(model, r.meta["input_tokens"], r.meta["output_tokens"])
        print(f"\ntest call -> {model}")
        print(f"  status={r.status} cached={r.cached} text={r.text!r}")
        print(f"  model_returned={r.meta['model_returned']} response_id={r.meta['response_id']} "
              f"created_at={r.meta['created_at']}")
        print(f"  tokens in/out/reasoning = {r.meta['input_tokens']}/{r.meta['output_tokens']}/"
              f"{r.meta['reasoning_tokens']}")
        print(f"  cost = {'unknown (not in COST_PER_M)' if cost is None else f'{cost:.8f} credits'}"
              f"{' (cache hit: nothing spent now)' if r.cached else ''}")

        results = {
            "models": models,
            "test_call": {
                "model_requested": model,
                "instructions": INSTRUCTIONS,
                "prompt": PROMPT,
                "max_output_tokens": max_output_tokens,
                "temperature": 0.0,
                "status": r.status,
                "output_text": r.text,
                "cached": r.cached,
                "response_meta": r.meta,
                "cost_credits": cost,
            },
            "price_table_credits_per_M": COST_PER_M,
        }
        write_result_json(out, results, MeteorConfig(), experiment="00_adaption_check",
                          note="meta.model_name/revision refer to the LOCAL channel model, "
                               "not the remote models listed here")
        client.save_credits()
        print(f"\nwrote {out.relative_to(REPO_ROOT)} and results/credits.json")
    finally:
        await client.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    asyncio.run(main(ap.parse_args().max_output_tokens))
