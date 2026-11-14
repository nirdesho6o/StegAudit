"""Adaption client for everything that only reads text and answers.

Idea in plain words
-------------------
The stego channel (Meteor) runs on the LOCAL model, because it needs exact,
bit-identical next-token distributions. Every other model role only needs
"read this text, give an answer":
  * the Sentinel (no key; tries to answer from the visible message),
  * the Receiver's answering step (after it has decoded the payload locally),
  * the LLM-judge detector ("does this text hide a message? 0-100").
These run on hosted models through Adaption's own Python SDK (`adaption`).
The call pattern is copied from apart-sprint/swarm-sprint (swarm/llm.py):
`AsyncAdaption(api_key=...)`, then `client.responses.create(model=...,
instructions=..., input=[{"role": "user", "content": ...}],
max_output_tokens=..., temperature=...)`. The only setting is
ADAPTION_API_KEY, read from the environment / .env and never written anywhere.

Reproducibility of remote answers comes from the CACHE, not from the model
--------------------------------------------------------------------------
Temperature 0 on a hosted model is NOT guaranteed to be deterministic:
batching, kernel choice and server-side changes can alter the output, and a
provider can change what model serves an id without changing the id. So:
  * every response is cached to cache/ keyed by a hash of
    (model, instructions, input, max_output_tokens, temperature);
  * a rerun reads the cache and gets exactly the same text at zero cost;
  * each cache entry also stores what the server said about itself: the model
    name as returned, the response id and the created_at timestamp, so a silent
    model change behind the same id is at least visible afterwards.
No test assumes two live calls return the same text.

Security assumption
-------------------
None for secrecy: remote models never see the key or the stego channel.
Assumption for honesty of results: the cache is the record of what the
models said. Deleting cache/ and rerunning gives a NEW sample, possibly with
different answers, and that rerun must be reported as such.

Cost control
------------
Tokens per call are summed per experiment into results/credits.json, priced
with COST_PER_M. Experiments with > 500 calls must call
`confirm_cost(...)` first, which prints an estimate and waits for "yes".
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "cache" / "adaption"
CREDITS_PATH = REPO_ROOT / "results" / "credits.json"

# Credits per 1M tokens (input, output). Copied from apart-sprint/swarm-sprint
# (its CLAUDE.md says these were checked against docs.adaptionlabs.ai, Nov 2026).
# A model missing here gets cost None ("unknown"), never a silent 0.
COST_PER_M: dict[str, tuple[float, float]] = {
    "qwen3.5-397b": (0.55, 3.50),
    "glm-5.2": (0.104, 4.40),
    "deepseek-v4-flash": (0.044, 0.133),
}

CONFIRM_THRESHOLD_CALLS = 500
MAX_RETRIES = 5


def call_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Credits for one call, or None if the model is not in the price table."""
    if model not in COST_PER_M:
        return None
    cin, cout = COST_PER_M[model]
    return input_tokens * cin / 1e6 + output_tokens * cout / 1e6


def cache_key(model: str, instructions: str, user_content: str,
              max_output_tokens: int, temperature: float) -> str:
    """SHA-256 over a canonical JSON of everything that defines the request."""
    req = {"model": model, "instructions": instructions, "input": user_content,
           "max_output_tokens": max_output_tokens, "temperature": temperature}
    return hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()


@dataclass
class Usage:
    calls: int = 0           # live API calls that succeeded
    cache_hits: int = 0      # answered from cache, cost 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    credits: float = 0.0
    unpriced_calls: int = 0  # calls to models missing from COST_PER_M


@dataclass
class RemoteResult:
    text: str
    status: str
    cached: bool
    meta: dict = field(default_factory=dict)  # response_id, model_returned, created_at, usage


class AdaptionClient:
    """Cached, retrying, credit-logging wrapper around AsyncAdaption.

    `sdk_client` can be injected (tests use a fake); by default a real
    AsyncAdaption client is built from ADAPTION_API_KEY.
    """

    def __init__(self, experiment: str, concurrency: int = 8,
                 cache_dir: Path = CACHE_DIR, credits_path: Path = CREDITS_PATH,
                 sdk_client: Any = None):
        self.experiment = experiment
        self.cache_dir = Path(cache_dir)
        self.credits_path = Path(credits_path)
        self.usage: dict[str, Usage] = {}
        self._sem = asyncio.Semaphore(concurrency)
        if sdk_client is None:
            load_dotenv(REPO_ROOT / ".env")
            api_key = os.environ.get("ADAPTION_API_KEY")
            if not api_key:
                raise RuntimeError("ADAPTION_API_KEY not set (put it in .env)")
            from adaption import AsyncAdaption
            sdk_client = AsyncAdaption(api_key=api_key)
        self._client = sdk_client

    # ---------- cache ----------
    def _cache_path(self, model: str, key: str) -> Path:
        return self.cache_dir / model.replace("/", "_") / f"{key}.json"

    def _usage(self, model: str) -> Usage:
        return self.usage.setdefault(model, Usage())

    # ---------- calls ----------
    async def ask(self, model: str, instructions: str, user_content: str,
                  max_output_tokens: int = 256, temperature: float = 0.0) -> RemoteResult:
        """One question to one hosted model. Served from cache when possible."""
        key = cache_key(model, instructions, user_content, max_output_tokens, temperature)
        path = self._cache_path(model, key)
        if path.exists():
            entry = json.loads(path.read_text())
            self._usage(model).cache_hits += 1
            return RemoteResult(entry["output_text"], entry["status"], True, entry["response_meta"])

        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            async with self._sem:
                try:
                    t0 = time.monotonic()
                    r = await self._client.responses.create(
                        model=model,
                        instructions=instructions,
                        input=[{"role": "user", "content": user_content}],
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                    )
                    latency = time.monotonic() - t0
                    break
                except Exception as e:  # SDK raises several error types; all are retried
                    last_err = e
            self._usage(model).retries += 1
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Adaption call failed after {MAX_RETRIES} retries: {last_err}")
            await asyncio.sleep(min(2 ** (attempt + 1) + random.random(), 30))

        meta = self._response_meta(r, latency)
        self._record_usage(model, meta)
        entry = {
            "request": {"model": model, "instructions": instructions, "input": user_content,
                        "max_output_tokens": max_output_tokens, "temperature": temperature},
            "output_text": r.output_text,
            "status": r.status,
            "response_meta": meta,
            "cached_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": self.experiment,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry, indent=2))
        return RemoteResult(r.output_text, r.status, False, meta)

    @staticmethod
    def _response_meta(r: Any, latency: float) -> dict:
        """Everything the server tells us about which model produced this answer."""
        u = getattr(r, "usage", None)
        details = getattr(u, "output_tokens_details", None) or {}
        # In adaption 0.19.0 output_tokens_details is a dict (not an object).
        reasoning = details.get("reasoning_tokens", 0) if isinstance(details, dict) else 0
        return {
            "response_id": getattr(r, "id", None),
            "model_returned": getattr(r, "model", None),
            "created_at": getattr(r, "created_at", None),
            "latency_s": round(latency, 3),
            "input_tokens": getattr(u, "input_tokens", 0) if u else 0,
            "output_tokens": getattr(u, "output_tokens", 0) if u else 0,
            "reasoning_tokens": int(reasoning or 0),
        }

    def _record_usage(self, model: str, meta: dict) -> None:
        u = self._usage(model)
        u.calls += 1
        u.input_tokens += meta["input_tokens"]
        u.output_tokens += meta["output_tokens"]
        u.reasoning_tokens += meta["reasoning_tokens"]
        c = call_cost(model, meta["input_tokens"], meta["output_tokens"])
        if c is None:
            u.unpriced_calls += 1
        else:
            u.credits += c

    async def list_models(self) -> list[dict]:
        """Hosted models this key can use, with every field the SDK returns."""
        res = await self._client.models.list(type="hosted")
        return [m.model_dump() if hasattr(m, "model_dump") else dict(m) for m in res.data]

    # ---------- bookkeeping ----------
    def save_credits(self) -> dict:
        """Add this client's usage to results/credits.json under the experiment name."""
        data = json.loads(self.credits_path.read_text()) if self.credits_path.exists() else {}
        exp = data.setdefault(self.experiment, {})
        for model, u in self.usage.items():
            prev = exp.get(model, asdict(Usage()))
            exp[model] = {k: prev.get(k, 0) + v for k, v in asdict(u).items()}
        self.credits_path.parent.mkdir(parents=True, exist_ok=True)
        self.credits_path.write_text(json.dumps(data, indent=2))
        return data

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close:
            await close()


def estimate_cost(model: str, n_calls: int, in_tokens_per_call: int,
                  out_tokens_per_call: int) -> float | None:
    c = call_cost(model, in_tokens_per_call, out_tokens_per_call)
    return None if c is None else c * n_calls


def confirm_cost(model: str, n_calls: int, in_tokens_per_call: int,
                 out_tokens_per_call: int, input_fn=input) -> bool:
    """Print a cost estimate. For > 500 calls, require the user to type 'yes'."""
    est = estimate_cost(model, n_calls, in_tokens_per_call, out_tokens_per_call)
    est_s = "unknown (model not in COST_PER_M)" if est is None else f"{est:.4f} credits"
    print(f"[cost] {model}: {n_calls} calls x ~{in_tokens_per_call} in / "
          f"~{out_tokens_per_call} out tokens -> {est_s} (cache hits cost 0)")
    if n_calls <= CONFIRM_THRESHOLD_CALLS:
        return True
    return input_fn("Type 'yes' to proceed: ").strip().lower() == "yes"
