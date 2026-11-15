"""remote.py with a FAKE SDK client: no network, no credits.

Deliberately NOT tested: that two live calls give the same text. Hosted
models are not guaranteed deterministic even at temperature 0; repeatability
comes from the cache, which is what these tests check.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from stegaudit import remote
from stegaudit.remote import AdaptionClient, cache_key, call_cost, confirm_cost


class FakeResponses:
    def __init__(self, fail_times=0):
        self.calls = 0
        self.fail_times = fail_times

    async def create(self, model, instructions, input, max_output_tokens, temperature):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("transient")
        return SimpleNamespace(
            id=f"resp_{self.calls}", model=model + "-served", created_at=1700000000 + self.calls,
            status="completed", output_text=f"answer #{self.calls}",
            usage=SimpleNamespace(input_tokens=1000, output_tokens=100, total_tokens=1100,
                                  output_tokens_details={"reasoning_tokens": 7}),
        )


class FakeModels:
    async def list(self, type):
        assert type == "hosted"
        m = SimpleNamespace(model_dump=lambda: {"id": "glm-5.2", "created": 1, "object": "model",
                                                "owned_by": "z", "display_name": None,
                                                "shutdown_date": None})
        return SimpleNamespace(data=[m])


def make_client(tmp_path, fail_times=0):
    sdk = SimpleNamespace(responses=FakeResponses(fail_times), models=FakeModels())
    c = AdaptionClient("unit", cache_dir=tmp_path / "cache", credits_path=tmp_path / "credits.json",
                       sdk_client=sdk)
    return c, sdk


def run(coro):
    return asyncio.run(coro)


def test_cache_miss_then_hit(tmp_path):
    c, sdk = make_client(tmp_path)
    r1 = run(c.ask("glm-5.2", "sys", "q"))
    r2 = run(c.ask("glm-5.2", "sys", "q"))
    assert not r1.cached and r2.cached
    assert r2.text == r1.text          # same text because it came from the cache
    assert sdk.responses.calls == 1    # second call never reached the API
    assert c.usage["glm-5.2"].cache_hits == 1


def test_cache_entry_stores_server_metadata(tmp_path):
    c, _ = make_client(tmp_path)
    run(c.ask("glm-5.2", "sys", "q"))
    [entry_path] = list((tmp_path / "cache").rglob("*.json"))
    e = json.loads(entry_path.read_text())
    m = e["response_meta"]
    assert m["response_id"] == "resp_1"
    assert m["model_returned"] == "glm-5.2-served"   # as returned, may differ from requested
    assert m["created_at"] == 1700000001
    assert m["reasoning_tokens"] == 7
    assert e["request"]["model"] == "glm-5.2" and e["request"]["temperature"] == 0.0


def test_cache_key_depends_on_every_request_field():
    base = ("m", "sys", "q", 16, 0.0)
    k = cache_key(*base)
    for i, alt in enumerate(["m2", "sys2", "q2", 17, 0.5]):
        changed = list(base)
        changed[i] = alt
        assert cache_key(*changed) != k


def test_retries_then_succeeds(tmp_path, monkeypatch):
    async def no_sleep(_):
        return None
    monkeypatch.setattr(remote.asyncio, "sleep", no_sleep)
    c, sdk = make_client(tmp_path, fail_times=2)
    r = run(c.ask("glm-5.2", "sys", "q"))
    assert r.status == "completed" and sdk.responses.calls == 3
    assert c.usage["glm-5.2"].retries == 2
    assert [e["type"] for e in c.errors] == ["ConnectionError", "ConnectionError"]
    assert c.errors[0]["model"] == "glm-5.2" and "transient" in c.errors[0]["message"]


def test_credits_logged_and_accumulated(tmp_path):
    c, _ = make_client(tmp_path)
    run(c.ask("glm-5.2", "sys", "q1"))
    run(c.ask("glm-5.2", "sys", "q2"))
    run(c.ask("glm-5.2", "sys", "q1"))  # cache hit, no cost
    data = c.save_credits()
    u = data["unit"]["glm-5.2"]
    assert u["calls"] == 2 and u["cache_hits"] == 1
    assert u["input_tokens"] == 2000 and u["output_tokens"] == 200
    assert u["credits"] == pytest.approx(2 * call_cost("glm-5.2", 1000, 100))
    # a second save adds to, not overwrites, the experiment's totals
    c.save_credits()
    assert json.loads((tmp_path / "credits.json").read_text())["unit"]["glm-5.2"]["calls"] == 4


def test_unknown_model_cost_is_none_not_zero(tmp_path):
    assert call_cost("some-unpriced-model", 1000, 100) is None
    c, _ = make_client(tmp_path)
    run(c.ask("some-unpriced-model", "sys", "q"))
    assert c.usage["some-unpriced-model"].unpriced_calls == 1


def test_list_models_keeps_all_fields(tmp_path):
    c, _ = make_client(tmp_path)
    [m] = run(c.list_models())
    assert set(m) >= {"id", "created", "owned_by", "display_name", "shutdown_date"}


def test_confirm_gate():
    assert confirm_cost("glm-5.2", 500, 100, 10, input_fn=lambda _: "no")      # <= 500: no prompt
    assert not confirm_cost("glm-5.2", 501, 100, 10, input_fn=lambda _: "no")
    assert confirm_cost("glm-5.2", 501, 100, 10, input_fn=lambda _: "yes")


def test_missing_key_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(remote, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("ADAPTION_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        AdaptionClient("x", cache_dir=tmp_path, credits_path=tmp_path / "c.json")


def test_cost_estimate_uses_billed_tokens_including_reasoning(tmp_path):
    """The fake returns 100 billed output tokens for a short visible answer;
    the estimate must use the 100, not the visible text length."""
    from stegaudit.remote import cached_token_stats, estimate_cost_from_cache
    c, _ = make_client(tmp_path)
    for q in ("a", "b", "c"):
        run(c.ask("glm-5.2", "sys", q))
    s = cached_token_stats("glm-5.2", experiment="unit", cache_dir=tmp_path / "cache")
    assert s["n"] == 3 and s["mean_output_tokens"] == 100 and s["mean_reasoning_tokens"] == 7
    est = estimate_cost_from_cache("glm-5.2", 1000, experiment="unit", cache_dir=tmp_path / "cache")
    assert est == pytest.approx(1000 * call_cost("glm-5.2", 1000, 100))
    # other experiments' entries are ignored; no pilot -> None (never a silent 0)
    assert cached_token_stats("glm-5.2", experiment="other", cache_dir=tmp_path / "cache")["n"] == 0
    assert estimate_cost_from_cache("glm-5.2", 10, experiment="other", cache_dir=tmp_path / "cache") is None


def test_real_client_gets_long_timeout(monkeypatch, tmp_path):
    """The SDK default (60 s) is too short for reasoning models; we pass DEFAULT_TIMEOUT_S."""
    import adaption
    seen = {}

    class Spy:
        def __init__(self, **kw):
            seen.update(kw)
    monkeypatch.setattr(adaption, "AsyncAdaption", Spy)
    monkeypatch.setenv("ADAPTION_API_KEY", "test-key")
    AdaptionClient("x", cache_dir=tmp_path, credits_path=tmp_path / "c.json")
    assert seen["timeout"] == remote.DEFAULT_TIMEOUT_S >= 300
