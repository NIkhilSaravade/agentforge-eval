"""Hosted-arm money handling: cost formula, persistent ledger, circuit-breaker, and the request shape sent to the API.
No network and no API key are used: the Anthropic client is replaced by a recording fake."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import humaneval.generate as gen
from humaneval.data import Problem
from humaneval.spend import BudgetExceeded, SpendGuard, cost_usd, ledger_total

P = Problem("HumanEval/x", 'def add(a, b):\n    """Add."""\n', "    return a + b\n",
            "def check(candidate):\n    assert candidate(1, 2) == 3\n", "add")


def test_cost_formula_uses_per_million_prices():
    assert cost_usd("claude-haiku-4-5", 1000, 1000) == pytest.approx(0.001 + 0.005)
    assert cost_usd("claude-opus-5", 1_000_000, 0) == pytest.approx(5.0)
    assert cost_usd("claude-opus-5", 0, 1_000_000) == pytest.approx(25.0)


def test_ledger_persists_and_breaker_trips_at_the_cap(tmp_path):
    led = tmp_path / "ledger.jsonl"
    g = SpendGuard(cap_usd=0.01, reserve_usd=0.0, path=led)
    assert not g.tripped
    g.record("a", "claude-haiku-4-5", 1000, 1000)          # $0.006
    assert not g.tripped and ledger_total(led) == pytest.approx(0.006)
    g.record("a", "claude-haiku-4-5", 1000, 1000)          # $0.012 total: over the cap
    assert g.tripped
    with pytest.raises(BudgetExceeded):
        g.check()
    # a NEW guard (a later run) sees the money already spent and starts tripped: the cap is project-wide
    assert SpendGuard(cap_usd=0.01, path=led).tripped


def test_reserve_holds_back_headroom_for_inflight_requests(tmp_path):
    g = SpendGuard(cap_usd=0.05, reserve_usd=0.045, path=tmp_path / "l.jsonl")
    g.record("a", "claude-haiku-4-5", 1000, 0)             # $0.001 + reserve 0.045 < 0.05
    assert not g.tripped
    g.record("a", "claude-haiku-4-5", 5000, 0)             # $0.006 total + reserve 0.045 = 0.051 >= 0.05
    assert g.tripped


class _FakeMessages:
    def __init__(self, stop_reason="end_turn"):
        self.calls, self.stop_reason = [], stop_reason

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="```python\ndef add(a, b):\n    return a + b\n```")],
            stop_reason=self.stop_reason, usage=SimpleNamespace(input_tokens=100, output_tokens=50))


def _cfg(**kw):
    return gen.GenConfig(model="claude-opus-5", label="t", provider="anthropic", **kw)


def test_hosted_request_sends_no_sampling_params_or_seed(monkeypatch, tmp_path):
    fake = SimpleNamespace(messages=_FakeMessages())
    monkeypatch.setattr(gen, "_anthropic_client", lambda: fake)
    guard = SpendGuard(10.0, path=tmp_path / "l.jsonl")
    row = gen.generate_one_anthropic(_cfg(temperature=0.8, top_p=0.95, base_seed=7), P, 0, guard)
    (call,) = fake.messages.calls
    assert not ({"temperature", "top_p", "top_k", "seed", "fallbacks"} & set(call)), call
    assert call["model"] == "claude-opus-5" and call["max_tokens"] == 512 and "thinking" not in call
    assert row["seed"] is None and row["finish_reason"] == "stop"
    assert row["cost_usd"] == pytest.approx(cost_usd("claude-opus-5", 100, 50))
    assert ledger_total(tmp_path / "l.jsonl") == pytest.approx(row["cost_usd"])


def test_thinking_disabled_is_sent_only_when_asked(monkeypatch, tmp_path):
    fake = SimpleNamespace(messages=_FakeMessages())
    monkeypatch.setattr(gen, "_anthropic_client", lambda: fake)
    guard = SpendGuard(10.0, path=tmp_path / "l.jsonl")
    gen.generate_one_anthropic(_cfg(thinking="disabled"), P, 0, guard)
    assert fake.messages.calls[0]["thinking"] == {"type": "disabled"}


def test_refusal_and_max_tokens_are_recorded_not_hidden(monkeypatch, tmp_path):
    guard = SpendGuard(10.0, path=tmp_path / "l.jsonl")
    for raw, mapped in (("refusal", "refusal"), ("max_tokens", "length")):
        monkeypatch.setattr(gen, "_anthropic_client", lambda raw=raw: SimpleNamespace(messages=_FakeMessages(raw)))
        row = gen.generate_one_anthropic(_cfg(), P, 0, guard)
        assert row["finish_reason"] == mapped and row["stop_reason_raw"] == raw


def test_tripped_breaker_makes_no_api_call(monkeypatch, tmp_path):
    def boom():
        raise AssertionError("the API client must not even be created once the breaker has tripped")
    monkeypatch.setattr(gen, "_anthropic_client", boom)
    g = SpendGuard(cap_usd=0.001, path=tmp_path / "l.jsonl")
    g.record("a", "claude-opus-5", 10_000, 0)
    with pytest.raises(BudgetExceeded):
        gen.generate_one_anthropic(_cfg(), P, 0, g)


def test_run_generation_skips_queued_work_after_the_breaker_trips(monkeypatch, tmp_path):
    fake = SimpleNamespace(messages=_FakeMessages())
    monkeypatch.setattr(gen, "_anthropic_client", lambda: fake)
    g = SpendGuard(cap_usd=cost_usd("claude-opus-5", 100, 50) * 2.5, path=tmp_path / "l.jsonl")   # room for 2 calls
    problems = [Problem(f"HumanEval/{i}", P.prompt, P.canonical_solution, P.test, "add") for i in range(6)]
    res = gen.run_generation(_cfg(n=1), problems, tmp_path / "c.jsonl", workers=1, guard=g)
    assert res["completed"] == 3 and res["budget_skipped"] == 3      # the 3rd call is the one that crosses the cap
    assert len(fake.messages.calls) == 3


def test_generation_order_is_sample_major(monkeypatch, tmp_path):
    """So a tripped breaker leaves balanced sample counts across problems."""
    fake = SimpleNamespace(messages=_FakeMessages())
    monkeypatch.setattr(gen, "_anthropic_client", lambda: fake)
    g = SpendGuard(10.0, path=tmp_path / "l.jsonl")
    problems = [Problem(f"HumanEval/{i}", P.prompt, P.canonical_solution, P.test, "add") for i in range(3)]
    gen.run_generation(_cfg(n=2), problems, tmp_path / "c.jsonl", workers=1, guard=g)
    import json
    order = [(r["task_id"], r["sample_idx"]) for r in map(json.loads, (tmp_path / "c.jsonl").read_text().splitlines())]
    assert [i for _, i in order] == [0, 0, 0, 1, 1, 1]


def test_run_config_never_stores_the_api_key():
    import inspect
    from humaneval import __main__ as cli
    assert '"api_key": None' in inspect.getsource(cli.cmd_generate)
