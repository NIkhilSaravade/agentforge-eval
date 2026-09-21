"""Phase 4: temperature + top-p sampling with a seedable RNG.

Three things must hold:
  1. Sampling works (correct distribution, top-p truncation, temperature).
  2. Greedy is untouched: golden fixtures still match exactly, including when sampled requests share
     the batch, and temperature=0 through the API is exactly the greedy path.
  3. Same seed + same prompt => the same output, twice.
"""
from __future__ import annotations

import uuid

import pytest
import torch
from conftest import BATCHES, FIXTURES, make_engine
from fastapi.testclient import TestClient

from engine.api import create_app
from engine.config import EngineConfig
from engine.request import Request
from engine.sampling import Sampler, pick_tokens


# --------------------------------------------------------------------------- unit: the sampler
def _logits_for(probs: list[float]) -> torch.Tensor:
    return torch.log(torch.tensor(probs))


def test_sampler_validates_arguments():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            Sampler(bad)
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            Sampler(1.0, bad)


def test_same_seed_same_draws_and_different_seed_differs():
    lg = _logits_for([0.25, 0.25, 0.25, 0.25])
    s1, s2, s3 = Sampler(1.0, seed=42), Sampler(1.0, seed=42), Sampler(1.0, seed=43)
    d1, d2, d3 = ([s.sample(lg) for _ in range(200)] for s in (s1, s2, s3))
    assert d1 == d2
    assert d1 != d3


def test_unseeded_sampler_reports_a_replayable_seed():
    lg = _logits_for([0.25] * 4)
    s = Sampler(1.0)
    first = [s.sample(lg) for _ in range(50)]
    replay = Sampler(1.0, seed=s.seed)
    assert [replay.sample(lg) for _ in range(50)] == first


def test_sampling_follows_the_softmax_distribution():
    probs = [0.5, 0.3, 0.15, 0.05]
    s, n = Sampler(1.0, seed=0), 20000
    counts = [0] * 4
    for _ in range(n):
        counts[s.sample(_logits_for(probs))] += 1
    for c, p in zip(counts, probs):
        assert abs(c / n - p) < 4 * (p * (1 - p) / n) ** 0.5   # within 4 sigma


def test_temperature_reshapes_the_distribution():
    lg = _logits_for([0.5, 0.3, 0.15, 0.05])
    cold = Sampler(0.05, seed=1)
    assert all(cold.sample(lg) == 0 for _ in range(200))         # ~argmax
    hot = Sampler(5.0, seed=1)
    assert len({hot.sample(lg) for _ in range(400)}) == 4        # tail is reachable


def test_top_p_truncates_the_tail():
    lg = _logits_for([0.5, 0.3, 0.15, 0.05])
    s = Sampler(1.0, top_p=0.7, seed=5)     # mass before token 2 is 0.8 >= 0.7, so only {0, 1} remain
    assert {s.sample(lg) for _ in range(500)} == {0, 1}
    tiny = Sampler(1.0, top_p=1e-6, seed=5)  # nucleus of one: the top token, always
    assert {tiny.sample(lg) for _ in range(100)} == {0}
    full = Sampler(1.0, top_p=1.0, seed=5)
    assert {full.sample(lg) for _ in range(2000)} == {0, 1, 2, 3}


def test_pick_tokens_without_samplers_is_exactly_argmax():
    lg = torch.randn(5, 100)
    want = torch.argmax(lg, dim=-1).tolist()
    assert pick_tokens(lg, None) == want
    assert pick_tokens(lg, [None] * 5) == want


def test_pick_tokens_mixed_rows_keep_greedy_rows_exact():
    lg = torch.randn(4, 100)
    want = torch.argmax(lg, dim=-1).tolist()
    out = pick_tokens(lg, [None, Sampler(5.0, seed=1), None, Sampler(5.0, seed=2)])
    assert out[0] == want[0] and out[2] == want[2]


# --------------------------------------------------------------------------- engine: greedy is untouched
def _eng(runner, **kw):
    kw.setdefault("max_batch", 8)
    return make_engine(runner, batching="continuous", **kw)


def test_greedy_golden_unchanged_when_neighbours_sample(engine):
    """The regression the greedy contract needs: rows without a sampler match the HF reference exactly
    even when every other row in the same batches is being sampled at a high temperature."""
    names = BATCHES["mixed_8"]
    fxs = [FIXTURES[n] for n in names]
    samplers = [None if i % 2 == 0 else Sampler(1.5, top_p=0.95, seed=100 + i) for i in range(len(fxs))]
    eng = _eng(engine, backend="paged", block_size=16, kv_budget_mib=256)
    outs = eng.generate_batch([f["prompt_token_ids"] for f in fxs],
                              [f["max_new_tokens"] for f in fxs], samplers)
    for i, (fx, out) in enumerate(zip(fxs, outs)):
        if samplers[i] is None:
            assert out == fx["expected_token_ids"], fx["name"]


def test_explicit_none_samplers_equal_the_default_path(engine):
    fxs = [FIXTURES[n] for n in BATCHES["mixed_8"]]
    eng = _eng(engine, backend="contiguous")
    outs = eng.generate_batch([f["prompt_token_ids"] for f in fxs],
                              [f["max_new_tokens"] for f in fxs], [None] * len(fxs))
    for fx, out in zip(fxs, outs):
        assert out == fx["expected_token_ids"]


# --------------------------------------------------------------------------- engine: sampling + reproducibility
def _run(runner, sampler_factory, **cfg):
    fx = FIXTURES["short_5_out300"]
    eng = _eng(runner, **cfg)
    return eng.generate_batch([fx["prompt_token_ids"]], [60], [sampler_factory()])[0]


def test_sampled_output_differs_from_greedy_and_is_reproducible_by_seed(engine):
    fx = FIXTURES["short_5_out300"]
    greedy = fx["expected_token_ids"][:60]
    a = _run(engine, lambda: Sampler(1.0, 0.95, seed=7), backend="contiguous")
    b = _run(engine, lambda: Sampler(1.0, 0.95, seed=7), backend="contiguous")
    c = _run(engine, lambda: Sampler(1.0, 0.95, seed=8), backend="contiguous")
    assert a == b                      # same seed, same prompt: identical
    assert a != c                      # different seed: different
    assert a != greedy                 # sampling actually samples
    assert len(a) == 60


def test_reproducible_across_backends_and_batch_neighbours(engine):
    """A request's draws come from its own generator, so neighbours must not perturb them."""
    alone = _run(engine, lambda: Sampler(0.8, 0.9, seed=11), backend="contiguous")
    paged = _run(engine, lambda: Sampler(0.8, 0.9, seed=11), backend="paged", block_size=16,
                 kv_budget_mib=256)
    assert alone == paged
    fx = FIXTURES["short_5_out300"]
    others = [FIXTURES[n] for n in ("long_400_out20", "block_exact_16")]
    eng = _eng(engine, backend="contiguous")
    outs = eng.generate_batch(
        [fx["prompt_token_ids"]] + [o["prompt_token_ids"] for o in others], [60, 20, 20],
        [Sampler(0.8, 0.9, seed=11), Sampler(1.0, seed=1), None])
    assert outs[0] == alone


def test_sampling_survives_preemption_and_recompute_deterministically(engine):
    """Tight KV forces eviction; the generator state travels with the request, so the same seeds give the
    same outputs on a repeat run under identical scheduling."""
    def once():
        eng = make_engine(engine, backend="paged", block_size=4, kv_budget_mib=40, preemption=True,
                          batching="continuous", max_batch=8)
        fxs = [FIXTURES[n] for n in BATCHES["mixed_8"]]
        reqs = [Request(uuid.uuid4().hex, list(f["prompt_token_ids"]), f["max_new_tokens"],
                        sampler=Sampler(1.0, 0.9, seed=200 + i)) for i, f in enumerate(fxs)]
        for r in reqs:
            eng.sched.submit(r)
        while eng.sched.has_work():
            eng.sched.step()
        assert eng.sched.metrics.preemptions > 0, "test is meaningless unless something was evicted"
        return [r.output_token_ids for r in reqs]
    assert once() == once()


# --------------------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client(engine):
    with TestClient(create_app(EngineConfig(max_batch=4))) as c:
        yield c


def _gen(client, **kw):
    return client.post("/generate", json={"prompt": "The harbour town", "max_new_tokens": 24, **kw}).json()


def test_api_temperature_zero_is_the_greedy_path(client):
    base = _gen(client)
    assert _gen(client, temperature=0)["token_ids"] == base["token_ids"]
    assert base["seed"] is None


def test_api_seeded_sampling_is_reproducible_and_seed_is_echoed(client):
    a = _gen(client, temperature=0.9, top_p=0.9, seed=5)
    b = _gen(client, temperature=0.9, top_p=0.9, seed=5)
    assert a["token_ids"] == b["token_ids"] and a["seed"] == 5
    assert _gen(client, temperature=0.9, top_p=0.9, seed=6)["token_ids"] != a["token_ids"]


def test_api_unseeded_request_reports_a_seed_that_replays_it(client):
    a = _gen(client, temperature=1.0)
    assert isinstance(a["seed"], int)
    assert _gen(client, temperature=1.0, seed=a["seed"])["token_ids"] == a["token_ids"]


@pytest.mark.parametrize("kw", [dict(temperature=0.5, top_p=0), dict(temperature=0.5, top_p=1.5),
                                dict(temperature=-1)])
def test_api_rejects_bad_sampling_values(client, kw):
    r = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 4, **kw})
    assert r.status_code == 400


def test_api_streaming_sampled_matches_non_streaming(client):
    body = {"prompt": "The harbour town", "max_new_tokens": 24, "temperature": 0.9, "seed": 9}
    full = client.post("/generate", json=body).json()
    lines = client.post("/generate", json={**body, "stream": True}).text.splitlines()
    import json as _json
    ids = [m["token_id"] for m in map(_json.loads, lines) if m.get("token_id") is not None]
    assert ids == full["token_ids"]


# --------------------------------------------------------------------------- the models that will be served
import json  # noqa: E402

from conftest import FIXTURE_DIR  # noqa: E402

QWEN_MODELS = ["qwen2.5-coder-0.5b", "qwen2.5-coder-1.5b"]
QWEN_ARRIVAL_CASES = ["short_5_out20", "block_exact_16", "long_400_out20", "cross_block_gen_10"]


@pytest.fixture(scope="module", params=QWEN_MODELS)
def qwen_case(request):
    from engine.qwen_runner import Qwen2Runner
    runner = Qwen2Runner(request.param, num_threads=4, max_context=1024)
    fx = {n: json.loads((FIXTURE_DIR / request.param / f"{n}.json").read_text())
          for n in QWEN_ARRIVAL_CASES}
    return runner, fx


def test_qwen_greedy_golden_unchanged_when_neighbours_sample(qwen_case):
    runner, fx = qwen_case
    names = QWEN_ARRIVAL_CASES
    samplers = [None, Sampler(1.5, 0.95, seed=1), None, Sampler(1.5, 0.95, seed=2)]
    eng = make_engine(runner, backend="paged", block_size=16, kv_budget_mib=256,
                      batching="continuous", max_batch=4)
    outs = eng.generate_batch([fx[n]["prompt_token_ids"] for n in names],
                              [fx[n]["max_new_tokens"] for n in names], samplers)
    for n, s, out in zip(names, samplers, outs):
        if s is None:
            assert out == fx[n]["expected_token_ids"], n


def test_qwen_same_seed_same_output_and_it_actually_samples(qwen_case):
    runner, fx = qwen_case
    f = fx["short_5_out20"]

    def run(seed, backend="contiguous", **kw):
        eng = make_engine(runner, backend=backend, batching="continuous", max_batch=2,
                          kv_budget_mib=256, **kw)
        return eng.generate_batch([f["prompt_token_ids"]], [40], [Sampler(1.0, 0.95, seed=seed)])[0]

    a, b, c = run(7), run(7), run(8)
    assert a == b                                              # same seed: identical
    assert a != c                                              # different seed: different
    assert a != runner.generate_cached(f["prompt_token_ids"], 40)   # not just greedy
    assert run(7, backend="paged", block_size=16) == a         # backend does not change the draws


# --------------------------------------------------------------------------- nucleus fast path is EXACT
def _reference_nucleus(logits: torch.Tensor, temperature: float, top_p: float):
    """The obvious full-vocabulary implementation (what Phase 4 originally shipped)."""
    probs = torch.softmax(logits.to(torch.float64) / temperature, dim=-1)   # float64 = the mathematical truth
    sp, idx = torch.sort(probs, descending=True, stable=True)
    cum = torch.cumsum(sp, dim=-1)
    sp = sp * ((cum - sp) < top_p)
    return idx, (sp / sp.sum()).to(torch.float32)


@pytest.mark.parametrize("scale,temperature,top_p", [
    (3.0, 0.8, 0.95), (3.0, 1.0, 0.5), (1.0, 0.7, 0.9), (6.0, 1.3, 0.99), (0.2, 1.0, 0.95)])
def test_nucleus_fast_path_equals_full_sort(scale, temperature, top_p):
    g = torch.Generator().manual_seed(1)
    for _ in range(25):
        logits = torch.randn(151936, generator=g) * scale
        s = Sampler(temperature, top_p, seed=0)
        idx, p = s.nucleus(logits)
        ridx, rp = _reference_nucleus(logits, temperature, top_p)
        n = int((rp > 0).sum())
        assert n >= 1
        kept = idx[p > 0]
        assert set(kept.tolist()) == set(ridx[:n].tolist())              # same nucleus
        ref = {int(i): float(q) for i, q in zip(ridx[:n], rp[:n])}
        for i, q in zip(kept.tolist(), p[p > 0].tolist()):
            assert abs(ref[i] - q) < 1e-5                                # same renormalised probabilities


def test_nucleus_falls_back_to_full_sort_on_flat_distributions():
    """5,000 equally likely tokens: the top-256 hold ~5% of the mass, far below top_p, so the fast path must
    NOT be used; the nucleus is ~95% of 5,000 tokens."""
    logits = torch.full((151936,), -1e9)
    logits[:5000] = 0.0
    s = Sampler(1.0, 0.95, seed=0)
    idx, p = s.nucleus(logits)
    assert int((p > 0).sum()) == pytest.approx(4750, abs=2)
    ridx, rp = _reference_nucleus(logits, 1.0, 0.95)
    assert int((rp > 0).sum()) == int((p > 0).sum())
    assert 0 <= s.sample(logits) < 5000


def test_sampling_is_fast_enough_for_batched_decode():
    """Regression guard for the defect the HumanEval smoke run exposed (11.8 ms/row/step with a full-vocab
    sort). Uses a peaked distribution like real LM logits (measured on Qwen: top-256 tokens hold >99.99% of the
    mass), not flat noise. Generous bound so it is not flaky; the old implementation was ~2x over it."""
    import time
    logits = torch.randn(151936) * 3
    logits[:20] += 15.0
    s = Sampler(0.8, 0.95, seed=1)
    s.sample(logits)
    t0 = time.perf_counter()
    for _ in range(40):
        s.sample(logits)
    assert (time.perf_counter() - t0) / 40 < 0.006
