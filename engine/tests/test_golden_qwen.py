"""Golden tests for Qwen2.5-Coder (0.5B and 1.5B Instruct): every test runs for every model that has
fixtures. Token ids must equal the HuggingFace fp32 greedy
reference exactly. Same rigor as test_golden.py / test_paged.py / test_preemption.py for GPT-2:
single-sequence cached path, static and continuous batching, requests joining and leaving mid
generation, paged KV at two block sizes, and forced preemption-and-recompute. No tolerance.

Fixtures: tests/fixtures/<model>/, from scripts/make_fixtures_qwen.py <model>.
The model is a hard requirement for these tests; nothing is skipped when weights are absent.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from conftest import FIXTURE_DIR, make_engine

from engine.request import Request

MODELS = ["qwen2.5-coder-0.5b", "qwen2.5-coder-1.5b"]
# A missing fixture directory raises at import (batches.json read), so a model can never be silently
# dropped from the suite.
QFIXES = {m: {p.stem: json.loads(p.read_text()) for p in sorted((FIXTURE_DIR / m).glob("*.json"))
              if p.stem != "batches"} for m in MODELS}
QBATCHES_ALL = {m: json.loads((FIXTURE_DIR / m / "batches.json").read_text()) for m in MODELS}
CASE_NAMES = sorted(QFIXES[MODELS[0]])     # both models are tested on the identical case set
ARRIVALS = [0, 0, 3, 5, 8, 13, 21, 34]
MAX_CONTEXT = 2560   # above the longest fixture (2000 + 20); the models themselves allow 32768
KV_MIB = 4096        # budget for contiguous slots: 8 slots of MAX_CONTEXT fit for both models


@pytest.fixture(scope="module", params=MODELS)
def model(request):
    return request.param


@pytest.fixture(scope="module")
def qwen(model):
    from engine.qwen_runner import Qwen2Runner
    return Qwen2Runner(model, num_threads=4, max_context=MAX_CONTEXT)


@pytest.fixture(scope="module")
def QFIX(model):
    return QFIXES[model]


@pytest.fixture(scope="module")
def QBATCHES(model):
    return QBATCHES_ALL[model]


def _exact(QFIX, name: str, out: list[int]) -> None:
    assert out == QFIX[name]["expected_token_ids"], name


def test_fixtures_present_and_cover_the_cases(QFIX, model):
    assert sorted(QFIX) == CASE_NAMES and len(QFIX) >= 10   # identical case set for both models
    for must in ("long_2000_out20", "agent_prompt_out64", "block_exact_16", "short_5_out300"):
        assert must in QFIX
    for fx in QFIX.values():
        assert fx["model"] == model   # the reference really was greedy, not the checkpoint's sampling defaults
        ref = fx["reference"]
        assert ref["do_sample"] is False and ref["repetition_penalty"] == 1.0 and ref["dtype"] == "float32"


SHAPES = {"qwen2.5-coder-0.5b": (24, 14, 2, 64), "qwen2.5-coder-1.5b": (28, 12, 2, 128)}


def test_spec_reflects_gqa(qwen, model):
    s = qwen.spec
    assert (s.n_layers, s.n_heads, s.n_kv_heads, s.head_dim) == SHAPES[model]
    # GQA: KV bytes/token is n_heads / n_kv_heads times smaller than an MHA model of the same width
    assert s.bytes_per_token == 2 * s.n_layers * 2 * s.head_dim * 4
    assert set(s.eos_ids) == {151645, 151643}


# --------------------------------------------------------------------------- M1: own forward
@pytest.mark.parametrize("name", CASE_NAMES)
def test_cached_matches_reference(name, qwen, QFIX):
    fx = QFIX[name]
    _exact(QFIX, name, qwen.generate_cached(fx["prompt_token_ids"], fx["max_new_tokens"]))


def test_eos_stops_generation(qwen, QFIX):
    """A short chat reply ends on EOS (<|im_end|>) well before max_new_tokens, for every model."""
    fx = QFIX["chat_ok_eos_out64"]
    assert len(fx["expected_token_ids"]) < fx["max_new_tokens"]
    assert fx["expected_token_ids"][-1] in qwen.spec.eos_ids
    _exact(QFIX, "chat_ok_eos_out64", qwen.generate_cached(fx["prompt_token_ids"], fx["max_new_tokens"]))


# --------------------------------------------------------------------------- M2 static
@pytest.fixture(scope="module")
def static_engine(qwen):
    return make_engine(qwen, backend="contiguous", batching="static", max_batch=8,
                       kv_budget_mib=KV_MIB)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_static_single_matches_reference(name, static_engine, QFIX):
    fx = QFIX[name]
    _exact(QFIX, name, static_engine.generate_greedy(fx["prompt_token_ids"], fx["max_new_tokens"]))


@pytest.mark.parametrize("batch", sorted(QBATCHES_ALL[MODELS[0]]))
def test_batch_independence(batch, static_engine, QFIX, QBATCHES):
    names = QBATCHES[batch]
    outs = static_engine.generate_batch([QFIX[n]["prompt_token_ids"] for n in names],
                                        [QFIX[n]["max_new_tokens"] for n in names])
    for n, out in zip(names, outs):
        _exact(QFIX, n, out)


# --------------------------------------------------------------------------- M3 continuous
@pytest.fixture(scope="module")
def cont_engine(qwen):
    return make_engine(qwen, backend="contiguous", batching="continuous", max_batch=8,
                       kv_budget_mib=KV_MIB)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_continuous_single_matches_reference(name, cont_engine, QFIX):
    fx = QFIX[name]
    _exact(QFIX, name, cont_engine.generate_greedy(fx["prompt_token_ids"], fx["max_new_tokens"]))


@pytest.mark.parametrize("batch", sorted(QBATCHES_ALL[MODELS[0]]))
def test_continuous_batch_independence(batch, cont_engine, QFIX, QBATCHES):
    names = QBATCHES[batch]
    outs = cont_engine.generate_batch([QFIX[n]["prompt_token_ids"] for n in names],
                                      [QFIX[n]["max_new_tokens"] for n in names])
    for n, out in zip(names, outs):
        _exact(QFIX, n, out)


@pytest.mark.parametrize("max_batch", [2, 3, 5])
def test_join_and_leave_mid_generation(max_batch, qwen, QFIX, QBATCHES):
    eng = make_engine(qwen, backend="contiguous", batching="continuous", max_batch=max_batch,
                       kv_budget_mib=KV_MIB)
    names = QBATCHES["mixed_8"]
    outs = eng.run_schedule([(QFIX[n]["prompt_token_ids"], QFIX[n]["max_new_tokens"], a)
                             for n, a in zip(names, ARRIVALS)])
    for n, out in zip(names, outs):
        _exact(QFIX, n, out)


# --------------------------------------------------------------------------- M4 paged
def paged(qwen, bs, **kw):
    kw.setdefault("batching", "continuous")
    kw.setdefault("max_batch", 8)
    kw.setdefault("kv_budget_mib", 256)
    return make_engine(qwen, backend="paged", block_size=bs, **kw)


@pytest.mark.parametrize("bs", [4, 16])
@pytest.mark.parametrize("name", CASE_NAMES)
def test_paged_single_matches_reference(name, bs, qwen, QFIX):
    fx = QFIX[name]
    _exact(QFIX, name, paged(qwen, bs).generate_greedy(fx["prompt_token_ids"], fx["max_new_tokens"]))


@pytest.mark.parametrize("bs", [4, 16])
@pytest.mark.parametrize("batch", sorted(QBATCHES_ALL[MODELS[0]]))
def test_paged_batch_independence(batch, bs, qwen, QFIX, QBATCHES):
    names = QBATCHES[batch]
    outs = paged(qwen, bs).generate_batch([QFIX[n]["prompt_token_ids"] for n in names],
                                          [QFIX[n]["max_new_tokens"] for n in names])
    for n, out in zip(names, outs):
        _exact(QFIX, n, out)


@pytest.mark.parametrize("bs", [4, 16])
def test_paged_join_and_leave(bs, qwen, QFIX, QBATCHES):
    eng = paged(qwen, bs, max_batch=3)
    names = QBATCHES["mixed_8"]
    outs = eng.run_schedule([(QFIX[n]["prompt_token_ids"], QFIX[n]["max_new_tokens"], a)
                             for n, a in zip(names, ARRIVALS)])
    for n, out in zip(names, outs):
        _exact(QFIX, n, out)


# --------------------------------------------------------------------------- M5 preemption
def test_preemption_happens_and_outputs_stay_exact(qwen, QFIX, QBATCHES):
    # Budget = 512 tokens of KV for this model (12 MiB for 0.5B, 28 MiB for 1.5B): the longest mixed_8
    # request (long_400_out20, 420 tokens) fits alone but the eight together need ~1000, so eviction is forced.
    eng = paged(qwen, 4, kv_budget_mib=512 * qwen.spec.bytes_per_token // 2**20, preemption=True)
    names = QBATCHES["mixed_8"]
    outs = eng.generate_batch([QFIX[n]["prompt_token_ids"] for n in names],
                              [QFIX[n]["max_new_tokens"] for n in names])
    assert eng.sched.metrics.preemptions > 0, "test is meaningless unless something was evicted"
    for n, out in zip(names, outs):
        _exact(QFIX, n, out)
    assert len(eng.manager.free_blocks) == eng.manager.num_blocks   # nothing leaked


def test_every_request_finishes_under_overcommit(qwen, QFIX, QBATCHES):
    eng = paged(qwen, 4, kv_budget_mib=512 * qwen.spec.bytes_per_token // 2**20, preemption=True)
    names = QBATCHES["mixed_8"] * 2
    reqs = [Request(uuid.uuid4().hex, list(QFIX[n]["prompt_token_ids"]), QFIX[n]["max_new_tokens"])
            for n in names]
    for r in reqs:
        eng.sched.submit(r)
    steps = 0
    while eng.sched.has_work():
        eng.sched.step()
        steps += 1
        assert steps < 20000, "livelock"
    for r, n in zip(reqs, names):
        assert r.output_token_ids == QFIX[n]["expected_token_ids"], n
