"""Golden tests for Qwen2.5-Coder-0.5B-Instruct: token ids must equal the HuggingFace fp32 greedy
reference exactly. Same rigor as test_golden.py / test_paged.py / test_preemption.py for GPT-2:
single-sequence cached path, static and continuous batching, requests joining and leaving mid
generation, paged KV at two block sizes, and forced preemption-and-recompute. No tolerance.

Fixtures: tests/fixtures/qwen2.5-coder-0.5b/, from scripts/make_fixtures_qwen.py.
The model is a hard requirement for these tests; nothing is skipped when weights are absent.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from conftest import FIXTURE_DIR, make_engine

from engine.request import Request

MODEL = "qwen2.5-coder-0.5b"
QDIR = FIXTURE_DIR / MODEL
QFIX = {p.stem: json.loads(p.read_text()) for p in sorted(QDIR.glob("*.json")) if p.stem != "batches"}
QBATCHES = json.loads((QDIR / "batches.json").read_text())
ARRIVALS = [0, 0, 3, 5, 8, 13, 21, 34]
MAX_CONTEXT = 4096   # above the longest fixture (2000 + 20); the model itself allows 32768


@pytest.fixture(scope="module")
def qwen():
    from engine.qwen_runner import Qwen2Runner
    return Qwen2Runner(MODEL, num_threads=4, max_context=MAX_CONTEXT)


def _exact(name: str, out: list[int]) -> None:
    assert out == QFIX[name]["expected_token_ids"], name


def test_fixtures_present_and_cover_the_cases():
    assert len(QFIX) >= 10
    for must in ("long_2000_out20", "agent_prompt_out64", "block_exact_16", "short_5_out300"):
        assert must in QFIX
    for fx in QFIX.values():   # the reference really was greedy, not the checkpoint's sampling defaults
        ref = fx["reference"]
        assert ref["do_sample"] is False and ref["repetition_penalty"] == 1.0 and ref["dtype"] == "float32"


def test_spec_reflects_gqa(qwen):
    s = qwen.spec
    assert (s.n_layers, s.n_heads, s.n_kv_heads, s.head_dim) == (24, 14, 2, 64)
    assert s.bytes_per_token == 2 * 24 * 2 * 64 * 4     # 24,576: 7x smaller than MHA at the same width
    assert set(s.eos_ids) == {151645, 151643}


# --------------------------------------------------------------------------- M1: own forward
@pytest.mark.parametrize("name", sorted(QFIX))
def test_cached_matches_reference(name, qwen):
    fx = QFIX[name]
    _exact(name, qwen.generate_cached(fx["prompt_token_ids"], fx["max_new_tokens"]))


def test_eos_stops_generation(qwen):
    """The agent-prompt fixture ends on EOS (<|im_end|>) well before max_new_tokens."""
    fx = QFIX["agent_prompt_out64"]
    assert len(fx["expected_token_ids"]) < fx["max_new_tokens"]
    assert fx["expected_token_ids"][-1] in qwen.spec.eos_ids
    _exact("agent_prompt_out64", qwen.generate_cached(fx["prompt_token_ids"], fx["max_new_tokens"]))


# --------------------------------------------------------------------------- M2 static
@pytest.fixture(scope="module")
def static_engine(qwen):
    return make_engine(qwen, backend="contiguous", batching="static", max_batch=8)


@pytest.mark.parametrize("name", sorted(QFIX))
def test_static_single_matches_reference(name, static_engine):
    fx = QFIX[name]
    _exact(name, static_engine.generate_greedy(fx["prompt_token_ids"], fx["max_new_tokens"]))


@pytest.mark.parametrize("batch", sorted(QBATCHES))
def test_batch_independence(batch, static_engine):
    names = QBATCHES[batch]
    outs = static_engine.generate_batch([QFIX[n]["prompt_token_ids"] for n in names],
                                        [QFIX[n]["max_new_tokens"] for n in names])
    for n, out in zip(names, outs):
        _exact(n, out)


# --------------------------------------------------------------------------- M3 continuous
@pytest.fixture(scope="module")
def cont_engine(qwen):
    return make_engine(qwen, backend="contiguous", batching="continuous", max_batch=8)


@pytest.mark.parametrize("name", sorted(QFIX))
def test_continuous_single_matches_reference(name, cont_engine):
    fx = QFIX[name]
    _exact(name, cont_engine.generate_greedy(fx["prompt_token_ids"], fx["max_new_tokens"]))


@pytest.mark.parametrize("batch", sorted(QBATCHES))
def test_continuous_batch_independence(batch, cont_engine):
    names = QBATCHES[batch]
    outs = cont_engine.generate_batch([QFIX[n]["prompt_token_ids"] for n in names],
                                      [QFIX[n]["max_new_tokens"] for n in names])
    for n, out in zip(names, outs):
        _exact(n, out)


@pytest.mark.parametrize("max_batch", [2, 3, 5])
def test_join_and_leave_mid_generation(max_batch, qwen):
    eng = make_engine(qwen, backend="contiguous", batching="continuous", max_batch=max_batch)
    names = QBATCHES["mixed_8"]
    outs = eng.run_schedule([(QFIX[n]["prompt_token_ids"], QFIX[n]["max_new_tokens"], a)
                             for n, a in zip(names, ARRIVALS)])
    for n, out in zip(names, outs):
        _exact(n, out)


# --------------------------------------------------------------------------- M4 paged
def paged(qwen, bs, **kw):
    kw.setdefault("batching", "continuous")
    kw.setdefault("max_batch", 8)
    kw.setdefault("kv_budget_mib", 256)
    return make_engine(qwen, backend="paged", block_size=bs, **kw)


@pytest.mark.parametrize("bs", [4, 16])
@pytest.mark.parametrize("name", sorted(QFIX))
def test_paged_single_matches_reference(name, bs, qwen):
    fx = QFIX[name]
    _exact(name, paged(qwen, bs).generate_greedy(fx["prompt_token_ids"], fx["max_new_tokens"]))


@pytest.mark.parametrize("bs", [4, 16])
@pytest.mark.parametrize("batch", sorted(QBATCHES))
def test_paged_batch_independence(batch, bs, qwen):
    names = QBATCHES[batch]
    outs = paged(qwen, bs).generate_batch([QFIX[n]["prompt_token_ids"] for n in names],
                                          [QFIX[n]["max_new_tokens"] for n in names])
    for n, out in zip(names, outs):
        _exact(n, out)


@pytest.mark.parametrize("bs", [4, 16])
def test_paged_join_and_leave(bs, qwen):
    eng = paged(qwen, bs, max_batch=3)
    names = QBATCHES["mixed_8"]
    outs = eng.run_schedule([(QFIX[n]["prompt_token_ids"], QFIX[n]["max_new_tokens"], a)
                             for n, a in zip(names, ARRIVALS)])
    for n, out in zip(names, outs):
        _exact(n, out)


# --------------------------------------------------------------------------- M5 preemption
def test_preemption_happens_and_outputs_stay_exact(qwen):
    # block_size 4; 12 MiB / (4 * 24,576 B) = 128 blocks = 512 tokens. The longest mixed_8 request
    # (long_400_out20, 420 tokens) fits alone but the eight together need ~1000, so eviction is forced.
    eng = paged(qwen, 4, kv_budget_mib=12, preemption=True)
    names = QBATCHES["mixed_8"]
    outs = eng.generate_batch([QFIX[n]["prompt_token_ids"] for n in names],
                              [QFIX[n]["max_new_tokens"] for n in names])
    assert eng.sched.metrics.preemptions > 0, "test is meaningless unless something was evicted"
    for n, out in zip(names, outs):
        _exact(n, out)
    assert len(eng.manager.free_blocks) == eng.manager.num_blocks   # nothing leaked


def test_every_request_finishes_under_overcommit(qwen):
    eng = paged(qwen, 4, kv_budget_mib=12, preemption=True)
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
