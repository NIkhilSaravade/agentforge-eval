"""/v1/chat/completions: OpenAI response shape, streaming and non-streaming, and that the tokens
are exactly what /generate (and therefore the golden-tested engine) produces for the same prompt."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from engine.api import ChatMessage, create_app, render_chat_prompt
from engine.config import EngineConfig

MSGS = [{"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Say hello"}]


@pytest.fixture(scope="module")
def client(engine):
    with TestClient(create_app(EngineConfig(max_batch=4))) as c:
        yield c


def _prompt_text(engine) -> str:
    return render_chat_prompt(engine.tokenizer, [ChatMessage(**m) for m in MSGS])


def _sse_events(text: str) -> list[str]:
    frames = [f for f in text.split("\n\n") if f]
    assert all(f.startswith("data: ") for f in frames)
    return [f[len("data: "):] for f in frames]


def test_non_streaming_shape(client):
    r = client.post("/v1/chat/completions",
                    json={"model": "gpt2", "messages": MSGS, "max_tokens": 12})
    assert r.status_code == 200
    j = r.json()
    assert j["object"] == "chat.completion"
    assert j["id"].startswith("chatcmpl-") and isinstance(j["created"], int)
    assert j["model"] == "gpt2"
    (choice,) = j["choices"]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] in ("stop", "length")
    u = j["usage"]
    assert u["completion_tokens"] <= 12
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    assert r.headers["x-request-id"]


def test_non_streaming_matches_generate(client, engine):
    """Same prompt, greedy: the chat route must return the same text as /generate."""
    prompt = _prompt_text(engine)
    ref = client.post("/generate", json={"prompt": prompt, "max_new_tokens": 16}).json()
    r = client.post("/v1/chat/completions", json={"messages": MSGS, "max_tokens": 16}).json()
    assert r["choices"][0]["message"]["content"] == ref["text"]
    assert r["choices"][0]["finish_reason"] == ref["finish_reason"]
    assert r["usage"]["prompt_tokens"] == len(engine.tokenizer.encode(prompt))
    assert r["usage"]["completion_tokens"] == len(ref["token_ids"])


def test_max_tokens_gives_length_finish(client):
    j = client.post("/v1/chat/completions", json={"messages": MSGS, "max_tokens": 3}).json()
    assert j["usage"]["completion_tokens"] <= 3
    if j["usage"]["completion_tokens"] == 3:
        assert j["choices"][0]["finish_reason"] == "length"


def test_streaming_shape_and_equals_non_streaming(client):
    body = {"model": "gpt2", "messages": MSGS, "max_tokens": 16}
    full = client.post("/v1/chat/completions", json=body).json()

    r = client.post("/v1/chat/completions", json={**body, "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(r.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert len({c["id"] for c in chunks}) == 1          # one id across the whole stream
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert chunks[-1]["choices"][0]["delta"] == {}
    assert chunks[-1]["choices"][0]["finish_reason"] == full["choices"][0]["finish_reason"]
    # only the final chunk carries a finish_reason
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])
    streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert streamed == full["choices"][0]["message"]["content"]


def test_streaming_include_usage(client):
    r = client.post("/v1/chat/completions", json={
        "messages": MSGS, "max_tokens": 8, "stream": True,
        "stream_options": {"include_usage": True}})
    events = _sse_events(r.text)
    assert events[-1] == "[DONE]"
    last = json.loads(events[-2])
    assert last["choices"] == []
    u = last["usage"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"] > 0


def test_streaming_multibyte_matches_token_decode(client, engine):
    """Streamed text must equal a plain decode of the tokens. (A U+FFFD may legitimately appear
    at the very end if max_tokens cuts a character in half, so we don't assert its absence.)"""
    msgs = [{"role": "user", "content": "Emoji test 😀 and 日本語:"}]
    prompt = render_chat_prompt(engine.tokenizer, [ChatMessage(**m) for m in msgs])
    ref = client.post("/generate", json={"prompt": prompt, "max_new_tokens": 24}).json()
    r = client.post("/v1/chat/completions",
                    json={"messages": msgs, "max_tokens": 24, "stream": True})
    chunks = [json.loads(e) for e in _sse_events(r.text)[:-1]]
    streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert streamed == engine.tokenizer.decode(ref["token_ids"]) == ref["text"]


def test_content_parts_list_is_accepted(client):
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Say "},
                                         {"type": "text", "text": "hello"}]}]
    r = client.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 4})
    assert r.status_code == 200


def test_sampling_params_accepted_but_greedy(client):
    """LiteLLM always sends temperature; we must not reject it. Output stays deterministic."""
    body = {"messages": MSGS, "max_tokens": 8, "temperature": 0.9, "top_p": 0.5}
    a = client.post("/v1/chat/completions", json=body).json()
    b = client.post("/v1/chat/completions", json=body).json()
    assert a["choices"][0]["message"] == b["choices"][0]["message"]


@pytest.mark.parametrize("body,code", [
    ({"messages": []}, 400),
    ({"messages": MSGS, "n": 2}, 400),
    ({"messages": [{"role": "user", "content": "word " * 1100}]}, 400),
    ({"model": "gpt2"}, 422),                                   # messages missing
])
def test_bad_requests(client, body, code):
    assert client.post("/v1/chat/completions", json=body).status_code == code


def test_template_fallback_for_tokenizer_without_chat_template(engine):
    text = _prompt_text(engine)
    assert text == "System: You are terse.\nUser: Say hello\nAssistant:"


def test_chat_template_used_when_tokenizer_has_one():
    class Tok:
        chat_template = "x"
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is True
            return "|".join(m["role"] for m in msgs) + "|<gen>"
    out = render_chat_prompt(Tok(), [ChatMessage(**m) for m in MSGS])
    assert out == "system|user|<gen>"
