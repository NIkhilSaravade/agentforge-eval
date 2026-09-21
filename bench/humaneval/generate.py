"""Generation: k completions per problem from any LiteLLM-reachable, OpenAI-compatible chat endpoint.

The self-hosted engine is addressed as `openai/<name>` with `api_base` pointing at its /v1 (Phase 1); a hosted
model is addressed by its own LiteLLM name. Same code path, same prompt, same parameters for both (Phase 6).
Per bench's gotchas: LiteLLM's own retries are disabled (they hid hangs), timeouts and max_tokens are explicit.

`provider="anthropic"` (Phase 6, hosted Claude arms) calls the official Anthropic SDK directly instead: LiteLLM's model
map can lag brand-new models' parameter rules, and Opus 5 / Sonnet 5 reject temperature/top_p/top_k, so NO sampling
parameters and no seed are sent for that provider (the API's own default sampling applies; results must say so). Every
call is costed from response.usage and checked against a project-wide SpendGuard before it is made. Refusal fallbacks are
deliberately NOT enabled: a fallback would silently substitute a different model into an anchor measurement; a refusal is
recorded (finish_reason "refusal") and scored as a fail.

Every completion is appended to a JSONL as soon as it finishes (resumable: (task_id, sample_idx) already present
are skipped). Sample seeds are a pure function of (base_seed, problem index, sample index), so a run is replayable.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import litellm

from humaneval.data import Problem
from humaneval.program import build_user_message
from humaneval.spend import BudgetExceeded, SpendGuard

SEED_STRIDE = 1000


@dataclass
class GenConfig:
    model: str                      # e.g. "openai/qwen2.5-coder-1.5b" or a hosted model name
    label: str                      # short name used in result files
    api_base: str | None = None
    api_key: str | None = None
    n: int = 10                     # completions per problem
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 512
    base_seed: int = 0
    timeout: int = 1800             # per request; queued requests wait inside the engine's scheduler
    drop_params: bool = False       # hosted providers that reject `seed`/`top_p` combos need this in Phase 6
    provider: str = "litellm"       # "litellm" (self-hosted engine) | "anthropic" (official SDK, hosted Claude)
    thinking: str | None = None     # anthropic only: None = model default, "disabled" = {"type": "disabled"}


def sample_seed(cfg: GenConfig, problem: Problem, sample_idx: int) -> int:
    return cfg.base_seed + SEED_STRIDE * problem.index + sample_idx


_STOP_MAP = {"end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop"}


def _anthropic_key() -> str:
    """Read ANTHROPIC_API_KEY from the environment, else from bench/.env. Never printed or logged."""
    import os
    from pathlib import Path
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        env = Path(__file__).resolve().parent.parent / ".env"
        for ln in env.read_text().splitlines() if env.exists() else []:
            if ln.startswith("ANTHROPIC_API_KEY="):
                key = ln.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (environment or bench/.env)")
    return key


_client = None
_client_lock = threading.Lock()


def _anthropic_client():
    global _client
    with _client_lock:
        if _client is None:
            import anthropic
            _client = anthropic.Anthropic(api_key=_anthropic_key(), timeout=cfg_timeout_default(), max_retries=2)
        return _client


def cfg_timeout_default() -> float:
    return 180.0


def generate_one_anthropic(cfg: GenConfig, problem: Problem, sample_idx: int, guard: SpendGuard) -> dict:
    guard.check()                                    # skip without calling the API once the cap is reached
    kwargs = dict(model=cfg.model, max_tokens=cfg.max_tokens,
                  messages=[{"role": "user", "content": build_user_message(problem)}])
    if cfg.thinking == "disabled":
        kwargs["thinking"] = {"type": "disabled"}
    t0 = time.perf_counter()
    started = time.time()
    resp = _anthropic_client().messages.create(**kwargs)
    latency = time.perf_counter() - t0
    usd = guard.record(cfg.label, cfg.model, resp.usage.input_tokens, resp.usage.output_tokens)
    text = "".join(b.text for b in resp.content if b.type == "text")
    return {
        "task_id": problem.task_id, "sample_idx": sample_idx, "seed": None, "model": cfg.label,
        "reply": text, "finish_reason": _STOP_MAP.get(resp.stop_reason, resp.stop_reason),
        "stop_reason_raw": resp.stop_reason,
        "prompt_tokens": resp.usage.input_tokens, "completion_tokens": resp.usage.output_tokens,
        "thinking_blocks": sum(1 for b in resp.content if b.type in ("thinking", "redacted_thinking")),
        "latency_s": round(latency, 4), "started_at": started, "cost_usd": usd,
    }


def generate_one(cfg: GenConfig, problem: Problem, sample_idx: int) -> dict:
    kwargs = dict(
        model=cfg.model,
        messages=[{"role": "user", "content": build_user_message(problem)}],
        temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_tokens,
        seed=sample_seed(cfg, problem, sample_idx),
        timeout=cfg.timeout, num_retries=0,
    )
    if cfg.api_base:
        kwargs["api_base"] = cfg.api_base
    if cfg.api_key is not None:
        kwargs["api_key"] = cfg.api_key
    if cfg.drop_params:
        kwargs["drop_params"] = True
    t0 = time.perf_counter()
    started = time.time()
    resp = litellm.completion(**kwargs)
    latency = time.perf_counter() - t0
    choice = resp.choices[0]
    usage = getattr(resp, "usage", None)
    try:
        cost = litellm.completion_cost(resp) or 0.0     # 0.0 when LiteLLM has no price (self-hosted)
    except Exception:
        cost = 0.0
    return {
        "task_id": problem.task_id, "sample_idx": sample_idx, "seed": kwargs["seed"], "model": cfg.label,
        "reply": choice.message.content or "", "finish_reason": choice.finish_reason,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "latency_s": round(latency, 4), "started_at": started, "cost_usd": cost,
    }


def load_done(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    return {(r["task_id"], r["sample_idx"]) for r in map(json.loads, path.read_text().splitlines() if path.stat().st_size else [])}


def run_generation(cfg: GenConfig, problems: list[Problem], out: Path, workers: int,
                   progress_every: int = 25, guard: SpendGuard | None = None) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    # Sample-major: every problem's sample 0, then every problem's sample 1, ... so if a spend breaker trips the
    # problems still have balanced sample counts (pass@k needs n >= k per problem).
    todo = [(p, i) for i in range(cfg.n) for p in problems if (p.task_id, i) not in done]
    lock = threading.Lock()
    errors: list[dict] = []
    n_done, t0, budget_skipped = 0, time.perf_counter(), 0
    with out.open("a") as fh, ThreadPoolExecutor(workers) as ex:
        if cfg.provider == "anthropic":
            assert guard is not None, "hosted runs require a SpendGuard"
            futs = {ex.submit(generate_one_anthropic, cfg, p, i, guard): (p, i) for p, i in todo}
        else:
            futs = {ex.submit(generate_one, cfg, p, i): (p, i) for p, i in todo}
        for fut in as_completed(futs):
            p, i = futs[fut]
            try:
                row = fut.result()
            except BudgetExceeded as e:
                budget_skipped += 1
                if budget_skipped == 1:
                    print(f"[gen] {e}; skipping the remaining queued requests", flush=True)
                continue
            except Exception as e:   # recorded, never silently dropped; the sample stays "not done" and reruns
                errors.append({"task_id": p.task_id, "sample_idx": i, "error": f"{type(e).__name__}: {e}"[:400]})
                continue
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                n_done += 1
                if n_done % progress_every == 0:
                    el = time.perf_counter() - t0
                    print(f"[gen] {n_done}/{len(todo)} done, {el:.0f}s elapsed", flush=True)
    return {"requested": len(todo), "completed": n_done, "already_done": len(done), "errors": errors,
            "budget_skipped": budget_skipped, "spent_usd": guard.spent if guard else None,
            "seconds": round(time.perf_counter() - t0, 1)}
