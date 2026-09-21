"""Generation: k completions per problem from any LiteLLM-reachable, OpenAI-compatible chat endpoint.

The self-hosted engine is addressed as `openai/<name>` with `api_base` pointing at its /v1 (Phase 1); a hosted
model is addressed by its own LiteLLM name. Same code path, same prompt, same parameters for both (Phase 6).
Per bench's gotchas: LiteLLM's own retries are disabled (they hid hangs), timeouts and max_tokens are explicit.

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


def sample_seed(cfg: GenConfig, problem: Problem, sample_idx: int) -> int:
    return cfg.base_seed + SEED_STRIDE * problem.index + sample_idx


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
                   progress_every: int = 25) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    todo = [(p, i) for p in problems for i in range(cfg.n) if (p.task_id, i) not in done]
    lock = threading.Lock()
    errors: list[dict] = []
    n_done, t0 = 0, time.perf_counter()
    with out.open("a") as fh, ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(generate_one, cfg, p, i): (p, i) for p, i in todo}
        for fut in as_completed(futs):
            p, i = futs[fut]
            try:
                row = fut.result()
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
            "seconds": round(time.perf_counter() - t0, 1)}
