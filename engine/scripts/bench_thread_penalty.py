"""Does it matter which thread loads the model and which thread runs it? (Found during the HumanEval run.)

    python scripts/bench_thread_penalty.py qwen2.5-coder-1.5b 32 results/thread_penalty_qwen2.5-coder-1.5b.json

Same load in three arrangements: 32 concurrent requests, ~160 prompt tokens, 100 new tokens each, no HTTP:
  main         model loaded and run on the main thread
  thread       model loaded on the main thread, run on another thread   (what the API server used to do)
  thread_all   model loaded AND run on the same non-main thread         (what the API server does now)
Each arrangement runs in a fresh process so nothing is shared between them.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEXT = ("def has_close_elements(numbers, threshold):\n    for idx, elem in enumerate(numbers):\n"
        "        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n") * 6


def child(model: str, n: int, mode: str) -> None:
    import threading

    import torch  # noqa: F401
    from transformers import AutoTokenizer

    from engine.config import EngineConfig
    from engine.qwen_runner import Qwen2Runner, resolve_model_dir
    from engine.request import Request
    from engine.scheduler import Engine

    cfg = EngineConfig(model=model, backend="paged", batching="continuous", max_batch=32, kv_budget_mib=2048,
                       max_context=4096, num_threads=20, preemption=True)
    ids = AutoTokenizer.from_pretrained(str(resolve_model_dir(model))).encode(TEXT)[:160]
    box: dict = {"runner": None if mode == "thread_all" else Qwen2Runner(model, 20, 4096)}

    def run() -> None:
        if box["runner"] is None:
            box["runner"] = Qwen2Runner(model, 20, 4096)
        eng = Engine(cfg, box["runner"])
        reqs = [Request(uuid.uuid4().hex, list(ids), 100, ignore_eos=True) for _ in range(n)]
        for r in reqs:
            eng.sched.submit(r)
        t0 = time.perf_counter()
        while eng.sched.has_work():
            eng.sched.step()
        wall = time.perf_counter() - t0
        m = eng.sched.metrics
        toks = sum(len(r.output_token_ids) for r in reqs)
        print("RESULT " + json.dumps({"mode": mode, "wall_seconds": round(wall, 1), "tokens_per_second": round(toks / wall, 1),
                                      "prefill_seconds": round(m.prefill_seconds, 1),
                                      "decode_seconds": round(m.decode_seconds, 1)}), flush=True)

    if mode == "main":
        run()
    else:
        t = threading.Thread(target=run, name="engine-loop")
        t.start()
        t.join()


def main() -> None:
    if len(sys.argv) > 4 and sys.argv[4] == "--child":
        child(sys.argv[1], int(sys.argv[2]), sys.argv[3])
        return
    model, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    results = []
    for mode in ("main", "thread", "thread_all"):
        p = subprocess.run([sys.executable, __file__, model, str(n), mode, "--child"], capture_output=True, text=True)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            raise SystemExit(f"{mode} failed:\n{p.stderr[-800:]}")
        results.append(json.loads(line[len("RESULT "):]))
        print(results[-1], flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"model": model, "concurrent_requests": n, "prompt_tokens": 160, "new_tokens": 100,
                                     "threads": 20, "results": results}, indent=1) + "\n")


if __name__ == "__main__":
    main()
