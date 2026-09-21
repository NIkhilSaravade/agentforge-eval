"""Decode/prefill step time vs batch size on the real runner + paged pool (no HTTP, no scheduler).

    python scripts/bench_decode_sweep.py qwen2.5-coder-1.5b 20 1,8,16,32 results/decode_sweep_qwen2.5-coder-1.5b.json

For each batch size b: prefill b sequences of CTX tokens in one step, then time 6 decode steps at b rows. This is the
'continuous batching pays off' measurement: one decode step costs far less than b separate single-row steps.
Numbers depend on the machine and thread count; both are recorded in the output.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402

from engine.cache import PagedPool  # noqa: E402
from engine.qwen_runner import Qwen2Runner  # noqa: E402

CTX = 400        # typical HumanEval prompt plus partial completion
BLOCK = 16


def main() -> None:
    model, threads, batches, out = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")], sys.argv[4]
    r = Qwen2Runner(model, num_threads=threads, max_context=2048)
    rows = []
    for b in batches:
        per = -(-(CTX + 8) // BLOCK)
        pool = PagedPool(b * per + 4, BLOCK, r.spec)
        tables = [list(range(i * per, (i + 1) * per)) for i in range(b)]
        prompt = [[(7 * i + j) % 1000 + 100 for j in range(CTX)] for i in range(b)]
        t0 = time.perf_counter()
        r.step_tokens(prompt, [0] * b, pool, tables)
        pre = time.perf_counter() - t0
        toks = [[123]] * b
        r.step_tokens(toks, [CTX] * b, pool, tables)          # warm
        n = 6
        t0 = time.perf_counter()
        for s in range(n):
            r.step_tokens(toks, [CTX + 1 + s] * b, pool, tables)
        dec = (time.perf_counter() - t0) / n
        rows.append({"batch": b, "prefill_seconds": round(pre, 3), "prefill_tokens_per_second": round(b * CTX / pre, 1),
                     "decode_step_seconds": round(dec, 4), "decode_tokens_per_second": round(b / dec, 1)})
        print(json.dumps(rows[-1]), flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "model": model, "threads": threads, "context_tokens": CTX, "block_size": BLOCK, "torch": torch.__version__,
        "platform": platform.platform(), "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
