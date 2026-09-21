"""Record a real scheduler trace for the results website's engine diagram.

    python scripts/record_scheduler_trace.py results/scheduler_trace.json

Runs the ACTUAL Scheduler and paged BlockManager (nothing simulated) on a small real GPT-2 workload: 8 requests of different
lengths arriving over time, a paged KV pool small enough that admission is optimistic and the pool runs dry, so the scheduler
has to preempt. The Scheduler/BlockManager methods are wrapped only to OBSERVE (allocate, append_slot, free, prefill, preempt,
emit); no engine code is changed. After every scheduler step the full state is snapshotted.

Two properties are asserted before anything is written, so the trace cannot be a picture of a broken run:
  1. at least one preemption-and-recompute actually happened;
  2. every request's output tokens are IDENTICAL to a single-request greedy run (preemption must not change results).
"""
from __future__ import annotations

import hashlib
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import EngineConfig  # noqa: E402
from engine.model_runner import ModelRunner  # noqa: E402
from engine.request import Request  # noqa: E402
from engine.scheduler import Engine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BLOCK_SIZE = 4
KV_BUDGET_MIB = 5          # GPT-2 is 73,728 B/token: 5 MiB / (4 tokens * 73,728 B) = 17 blocks (68 token slots)
MAX_BATCH = 4
TEXT = ("The harbour town woke slowly that morning, as it always did in late autumn. Fishing boats rocked against their "
        "moorings while gulls argued over scraps on the quay. In the bakery on Mill Street, Marta pulled the first loaves "
        "from the oven and set them on the rack to cool, the smell drifting out into the cold air.")
# (name, prompt tokens, new tokens, arrives at scheduler step)
WORKLOAD = [("A", 10, 18, 0), ("B", 6, 24, 0), ("C", 18, 14, 1), ("D", 8, 28, 2),
            ("E", 12, 20, 4), ("F", 5, 16, 6), ("G", 14, 22, 9), ("H", 7, 14, 12)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results" / "scheduler_trace.json")
    runner = ModelRunner()
    base = runner.tokenizer.encode(TEXT)
    cfg = EngineConfig(backend="paged", batching="continuous", max_batch=MAX_BATCH, kv_budget_mib=KV_BUDGET_MIB,
                       block_size=BLOCK_SIZE, preemption=True)
    eng = Engine(cfg, runner)
    sched, mgr = eng.sched, eng.manager
    num_blocks = mgr.num_blocks

    reqs, name_of = {}, {}
    for i, (name, plen, nnew, at) in enumerate(WORKLOAD):
        prompt = base[i:i + plen]
        assert len(prompt) == plen, "workload text too short"
        r = Request(uuid.uuid4().hex, prompt, nnew, ignore_eos=True)   # fixed output length, as the benchmarks do
        reqs[name], name_of[r.request_id] = r, name

    events: list[dict] = []
    cur_step = {"n": -1}

    def ev(kind: str, **kw) -> None:
        events.append({"step": cur_step["n"], "type": kind, **kw})

    # ---- observation wrappers (behaviour untouched)
    o_alloc, o_append, o_free = mgr.allocate, mgr.append_slot, mgr.free
    o_prefill, o_preempt, o_emit = sched._prefill, sched._preempt, sched._emit

    def allocate(req):
        o_alloc(req)
        ev("allocate", id=name_of[req.request_id], blocks=list(req.block_table), recompute=bool(req.output_token_ids))

    def append_slot(req):
        before = len(req.block_table)
        ok = o_append(req)
        if len(req.block_table) > before:
            ev("grow", id=name_of[req.request_id], block=req.block_table[-1])
        elif not ok:
            ev("out_of_blocks", id=name_of[req.request_id])
        return ok

    def free(req):
        blocks = list(req.block_table)
        o_free(req)
        ev("free", id=name_of[req.request_id], blocks=blocks, reason="finished" if req.finished else "preempted")

    def prefill(batch):
        for r in batch:
            ev("prefill", id=name_of[r.request_id], tokens=len(r.prompt_token_ids) + len(r.output_token_ids),
               recompute=bool(r.output_token_ids))
        o_prefill(batch)

    def preempt(req):
        ev("preempt", id=name_of[req.request_id], generated=len(req.output_token_ids),
           kv_tokens_discarded=req.seq_len, blocks=list(req.block_table))
        o_preempt(req)

    def emit(req, tok):
        o_emit(req, tok)
        ev("token", id=name_of[req.request_id], n=len(req.output_token_ids), finished=req.finished)

    mgr.allocate, mgr.append_slot, mgr.free = allocate, append_slot, free
    sched._prefill, sched._preempt, sched._emit = prefill, preempt, emit

    def snapshot() -> dict:
        used = {b for r in sched.running for b in r.block_table}
        return {
            "running": [{"id": name_of[r.request_id], "seq_len": r.seq_len, "generated": len(r.output_token_ids),
                         "max_new": r.max_new_tokens, "blocks": list(r.block_table), "preempted": r.preempt_count}
                        for r in sorted(sched.running, key=lambda r: r.admit_seq)],
            "waiting": [{"id": name_of[r.request_id], "prompt": len(r.prompt_token_ids), "generated": len(r.output_token_ids),
                         "max_new": r.max_new_tokens, "preempted": r.preempt_count} for r in sched.waiting],
            "free_blocks": sorted(mgr.free_blocks), "used_blocks": len(used), "preemptions_total": sched.metrics.preemptions,
        }

    steps = []
    pending = sorted(WORKLOAD, key=lambda w: w[3])
    n = 0
    while pending or sched.has_work():
        cur_step["n"] = n
        arrived = []
        while pending and pending[0][3] <= n:
            name = pending.pop(0)[0]
            sched.submit(reqs[name])
            arrived.append(name)
            ev("arrive", id=name, prompt=len(reqs[name].prompt_token_ids), new_tokens=reqs[name].max_new_tokens)
        sched.step()
        steps.append({"step": n, "arrived": arrived, "state": snapshot()})
        n += 1
        assert n < 5000, "scheduler made no progress"

    # ---- assertions
    assert sched.metrics.preemptions > 0, "no preemption happened; the workload does not exercise the mechanism"
    assert len(mgr.free_blocks) == num_blocks, "blocks leaked"
    for name, r in reqs.items():
        assert r.finished, name
        want = runner.generate_cached(list(r.prompt_token_ids), r.max_new_tokens, ignore_eos=True)
        assert r.output_token_ids == want, f"request {name}: output differs from the single-request greedy run"

    trace = {
        "about": "A real trace of engine/engine/scheduler.py + block_manager.py serving GPT-2 (paged KV, continuous batching, "
                 "preempt-and-recompute). Nothing here is simulated; see engine/scripts/record_scheduler_trace.py.",
        "config": {"model": "gpt2", "backend": cfg.backend, "batching": cfg.batching, "max_batch": cfg.max_batch,
                   "block_size": BLOCK_SIZE, "num_blocks": num_blocks, "kv_budget_mib": KV_BUDGET_MIB,
                   "preemption": cfg.preemption, "ignore_eos": True},
        "requests": [{"id": name, "prompt_tokens": plen, "new_tokens": nnew, "arrives_at_step": at}
                     for name, plen, nnew, at in WORKLOAD],
        "steps": steps, "events": events,
        "totals": {"steps": len(steps), "preemptions": sched.metrics.preemptions,
                   "tokens_generated": sum(len(r.output_token_ids) for r in reqs.values()),
                   "outputs_identical_to_single_request_greedy": True, "blocks_leaked": 0},
        "provenance": {p: sha(ROOT / p) for p in ("engine/scheduler.py", "engine/block_manager.py", "engine/cache.py")},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trace, separators=(",", ":")) + "\n")
    print(f"wrote {out_path}: {len(steps)} steps, {sched.metrics.preemptions} preemption(s), "
          f"{trace['totals']['tokens_generated']} tokens, {num_blocks} blocks of {BLOCK_SIZE}; "
          f"all outputs identical to single-request greedy")


if __name__ == "__main__":
    main()
