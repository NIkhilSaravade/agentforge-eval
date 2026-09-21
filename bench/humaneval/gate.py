"""Harness gate (same idea as bench's gold/empty gate, T6): before trusting any model score, the dataset's own
canonical solutions must all pass through THIS pipeline (sandbox, assembly, scoring), and an empty completion must
not. If either fails, the harness is wrong, not the model.

    uv run python -m humaneval.gate [--workers 8] [--out results/humaneval_gate.json]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from humaneval.data import load_problems
from humaneval.executor import run_program
from humaneval.program import assemble_canonical, assemble_program
from humaneval.sandbox import PythonDockerSandbox


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N problems (smoke)")
    ap.add_argument("--out", default="results/humaneval_gate.json")
    args = ap.parse_args()

    problems, prov = load_problems()
    if args.limit:
        problems = problems[:args.limit]
    sb = PythonDockerSandbox()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as ex:
        gold = list(ex.map(lambda p: run_program(sb, p, assemble_canonical(p)), problems))
        empty = list(ex.map(lambda p: run_program(sb, p, assemble_program(p, "")[0]), problems))
    secs = time.perf_counter() - t0

    gold_pass = sum(r.passed for r in gold)
    empty_pass = sum(r.passed for r in empty)
    report = {
        "dataset": prov, "n_problems": len(problems), "seconds": round(secs, 1),
        "gold_passed": gold_pass, "gold_failed_ids": [p.task_id for p, r in zip(problems, gold) if not r.passed],
        "gold_outcomes": dict(Counter(r.outcome for r in gold)),
        "empty_passed": empty_pass, "empty_passed_ids": [p.task_id for p, r in zip(problems, empty) if r.passed],
        "empty_outcomes": dict(Counter(r.outcome for r in empty)),
        "gold_failure_details": {p.task_id: r.detail for p, r in zip(problems, gold) if not r.passed},
        "gate_ok": gold_pass == len(problems) and empty_pass == 0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "dataset"}, indent=1))
    return 0 if report["gate_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
