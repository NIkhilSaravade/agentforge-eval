"""Export one real problem's real samples for the results website's request-trace diagram.

    uv run python -m humaneval.export_examples

Selection rule (fixed, stated in the output, not a hand pick): among the self-hosted sampled run's problems, take the
LOWEST-INDEX problem whose 10 samples include both passes and failures with 4 to 7 passes ("a problem the model sometimes
solves"). From it take the lowest-index passing sample and the lowest-index failing sample. Everything else is copied
verbatim from the run's results.jsonl; the prompt, assembled program and test file are rebuilt with the same functions the
harness used.
"""
from __future__ import annotations

import json
from pathlib import Path

from humaneval.data import load_problems
from humaneval.program import PROMPT_TEMPLATE, assemble_program, build_test_file, build_user_message, extract_code

ROOT = Path(__file__).resolve().parent.parent / "results" / "humaneval"
RUN = "qwen2.5-coder-1.5b"


def main() -> None:
    problems, prov = load_problems()
    by_id = {p.task_id: p for p in problems}
    rows = [json.loads(x) for x in (ROOT / RUN / "results.jsonl").read_text().splitlines() if x.strip()]
    per: dict[str, list[dict]] = {}
    for r in rows:
        per.setdefault(r["task_id"], []).append(r)
    chosen = None
    for p in problems:
        rs = sorted(per[p.task_id], key=lambda r: r["sample_idx"])
        c = sum(r["passed"] for r in rs)
        if 4 <= c <= 7:
            chosen = (p, rs, c)
            break
    assert chosen, "no mixed problem found"
    problem, rs, c = chosen

    def sample(r: dict) -> dict:
        code = extract_code(r["reply"], problem.entry_point)
        program, mode = assemble_program(problem, code)
        return {
            "sample_idx": r["sample_idx"], "seed": r["seed"], "reply": r["reply"], "extracted_code": code,
            "assembly_mode": mode, "program": program, "test_file": build_test_file(problem, program),
            "finish_reason": r["finish_reason"], "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"], "client_latency_seconds": r["latency_s"],
            "exec_seconds": r["exec_seconds"], "passed": r["passed"], "outcome": r["outcome"], "detail": r["detail"],
        }

    out = {
        "rule": "lowest-index problem in the self-hosted sampled run with 4 to 7 passing samples out of 10; the lowest-index passing "
                "sample and the lowest-index failing sample of that problem",
        "run": RUN, "dataset": prov,
        "problem": {"task_id": problem.task_id, "entry_point": problem.entry_point, "prompt": problem.prompt,
                    "test": problem.test, "passes_out_of_n": c, "n": len(rs)},
        "prompt_template": PROMPT_TEMPLATE, "user_message": build_user_message(problem),
        "passing": sample(next(r for r in rs if r["passed"])),
        "failing": sample(next(r for r in rs if not r["passed"])),
    }
    path = ROOT / "site_examples.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", path, "| problem", problem.task_id, f"passes {c}/{len(rs)}",
          "| passing sample", out["passing"]["sample_idx"], "| failing sample", out["failing"]["sample_idx"], out["failing"]["outcome"])


if __name__ == "__main__":
    main()
