"""CLI:  uv run python -m humaneval <generate|execute|report> ...

  generate  k completions per problem -> <dir>/completions.jsonl   (resumable)
  execute   run each completion in the sandbox -> <dir>/results.jsonl (resumable)
  report    pass@k + CI + latency/tokens/throughput -> <dir>/summary.json

Generation and execution are separate stages on purpose: generation wall-clock and throughput are measured on
their own, without sandbox containers competing for the CPU the engine is using.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from humaneval.data import load_problems
from humaneval.executor import run_program
from humaneval.generate import GenConfig, run_generation
from humaneval.program import PROMPT_TEMPLATE, assemble_program, extract_code
from humaneval.report import summarize
from humaneval.sandbox import PythonDockerSandbox
from humaneval.spend import SpendGuard


def _select(problems, spec: str | None):
    if not spec or spec == "all":
        return problems
    if "," in spec:
        want = {int(x) for x in spec.split(",")}
        return [p for p in problems if p.index in want]
    return problems[: int(spec)]


def _engine_get(url: str, path: str, method: str = "GET"):
    try:
        req = urllib.request.Request(url.rstrip("/") + path, method=method, data=b"" if method == "POST" else None)
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _hardware() -> dict:
    cpu = ""
    try:
        cpu = next(ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name"))
    except Exception:
        pass
    mem = ""
    try:
        mem = next(ln.split(":", 1)[1].strip() for ln in open("/proc/meminfo") if ln.startswith("MemTotal"))
    except Exception:
        pass
    return {"cpu": cpu, "logical_cpus": subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip(),
            "mem_total": mem, "platform": platform.platform()}


def cmd_generate(a) -> None:
    problems, prov = load_problems()
    problems = _select(problems, a.problems)
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    cfg = GenConfig(model=a.model, label=a.label, api_base=a.api_base, api_key=a.api_key, n=a.n,
                    temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, base_seed=a.seed,
                    drop_params=a.drop_params, provider=a.provider, thinking=a.thinking)
    guard = None
    if a.provider == "anthropic":
        guard = SpendGuard(a.budget_cap_usd, reserve_usd=a.reserve_usd)
        print(f"[spend] project ledger total ${guard.spent:.4f}; cap ${a.budget_cap_usd:.2f}; "
              f"reserve ${a.reserve_usd:.2f}; tripped={guard.tripped}", flush=True)
    if a.engine_url:
        _engine_get(a.engine_url, "/stats/reset", "POST")     # engine counters cover exactly this run
    t0 = time.time()
    res = run_generation(cfg, problems, d / "completions.jsonl", a.workers, guard=guard)
    meta_path = d / "run.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"segments": []}
    meta.update({
        "label": a.label, "config": cfg.__dict__ | {"api_key": None}, "dataset": prov,
        "prompt_template_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest(),
        "prompt_template": PROMPT_TEMPLATE, "problems_selected": [p.task_id for p in problems],
        "workers": a.workers, "hardware": _hardware(),
        "engine_version": _engine_get(a.engine_url, "/version") if a.engine_url else None})
    meta["segments"].append({"started_at": t0, "wall_seconds": res["seconds"], "requested": res["requested"],
                             "completed": res["completed"], "errors": len(res["errors"]),
                             "engine_stats": _engine_get(a.engine_url, "/stats") if a.engine_url else None})
    meta["errors"] = meta.get("errors_detail", []) + res["errors"]
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "errors"}), f"errors={len(res['errors'])}")
    if guard is not None:
        print(f"[spend] project ledger total now ${guard.spent:.4f} of ${a.budget_cap_usd:.2f}", flush=True)
    if res["errors"]:
        print("first errors:", json.dumps(res["errors"][:3], indent=1))
        raise SystemExit(1)


def cmd_execute(a) -> None:
    problems, _ = load_problems()
    by_id = {p.task_id: p for p in problems}
    d = Path(a.dir)
    comps = [json.loads(x) for x in (d / "completions.jsonl").read_text().splitlines() if x]
    out = d / "results.jsonl"
    done = {(r["task_id"], r["sample_idx"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    todo = [c for c in comps if (c["task_id"], c["sample_idx"]) not in done]
    sb = PythonDockerSandbox()

    def one(c: dict) -> dict:
        p = by_id[c["task_id"]]
        code = extract_code(c["reply"], p.entry_point)
        program, mode = assemble_program(p, code)
        r = run_program(sb, p, program, timeout=a.timeout)
        return c | {"assembly_mode": mode, "passed": r.passed, "outcome": r.outcome, "detail": r.detail,
                    "exec_seconds": round(r.seconds, 3)}

    n = 0
    with out.open("a") as fh, ThreadPoolExecutor(a.workers) as ex:
        for row in ex.map(one, todo):
            fh.write(json.dumps(row) + "\n")
            n += 1
    print(f"executed {n} new samples ({len(done)} already done)")


def cmd_report(a) -> None:
    d = Path(a.dir)
    rows = [json.loads(x) for x in (d / "results.jsonl").read_text().splitlines() if x]
    meta = json.loads((d / "run.json").read_text())
    wall = sum(s["wall_seconds"] for s in meta["segments"])
    ks = [int(k) for k in a.k.split(",")]
    summary = summarize(rows, meta["label"], ks, wall_seconds=wall, usd_per_hour=a.usd_per_hour)
    summary["run"] = {k: meta[k] for k in ("config", "dataset", "prompt_template_sha256", "hardware", "engine_version", "workers")}
    summary["engine_stats_last_segment"] = meta["segments"][-1].get("engine_stats")
    (d / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("n_problems", "n_samples", "pass_at_k", "outcomes")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(prog="humaneval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--label", required=True)
    g.add_argument("--api-base")
    g.add_argument("--api-key", default=None)
    g.add_argument("--engine-url", help="engine root URL (http://localhost:8000): resets/reads /stats, /version")
    g.add_argument("--dir", required=True)
    g.add_argument("--problems", default="all", help="'all', N (first N), or 'i,j,k'")
    g.add_argument("--n", type=int, default=10)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--workers", type=int, default=16)
    g.add_argument("--drop-params", action="store_true")
    g.add_argument("--provider", choices=["litellm", "anthropic"], default="litellm")
    g.add_argument("--thinking", choices=["disabled"], default=None,
                   help="anthropic only: send thinking={type: disabled}; omit for the model's default")
    g.add_argument("--budget-cap-usd", type=float, default=6.50, help="project-wide hard cap for hosted spend")
    g.add_argument("--reserve-usd", type=float, default=0.05, help="held back for requests in flight")
    g.set_defaults(fn=cmd_generate)
    e = sub.add_parser("execute")
    e.add_argument("--dir", required=True)
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--timeout", type=int, default=30)
    e.set_defaults(fn=cmd_execute)
    r = sub.add_parser("report")
    r.add_argument("--dir", required=True)
    r.add_argument("--k", default="1,10")
    r.add_argument("--usd-per-hour", type=float, default=None, help="ASSUMED hourly hardware rate; omit = no cost computed")
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
