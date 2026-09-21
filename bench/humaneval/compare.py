"""Phase 6: self-hosted vs hosted comparison from the saved per-sample results (no API calls, no engine needed).

    uv run python -m humaneval.compare

Every number is recomputed from results.jsonl / completions.jsonl / run.json under results/humaneval/. Nothing is typed by hand.
pass@k is computed with bench's pipeline.stats estimator per problem (n may differ per arm: 10, 10, 3), with a problem-level
bootstrap CI. Cost of hosted arms = sum of per-call cost computed from response.usage x the live price table (humaneval/spend.py).
"""
from __future__ import annotations

import json
from pathlib import Path

from humaneval.report import bootstrap_ci, per_problem_pass_at_k

ROOT = Path(__file__).resolve().parent.parent / "results" / "humaneval"
ARMS = {
    "self_hosted_sampled": "qwen2.5-coder-1.5b",
    "self_hosted_greedy": "qwen2.5-coder-1.5b-greedy",
    "haiku_4_5": "claude-haiku-4-5",
    "opus_5": "claude-opus-5",
}
COMMON_KS = (1, 2, 3)


def load(d: str) -> tuple[list[dict], dict]:
    rows = [json.loads(x) for x in (ROOT / d / "results.jsonl").read_text().splitlines() if x.strip()]
    return rows, json.loads((ROOT / d / "run.json").read_text())


def pass_table(rows: list[dict], ks) -> dict:
    out = {}
    for k in ks:
        vals = per_problem_pass_at_k(rows, k)
        if not vals:
            continue
        lo, hi = bootstrap_ci(list(vals.values()))
        out[f"pass@{k}"] = {"value": sum(vals.values()) / len(vals), "ci95": [lo, hi], "n_problems": len(vals)}
    return out


def main() -> None:
    arms: dict[str, dict] = {}
    for name, d in ARMS.items():
        rows, meta = load(d)
        n = len(rows)
        trunc = [r for r in rows if r.get("finish_reason") == "length"]
        passed = sum(r["passed"] for r in rows)
        samples_per_problem = sorted({len([x for x in rows if x["task_id"] == t]) for t in {r["task_id"] for r in rows}})
        arm = {
            "run_dir": d, "n_samples": n, "samples_per_problem": samples_per_problem,
            "passed_samples": passed, "sample_pass_rate": passed / n,
            "pass_at_k": pass_table(rows, (1, 2, 3, 5, 10)),
            "outcomes": {o: sum(1 for r in rows if r["outcome"] == o) for o in sorted({r["outcome"] for r in rows})},
            "truncated_at_max_tokens": len(trunc),
            "truncation_upper_bound_sample_pass_rate": (passed + sum(1 for r in trunc if not r["passed"])) / n,
            "max_tokens": meta["config"]["max_tokens"],
            "tokens": {"prompt_total": sum(r["prompt_tokens"] for r in rows),
                       "completion_total": sum(r["completion_tokens"] for r in rows)},
            "protocol": {"provider": meta["config"].get("provider"), "model": meta["config"]["model"],
                         "temperature": meta["config"]["temperature"] if meta["config"].get("provider") != "anthropic" else "API default (not sent)",
                         "top_p": meta["config"]["top_p"] if meta["config"].get("provider") != "anthropic" else "API default (not sent)",
                         "seed": "per-sample deterministic" if meta["config"].get("provider") != "anthropic" else "none (API has no seed)",
                         "thinking": meta["config"].get("thinking") or "model default (n/a for the self-hosted model)"},
            "dataset_sha256": meta["dataset"]["sha256"], "prompt_template_sha256": meta["prompt_template_sha256"],
        }
        if meta["config"].get("provider") == "anthropic":
            usd = sum(r["cost_usd"] for r in rows)
            arm["api_cost"] = {"usd_total": usd, "usd_per_sample": usd / n,
                               "usd_per_expected_solved_sample": usd / passed if passed else None,
                               "note": "computed from response.usage x the live price table (humaneval/spend.py)"}
        else:
            wall = sum(s["wall_seconds"] for s in meta["segments"])
            arm["self_hosted"] = {"wall_seconds": wall, "wall_hours": wall / 3600,
                                  "completion_tokens_per_second": arm["tokens"]["completion_total"] / wall,
                                  "engine": meta["segments"][-1].get("engine_stats"), "hardware": meta["hardware"]}
        arms[name] = arm

    # identical task set and prompt across every arm
    checks = {
        "same_dataset_sha256_all_arms": len({a["dataset_sha256"] for a in arms.values()}) == 1,
        "same_prompt_template_sha256_all_arms": len({a["prompt_template_sha256"] for a in arms.values()}) == 1,
        "same_164_problems_all_arms": all(a["pass_at_k"]["pass@1"]["n_problems"] == 164 for a in arms.values()),
    }
    assert all(checks.values()), checks

    # break-even hourly rate for the self-hosted machine: below this rate self-hosting costs less than the hosted API
    s = arms["self_hosted_sampled"]
    hours = s["self_hosted"]["wall_hours"]
    breakeven = {}
    for key in ("haiku_4_5", "opus_5"):
        a = arms[key]
        per_sample = a["api_cost"]["usd_per_sample"]
        hosted_equiv_same_samples = per_sample * s["n_samples"]                    # hosted cost of the same 1,640 samples
        hosted_per_solved = a["api_cost"]["usd_per_expected_solved_sample"]
        breakeven[key] = {
            "hosted_cost_for_1640_samples_usd": hosted_equiv_same_samples,
            "usd_per_hour_below_which_selfhosting_is_cheaper_per_sample": hosted_equiv_same_samples / hours,
            "usd_per_hour_below_which_selfhosting_is_cheaper_per_solved_sample":
                hosted_per_solved * s["passed_samples"] / hours,
        }

    ledger = ROOT / "spend_ledger.jsonl"
    total_usd = sum(json.loads(x)["usd"] for x in ledger.read_text().splitlines() if x.strip())
    result = {
        "task": "HumanEval, 164 problems, identical prompt, sandbox and scorer for every arm",
        "checks": checks, "arms": arms,
        "comparison_at_common_k": {
            f"pass@{k}": {name: arms[name]["pass_at_k"].get(f"pass@{k}") for name in arms} for k in COMMON_KS},
        "break_even_self_hosting_rate": breakeven,
        "total_hosted_spend_usd_including_pilots": total_usd,
        "budget": {"user_funded_usd": 6.80, "hard_cap_usd": 6.50},
        "caveats": [
            "Sampling differs by design: the self-hosted arm samples at T=0.8, top_p=0.95 (seeded); the hosted arms use the API's default "
            "sampling because Opus 5 / Sonnet 5 reject temperature/top_p. Hosted runs are not seed-reproducible; scoring is reproducible from the saved completions.",
            "Sample counts differ: 10 per problem (self-hosted, Haiku) vs 3 (Opus 5, budget). Compare at common k (1-3); pass@k from n=10 is the unbiased "
            "estimator, from n=3 it is the plain fraction solved in 3 draws.",
            "max_tokens=512 for every arm (the self-hosted protocol). It truncated 15/1640 self-hosted, 74/1640 Haiku and 12/492 Opus samples, every "
            "truncated sample that ran failed except a few Haiku ones; see truncation_upper_bound_sample_pass_rate for the most the cap could be costing each arm.",
            "Opus 5 ran with thinking disabled (cost); default adaptive thinking was measured on only 2 calls and projected over the cap.",
            "HumanEval is public and very likely in every model's training data; absolute scores overstate real-world ability for all arms.",
            "pass@k CIs are problem-level bootstrap (the sampling unit for 'a different set of problems'); they do not include seed-to-seed variance.",
            "The self-hosted machine has no dollar cost here: no hourly rate was supplied. break_even_self_hosting_rate is the rate-free comparison.",
        ],
    }
    out = ROOT / "phase6_comparison.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", out)
    for k in COMMON_KS:
        print(f"pass@{k}:", {n: (round(v['value'] * 100, 2) if v else None) for n, v in result["comparison_at_common_k"][f"pass@{k}"].items()})
    print("break-even $/h (per sample):", {k: round(v["usd_per_hour_below_which_selfhosting_is_cheaper_per_sample"], 2) for k, v in breakeven.items()})
    print("break-even $/h (per solved):", {k: round(v["usd_per_hour_below_which_selfhosting_is_cheaper_per_solved_sample"], 2) for k, v in breakeven.items()})
    print("hosted spend incl. pilots: $%.4f" % total_usd)


if __name__ == "__main__":
    main()
