"""Scoring and summary: pass@k via bench's pipeline.stats (unbiased estimator + per-task bootstrap), plus a
problem-level bootstrap confidence interval for the mean pass@k, plus latency / token / throughput accounting.

The CI resamples PROBLEMS with replacement (the sampling unit for "how would this number move on a different
set of problems"), B=10,000, fixed seed, percentile interval. pipeline.stats' own stderr is also reported.
"""
from __future__ import annotations

import random
import statistics
from collections import Counter

from pipeline.stats import compute_model_stats, pass_at_k


def per_problem_pass_at_k(rows: list[dict], k: int) -> dict[str, float]:
    by: dict[str, list[bool]] = {}
    for r in rows:
        by.setdefault(r["task_id"], []).append(bool(r["passed"]))
    return {t: pass_at_k(len(o), sum(o), k) for t, o in by.items() if len(o) >= k}


def bootstrap_ci(values: list[float], b: int = 10_000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return means[int(b * alpha / 2)], means[int(b * (1 - alpha / 2)) - 1]


def percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def summarize(rows: list[dict], label: str, ks: list[int], wall_seconds: float | None = None,
              usd_per_hour: float | None = None) -> dict:
    """rows: one per (problem, sample), with `passed`, `outcome`, `latency_s`, token counts, `cost_usd`."""
    records = [{"model": label, "instance_id": r["task_id"], "resolved": r["passed"],
                "cost_usd": r.get("cost_usd") or 0.0} for r in rows]
    n_per = Counter(r["task_id"] for r in rows)
    out: dict = {"label": label, "n_problems": len(n_per), "n_samples": len(rows),
                 "samples_per_problem": dict(Counter(n_per.values()))}
    passk = {}
    for k in ks:
        vals = per_problem_pass_at_k(rows, k)
        if not vals:
            continue
        ms = compute_model_stats(records, label, k, n_bootstrap=1000, seed=0)
        lo, hi = bootstrap_ci(list(vals.values()))
        mean = sum(vals.values()) / len(vals)
        passk[f"pass@{k}"] = {"value": mean, "ci95": [lo, hi], "n_problems": len(vals),
                              "stderr_across_problems": ms.mean_pass_at_k_stderr,
                              "problems_solved_at_least_once": sum(1 for r_ in _by_task(rows).values() if any(r_))}
    out["pass_at_k"] = passk
    out["outcomes"] = dict(Counter(r["outcome"] for r in rows))
    out["assembly_modes"] = dict(Counter(r.get("assembly_mode") for r in rows))
    out["truncated_by_max_tokens"] = sum(1 for r in rows if r.get("finish_reason") == "length")
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    ctoks = [r["completion_tokens"] for r in rows if r.get("completion_tokens") is not None]
    ptoks = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens") is not None]
    out["tokens"] = {"prompt_total": sum(ptoks), "completion_total": sum(ctoks),
                     "completion_mean": statistics.mean(ctoks) if ctoks else None,
                     "completion_max": max(ctoks) if ctoks else None}
    if lat:
        out["request_latency_s"] = {"mean": statistics.mean(lat), "p50": percentile(lat, .5),
                                    "p90": percentile(lat, .9), "p99": percentile(lat, .99), "max": max(lat),
                                    "note": "includes time queued inside the engine behind other requests"}
    if wall_seconds:
        out["generation_wall_seconds"] = wall_seconds
        out["completion_tokens_per_second"] = sum(ctoks) / wall_seconds if ctoks else None
        out["samples_per_second"] = len(rows) / wall_seconds
        solved = out["pass_at_k"].get("pass@1", {}).get("value")
        if usd_per_hour is not None:
            cost = wall_seconds / 3600 * usd_per_hour
            out["self_hosted_cost"] = {
                "assumed_usd_per_hour": usd_per_hour, "usd_total": cost,
                "usd_per_million_completion_tokens": cost / max(sum(ctoks), 1) * 1e6,
                "usd_per_sample": cost / len(rows),
                "usd_per_expected_solved_sample": (cost / (solved * len(rows))) if solved else None,
                "note": "wall-clock x an ASSUMED hourly rate; the rate is an input, not a measurement"}
        else:
            out["self_hosted_cost"] = "not computed: no --usd-per-hour supplied (no rate is assumed by default)"
    api_cost = sum(r.get("cost_usd") or 0.0 for r in rows)
    out["api_cost_usd_total"] = api_cost
    return out


def _by_task(rows: list[dict]) -> dict[str, list[bool]]:
    by: dict[str, list[bool]] = {}
    for r in rows:
        by.setdefault(r["task_id"], []).append(bool(r["passed"]))
    return by
