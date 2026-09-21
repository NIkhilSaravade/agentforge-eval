# AgentForge Eval

A from-scratch CPU inference engine (continuous batching, paged KV cache) pointed at a real coding benchmark, then compared
against hosted Claude models on the same task set, with real money spent on the comparison.

**The short version:** the engine works and is verified byte-for-byte against the HuggingFace reference, and it makes running many
concurrent generations on one CPU practical. But the model it can serve on that CPU, Qwen2.5-Coder-1.5B, is **clearly worse than
even the cheapest hosted model**: 66.89% pass@1 on HumanEval versus 92.74% for Claude Haiku 4.5 and 96.95% for Claude Opus 5. On this
hardware, self-hosting is cheaper than the hosted API only if the machine costs less than about $0.89/hour (against Haiku 4.5).

That is not the result the project set out to find, and this README says so up front. The full story is below, including the original
plan that did not survive contact with the data.

## Results

Task: HumanEval, 164 Python problems. Every arm uses the same problems, the same fixed prompt, the same sandbox and the same scorer
(checked by hash in the results file). pass@k uses the unbiased estimator; intervals are 95% bootstrap CIs that resample problems.
Source of every number below: [`bench/results/humaneval/phase6_comparison.json`](bench/results/humaneval/phase6_comparison.json).

| Arm | Samples | pass@1 | pass@3 | pass@10 |
|---|---|---|---|---|
| Self-hosted Qwen2.5-Coder-1.5B, sampled (T=0.8, top_p=0.95) | 1,640 (n=10) | 66.89% [61.34, 72.50] | 80.71% [75.39, 85.83] | 87.80% [82.93, 92.68] |
| Self-hosted Qwen2.5-Coder-1.5B, greedy | 164 (n=1) | 72.56% [65.85, 79.27] | n/a | n/a |
| Claude Haiku 4.5 | 1,640 (n=10) | 92.74% [88.90, 96.10] | 94.30% [90.71, 97.36] | 95.73% [92.68, 98.78] |
| Claude Opus 5 (thinking off) | 492 (n=3) | 96.95% [93.90, 99.39] | 96.95% [93.90, 99.39] | not measured (n=3) |

A sanity check on the whole stack: Qwen's technical report (Table 16) lists 70.7 for this model on HumanEval; our greedy run scored 72.56%
(119 of 164), inside our interval. The report's fetched text does not state its decoding setting, so this is a consistency check, not an
exact reproduction.

### Cost and speed

| Arm | Cost | Wall-clock | Notes |
|---|---|---|---|
| Claude Haiku 4.5 | $2.304695 for 1,640 samples | 354.0 s | $0.001405 per sample |
| Claude Opus 5 | $3.59587 for 492 samples | 219.8 s (145.8 + 74.0) | $0.007309 per sample |
| Self-hosted 1.5B | no dollar figure (see below) | 9,351.8 s (2.60 h) for 1,640 samples | 29.08 completion tokens/s |

Total hosted spend for the whole project, pilots included: **$6.064139** against a hard cap of $6.50 (the account was funded with $6.80).

**No dollar cost is reported for the self-hosted run**, because no hourly rate for the machine was supplied and none is assumed.
The rate-free comparison, from the measured 2.60 hours, is the break-even hourly rate: below it, self-hosting the same 1,640 samples
costs less than the hosted API.

| Against | Hosted cost for 1,640 samples | Break-even, per sample | Break-even, per solved sample |
|---|---|---|---|
| Haiku 4.5 | $2.30 | $0.89 / hour | $0.64 / hour |
| Opus 5 | $11.99 (extrapolated from the measured per-sample cost) | $4.61 / hour | $3.18 / hour |

Those break-evens ignore the quality gap, which is large. The per-solved-sample column is lower because the self-hosted model solves fewer of
its samples, so the hardware would have to be even cheaper for self-hosting to win per solved sample.

### Where the self-hosted model fell short

- **Quality.** 25.9 points behind Haiku 4.5 and 30.1 points behind Opus 5 at pass@1, and still behind at pass@3 (80.71% vs 94.30% vs 96.95%).
  Of 164 problems it solved at least one of ten samples on 144 and never solved 20; 61 problems it solved ten times out of ten.
- **Speed.** The same 1,640 samples took 2.6 hours on this CPU versus 354 seconds (Haiku 4.5) and about 220 seconds for 492 samples (Opus 5).
- **Cost.** Beats Haiku 4.5 only below about $0.89/hour of hardware (about $0.64/hour per solved sample). It is not a cheaper way to get
  the same quality, because the quality is not the same.
- **It never had a chance at the original benchmark.** See the history below.

### Where self-hosting still helps

No per-token bill and no data leaving the machine. The engine's continuous batching is what makes 2.6 hours possible at all: on the same
CPU, one decode step for 32 concurrent sequences costs 3.4x the time of one sequence but produces 32 tokens instead of 1.
From [`engine/results/decode_sweep_qwen2.5-coder-1.5b.json`](engine/results/decode_sweep_qwen2.5-coder-1.5b.json) (20 threads, 400-token context):

| Batch | Decode step | Decode throughput | Prefill throughput |
|---|---|---|---|
| 1 | 0.1126 s | 8.9 tok/s | 215.9 tok/s |
| 8 | 0.2587 s | 30.9 tok/s | 373.1 tok/s |
| 16 | 0.2395 s | 66.8 tok/s | 328.2 tok/s |
| 32 | 0.3824 s | 83.7 tok/s | 338.5 tok/s |

That is one run per row and it is noisy (an earlier run measured batch 8 at about 42 tok/s). In the real HumanEval workload the engine ran at
an average batch of 31.7 and a decode step took 0.96 s, well above the 0.38 s of the uniform benchmark. Attention padding waste was 37%
(mixed-length sequences padded to the longest), a suspect that was not verified. Prefill is not shared between the ten samples of one prompt
(no prefix caching), and took 1,094.6 s of the 9,340.8 s of engine time.

## How we got here (the original plan did not survive)

The project began as: run [ts-bench](bench/) (a SWE-bench-style benchmark of real bug fixes in TypeScript, Python and Java repos) against a
self-hosted model, use the dollars saved by cheap concurrent generation to run enough trials for a statistically meaningful pass@k, and spend
real API budget only on one frontier anchor.

That failed before it started, and the evidence was already in the repo. ts-bench's own earlier runs scored local open-weight models 10x
larger than anything this CPU can serve at **0 resolved out of 647 attempts** (qwen2.5-coder:14b 0/240, codestral about 0/240, qwen3:14b 0/167),
while Claude Haiku 4.5 resolved 6 of 20. A 0.5-1.5B model failing on that task class is a certainty, not a finding, and more trials of a model
that cannot solve anything only produce more zeros. So the plan changed twice, each time with the user's approval and each time recorded on
[the task board](docs/agentforge-task-board.md):

1. First reframe: stop measuring solve rate and measure the engine's throughput under agent-shaped traffic.
2. Then a better pivot: evaluate the model on a task class it is actually built for (HumanEval, function-level Python), with real seeded
   sampling and real pass@k. ts-bench's SWE harness was left untouched.

The result above is the outcome of the second plan: even on the task class small code models are built for, the self-hosted model loses to
the hosted ones.

## What was built

- **Engine** ([`engine/`](engine/), from llm-serve): a hand-written forward pass, continuous-batching scheduler and paged KV cache, no
  `generate()` in the serving path. This project added an OpenAI-compatible `/v1/chat/completions` (streaming and not), a hand-written
  Qwen2 forward pass (RMSNorm, RoPE, grouped-query attention, SwiGLU) for Qwen2.5-Coder 0.5B and 1.5B, and seeded temperature/top-p sampling.
  The greedy path is verified byte-identical to the HuggingFace fp32 reference for both models across single-sequence, static and continuous
  batching, requests joining and leaving mid-generation, paged KV at two block sizes, and forced preemption-and-recompute (162 golden tests).
  The full engine suite is 354 tests.
- **HumanEval harness** ([`bench/humaneval/`](bench/humaneval/)): generate k completions per problem from any endpoint, execute each in an
  isolated container (no network, non-root, read-only root filesystem, memory/pid/CPU caps), score with the dataset's own tests. It is gated:
  the dataset's 164 canonical solutions must pass (164/164) and an empty completion must not (0/164), through the same pipeline. Bench suite: 79 tests.
- **Spend control** ([`bench/humaneval/spend.py`](bench/humaneval/spend.py)): prices from Anthropic's pricing page, cost computed from each
  response's token usage, an append-only ledger shared by every hosted call, and a circuit breaker that stops requests before they are made.

## Things found by measuring

Each was found while running the experiment, and each has a test or a saved measurement:

- **The gate caught a harness bug.** The first gate run passed 160 of 164 gold solutions: four problems' tests call helpers defined in the
  prompt, and the first harness put program and tests in separate namespaces. Fixed; then 164/164.
- **The stop token leaked into chat replies** (`...` followed by the literal text `<|im_end|>`). Fixed on the chat path only.
- **A sampling slowdown I introduced.** Top-p sorted the whole 152K-token vocabulary per row per step (11.8 ms/row/step). An exact fast path
  over the top 256 tokens brought it to 0.45 ms on real logits. My first version disagreed with the reference on one razor-edge case because
  of float32 rounding, so the decision arithmetic is float64.
- **Loading the model on one thread and serving from another made decoding 45% slower.** From
  [`engine/results/thread_penalty_qwen2.5-coder-1.5b.json`](engine/results/thread_penalty_qwen2.5-coder-1.5b.json) (32 concurrent requests,
  no HTTP): 60.9 s all on the main thread, 88.1 s loaded on the main thread and run on another, 60.0 s loaded and run on the same other thread.
  The API server used to do the slow one and now builds the engine on its own thread. Raw PyTorch matmuls are the same speed on either thread,
  so the mechanism is not established; the design follows the measurement.
- **A `no_report` outcome that turned out to be a memory bomb.** Two self-hosted samples of HumanEval/100 grew a list forever; `docker inspect`
  showed `OOMKilled=true` (exit 137). Scored as a model failure and labelled `resource_killed`, not an infrastructure error.

## Caveats you should hold onto

- **Contamination.** HumanEval is public and very likely in every model's training data, so absolute scores for all arms overstate real-world
  ability. The comparison between arms is less affected than the absolute numbers.
- **The 512-token output cap truncated hosted replies more than self-hosted ones.** It cut 15 of 1,640 self-hosted samples, 74 of 1,640 Haiku
  samples and 12 of 492 Opus samples (Haiku and Opus write longer answers). Four of Opus 5's five failing problems are truncations. If every
  truncated sample had passed, the sample pass rates would be at most 67.80% (self-hosted), 97.07% (Haiku) and 99.39% (Opus): the hosted
  scores are understated, and the ranking cannot change. Re-running just the truncated hosted samples at a larger cap was not affordable within the cap.
- **Sampling differs by design.** The self-hosted arm sampled at T=0.8, top_p=0.95 with fixed seeds. The hosted arms used the API's default
  sampling because Opus 5 rejects `temperature` and `top_p`. Hosted runs cannot be re-generated identically; scoring is exactly reproducible
  from the saved completions.
- **Sample counts differ** (10, 10 and 3 per problem), so compare at k of 3 or less. Opus 5's failures are all-or-nothing per problem, so its
  pass@1, pass@2 and pass@3 are identical and carry no extra information.
- **Opus 5 ran with thinking disabled** to fit the budget. Default thinking was measured on only two calls and projected over the cap.
- **One machine, one run.** Timing depends on this i7-14700K under WSL2 with 16 GB; the intervals resample problems and do not include
  run-to-run or seed-to-seed variance.
- The engine is CPU-only and greedy or seeded-sampling only, with no prefix caching. That is a deliberate scope, not a claim about production serving.

## Reproduce

```bash
# engine tests (needs the model weights; downloaded from Hugging Face on first use)
cd engine && make setup && make test

# serve the 1.5B, exactly as used for the HumanEval run
LLM_SERVE_CONFIG='{"model":"qwen2.5-coder-1.5b","backend":"paged","batching":"continuous","max_batch":32,"kv_budget_mib":2048,"max_context":4096,"num_threads":20,"preemption":true}' \
  .venv/bin/python -m uvicorn engine.api:app --host 127.0.0.1 --port 8000

# HumanEval harness
cd bench && docker build -t agentforge-humaneval:py312 humaneval/
uv run python -m humaneval.gate                               # 164/164 gold, 0/164 empty
uv run python -m humaneval generate --model openai/qwen2.5-coder-1.5b --label qwen2.5-coder-1.5b \
  --api-base http://127.0.0.1:8000/v1 --api-key none --engine-url http://127.0.0.1:8000 \
  --dir results/humaneval/qwen2.5-coder-1.5b --n 10 --temperature 0.8 --top-p 0.95 --max-tokens 512 --workers 40
uv run python -m humaneval execute --dir results/humaneval/qwen2.5-coder-1.5b
uv run python -m humaneval report  --dir results/humaneval/qwen2.5-coder-1.5b --k 1,5,10
uv run python -m humaneval.compare                            # rebuilds phase6_comparison.json from saved results

# hosted arms spend real money; ANTHROPIC_API_KEY goes in bench/.env (gitignored). A project-wide cap is enforced:
uv run python -m humaneval generate --provider anthropic --model claude-haiku-4-5 --label claude-haiku-4-5 \
  --dir results/humaneval/claude-haiku-4-5 --n 10 --max-tokens 512 --budget-cap-usd 6.50
```

Scoring is exactly reproducible from the saved `completions.jsonl` files. Regenerating hosted completions will give different samples.

## Repository map

- [`docs/agentforge-task-board.md`](docs/agentforge-task-board.md): the live log of every phase: what was built, problems hit, and the
  real output of each done-when check. The place to look for the reasoning behind any number here.
- [`engine/`](engine/): the inference engine and its tests (`tests/test_golden*.py`, `test_sampling.py`, `test_openai_api.py`).
- [`bench/humaneval/`](bench/humaneval/): the HumanEval harness, sandbox, spend control and comparison; results in `bench/results/humaneval/`.
- [`bench/`](bench/) (rest): ts-bench, the SWE-bench-style benchmark this project started from; untouched.
- [`CLAUDE.md`](CLAUDE.md): the project rules (no `generate()` in the serving path, correct before fast, no spend without approval,
  negative results published with their root cause).
