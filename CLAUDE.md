# CLAUDE.md — agentforge-eval

## What this project is

Combines two existing projects into one: engine/ (from llm-serve — a from-scratch GPT-2
inference server with a continuous-batching scheduler and paged KV-cache manager) and bench/
(from ts-bench — a SWE-bench-style benchmark for AI coding agents across TypeScript/Python/Java).

The thesis (revised again 2026-09-21, see docs/agentforge-task-board.md, PIVOT): a model small enough to serve
on CPU (Qwen2.5-Coder 0.5B/1.5B) cannot realistically solve bench's SWE-bench-style repo tasks (bench's own 14B-22B
local models scored 0/647), so it is NOT evaluated there. It is evaluated on HumanEval (function-level Python),
where small code models are competent, with real seeded sampling and pass@k (bench's `pipeline/stats.py`). The
engine's cost/throughput advantage (continuous batching + paged KV cache, CPU-only) is then compared against a
frontier anchor on the same 164 problems with the same k, with real spend only after explicit approval.

## Non-negotiables (inherited from both parents)

- No model.generate()/HF generation helpers in engine/'s serving path — hand-rolled forward
  pass only, HF reference allowed in tests only.
- Correct before fast: a change isn't done without its test passing and, where relevant, a real
  number written to a results file.
- No silent scope changes — the model swap was the biggest one and is now confirmed (Qwen2.5-Coder
  0.5B tests / 1.5B serving, CPU-only; rule 6 no-GPU-serving stays). Anything else: propose first.
- Never spend real API budget without explicit go-ahead in conversation.
- Never tune a benchmark to look good — a negative result gets published with its root cause, not
  hidden or reworded.

## Repo layout

agentforge-eval/
  CLAUDE.md
  docs/agentforge-task-board.md   (live status, same style as ts-bench's task board)
  engine/    (from llm-serve — inference engine, scheduler, KV cache)
  bench/     (from ts-bench — agent loop, harness, eval pipeline)

## How to work here

Work one phase at a time from docs/agentforge-task-board.md. Update it after every phase and
every non-trivial sub-step: what was built (real file/function names), problems hit and how
fixed, the Done-when check and its actual output. Never mark something done without running the
check and showing the result.

Commit and push after every phase and every meaningful sub-step, not just at the end. Do not let
uncommitted work accumulate. Each commit message states honestly what is verified and what is not
(with the real test output); code and its task-board update go in the same commit where that makes
sense. Never skip hooks or force-push.
