# CLAUDE.md — agentforge-eval

## What this project is

Combines two existing projects into one: engine/ (from llm-serve — a from-scratch GPT-2
inference server with a continuous-batching scheduler and paged KV-cache manager) and bench/
(from ts-bench — a SWE-bench-style benchmark for AI coding agents across TypeScript/Python/Java).

The thesis: ts-bench's real bottleneck is dollars-per-rollout on hosted APIs — a small budget
buys only a handful of trials, nowhere near enough for a statistically meaningful pass@k against a
real frontier model. The engine's whole value proposition is running many concurrent generations
cheaply via continuous batching + paged KV cache. This project self-hosts an efficient inference
engine and points the bench's agent loop at it, using the freed-up budget to run enough trials for
a real result, then spending real API budget surgically on one frontier model as an anchor.

## Non-negotiables (inherited from both parents)

- No model.generate()/HF generation helpers in engine/'s serving path — hand-rolled forward
  pass only, HF reference allowed in tests only.
- Correct before fast: a change isn't done without its test passing and, where relevant, a real
  number written to a results file.
- No silent scope changes — the model swap (engine can't serve a coding agent with GPT-2 small)
  is the biggest one; propose and get confirmation before acting on it.
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