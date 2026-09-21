# AgentForge Eval — Task Board

Live status. Updated after every phase and non-trivial sub-step. A phase is only marked DONE after
its Done-when check has been run and the real output is pasted here.

Ground rules: no real API spend without explicit go-ahead; no silent scope changes (Phase 2 model
swap needs confirmation); negative results are published with root cause.

| Phase | Title | Status |
|---|---|---|
| 1 | OpenAI-compatible `/v1/chat/completions` on the engine | DONE (2026-09-21) |
| 2 | Model capability decision (STOP, needs confirmation) | DONE (confirmed 2026-09-21) |
| 3 | Golden test for the new model (Qwen2.5-Coder-0.5B) | IN PROGRESS |
| 4 | Real sampling (temperature + top-p, seedable) in the engine | NOT STARTED (new) |
| 5 | HumanEval harness: generate k, test, pass@k against the self-hosted model | NOT STARTED (replaces old Phase 4) |
| 6 | Budget experiment: same 164 problems, same k, frontier anchor (needs explicit spend approval) | NOT STARTED (replaces old Phase 5) |
| 7 | Write-up | NOT STARTED (was Phase 6) |

---

## Phase 1 — OpenAI-compatible endpoint

Done-when: an OpenAI-shaped chat-completion request returns a correctly formatted response from the
existing engine; streaming and non-streaming both covered by tests.

### Log
- (2026-09-21) Board created. Read `engine/engine/api.py`: `/generate` builds a `Request`, submits it
  to `EngineLoop`, and consumes tokens from an asyncio queue fed by `req.sink`. Plan: factor that
  submit/consume machinery so `/generate` and `/v1/chat/completions` share it.

### Built
- `engine/engine/api.py`
  - `_start()` (inside `create_app`): the submit/consume machinery formerly inline in `/generate`
    (413/429 admission control, `Request` + sink, `IncrementalDetokenizer`, cancellation, metrics,
    `finish()` logging). `/generate` now calls it; behaviour unchanged (existing tests pass).
  - `POST /v1/chat/completions`: non-streaming returns `chat.completion` (choices/message/
    finish_reason/usage); `stream=true` returns SSE `chat.completion.chunk` frames (role chunk,
    content deltas, finish chunk, optional usage chunk for `stream_options.include_usage`, then
    `data: [DONE]`).
  - `ChatIn` / `ChatMessage` / `render_chat_prompt()`: uses the tokenizer's `chat_template` when
    present (instruct models, Phase 2+), else a plain `System: ..\nUser: ..\nAssistant:` transcript
    (GPT-2 has no template).
- `engine/tests/test_openai_api.py`: 14 tests (response shape, stream == non-stream, chat ==
  /generate token-for-token, include_usage, multibyte streaming, content-parts, greedy determinism,
  400/422 paths, template logic).

### Known limitations (deliberate, carried forward)
- Decoding is greedy only: `temperature`/`top_p` are accepted and ignored (LiteLLM always sends
  them). **Matters for Phase 5:** pass@k over k trials with greedy decoding gives k identical
  samples per task unless sampling is added. Flagging now; needs a decision before Phase 5.
- `stop` is accepted but not enforced. Revisit in Phase 4 if bench's agent loop needs it.
- Errors use FastAPI's `{"detail": ..}` shape, not OpenAI's `{"error": ..}`. Revisit in Phase 4.
- Context is still capped at 1024 tokens (GPT-2 position table); a prompt >= 1024 tokens is a 400.
  A coding-agent loop will hit this fast; part of the Phase 2 model decision.

### Problems hit
- The engine venv is Linux-only, so tests run via WSL (`wsl.exe -d Ubuntu -- bash -lc ...`).
- First test run: my multibyte streaming test wrongly asserted no U+FFFD. A `max_tokens` cut
  mid-character legitimately ends in U+FFFD (same as a plain decode). Fixed the test to compare
  against `/generate`'s text instead; the code was correct.
- Scripted edits turned `\n` escapes into literal newlines (SyntaxError at import). Fixed by hand and
  re-verified with `ast.parse`.

### Done-when check (actual output)
```
$ .venv/bin/python -m pytest -m 'not perf' -q
155 passed, 1 deselected, 2 warnings in 156.93s (0:02:36)
```
141 pre-existing + 14 new. The one deselected test is the noisy wall-clock perf guard, excluded by
the engine's own `make test`. Changes are uncommitted in the working tree.

---

## Phase 2 — Model capability decision (STOP: awaiting explicit confirmation)

Done-when: a written proposal exists and has been explicitly confirmed. **CONFIRMED 2026-09-21 (see
"Confirmed decisions" at the end of this section).**

### What the bench agent loop actually demands (read from `bench/agent/prompts.py`, `agent/loop.py`)
- Plain-text protocol, deliberately NOT provider tool-calling: the model must emit exactly one
  ```` ```bash ```` block per turn; last block wins; submit = a block containing only `echo TSBENCH_DONE`.
- `temperature=0`, `max_tokens=1024` per turn, up to 40 turns / 900 s per attempt. The full history is
  resent every turn, so prompt length grows every turn (observations are truncated by `_truncate`).
  Real API runs in bench's own log used ~390K-540K cumulative prompt+completion tokens per task.
- Fairness contract clause 2: the prompt is identical for every model. Nothing gets tuned for ours.

### Evidence already in the repo that sets expectations (bench/docs/step7-real-model-run.md)
| Model | Attempts | Resolved |
|---|---|---|
| qwen2.5-coder:14b (local) | 240 | 0 |
| codestral 22B (local, partial) | ~240 | 0 |
| qwen3:14b (local, partial) | 167 | 0 |
| claude-haiku-4.5 (paid) | 20 (1 each) | 6 (30%) |

**Consequence, stated up front:** models 10x larger than anything CPU-feasible here scored 0/647 on this
dataset under this scaffold. A ~0.5-3B model will very probably score 0% too. If it does, pass@k is 0
for every k and the self-hosted-vs-anchor comparison has no signal on the self-hosted side. That would
be a real, publishable negative result (with root cause), not something to hide, but it should not be a
surprise at Phase 5. This proposal is about the best *realistic* chance, not a promise.

### Two engine facts that shape every option
1. **The forward pass is GPT-2-specific** (`engine/model_runner.py`: Conv1D weights, learned absolute
   position table, LayerNorm, MHA, GELU; `engine/cache.py`: `N_LAYERS=12, N_HEADS=12, HEAD_DIM=64`
   module constants). Any modern instruct/code model (RoPE, RMSNorm, GQA, SwiGLU) means a new hand-written
   forward pass plus a config-driven cache. The no-`generate()` rule stays; HF is reference-only in tests.
2. **Context is hard-capped at 1024 tokens** (`MAX_CONTEXT`, the `wpe` table, `/generate` 400 above it).
   A 40-turn agent history needs thousands of tokens. Fixed by RoPE models, but every place that assumes 1024
   must go.
Also: there is no prefix caching, so each turn re-prefills the whole history. On CPU that is likely the
dominant cost (rough estimate, NOT measured: a 1.5B model at ~3 GFLOP/token and ~10K-token late-episode
prompts is tens of seconds of prefill per turn). Phase 4 will measure it; we should expect
attempts to take minutes, not seconds, and pass@k trial counts to be bounded by wall clock, not dollars.

Machine (WSL2): 28 logical cores, 15 GB RAM, torch CPU fp32. Numbers below are from my memory of the model
cards; the exact values get verified from each model's `config.json` before anything is downloaded.

### Candidates (all share ONE architecture: Qwen2 = RoPE + RMSNorm + GQA + SwiGLU + QKV bias, tied embeddings)
| | A. Qwen2.5-Coder-0.5B-Instruct | B. Qwen2.5-Coder-1.5B-Instruct (**recommended**) | C. Qwen2.5-Coder-3B-Instruct |
|---|---|---|---|
| Params / fp32 weights | ~0.5B / ~2 GB | ~1.5B / ~6 GB | ~3B / ~12 GB |
| Layers, q/kv heads | 24, 14/2 | 28, 12/2 | 36, 16/2 |
| Context | 32K | 32K | 32K |
| KV bytes/token (fp32) | ~24 KB | ~57 KB | ~74 KB (est.) |
| License | Apache-2.0 | Apache-2.0 | Qwen Research (not Apache; check terms) |
| Fits 15 GB RAM in fp32 | easily | yes, room for a large paged KV pool | barely, no headroom; bf16 would break the byte-identical-vs-HF exactness rule unless the golden is also bf16 |
| Instruction following | weak | modest | somewhat better |
| Realistic role | fast golden-test fixture, wiring smoke tests | primary served model | stretch only, likely infeasible on 15 GB fp32 |

GQA (2 KV heads) is a real win for the engine's thesis: KV cache per token is ~5-10x smaller than an MHA
model of the same size, so the paged pool holds far more concurrent long histories.

Why not the others (short): Llama-3.2-3B-Instruct is general (not code-tuned), gated behind a license-accept/
HF login, and larger; DeepSeek-Coder-1.3B is only 16K context, older, MHA (bigger KV), and weaker at
instruction following; SmolLM2-1.7B-Instruct is Apache and Llama-arch but not code-tuned. Any of these
would still need the RoPE/RMSNorm/SwiGLU path, so Qwen2 costs no more engine work and is the strongest code model at the size.

### Recommendation
Implement the Qwen2 forward pass once. Use **A (0.5B)** for the fast golden test and Phase 4 wiring smoke
tests (same code path as B, ~3x cheaper per run), and serve **B (1.5B)** for the real runs. C is out unless
you want to relax a rule.

### Decisions needed from you
1. Confirm model: A for tests + B for serving (recommended), or something else?
2. **GPU:** the machine has an RTX 5060 Ti (16 GB) that bench already uses via Ollama. Engine rule 6 says
   "no GPU code paths". CPU-only keeps that rule and the exactness story; GPU would make a 3B+ model and
   long prompts practical but is a scope change I will NOT make without your say-so. Recommendation: stay CPU.
3. **Sampling:** the engine is greedy-only, bench pins `temperature=0`. With greedy decoding repeated
   trials of a deterministic engine yield identical outcomes, so "pass@k for k>1" on the self-hosted model
   would be k copies of one sample. Options: (a) add temperature sampling to the engine (deviates from the
   greedy-only golden guarantees, must be a separate, seeded path) and run the self-hosted trials at T>0
   (a fairness-contract question, since the anchor runs at T=0); (b) keep greedy and report pass@1 only for
   the self-hosted model. Not needed until Phase 5, but it affects design, so flagging it now.
4. **Expectation check:** are you OK proceeding knowing 0% pass@k is the likely outcome for these sizes?
   If not, the honest alternative is a different thesis framing (e.g., throughput/cost-per-token vs
   quality) rather than a bigger model I can't serve.

### Confirmed decisions (user, 2026-09-21)
1. **Models:** A = Qwen2.5-Coder-0.5B-Instruct for golden tests / smoke tests; B = Qwen2.5-Coder-1.5B-Instruct
   for serving. C (3B) ruled out (license, fp32-vs-bf16 golden consistency). One Qwen2 forward pass covers both.
2. **CPU-only serving. Rule 6 (no GPU code paths) is NOT lifted.** It is core to the thesis: paged KV-cache
   management matters because CPU RAM is scarce. No fine-tuning is planned, so GPU training is not in play.
3. **Sampling: option (b).** Self-hosted model is pass@1 only. No seeded sampling is added to the engine (it
   would touch the rigorously golden-tested greedy path for a feature this experiment doesn't need).
   Methodology note to publish: the asymmetry is real and stated (self-hosted = pass@1; frontier anchor = real
   pass@k because hosted APIs support temperature>0). No forced parity.
   **Deferred / future scope:** seeded temperature sampling in the engine.
4. **Proceed, with a reframe (below).**

### THESIS REFRAME (replaces the original "more trials buys statistically meaningful pass@k" thesis)
Reason: bench's own prior runs show 14B-22B local models resolving 0/647 attempts. A 0.5-1.5B model failing
too is a near-certainty, not a finding; more trials of a structurally incapable model just yields more zeros,
and "close the pass@k gap via volume" has no gap to characterize.

New thesis: **characterize the self-hosted engine's throughput, cost and latency under real multi-turn agentic
traffic** (bench's actual request shape: up to 40 turns, growing context, greedy, max_tokens 1024), i.e. stress
continuous batching + paged KV cache with a realistic agent workload, independent of whether the served model
can solve anything.
- **Phase 4** is now: wire bench to the engine (LiteLLM custom provider) and run real bench attempts to
  prove the wiring and to record real per-turn/per-attempt latency, tokens, prefill vs decode time and KV usage.
  The scored result (almost surely resolved=False) is still logged honestly, with pass/fail.
- **Phase 5** is now: (i) engine load characterization with concurrent agent rollouts (throughput / latency /
  KV-cache occupancy / preemptions vs concurrency, with the paged vs non-paged ablation where it applies), and
  (ii) the paid frontier-model anchor, repurposed: a real pass@k + cost baseline from a capable model *on its own
  terms*, not something the self-hosted path is expected to match. Still needs explicit spend approval and a
  design presented first.
- **Reporting:** the self-hosted near-zero solve rate is reported plainly and cross-referenced to bench's prior
  0/647 (qwen2.5-coder:14b 0/240, codestral ~0/240, qwen3:14b 0/167) as corroboration, not a surprise.

---

## PIVOT (user, 2026-09-21): evaluate the self-hosted model on HumanEval, not on bench's SWE tasks

This supersedes the "THESIS REFRAME" above (throughput/latency characterization under 40-turn agent
traffic) and confirmed decision 3 (pass@1 only, no sampling). Decisions 1 and 2 (Qwen2.5-Coder 0.5B tests /
1.5B serving, CPU-only, rule 6 stays) are unchanged. The engine-load/throughput measurements are no longer
the headline; latency, tokens and cost-per-token on this hardware are still logged in Phase 5.

**Rationale.** Phase 2 predicted, and bench's own history confirms (14B-22B local models: 0/647), that a
0.5-1.5B model scores a certain 0% on SWE-bench-style repo-patch tasks. That is a task/model mismatch, not a
result worth a budget. Change: keep bench's SWE harness untouched (still valid, still useful elsewhere) but do
NOT route the self-hosted model through it. Evaluate on **HumanEval** (164 Python function-level problems:
signature + docstring + hidden unit tests), a task class small code models are actually built for. This is a
capability match, not a lowered bar to manufacture a better number.

**Sampling.** HumanEval's standard protocol is pass@k with real sampling (temperature > 0, k independent
generations per problem), the statistic bench's `pipeline/stats.py` already implements (unbiased estimator +
bootstrap). Short independent generations are a much safer place to add sampling than the long agentic loop.
This reverses decision 3(b): seeded sampling is now in scope (Phase 4), and the earlier "pass@1 vs pass@k
asymmetry" methodology note no longer applies (both sides get pass@k).

**Constraint that carries over:** the greedy decode path is golden-tested and must stay byte-identical; Phase 4
must add a regression test proving that.

### Revised phases
- **Phase 3 (unchanged):** golden greedy fixtures for Qwen2.5-Coder-0.5B-Instruct from HF; byte-identical engine output.
- **Phase 4 (new):** temperature + top-p sampling with seedable RNG, short generations only.
  Done-when: sampled generation works; greedy golden tests pass unchanged; fixed-seed reproducibility test passes.
- **Phase 5 (replaces old 4):** single-shot generate-and-test harness: 164 problems x k completions via the Phase 1
  endpoint, executed against each problem's real tests reusing bench's `PythonAdapter` test-execution plumbing
  (no agent loop, no tool calls), pass@k via `pipeline/stats.py`. Fully self-hosted, zero spend.
  Done-when: all 164 run end-to-end; real pass@k with bootstrap CI; latency, tokens, cost-per-token logged.
- **Phase 6 (replaces old 5, needs explicit approval):** same 164 problems, same k, 1-2 frontier models; get a real
  cost estimate before asking. Compare pass@k with CIs and cost-per-solved-problem. Done-when: results file with
  both sides on the same task set.
- **Phase 7:** README with Phase 6 numbers verbatim, including where the self-hosted model fell short.

### Open items I am flagging, not acting on
- **Serving model for HumanEval (0.5B vs 1.5B):** Phase 3 golden covers 0.5B only, as specified. I started to
  download the 1.5B for its golden set and the download was declined. I have not downloaded it. If the 1.5B is
  the model to serve/evaluate, it needs its own golden fixtures; tell me when to fetch it.
- **HumanEval is not in the repo** and needs a download (openai/human-eval or the HF dataset) in Phase 5;
  also note HumanEval is a public benchmark with known training-data contamination risk for code models, which
  the write-up should state.
- **Executing model-generated code** in Phase 5 needs a sandbox/timeout (bench's Python adapter/sandbox will be reused; verify its isolation).
