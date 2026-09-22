# AgentForge Eval — Task Board

Live status. Updated after every phase and non-trivial sub-step. A phase is only marked DONE after
its Done-when check has been run and the real output is pasted here.

Ground rules: no real API spend without explicit go-ahead; no silent scope changes (Phase 2 model
swap needs confirmation); negative results are published with root cause.

| Phase | Title | Status |
|---|---|---|
| 1 | OpenAI-compatible `/v1/chat/completions` on the engine | DONE (2026-09-21) |
| 2 | Model capability decision (STOP, needs confirmation) | DONE (confirmed 2026-09-21) |
| 3 | Golden test for the new model (Qwen2.5-Coder-0.5B) | DONE (2026-09-21) |
| 4 | Real sampling (temperature + top-p, seedable) in the engine | DONE (2026-09-21) |
| 5 | HumanEval harness: generate k, test, pass@k against the self-hosted model | DONE (2026-09-22) |
| 6 | Budget experiment: same 164 problems, same k, frontier anchor (needs explicit spend approval) | DONE (2026-09-22), $6.06 spent of $6.50 cap |
| 7 | Write-up | DONE (2026-09-22) |

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

---

## Phase 3 — Golden test for Qwen2.5-Coder-0.5B-Instruct (DONE 2026-09-21)

Done-when: a golden test for the new model passes, same rigor as the GPT-2 golden test.

### Built
- `engine/engine/spec.py`: `ModelSpec` (layers, q/kv heads, head_dim, max_context, EOS ids, `bytes_per_token`)
  and `GPT2_SPEC`, which reproduces the old module constants exactly.
- Parameterized on the spec instead of hard-coded GPT-2 dims / 1024 context: `cache.py` (`ContiguousKVCache`,
  `SlotPool`, `PagedPool`, `BatchAccess`), `block_manager.py` (`slots_for_budget`, `blocks_for_budget`),
  `scheduler.py` (max context, multi-EOS, pool sizing), `api.py` (context limit), `config.py` (`model`,
  `max_context`), `observability.py` (model name), `model_runner.py` (`spec`, `load_runner()`).
- `engine/engine/qwen_runner.py`: `Qwen2Runner`. Hand-written RMSNorm, RoPE, GQA (2 KV heads), SwiGLU, QKV bias,
  tied embeddings. Weights come straight from safetensors (bf16 upcast to fp32, same values as HF
  `dtype=float32`), no HF model object kept, transformers used for the tokenizer only. Ops mirror HF's order and
  kernels (fp32 RMSNorm, `F.linear`, RoPE from `inv_freq @ positions`, `enable_gqa` when unmasked, `repeat_kv`
  when masked).
- `engine/scripts/make_fixtures_qwen.py` (HF `generate()` allowed here only) and
  `engine/tests/fixtures/qwen2.5-coder-0.5b/`: the 8 GPT-2 cases re-derived with the Qwen tokenizer, plus
  `long_2000_out20` (RoPE/prefill at ~2K positions) and `agent_prompt_out64` (a real bench system prompt through the
  chat template; ends on EOS after 11 tokens, so the stop path is covered). Each fixture records snapshot hash,
  dtype, transformers/torch versions and the greedy settings.
- `engine/tests/test_golden_qwen.py`: 76 tests: cached single-sequence, static, continuous, batch independence,
  join/leave mid-generation, paged (block sizes 4 and 16, single, batch, join/leave), forced preemption-and-recompute
  with no leaked blocks, EOS stop, GQA spec check.

### Problems hit / things worth knowing
- The checkpoint's `generation_config.json` has `do_sample=True`, `repetition_penalty=1.05`, temperature 0.7, top_p 0.8,
  top_k 20. `generate(do_sample=False)` alone would still apply the repetition penalty, so "greedy" would not be
  greedy. The fixture script overrides all of it and the test asserts it (`do_sample False`, penalty 1.0, fp32).
- Two EOS ids (151645 `<|im_end|>`, 151643 `<|endoftext|>`); the engine now checks `tok in spec.eos_ids`.
- The `naive` (M0) backend is GPT-2-only: it is defined as HF-forward-recompute, so there is nothing to compare for
  Qwen. `Qwen2Runner.run` raises `NotImplementedError`.
- Served context for Qwen defaults to 8192 (`DEFAULT_MAX_CONTEXT`), not the model's 32768; contiguous slots reserve
  the full context each. Overridable via `EngineConfig.max_context`.
- The numerics matched HF on the first run, including the masked/batched/paged paths. No tolerance is used.

### Done-when check (actual output)
```
$ pytest tests/test_golden_qwen.py -q
76 passed in 315.63s (0:05:15)

$ pytest -m 'not perf' -q          # full engine suite, after the refactor
231 passed, 1 deselected, 2 warnings in 462.85s (0:07:42)
```
231 = 155 (GPT-2 + chat endpoint + ops, unchanged and passing) + 76 (Qwen golden). The deselected test is the noisy
perf guard.

### Phase 3 addendum: Qwen2.5-Coder-1.5B golden verification (2026-09-21)

**Why the 1.5B was not done originally.** Not a technical constraint. My download tool call was rejected by the user
(interrupted, no reason given); I had not checked disk space and had no evidence of any blocker. The user then asked
for the check and the download. Facts at that point: 852 GB free disk (11% used), ~12 GB available RAM, 1.5B fp32 is
~6 GB. No blocker, so it was downloaded (HF cache 1.8 GB -> 4.6 GB) and verified exactly like the 0.5B.

**Done:**
- `scripts/make_fixtures_qwen.py qwen2.5-coder-1.5b` -> `tests/fixtures/qwen2.5-coder-1.5b/` (HF fp32 greedy reference,
  same 10 cases, same overrides of the checkpoint's sampling defaults).
- New case `chat_ok_eos_out64` for both models: a short chat reply that ends on EOS (2 tokens). Reason: the bench agent
  prompt ends on EOS for the 0.5B (11 tokens) but runs the full 64 tokens for the 1.5B, so it cannot be the shared
  EOS-path check. Regenerating the 0.5B fixtures left every existing file byte-identical (git showed only the new
  file), a free determinism check on the HF reference.
- `tests/test_golden_qwen.py` is now parametrized over both models: identical checks, same 11 cases, same batches.
  Preemption budget is derived from each model's KV bytes/token (512 tokens: 12 MiB for 0.5B, 28 MiB for 1.5B).
  `MAX_CONTEXT` for the tests is 2560 (above the longest fixture, 2020).
- Shapes verified in `test_spec_reflects_gqa`: 1.5B = 28 layers, 12 q heads, 2 KV heads, head_dim 128
  (57,344 KV bytes/token fp32; 0.5B is 24,576).

**Result (actual output):**
```
$ pytest tests/test_golden_qwen.py -q
162 passed in 1313.19s (0:21:53)
```
162 = 81 per model, both models, all paths: single-sequence cached, static, continuous, batch independence,
join/leave mid-generation (max_batch 2/3/5), paged KV (block sizes 4 and 16: single, batch, join/leave), forced
preemption-and-recompute, EOS stop, GQA spec. No tolerance, exact token-id equality with HF. **Both models pass.**

### Still not covered
- The 1.5B agent prompt fixture does not exercise EOS (only `chat_ok_eos_out64` does).
- The full engine suite was not re-run after this change (only the golden file changed since the 231-pass run).
- Serving-model choice for Phase 5 (0.5B vs 1.5B) is still undecided; that is the user's call.

---

## Serving-model decision (user, 2026-09-21)

**The 1.5B (Qwen2.5-Coder-1.5B-Instruct) is the served model for Phase 5 and Phase 6.** The 0.5B is used only for fast
wiring/smoke tests of the HumanEval harness before the full 164-problem run on the 1.5B. Both are golden-verified.
After Phase 5, stop and present the Phase 6 budget proposal; no spend before explicit approval.

---

## Phase 4 — Seeded temperature + top-p sampling (DONE 2026-09-21)

Done-when: sampled generation works, greedy path's existing golden tests still pass unchanged, a fixed-seed
reproducibility test passes.

### Built
- `engine/engine/sampling.py`: `Sampler(temperature, top_p, seed)` (own `torch.Generator`, nucleus filtering keeps the
  smallest prefix reaching top_p and always the top token; unseeded requests get a concrete random seed so they can be
  replayed) and `pick_tokens(logits, samplers)`. With no samplers, or a row with `None`, it is exactly the old
  `torch.argmax(logits, dim=-1)`, so the greedy path is the same code as before, not a re-implementation.
- Wiring: `Request.sampler`, `step_tokens(..., samplers=None)` in both `ModelRunner` (GPT-2) and `Qwen2Runner`,
  scheduler `_prefill`/`_decode`, `Engine.generate_batch(..., samplers=None)`. Greedy and sampled requests can share
  a batch. The generator state lives on the request, so it survives preemption-and-recompute.
- API (`engine/engine/api.py`): `/generate` and `/v1/chat/completions` take `temperature`, `top_p`, `seed`.
  temperature absent or 0 = greedy; > 0 samples; invalid values (top_p outside (0,1], negative temperature) are a 400,
  not a silent fallback. `/generate` returns the `seed` used so an unseeded run can be replayed. This replaces Phase 1's
  "sampling knobs are accepted and ignored" behaviour.
- `engine/tests/test_sampling.py` (24 tests): sampler unit tests (empirical distribution within 4 sigma, temperature
  reshaping, top-p truncation, validation, replayable unseeded seed); `pick_tokens` greedy rows exact; greedy golden
  unchanged when neighbours sample (GPT-2 and both Qwen models); same seed => same output; different seed differs;
  sampled != greedy; same draws across contiguous vs paged backends and with different batch neighbours; determinism
  through forced preemption; API-level seed echo/replay, streaming == non-streaming, 400s.
- `engine/tests/test_openai_api.py`: replaced the obsolete "accepted but greedy" test with temperature 0 == greedy and
  seed reproducibility.

### Problems hit
- None in the implementation (all sampling tests passed on the first run). Two test-hygiene fixes before trusting them:
  a sloppy `... or ...` assertion and a tautology-prone leftover were tightened; and Qwen-level sampling tests were
  added because the first draft only exercised GPT-2, and the Qwen models are what will be served.
- Process slip: I first launched the full-suite run in a way that discarded its output; killed it and re-ran it with
  output captured. The number below is from the captured run.

### Limitation (also in `sampling.py`)
Same-seed equality is pinned under the tested scheduling conditions (same backend, other backends, different batch
neighbours, preemption), but is not proven in general: batch composition can change the low bits of the logits, which
could flip a draw that lands exactly on a probability boundary. Greedy exactness across batchings is separately pinned
by the golden tests. Sampling is intended for short independent generations; it is not restricted to them in code.
M1 `generate_cached` remains greedy-only (test/reference path).

### Done-when check (actual output)
```
$ pytest tests/test_sampling.py -q
24 passed in 84.05s

$ pytest -m 'not perf' -q          # full engine suite
341 passed, 1 deselected, 2 warnings in 1533.18s (0:25:33)
```
341 = 155 (GPT-2 golden/paged/preemption/ops + chat endpoint, unchanged) + 162 (Qwen 0.5B + 1.5B golden) + 24 (sampling).
`git diff c84a046 HEAD` shows `test_golden.py`, `test_paged.py`, `test_preemption.py`, `test_ops.py`, `conftest.py` and
the GPT-2 fixtures are untouched since the baseline commit. The deselected test is the noisy perf guard.

---

## Phase 5 — HumanEval harness against the self-hosted engine (DONE 2026-09-22)

Done-when: all 164 HumanEval problems run end-to-end against the self-hosted model, producing a real pass@k with a
bootstrap CI, logged with latency, tokens and cost-per-token on this hardware.

### Built (`bench/humaneval/`, new package; existing bench code untouched)
- `data.py`: 164 problems from the pinned HF dataset (`openai/openai_humaneval`, revision `7dce6050...`, file sha256 recorded in every run).
- `program.py`: one FIXED chat prompt for every model (sha256 recorded), code extraction (first fenced block defining the entry
  point), program assembly (prompt + reply, so prompt helpers stay defined). All 1,640 replies were `full_function` mode.
- `sandbox.py` + `Dockerfile`: `PythonDockerSandbox`, same `run()` interface as bench's sandboxes: no network, non-root, cap-drop
  ALL, read-only rootfs, memory/pid/cpu/file-size caps, pinned pytest 8.3.5. Needed because bench's `LocalSandbox` has no isolation
  and its `DockerSandbox` is Node-specific.
- `executor.py`: program + tests in ONE namespace (as the original human-eval), scored from the pytest JSON report through bench's
  own `PythonAdapter.parse_results`.
- `generate.py`: LiteLLM client against any OpenAI-compatible endpoint (`num_retries=0` per bench's gotchas), per-sample seeds are a pure
  function of (base seed, problem, sample), resumable. `report.py`: `pipeline.stats.pass_at_k` plus a problem-level bootstrap CI
  (B=10,000, fixed seed). `gate.py`, `__main__.py` (`generate` / `execute` / `report`).
- `tests/test_humaneval.py`: 23 tests, including probes that the sandbox blocks the network, has a read-only root FS, runs non-root, kills
  infinite loops, and reports an OOM as `resource_killed`.
- Reuse from bench: `PythonAdapter.parse_results`, the Sandbox `run()` shape, `pipeline.stats`. NOT reused: `PythonAdapter.install/run_tests`
  (they build a host venv that cannot be mounted into a network-less container).

### Gate (harness correctness before trusting any score) — `bench/results/humaneval_gate.json`
gold (canonical solutions) **164/164**, empty completion **0/164**, 35 s. The FIRST run failed 160/164: HumanEval/32, 33, 38, 50 raised
`NameError`, because their tests call helpers defined in the prompt and my first layout split program and tests into two modules. Fixed by
running both in one namespace. That was a harness bug, caught by the gate, not a model result.

### Run configuration (identical for the sampled and greedy runs except where stated)
Model Qwen2.5-Coder-1.5B-Instruct, fp32, CPU-only. Engine: paged KV (block 16), continuous batching, `max_batch=32`, `num_threads=20`,
`max_context=4096`, 2048 MiB KV budget, preemption on. Machine: Intel i7-14700K, 28 logical CPUs, 16 GB RAM, WSL2. Client: 40 concurrent requests.
- **Sampled:** n=10 per problem, temperature 0.8, top_p 0.95, max_tokens 512, seeds 0-based per (problem, sample).
- **Greedy (secondary reference):** n=1, temperature 0 (the golden-tested argmax path). Added by me as a labelled extra, because it is cheap and directly
  comparable to published single-sample numbers.

### RESULTS (real numbers; files in `bench/results/humaneval/`)
**Sampled, 164 problems x 10 samples = 1,640** (`qwen2.5-coder-1.5b/summary.json`):

| metric | value | 95% CI (bootstrap over problems) |
|---|---|---|
| pass@1 | **66.89%** | 61.34 - 72.50 |
| pass@5 | **84.27%** | 79.03 - 89.17 |
| pass@10 | **87.80%** | 82.93 - 92.68 |

Outcomes: 1,097 passed, 510 failed, 5 error, 22 collection_error, 4 timeout, 2 resource_killed (=1,640). The failed/error split is a traceback-text
heuristic and is not used for scoring. 144 of 164 problems solved at least once; 20 never solved: HumanEval/26, 32, 83, 91, 93, 101, 108, 113, 115,
119, 120, 126, 127, 129, 130, 134, 135, 145, 160, 163. Per-problem passes out of 10: 61 problems 10/10, 83 mixed, 20 zero.
15 samples hit the 512-token cap (`finish_reason=length`); all 15 failed.

**Greedy, n=1** (`qwen2.5-coder-1.5b-greedy/summary.json`): pass@1 **72.56%** (119/164), CI 65.85 - 79.27 (problem sampling only). 45 problems failed
under greedy; 25 of those were solved by at least one sampled completion. No problem that greedy solved was 0/10 under sampling.

**External sanity check.** The Qwen2.5-Coder technical report (arXiv 2409.12186, Table 16) lists 70.7 for Qwen2.5-Coder-1.5B-Instruct on HumanEval. The
fetched text does not state the decoding setting. Our greedy 72.56% is 1.9 points above it and well inside our CI (116 vs 119 problems), so the engine, prompt
and harness are not systematically off. Sampled pass@1 at T=0.8 is lower than greedy, as expected. Prior published figures for small Qwen coder models vary by
harness/prompt, so this is a consistency check, not an exact reproduction.

### Throughput, latency, tokens (sampled run; `run.json` and engine `/stats`)
- Tokens: 310,330 prompt, 271,921 completion (mean 165.8, max 512).
- Generation wall-clock **9,351.8 s (2.60 h)** for 1,640 samples: **29.08 completion tok/s**, 0.175 samples/s. Greedy run: 1,068 s, 27.75 tok/s.
- Engine: 8,591 decode steps at average batch 31.7 (peak 32, slot utilisation 98%), decode 8,246 s (0.96 s/step), prefill 1,095 s, 0 preemptions,
  0 rejections, KV utilisation 27%, attention padding waste 37%.
- Request latency (includes time queued in the engine): mean 226 s, p50 198 s, p90 388 s, p99 648 s, max 873 s. This is a THROUGHPUT configuration
  (40 requests in flight against a batch of 32); it says nothing about single-request latency (~0.1 s/token alone, measured separately).
- **Cost-per-token: no dollar figure is computed, on purpose.** It needs an hourly hardware rate and none was supplied; I do not assume one. The
  measured inputs are above. Arithmetic: 1e6 completion tokens takes 34,388 s = 9.55 h at 29.08 tok/s, so the self-hosted cost is **about $9.55 per
  million completion tokens for every $1/hour assumed** (all-in: prefill included in the wall-clock). `report --usd-per-hour R` computes the full cost block.
  The rate is an input to Phase 6.

### Problems hit and how fixed
1. Gate failed 160/164 (NameError on prompt helpers): one-namespace program+test. See Gate.
2. **Engine bug: stop token leaked into chat replies** (`...```<|im_end|>`). Chat path now hides the stop token from content (still in the token stream
   and usage); `/generate` and golden semantics unchanged. Tests on the 0.5B, streaming and not.
3. **Engine performance bug from my own Phase 4: top-p sorted the full 152K vocab** (11.8 ms/row/step, ~190 ms/step at batch 16). Exact fast path via top-256
   (0.45 ms on real logits, 26x faster). First version used fp32 and disagreed with the reference on one razor-edge case (true cumulative mass 0.5000021 vs
   top_p 0.5); decision arithmetic is now float64 and tests compare against a float64 reference. Same seed now gives different tokens than the earlier Phase 4
   sampler (nothing pinned specific values).
4. **Serving penalty: 45% slower decode** when weights are loaded on one thread and served from another (Qwen 1.5B, 32 concurrent requests, no HTTP):
   65.2 s main thread; 90.5 s load-on-main + engine-on-thread; 85.5 s with `set_num_threads` inside the thread; 88.3 s with `OMP_NUM_THREADS`/`MKL_NUM_THREADS`;
   59.1 s load and run on the same thread. The API server did the slow thing. `EngineLoop(build=...)` now builds on the loop thread; through the real server
   the same load went 90.5 s -> 59.4 s. Raw ops (linear/sdpa/matmul) are the same on either thread, so this is not a general thread penalty; the mechanism is
   NOT established.
5. `no_report` on 2 samples (HumanEval/100, an infinite loop appending to a list): reproduced, `docker inspect` says `OOMKilled=true`, exit 137. A model failure the
   sandbox contained, not infra. Relabelled `resource_killed`; those 2 rows re-executed; pass@k unchanged.
6. Process slips (no effect on results): `pkill -f` matched its own shell three times; one run's output was initially discarded and re-run; shell `\n` escaping
   corrupted scripted edits several times (each caught by a syntax check).

### Limitations and caveats (state these in the write-up)
- **Contamination.** HumanEval is public and very likely in Qwen2.5-Coder's training data; these scores probably overstate real out-of-distribution ability.
- **Real-workload decode was 0.96 s/step vs 0.41 s/step in my uniform synthetic test at the same batch size.** Padding waste (37%) is a suspect (the paged
  `gather` copies KV for all rows up to the longest row), but I did not verify it. The published throughput is what was measured, not a tuned number.
- No prefix caching: the 10 samples of one prompt each re-prefill it (prefill was 1,095 s, 12% of engine time).
- The sampled pass@1 CI resamples problems; it does not include seed-to-seed variance. pass@10 with n=10 is the fraction of problems solved at least once
  (the unbiased estimator degenerates at n=k), so its CI is only the problem-sampling uncertainty.
- The smoke artifacts `results/humaneval/smoke_0.5b/` were produced BEFORE fixes 2-3 (stop token in replies, slow sampler); they prove wiring only and are not results.
- Sample ordering/queueing: concurrency 40 vs batch 32 kept the engine full; latency numbers reflect that.

### Done-when check (actual output)
```
$ python -m humaneval.gate            -> gold 164/164, empty 0/164 (35.1 s)
$ pytest bench/tests/test_humaneval.py -> 23 passed
$ python -m humaneval generate ...    -> requested 1640 completed 1640 errors 0 (9351.8 s)
$ python -m humaneval report          -> pass@1 66.89 [61.34, 72.50]  pass@5 84.27  pass@10 87.80 [82.93, 92.68]
$ engine: pytest -m 'not perf'         -> 354 passed, 1 deselected in 1559.58s (25:59)
$ bench:  pytest                       -> 69 passed in 35.17s
```
354 = 341 (prior full pass) + 13 new (stop-token 3, sampler fast-path/speed 7, engine-thread 3). All 164 problems ran end-to-end with real pass@k and CIs.

---

## Phase 6 — Budget experiment: PROPOSAL (awaiting explicit approval; NOTHING SPENT, no API call made)

Done-when: a results file with real, reproducible pass@k and cost numbers for both the self-hosted model and the frontier anchor, on the same task set.

### Design
- **Task set and scoring identical to Phase 5:** the same 164 HumanEval problems (dataset revision and file sha256 checked equal), the same fixed prompt (sha256 checked
  equal), the same sandbox and scorer, the same n=10 samples per problem, so pass@1/5/10 with the same problem-level bootstrap CI are directly comparable.
- **Harness:** the existing `humaneval generate/execute/report` via LiteLLM (`anthropic/<model>`), with two additions: (1) cost computed from `response.usage` tokens x the
  live price table (not LiteLLM's price map, which may lag new models), (2) a spend circuit-breaker that tracks cumulative cost and aborts at a set threshold.
- **Stages, each gated:** (0) check credentials without printing them; (1) PILOT per arm: 10 problems x 2 samples = 20 calls, to measure real tokens/sample (including any
  thinking tokens) and confirm LiteLLM handles the model's parameters; (2) project the full-run cost from the pilot and compare to the approved cap; (3) full run only if the
  projection is within the cap, otherwise STOP and ask; (4) execute, report, write `bench/results/humaneval/phase6_comparison.json`.
- **Order:** the cheaper arm first, so parameter/param-drop problems surface at trivial cost.

### Measured token volume (Phase 5, Qwen tokenizer): 310,330 prompt + 271,921 completion tokens over 1,640 samples (189 / 166 per sample).

### Cost estimate (live prices from platform.claude.com/docs/en/about-claude/pricing, fetched 2026-09-22)
Assumes NO thinking tokens, and a deliberately conservative x1.35 on token counts (Claude 4.7-and-later tokenizers produce ~30% more tokens for the same text; Haiku 4.5 uses the older one).
Tokens used: 418,946 input, 367,093 output for 1,640 samples.

| Model | $/MTok in / out | Est. cost, 164 x 10, no thinking | With Batch API (50% off) |
|---|---|---|---|
| Claude Haiku 4.5 | 1 / 5 | **$2.25** | $1.13 |
| Claude Sonnet 5 | 2 / 10 | **$4.51** | $2.26 |
| Claude Opus 5 | 5 / 25 | **$11.27** | $5.64 |
| Claude Fable 5.1 | 10 / 50 | **$22.54** | $11.27 |

**Thinking is the big unknown.** Opus 5 and Sonnet 5 think by default (adaptive); thinking tokens are billed as output. I do not know how many they will use on HumanEval. Illustrative only
(the multipliers are guesses, the pilot measures the real value): Opus 5 with thinking at 3x output = ~$29.6, at 6x = ~$57. Fable 5.1 has thinking always on and cannot be turned off.

### Recommended arms and cap
- **Opus 5** (frontier anchor) and **Haiku 4.5** (cheapest hosted model: it answers "does self-hosting beat the cheapest API?", which is the thesis question). Estimated $13.5 without thinking.
- **Hard cap $35 total** (pilots, full runs, retries), circuit-breaker aborting at 80% of remaining budget; stop and ask if the pilot projects over the cap.
- Optional extras, only if you want them: Sonnet 5 (+$4.5), Fable 5.1 (+$22.5, thinking forced on, excluded from the recommendation).

### Methodology differences that must be stated in the results (not hidden)
1. **Sampling parameters cannot match.** The API reference states Opus 5, Sonnet 5 and Fable reject `temperature`/`top_p`/`top_k` (HTTP 400), so the hosted arms cannot run at T=0.8, top_p=0.95.
   They sample at the API's own default; the self-hosted arm uses T=0.8, top_p=0.95 (and its greedy pass@1 is reported separately). pass@k is defined for both, but the sampling distributions differ.
2. **Not seed-reproducible.** The hosted API has no seed. Saved completions make SCORING exactly reproducible; regenerating them will give different samples.
3. **Thinking policy for Opus 5** (needs your decision): "on its own terms" = default adaptive thinking (higher cost, `max_tokens` must be large enough to hold thinking, so the visible-reply cap differs
   from the self-hosted 512) vs thinking disabled (matches the self-hosted no-thinking regime; documented failure modes are about tool calls and `<thinking>` tag leakage, which the fenced-block
   extractor mostly tolerates). Recommendation: pilot both on 10 problems x 2 samples, run the "own terms" arm if the projection stays under the cap, and report which was used.
4. The contamination caveat applies to both sides (HumanEval is public).
5. Refusals (`stop_reason: refusal`) are recorded and counted as fails; expected to be ~0 for this prompt.

### Self-hosted cost side (needs your input)
Measured: 2.60 h wall-clock for the 1,640 samples on this machine. No dollar cost is computed without an hourly rate. Rate-free comparison from measured numbers: self-hosting the 1,640 samples
costs less than the hosted estimate only if the hardware rate is below (hosted cost / 2.60 h): **Haiku 4.5 ~$0.87/h, Sonnet 5 ~$1.73/h, Opus 5 ~$4.33/h** (no-thinking estimates; before any quality difference).
Hosted runs take minutes at these sizes; the self-hosted CPU run took 2.6 hours. The comparison the results file will show: pass@k with CIs, cost per sample and cost per expected-solved sample for each
arm, and this break-even hourly rate. If you give me an hourly rate (cloud-equivalent machine price, or electricity for the hardware you own), I will also compute the concrete self-hosted dollar figure.

### Decisions needed before any spend
1. Which arms (recommended: Opus 5 + Haiku 4.5)?
2. Budget cap (proposed: $35 hard cap, abort at 80%)?
3. Opus 5 thinking policy (own terms if projection fits the cap, else thinking off)?
4. Accept the sampling-parameter asymmetry (hosted arms at API-default sampling)?
5. An hourly rate for the self-hosted machine, or accept the break-even presentation?
6. Confirm an `ANTHROPIC_API_KEY` (or `ant auth login` profile) is available in this environment; I will check without printing it and ask if absent.

---

## Phase 6 — Budget experiment (DONE 2026-09-22): real hosted runs, real spend

Done-when: a results file with real, reproducible pass@k and cost numbers for both the self-hosted model and the frontier anchor exists, on the same task set.
**Results file: `bench/results/humaneval/phase6_comparison.json`** (every number recomputed from saved per-sample results by `python -m humaneval.compare`; nothing typed by hand).

### Approval and scope actually executed (user decisions, 2026-09-22)
- The user chose **Option C** (Haiku 4.5 at n=10 + Opus 5 at reduced n) and funded **$6.80** (not the $10 mentioned earlier), so I set the hard cap to **$6.50** with a project-wide spend ledger.
- **Total spent: $6.0641** (Haiku full run $2.3047, Opus full run $3.5959, three pilots $0.1636). Under the cap and under the funded balance.
- Changes forced by measurement, all inside the approved option: Opus 5 ran at **n=3** (the plan was n=3, first executed as n=2, then the third pass added once the measured cost showed it fit); **thinking disabled**
  for Opus (default thinking measured on only 2 calls at $0.0091/call, projecting a total over the cap); refusal fallbacks deliberately NOT enabled (a fallback would silently substitute a different model into the anchor).
- Hosted arms call the **official Anthropic SDK** (`anthropic` 1.7.0, added to bench's `pyproject.toml`/`uv.lock`), not LiteLLM: LiteLLM's model map can lag new models' parameter rules. No `temperature`/`top_p`/`seed` are sent
  (Opus 5 / Sonnet 5 reject them; hosted arms therefore use the API's default sampling).

### Results (same 164 problems, same prompt sha256, same sandbox and scorer; checks asserted in the file)
pass@k with 95% problem-level bootstrap CIs:

| arm | samples | pass@1 | pass@3 | pass@10 |
|---|---|---|---|---|
| Self-hosted Qwen2.5-Coder-1.5B, sampled (T=0.8, top_p=0.95) | 1,640 (n=10) | **66.89%** [61.3, 72.5] | **80.71%** [75.4, 85.8] | **87.80%** [82.9, 92.7] |
| Self-hosted Qwen2.5-Coder-1.5B, greedy | 164 (n=1) | **72.56%** [65.9, 79.3] | n/a | n/a |
| Claude Haiku 4.5 | 1,640 (n=10) | **92.74%** [88.9, 96.1] | **94.30%** [90.7, 97.4] | **95.73%** |
| Claude Opus 5 (thinking off) | 492 (n=3) | **96.95%** [93.9, 99.4] | **96.95%** [93.9, 99.4] | not measured (n=3) |

The self-hosted model is **25.9 points behind Haiku 4.5 and 30.1 points behind Opus 5 at pass@1**, and stays behind at pass@3 (80.7 vs 94.3 vs 97.0). This is the honest headline: on function-level Python the self-hosted 1.5B model is
clearly worse than even the cheapest hosted model. Opus 5's failures are all-or-nothing per problem (pass@1 = pass@2 = pass@3), so at n=3 its higher-k numbers carry no extra information.

### Cost and speed (measured)
| arm | cost | per sample | per expected-solved sample | wall-clock |
|---|---|---|---|---|
| Haiku 4.5 (1,640 samples) | **$2.3047** | $0.001405 | $0.001515 | 354 s (0.22 s/sample, 8 workers) |
| Opus 5 (492 samples) | **$3.5959** | $0.007309 | $0.007539 | ~220 s |
| Self-hosted (1,640 samples) | **no dollar figure: no hourly rate supplied** | n/a | n/a | 9,352 s (2.6 h) |

Hosted generation was ~26x (Haiku) and ~13x (Opus) faster than the self-hosted CPU run for the same sample count, at the concurrency used.
**Break-even hourly rate for the self-hosted machine** (below this rate, self-hosting the same 1,640 samples costs less than the hosted API), from the measured 2.598 h:
- vs Haiku 4.5: **$0.89/h per sample**, **$0.64/h per solved sample** (hosted cost for 1,640 samples $2.30).
- vs Opus 5: **$4.61/h per sample**, **$3.18/h per solved sample** (hosted equivalent for 1,640 samples $11.99, extrapolated from the measured $0.007309/sample).
So self-hosting on this CPU is cheaper than Haiku only if the machine costs under ~$0.89/h all-in (e.g. hardware you already own and marginal electricity), and cheaper than Opus 5 if under ~$4.61/h. It is never better in quality.
Where self-hosting genuinely helps: no per-token bill, no data leaving the machine, and the engine's batching (decode 0.10 s/step alone vs 0.25 s/step at batch 16 for the 1.5B, measured in Phase 5) is what makes 2.6 h possible at all.
It does not win on cost-per-solved-problem against Haiku unless the hardware is nearly free.

### Truncation caveat (the 512-token cap, applied to every arm to keep the protocol identical)
| arm | samples cut at 512 tokens | passed among them | most the cap could be costing (upper bound on sample pass rate) |
|---|---|---|---|
| Self-hosted | 15 / 1,640 | 0 | 67.8% (vs 66.9% measured) |
| Haiku 4.5 | 74 / 1,640 | 3 | 97.1% (vs 92.7%) |
| Opus 5 | 12 / 492 | 0 | 99.4% (vs 96.95%) |
Hosted models are more verbose, so the cap penalizes them more. Four of Opus 5's five failing problems (68, 81, 109, 124) are truncations; only HumanEval/103 is a genuine failure. **The ranking cannot change**
(even the upper bounds leave the self-hosted arm 30 points behind), but the hosted scores are understated. A rerun of just the truncated hosted samples at a larger cap would cost roughly $0.6, which would exceed the $6.50 cap, so it was not done.

### Methodology differences that must accompany any use of these numbers
1. Sampling differs by design (self-hosted T=0.8/top_p=0.95 seeded; hosted at the API's default). 2. Hosted runs are not seed-reproducible: SCORING is reproducible from the saved completions, regenerating gives different samples.
3. Sample counts differ (10, 10, 3): compare at k <= 3; n=3 pass@k is a plain fraction, n=10 is the unbiased estimator. 4. Opus 5 thinking off; default thinking was measured on 2 calls only ($0.0091/call, one call used an adaptive thinking block).
5. HumanEval is public and very likely in every model's training data; absolute scores overstate real-world ability for all arms. 6. CIs resample problems, not seeds.

### Pilots (real calls, evidence for decisions)
Haiku 20 calls $0.0211 (20/20 pass, 152 input / 180 output tokens per call); Opus 5 thinking off 20 calls $0.1454 (20/20 pass, $0.00622/call, 192 in / 210 out); Opus 5 default thinking, HumanEval/32 and /50, 2 calls $0.0182 (one used a thinking block; 517 vs 95 output tokens).
The pilots' first 10 problems are easier than average, so pilot per-call costs understated the full runs by ~33% (Haiku $0.00105 -> $0.001405). That is why Opus was first run at n=2 and the third pass added only after the measured cost showed it fit.

### Problems hit
- Pilot cost projections were optimistic (see above); handled by staging the Opus run instead of trusting the projection.
- A test of my own breaker had a float-boundary mistake (0.005 + 0.045 landed just under 0.05); fixed the test, the breaker was right.
- Process: none that affected results. The API key was read from `bench/.env` (gitignored, untracked); a scan of every results file, source file and test for the key prefix found nothing.

### Done-when check (actual output)
```
$ python -m humaneval generate --provider anthropic (Haiku)   -> completed 1640, errors 0, 354.0 s, ledger $2.4683
$ python -m humaneval generate --provider anthropic (Opus n=2) -> completed 328, errors 0, 145.8 s, ledger $4.8640
$ python -m humaneval generate --provider anthropic (Opus n=3) -> completed 164 (+328 done), errors 0, 74.0 s, ledger $6.0641 of $6.50
$ python -m humaneval.compare  -> pass@1 self 66.89 | greedy 72.56 | haiku 92.74 | opus 96.95 ; break-even $/h: haiku 0.89, opus 4.61 ; spend $6.0641
$ bench: pytest                 -> 79 passed in 37.75 s (69 prior + 10 new: cost formula, ledger, breaker, request shape, no key stored)
```
The engine suite was not re-run: no engine code changed in Phase 6 (last full pass: 354 passed).

---

## Phase 7 — Write-up (DONE 2026-09-22)

Done-when: a top-level README with the real story and numbers verbatim from the results file, including an honest statement of where the self-hosted model fell short.

### Built
- `README.md` (repo root): the headline (self-hosted 66.89% pass@1 vs Haiku 4.5 92.74% vs Opus 5 96.95%), results tables, cost/speed, break-even hourly rates, a "Where the self-hosted model
  fell short" section, "Where self-hosting still helps" with the measured batching numbers, the history of the original plan that did not survive (ts-bench 0/647 evidence, the two reframes),
  what was built, "Things found by measuring", caveats, reproduce commands, repo map.
- To make every measured number in the README traceable to a committed file, two measurements that previously existed only as scratch output were turned into committed scripts with saved JSON:
  `engine/scripts/bench_decode_sweep.py` -> `engine/results/decode_sweep_qwen2.5-coder-1.5b.json` and `engine/scripts/bench_thread_penalty.py` -> `engine/results/thread_penalty_qwen2.5-coder-1.5b.json`.

### Verification (actual output)
A script recomputed 40 numbers from `bench/results/humaneval/phase6_comparison.json`, the decode-sweep JSON and the thread-penalty JSON and checked each appears in the README:
```
checked 40 numbers; all present in README
```
(It first reported 3 misses, all comma formatting: 9351.8 vs 9,351.8; the values were right. The checker was adjusted, not the README.)
Two factual errors in my first draft were caught on re-reading and fixed before commit: the OOM-killed HumanEval/100 samples were self-hosted (not Haiku's), and a sentence about per-solved-sample break-even was worded backwards.

### Numbers in the README that come from re-runs, not the original run
The decode sweep was re-run for the committed file and differs from my earlier ad-hoc run (batch 8: 30.9 vs ~41.8 tok/s), which the README says openly. The thread experiment was also re-run in fresh processes:
main 60.9 s, loaded-on-main-run-on-thread 88.1 s (45% slower), loaded-and-run-on-same-thread 60.0 s (the original ad-hoc numbers were 65.2 / 90.5 / 59.1).

### Project status
All seven phases done. Total hosted spend $6.064139 (cap $6.50). Engine suite 354 passed (last full run, before Phase 6; no engine code changed since except two new measurement scripts). Bench suite 79 passed.
Known open items (not done, listed so they are not lost): re-run truncated hosted samples at a larger token cap (~$0.6, over the cap); Opus 5 with default thinking at scale; n=10 for Opus 5; an hourly rate for the self-hosted machine to turn the break-even into a dollar figure;
prefix caching and a padding-free attention path in the engine (the 0.96 s/step real-workload decode vs 0.38 s uniform benchmark is unexplained); Qwen2.5-Coder-3B; the SWE-bench-style tasks with a stronger model.

---

# Website track (W1-W5): the results site (`web/`)

A portfolio site built from the real results: React + TypeScript + Vite, standalone (own `package.json`, no imports from `engine/` or `bench/`; it reads their result files through a generated `data.json`).
Rules inherited from the project: every number traces to a file; the negative result has the same visual weight as anything else; nothing is tuned to look better than it is. Work one phase at a time; commit and push after each.

| Phase | What | Status |
|---|---|---|
| W1 | Scaffold, data pipeline (`web/scripts/build-data.mjs` -> `web/src/data.json`), data tests | DONE (2026-09-22) |
| W2 | Real scheduler trace recorded from the actual engine + the animated engine diagram | DONE (2026-09-22) |
| W3 | Opening, pivot story, eval-architecture trace (a real request's path) | DONE (2026-09-22) |
| W4 | Results: table, CI chart, per-problem strips, break-even chart, caveats at equal weight | DONE (2026-09-22) |
| W5 | Close, polish, "AI tell" lint, number-provenance lint, screenshots reviewed, mobile check | DONE (see W5 below) |

## Decisions made up front (recorded so they are not silent)
- **Location:** `web/` at the repo root (standalone project inside the monorepo, so the data script can read the result files). The site name/path is easy to move.
- **Visual language (held throughout):** editorial paper `#f2eee4`, ink `#161410`, hairline rules, ONE loud accent. Type: Newsreader (display + text, self-hosted via fontsource) with IBM Plex Mono for every number, label and code. No gradients, no
  glass, no shadows, no rounded cards, no emoji, no icon grid, no particle backgrounds. Layout is asymmetric editorial: a sticky section index in the left rail, a narrow text column, and figures that break out wide.
- **Series colors (validated, not eyeballed):** `#C4411D` vermilion = self-hosted (the arm that loses gets the loudest color, on purpose), `#2B58A6` cobalt = Claude Haiku 4.5, `#A56E00` ochre = Claude Opus 5. Ran
  `validate_palette.js --mode light --surface "#f2eee4"`: ALL CHECKS PASS (lightness band, chroma floor >= 0.10, worst adjacent CVD dE 21.3 protan / 22.0 tritan, normal-vision floor worst 28.0, contrast >= 3:1). Near-black ink was rejected as a
  series color: it fails the chroma floor. The greedy self-hosted arm reuses the vermilion hue with a hollow marker (same entity, different decoding), never a fourth hue.
- **Diagrams show real state, not decoration.** The engine diagram animates a trace recorded from the actual scheduler (`engine/scripts/record_scheduler_trace.py`: real requests, real paged block tables, a real preemption), scrubbed by scroll.
  The eval diagram follows one real sample's real timings from the results files.
- **Data integrity:** `build-data.mjs` reads the result JSON/JSONL and parses the prose-only facts (0/647, 6 of 20, test counts) out of the docs with regexes that FAIL THE BUILD if the text no longer matches. A lint forbids digits in JSX text, so no number can be typed into a component.
- **Contact details:** the site can name the repo (public GitHub URL from the git remote). Whether to publish an email or any other contact is the user's call; it is a config field that stays empty until supplied.
- **Out of scope unless asked:** deployment/hosting, analytics, a CMS.

## Correction found while planning (fixed in `README.md`)
The README said codestral scored "about 0/240". The source doc (`bench/docs/step7-real-model-run.md`) says codestral completed 240/240 with 0 resolved (240 + 240 + 167 = 647), and states the claim is narrow: one plain bash-loop scaffold, a 24-instance set,
no evidence about other scaffolds. README corrected; the site uses the doc's wording.

## W1 — Scaffold and data pipeline (DONE 2026-09-22)

### Built
- `web/` standalone project: React 19, Motion 13, Vite 8, TypeScript (strict), vitest, fontsource (Newsreader, IBM Plex Mono, self-hosted), Playwright for screenshots. `package.json`, `tsconfig.json`, `vite.config.ts`.
- `bench/humaneval/export_examples.py` -> `bench/results/humaneval/site_examples.json`: one real problem's real samples for the eval-architecture diagram. Selection rule is fixed and stated in the file (not a hand pick): the lowest-index problem in the
  self-hosted sampled run with 4 to 7 passes out of 10, then its lowest-index passing and failing samples. It picked **HumanEval/6** (`parse_nested_parens`, 7/10 pass; passing sample 0, failing sample 1 with a real assertion diff).
- `web/scripts/build-data.mjs` -> `web/src/data.json` (72 KB, deterministic, no timestamps). Reads 14 result files (sha256 of each is recorded in `meta.sources`). Numbers that exist only in prose are parsed with regexes that throw if the text
  changes: the ts-bench evidence (0 of 647; per-model 240/240, 240/240, 167/240; Haiku 6 of 20; the doc's own "narrow claim" sentence), the test counts (162 / 354 / 79), and the phase table.
- Independent cross-checks that fail the build: per-problem counts recomputed into pass@1/pass@3/pass@10 must equal the reported values (1e-9); break-even must satisfy hours x rate = hosted cost; the spend ledger must sum to the reported total;
  per-model attempts must sum to 647.
- `web/tests/data.test.ts`, 14 tests: regeneration is byte-identical; each arm's pass@1 and CI equals the task board's Phase 6 table; pass@3 equals the board's; cost lines equal the board's; break-even equals the board's; every README pass@k cell equals
  data.json's; all arms have 164 problems; the self-hosted CI does not overlap Haiku's.

### Problems hit
- `npm install` failed with `npm error No workspaces found!`. Root cause: the user-level `~/.npmrc` contains `workspaces=true`, which forces workspace mode on every npm command. This is very probably the real cause of the npm
  "No workspaces found!" errors bench's `step7-real-model-run.md` (Bug #6 and Bug #8) investigated at length; I did not test that. I did NOT edit `~/.npmrc`; I set `npm_config_workspaces=false` per command. Worth the user's attention: removing that line would fix `npm` everywhere.
- My first test matched the wrong board row (a cost-estimate row from the Phase 6 proposal with the same arm name); fixed by requiring an actual percentage cell.

### Check (actual output)
```
$ node scripts/build-data.mjs   -> wrote src/data.json (14 source files hashed; scheduler trace not present yet)
$ vitest run                    -> Test Files 1 passed | Tests 14 passed
```

## W2 — Real scheduler trace and the animated engine diagram (DONE 2026-09-22)

### Built
- `engine/scripts/record_scheduler_trace.py` -> `engine/results/scheduler_trace.json`. Wraps the LIVE `Scheduler`/`BlockManager` (allocate, append_slot, free, `_prefill`, `_preempt`, `_emit`) to observe only; no engine code changed. Workload: 8 real GPT-2 requests,
  block size 4, a 17-block pool (5 MiB), max batch 4. Result: **60 steps, 4 evictions, 156 tokens, pool peaks at 17/17**. The script asserts before writing: at least one preemption happened, no blocks leaked, and every request's output equals a
  single-request greedy run. The FIRST attempt (7 MiB, 24 blocks) never exhausted the pool and the assertion refused to write a trace, so the workload was tightened (5 MiB, longer outputs). The assertion did its job.
- Snapshot semantics were verified from the data, not assumed: in all 144 running snapshots `len(block_table) == ceil((seq_len - 1) / block_size)`, so a request's cache holds `seq_len - 1` tokens (the newest sampled token is not cached until the next decode). The diagram fills token cells by that rule.
- `web/src/components/EngineTrace.tsx`: an SVG driven only by the recorded snapshots and events. Letter chips travel between the waiting queue, the four batch rows and back; each KV block shows its owner and logical index (`C2` = request C's second block) and its
  token cells fill as tokens are cached; free blocks are hatched; an eviction marks the discarded blocks with fading X's; the step text is generated from that step's own events. Scroll drives it (sticky figure inside a tall section); Prev, Next, Play, "Next eviction" and a range slider are alternatives;
  the last input wins. Reduced-motion users get instant transitions and a static figure.
- `web/src/sections/Engine.tsx`, `Opening.tsx`, `components/Chrome.tsx` (sticky section rail, section header, restrained reveal), `lib/{data,fmt,trace}.ts`, `styles.css` (paper/ink tokens, type scale, editorial layout).
- `web/scripts/shots.mjs`: Playwright screenshots of desktop and mobile, plus the engine figure at scroll positions and at an eviction moment.
- The section is honest about scope: the diagram uses a deliberately tiny pool to force evictions; the real HumanEval run used a 2,048 MiB pool of 16-token blocks and had 0 evictions at 27.07% mean utilisation, and the page says so. It also reports the real-run decode step (0.96 s) against the uniform
  benchmark (0.38 s) as an unexplained gap, and the 45% thread-arrangement finding.

### Problems hit
- Chromium would not start in WSL: `libnspr4`, `libnss3`, `libnssutil3`, `libasound2` were missing and there is no passwordless sudo. Worked around without root: `apt-get download` the three packages, `dpkg -x` them into `~/.local/chromium-libs/root`, and set `LD_LIBRARY_PATH` when running `shots.mjs`. Nothing installed system-wide.
- Two text collisions found only by LOOKING at screenshots (the "EVICTED n times" tag overlapped the row title, then the block-table line). Fixed by giving the tag its own line. The first score card's label wrapped and misaligned the three big numbers; fixed with a min-height.
- The dataviz palette validator forbids near-black ink as a series color (fails the chroma floor); ink is used for text and structure only.

### Check (actual output)
```
$ tsc -b            -> clean
$ vitest run        -> 14 passed
$ vite build        -> built (466 KB JS gzip 133 KB; fonts self-hosted)
$ shots.mjs         -> no page errors (desktop 1440x900 and mobile 390x844)
```
Screenshots reviewed: the opening, the engine figure at 7 scroll positions, a mid-animation eviction (ghost X's on the 3 evicted blocks, chip returning to the queue tagged RECOMPUTE, red step text), and the final state.

## W3 — The pivot story and the eval-architecture trace (DONE 2026-09-22)

### Built
- `web/src/sections/Pivot.tsx` + `components/AttemptsGrid.tsx`: the pivot told as problem, reasoning, decision. The centrepiece is every one of ts-bench's earlier attempts as one square: 647 hollow squares for the three local models (the partial run's unrun cells shown as
  shaded, not attempted) and Haiku 4.5's 20 with 6 filled. The doc's own "narrow claim" sentence and its caveat are quoted, from `data.json`, not paraphrased. The model-size sentence uses only sizes the data states (two of the three local models are 14B-tagged, 9.3x the served 1.5B; the third, codestral, is a
  12 GB download against 9 GB), and says so.
- `web/src/components/RequestTrace.tsx` + `sections/Harness.tsx`: one REAL sample's path through the harness in seven stages (prompt, engine, reply, assemble, sandbox, score, aggregate), scroll-driven, with a pass/fail toggle. It uses HumanEval/6 from `site_examples.json`: the actual prompt sent, the request
  parameters and per-sample seed, the model's raw reply, the dataset's hidden test, the docker flags, the real assertion failure of the failing sample (`assert [1, 2, 2, 2, 2, 2, ...] == [2, 3, 1, 3]`), the real outcome counts across all 1,640 samples, and the aggregation with the real pass@k, bootstrap draws, and the harness gate (164/164 gold, 0/164 empty, and its first run at 160/164).
- `web/scripts/build-data.mjs` extended, all regex-verified (build fails if the text changes): model sizes and download sizes, the doc's caveat sentence, the failed first gate run, the sandbox flags parsed from `bench/humaneval/sandbox.py`, the bootstrap draw count and seed parsed from `report.py`, the client's in-flight request count, this problem's own pass@1 and pass@3.
- Rail: the active section is now computed from scroll position.

### Problems hit (all found by reading the page or the source, not by tests)
- The rail highlighted "03 The harness" while the reader was in section 02: the IntersectionObserver kept stale state when no heading was in its narrow band. Replaced with a scroll-position computation.
- The request-trace panel changed height between stages and left half the viewport blank. Fixed with a constant panel height and real supporting content (what the sandbox tests prove; the outcome table).
- Three factual overstatements in my own draft prose, caught on re-reading against the data: "the smallest of those local models" (codestral's size was not in the data), "the lowest-numbered problem the model solves in 7 of 10 tries" (the rule is 4 to 7 passes; it LANDED on 7), and "the same code drives all three models" (generation differs by provider: LiteLLM for the engine, the Anthropic SDK for hosted; prompt, sandbox and scorer are shared). All corrected.
- One data.json commit broke a test (the doc hash changed with every board edit); fixed earlier by hashing the parsed facts instead. Chained shell commands now use `&&`.
- Screenshot tooling: `html { scroll-behavior: smooth }` made captures land on the wrong stage; the capture script disables it for itself only. An earlier inline patch also corrupted the script through shell interpolation; repaired.

### Check (actual output)
```
$ tsc -b       -> clean
$ vitest run   -> 14 passed (data.json fresh; matches the board and README tables)
$ vite build   -> built
$ shots.mjs    -> no page errors (desktop + mobile)
```
Screenshots reviewed: pivot (attempts grid, quote, reasoning), all seven trace stages for the passing sample, the Score stage for both samples.

## W4 — The results section (DONE 2026-09-22)

### Built (`web/src/sections/Results.tsx`, `components/ResultsCharts.tsx`, `lib/palette.ts`)
- The results table: all four arms with pass@1, pass@3, pass@10 and their 95% intervals, samples, cost, wall-clock. The self-hosted rows say "no $ figure, no hourly rate supplied" rather than a made-up number.
- An interval chart (pass@1 and pass@3, full 0 to 100% axes, so the gap is not exaggerated by a truncated axis): thin marks, direct value labels, a legend, hover states, and the table beside it as the accessible view.
- Per-problem strips: one cell per problem for each arm, shaded by how many of that problem's samples passed. This is where the negative result is most visible: the self-hosted rows are patchy, Haiku 4.5 and Opus 5 nearly solid. Counts (never / sometimes / every time) are derived from `data.strips`.
- A break-even chart with a slider: cost of self-hosting = measured wall-clock hours x an hourly rate the reader chooses, against Haiku 4.5's and Opus 5's measured costs; toggle between "every sample" and "solved samples"; circles mark the two break-evens. No hourly rate is assumed anywhere.
- The caveats block, at the same heading size, rule weight and body type as the results (not small print): the seven caveats from the results file written as readable prose with every number interpolated from data, beside a truncation table (cut-off counts, pass rate, and the most the cap could be costing each arm: up to 0.9, 4.3 and 2.4 points) and a short "where self-hosting still helps".
  The page THROWS at render if the results file gains or loses a caveat, so none can be silently dropped or softened.
- Colors were validated, not eyeballed: the series palette (all checks pass) and, after a first attempt failed, a 3-step ramp per hue for the strips. The five-equal-step opacity ramp FAILED (light end 1.4 to 1.6:1, under the 2:1 floor; dark steps too close), so the strips use three steps plus a hollow class for "never".
- `build-data.mjs` gained `overlap` facts computed from the per-problem counts; 2 new tests (16 total): overlap facts agree with the counts; the caveat count is pinned.

### A wrong claim caught before shipping
I had written under the strips: "the problems the self-hosted model misses are, for the most part, not the ones the hosted models miss." I had not checked it. Checked against the data: of the 20 problems the self-hosted model never solved, Opus 5 solved all 20 every time, Haiku 4.5 solved 10 every time (and never solved 3). The
5 problems Opus 5 missed were each solved at least once by the self-hosted model, and 4 of those 5 have an Opus 5 reply cut off by the shared 512-token cap. The unverified sentence was deleted and replaced with these computed facts, including the one point in the small model's favor and why it is mostly an artifact of the cap.

### Other problems found by reading the page
- The PASS@3 panel was clipped (value labels cut off at the right edge); the plot area now leaves room.
- The break-even chart's y ticks rounded 3.75 to $4 and 11.25 to $11 (misleading); ticks now use round steps ($0, $5, $10, ...). The break-even label was crossed by a line; moved.
- The caveats first rendered the raw strings from the results file, which contain code identifiers (`truncation_upper_bound_sample_pass_rate`); rewritten as prose. One sentence said "Opus 5 and Sonnet 5 reject temperature/top_p": Sonnet 5 was never run here and the claim comes from Anthropic's API reference, so it now says that.
- Section headings (h3) were barely larger than body text; enlarged.

### Check (actual output)
```
$ tsc -b       -> clean
$ vitest run   -> 16 passed
$ vite build   -> built
$ shots.mjs    -> no page errors
```
Screenshots reviewed: the table, interval chart, all four strips, break-even chart at its default, caveats and truncation table.

## W5: close, polish, lints, mobile — DONE

**Built:** `web/src/sections/Close.tsx` (stack deflist, repo link, corrections, provenance; contact shown only if `web/site.config.json` supplies it — currently empty on purpose); `web/scripts/lint-tells.mjs` (gradient, blur/glass, shadow, sparkle, Inter, endless/bounce animation, canvas/WebGL, particles, radius >2px, emoji, purple hues, banned deps); `web/scripts/lint-numbers.mjs` (TypeScript-AST scan: no digit typed into JSX text, static child strings or aria-label/title/alt); `web/tests/lint.test.ts` (18 tests proving each lint catches what it claims); `web/scripts/shots.mjs` (desktop 1440 and mobile 390 captures, horizontal-overflow detector).

**Problems hit and fixes:**
- TypeScript 7 has no JS API, so the number lint uses a `ts5` alias (typescript@^5) used only by the lint.
- lint-numbers found 15 real violations (typed "95%", "1.5-billion", an ad-hoc batch sentence). Fixed: `CONF` is now read from report.py alpha, the others come from data.json.
- Mobile: 61px horizontal overflow from grid min-content tracks; fixed with `minmax(0,1fr)`. Diagrams unreadable when scaled to 390px; now `min-width:700px` inside a horizontally scrolling frame.
- Factual overstatements caught in review and corrected (see W4 notes for the wording list).

**Done-when check, actual output:**
- `build-data.mjs`: data.json regenerated from 22 source files.
- vitest: 2 files, 34 tests passed.
- lint-tells: clean (18 files). lint-numbers: clean (13 components).
- `tsc -b`: clean. `vite build`: built.
- `shots.mjs`: "no page errors" (no console errors, no page overflow at 390px).
- Manual AI-tell review (not lintable): no centered hero with two pill buttons; no repeated icon-heading-paragraph grid (the one 3-score strip in the opening is used once, no icons); negative result sits in the opening and at equal weight in Results.

**Open items for the owner:** contact details (add to `web/site.config.json`); confirm repo is public / accepts issues; keep or remove the "Built with Claude Code" line; deployment not done (out of scope); `workspaces=true` in ~/.npmrc breaks per-project npm (we set `npm_config_workspaces=false`).

## W6: Clerk-influenced visual pass (rounded cards, alternating light/dark sections) — DONE

User asked the site to feel like clerk.com: rounded bordered cards, tighter type/spacing, and a scroll-driven
alternation between white and black sections with a smooth transition. Confirmed with the user first, since it
directly reversed a W1 non-negotiable ("no gradients, no shadows, no rounded cards, no blur"); the user explicitly
lifted the ban on rounded corners, shadows and a blue/purple gradient (blur/glassmorphism, emoji/sparkle icons,
non-brand typefaces and endless animation stay banned).

**Built:**
- `web/src/components/Chrome.tsx` — `SECTIONS` now carries a `theme: "light" | "dark"` per section (alternating:
  result/pivot/results light, engine/harness/close dark); new `ThemeWatcher` component reuses the existing scroll
  `useActive` hook and sets `document.body.dataset.theme` as the active section changes.
- `web/src/styles.css` — `body[data-theme="dark"]` overrides the same custom properties every component already
  reads (`--paper`, `--ink`, `--rule`, `--self`, `--haiku`, `--opus`, plus new `--card`/`--card-border`/`--shadow`/
  `--radius`), so the whole page recolors from one attribute flip, and a scoped `transition` on `body` and its
  descendants crossfades the swap instead of cutting. `--self`/`--haiku`/`--opus` are stepped brighter in the dark
  palette to hold contrast (not re-validated with the dataviz palette script — noted as an open item below). Cards
  (`.score`, `.figure .well`, `.stage-data`, `.table-wrap`, `.deflist > div`, `.btn`, the active rail item) now use
  `var(--radius)`/`var(--card)`/`var(--card-border)`/`var(--shadow)`. Section top padding increased for more
  product-page rhythm.
- `web/scripts/lint-tells.mjs` — removed the gradient, shadow, border-radius and purple-hue rules (now the sites


## W6: Clerk-influenced visual pass (rounded cards, alternating light/dark sections) — DONE

User asked the site to feel like clerk.com: rounded bordered cards, tighter type/spacing, and a scroll-driven
alternation between white and black sections with a smooth transition. Confirmed with the user first, since it
directly reversed a W1 non-negotiable ("no gradients, no shadows, no rounded cards, no blur"); the user explicitly
lifted the ban on rounded corners, shadows and a blue/purple gradient (blur/glassmorphism, emoji/sparkle icons,
non-brand typefaces and endless animation stay banned).

**Built:**
- `web/src/components/Chrome.tsx` — `SECTIONS` now carries a `theme: "light" | "dark"` per section (alternating:
  result/pivot/results light, engine/harness/close dark); new `ThemeWatcher` component reuses the existing scroll
  `useActive` hook and sets `document.body.dataset.theme` as the active section changes.
- `web/src/styles.css` — `body[data-theme="dark"]` overrides the same custom properties every component already
  reads (`--paper`, `--ink`, `--rule`, `--self`, `--haiku`, `--opus`, plus new `--card`/`--card-border`/`--shadow`/
  `--radius`), so the whole page recolors from one attribute flip, and a scoped `transition` on `body` and its
  descendants crossfades the swap instead of cutting. `--self`/`--haiku`/`--opus` are stepped brighter in the dark
  palette to hold contrast (not re-validated with the dataviz palette script — noted as an open item below). Cards
  (`.score`, `.figure .well`, `.stage-data`, `.table-wrap`, `.deflist > div`, `.btn`, the active rail item) now use
  `var(--radius)`/`var(--card)`/`var(--card-border)`/`var(--shadow)`. Section top padding increased for more
  product-page rhythm.
- `web/scripts/lint-tells.mjs` — removed the gradient, shadow, border-radius and purple-hue rules (now the site's
  intentional design); kept glass/blur, sparkle/emoji, Inter, endless-animation and canvas/particle bans.
  `web/tests/lint.test.ts` updated to match: replaced the five stale "flags X" cases with one case asserting
  rounded/shadow/gradient/purple are now accepted.

**Problems hit:** first rebuild failed lint-tells because the new CSS file-header comment used the word "sparkle"
inside a sentence explaining what is still banned — the lint cannot tell negation from advocacy, correctly flagged
its own literal match; reworded the comment instead of weakening the rule.

**Done-when check, actual output:**
- `build-data.mjs`: data.json regenerated (22 source files hashed).
- vitest: 2 files, 29 tests passed.
- lint-tells: clean (18 files). lint-numbers: clean (13 components).
- `tsc -b`: clean. `vite build`: built.
- `shots.mjs`: "no page errors" on both desktop (1440) and mobile (390) after the restyle.
- Visual check (screenshots, not lintable): opening/pivot/results render light with rounded score/table/chart
  cards; engine/harness/close render dark with the same card language; the engine and harness diagrams read
  clearly against the dark surface; rail's active item now shows as a bordered highlight instead of a hairline;
  mobile captures show the theme flip and card stacking with no horizontal overflow.

**Open items:** none — resolved same day. The dark-mode step of `--self`/`--haiku`/`--opus` was re-run through the
dataviz skill's `validate_palette.js` as its own categorical palette (`--mode dark`, surface `#1a1a19`). First pass
(`#ff6a3d`/`#6c9fe8`/`#e0ac3f`) failed the OKLCH lightness band (L 0.48-0.67 expected; measured 0.70-0.77 - too light
against a near-black surface). Retuned to `#e8552e`/`#5a8fe0`/`#b8871f`: all five checks pass (lightness band, chroma
floor, CVD separation worst-adjacent dE 25.1, normal-vision floor dE 26.1, contrast >= 3:1 on `#1a1a19`). Light-mode
trio was also validated as its own palette (unchanged from W1-W4: all-pass, worst adjacent CVD dE 21.3). Verified
again after the retune: vitest 29/29, both lints clean, `tsc -b` clean, `vite build` built, `shots.mjs` no page
errors; engine/harness dark-section screenshots re-checked by eye for legibility.


## W7: black throughout (dropped the light/dark alternation) — DONE

User asked to keep the page black throughout rather than alternating light/dark per section.

**Built:** `web/src/styles.css` — the validated dark palette (from W6's palette-validator fix) moved into `:root`
as the only palette; the light `:root` block and the `body[data-theme="dark"]` override block are gone, along
with the now-unneeded `body`/`body :where(*)` crossfade-transition rules. `web/src/components/Chrome.tsx` — removed
the `ThemeWatcher` component and the per-section `theme` field on `SECTIONS` (nothing toggles `data-theme` anymore).
`web/src/App.tsx` — dropped the `<ThemeWatcher />` mount.

**Done-when check, actual output:**
- `build-data.mjs`: data.json regenerated (22 source files hashed).
- vitest: 2 files, 29 tests passed. lint-tells: clean (18 files). lint-numbers: clean (13 components).
- `tsc -b`: clean. `vite build`: built. `shots.mjs`: no page errors.
- Visual check: opening, pivot, harness, results and close all render on the same black surface with the same
  bordered/rounded card language; confirmed on both desktop (1440) and mobile (390) captures.
