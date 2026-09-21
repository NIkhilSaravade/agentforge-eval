// Generates src/data.json, the ONLY source the site renders from, out of the repo's real result files.
//   node scripts/build-data.mjs [--out path]
// Rules: nothing here is typed by hand. Every number is read from a result file, or parsed out of a doc with a regex that
// FAILS THE BUILD if the text no longer matches, or derived by arithmetic on those. The output is deterministic (no timestamps);
// `sources` records the sha256 of every input so a stale data.json is detectable.
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../..");
const outArg = process.argv.indexOf("--out");
const OUT = outArg > 0 ? resolve(process.argv[outArg + 1]) : resolve(HERE, "../src/data.json");

const R = "bench/results/humaneval";
const sources = {};
const read = (rel, optional = false, hash = true) => {
  const p = resolve(ROOT, rel);
  if (!existsSync(p)) {
    if (optional) return null;
    throw new Error(`missing source file: ${rel}`);
  }
  const buf = readFileSync(p);
  if (hash) sources[rel] = createHash("sha256").update(buf).digest("hex");
  return buf.toString("utf8");
};
const json = (rel, optional = false) => {
  const t = read(rel, optional);
  return t === null ? null : JSON.parse(t);
};
const jsonl = (rel) => read(rel).split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
const must = (cond, msg) => {
  if (!cond) throw new Error(`data integrity: ${msg}`);
};
const match1 = (text, re, what) => {
  const m = text.match(re);
  must(m, `could not parse "${what}" out of its doc (regex ${re}); the doc text changed, so this fact is no longer verified`);
  return m;
};

// ---------------------------------------------------------------- the comparison file (single source for the results)
const cmp = json(`${R}/phase6_comparison.json`);
must(Object.values(cmp.checks).every(Boolean), "phase6_comparison.json reports failed identical-task-set checks");

const ARMS = [
  { id: "self", key: "self_hosted_sampled", dir: "qwen2.5-coder-1.5b", label: "Self-hosted Qwen2.5-Coder-1.5B", short: "Self-hosted, sampled", kind: "self", decoding: "T=0.8, top_p=0.95, seeded", marker: "solid" },
  { id: "greedy", key: "self_hosted_greedy", dir: "qwen2.5-coder-1.5b-greedy", label: "Self-hosted Qwen2.5-Coder-1.5B", short: "Self-hosted, greedy", kind: "self", decoding: "greedy (argmax)", marker: "hollow" },
  { id: "haiku", key: "haiku_4_5", dir: "claude-haiku-4-5", label: "Claude Haiku 4.5", short: "Claude Haiku 4.5", kind: "hosted", decoding: "API default sampling", marker: "solid" },
  { id: "opus", key: "opus_5", dir: "claude-opus-5", label: "Claude Opus 5", short: "Claude Opus 5", kind: "hosted", decoding: "API default sampling, thinking off", marker: "solid" },
];

const pk = (a, k) => {
  const v = a.pass_at_k[`pass@${k}`];
  return v ? { value: v.value, lo: v.ci95[0], hi: v.ci95[1], problems: v.n_problems } : null;
};
const passAtK = (n, c, k) => {
  if (n - c < k) return 1;
  let p = 1;
  for (let i = 0; i < k; i++) p *= (n - c - i) / (n - i);
  return 1 - p;
};

const arms = [];
const strips = {};
for (const A of ARMS) {
  const a = cmp.arms[A.key];
  const rows = jsonl(`${R}/${A.dir}/results.jsonl`);
  const per = new Map();
  for (const r of rows) {
    const e = per.get(r.task_id) ?? { n: 0, c: 0 };
    e.n += 1;
    e.c += r.passed ? 1 : 0;
    per.set(r.task_id, e);
  }
  const strip = [...per.entries()]
    .map(([task_id, e]) => ({ task_id, index: Number(task_id.split("/")[1]), n: e.n, c: e.c }))
    .sort((x, y) => x.index - y.index);
  must(strip.length === 164, `${A.id}: expected 164 problems, got ${strip.length}`);
  must(strip.reduce((s, e) => s + e.n, 0) === a.n_samples, `${A.id}: strip sample total != comparison n_samples`);
  must(strip.reduce((s, e) => s + e.c, 0) === a.passed_samples, `${A.id}: strip pass total != comparison passed_samples`);
  // independent recomputation of pass@k from the per-problem counts must match the reported values
  for (const k of [1, 3, 10]) {
    const rep = a.pass_at_k[`pass@${k}`];
    if (!rep) continue;
    const elig = strip.filter((e) => e.n >= k);
    const mine = elig.reduce((s, e) => s + passAtK(e.n, e.c, k), 0) / elig.length;
    must(Math.abs(mine - rep.value) < 1e-9, `${A.id}: recomputed pass@${k} ${mine} != reported ${rep.value}`);
  }
  strips[A.id] = strip;
  arms.push({
    id: A.id, label: A.label, short: A.short, kind: A.kind, decoding: A.decoding, marker: A.marker,
    nSamples: a.n_samples, samplesPerProblem: a.samples_per_problem, passedSamples: a.passed_samples,
    pass1: pk(a, 1), pass3: pk(a, 3), pass10: A.id === "opus" ? null : pk(a, 10),
    outcomes: a.outcomes,
    truncated: a.truncated_at_max_tokens, truncationUpperBound: a.truncation_upper_bound_sample_pass_rate,
    maxTokens: a.max_tokens, tokens: a.tokens, protocol: a.protocol,
    cost: a.api_cost ? { usdTotal: a.api_cost.usd_total, usdPerSample: a.api_cost.usd_per_sample, usdPerSolved: a.api_cost.usd_per_expected_solved_sample } : null,
    selfHosted: a.self_hosted ? { wallSeconds: a.self_hosted.wall_seconds, wallHours: a.self_hosted.wall_hours, tokensPerSecond: a.self_hosted.completion_tokens_per_second } : null,
  });
}
const hostedWall = (dir) => jsonl(`${R}/${dir}/completions.jsonl`).length; // provenance touch; wall time comes from run.json below
void hostedWall;
const wall = (dir) => json(`${R}/${dir}/run.json`).segments.reduce((s, x) => s + x.wall_seconds, 0);
for (const a of arms) if (a.kind === "hosted") a.wallSeconds = wall(a.id === "haiku" ? "claude-haiku-4-5" : "claude-opus-5");
for (const a of arms) if (a.kind === "self") a.wallSeconds = a.selfHosted.wallSeconds;

// ---------------------------------------------------------------- cost model and break-even
const selfHours = cmp.arms.self_hosted_sampled.self_hosted.wall_hours;
const be = cmp.break_even_self_hosting_rate;
const costModel = {
  selfHostedHoursFor1640: selfHours,
  samplesCompared: cmp.arms.self_hosted_sampled.n_samples,
  hostedCostFor1640: { haiku: be.haiku_4_5.hosted_cost_for_1640_samples_usd, opus: be.opus_5.hosted_cost_for_1640_samples_usd },
  opusIsExtrapolated: true,
  breakEven: {
    haiku: { perSample: be.haiku_4_5.usd_per_hour_below_which_selfhosting_is_cheaper_per_sample, perSolved: be.haiku_4_5.usd_per_hour_below_which_selfhosting_is_cheaper_per_solved_sample },
    opus: { perSample: be.opus_5.usd_per_hour_below_which_selfhosting_is_cheaper_per_sample, perSolved: be.opus_5.usd_per_hour_below_which_selfhosting_is_cheaper_per_solved_sample },
  },
};
// the break-even rate is where selfHours * rate == hosted cost: recompute it and compare
must(Math.abs(selfHours * costModel.breakEven.haiku.perSample - costModel.hostedCostFor1640.haiku) < 1e-6, "haiku break-even does not satisfy hours*rate = cost");
must(Math.abs(selfHours * costModel.breakEven.opus.perSample - costModel.hostedCostFor1640.opus) < 1e-6, "opus break-even does not satisfy hours*rate = cost");

// ---------------------------------------------------------------- spend
const ledger = jsonl(`${R}/spend_ledger.jsonl`);
const ledgerTotal = ledger.reduce((s, r) => s + r.usd, 0);
must(Math.abs(ledgerTotal - cmp.total_hosted_spend_usd_including_pilots) < 1e-9, "ledger total != comparison total spend");
const spend = {
  totalUsd: ledgerTotal, capUsd: cmp.budget.hard_cap_usd, fundedUsd: cmp.budget.user_funded_usd, calls: ledger.length,
  haikuRunUsd: cmp.arms.haiku_4_5.api_cost.usd_total, opusRunUsd: cmp.arms.opus_5.api_cost.usd_total,
  pilotsUsd: ledgerTotal - cmp.arms.haiku_4_5.api_cost.usd_total - cmp.arms.opus_5.api_cost.usd_total,
};

// ---------------------------------------------------------------- engine measurements
const sweep = json("engine/results/decode_sweep_qwen2.5-coder-1.5b.json");
const thread = json("engine/results/thread_penalty_qwen2.5-coder-1.5b.json");
const b1 = sweep.rows.find((r) => r.batch === 1);
const b32 = sweep.rows.find((r) => r.batch === 32);
const byMode = Object.fromEntries(thread.results.map((r) => [r.mode, r]));
const eng = cmp.arms.self_hosted_sampled.self_hosted.engine;
const engine = {
  model: sweep.model, threads: sweep.threads, contextTokens: sweep.context_tokens, blockSize: sweep.block_size,
  decodeSweep: sweep.rows,
  batchingGain: { tokensPerSecondRatio: b32.decode_tokens_per_second / b1.decode_tokens_per_second, stepTimeRatio: b32.decode_step_seconds / b1.decode_step_seconds },
  threadPenalty: {
    concurrentRequests: thread.concurrent_requests, promptTokens: thread.prompt_tokens, newTokens: thread.new_tokens,
    modes: thread.results, slowdown: byMode.thread.wall_seconds / byMode.main.wall_seconds - 1,
  },
  realWorkload: {
    decodeSteps: eng.decode_steps, peakBatch: eng.peak_batch, averageBatch: eng.output_tokens / eng.decode_steps,
    decodeStepSeconds: eng.decode_seconds / eng.decode_steps, prefillSeconds: eng.prefill_seconds, decodeSeconds: eng.decode_seconds,
    attnPaddingWaste: eng.attn_padding_waste, kvUtilisation: eng.kv_utilisation, preemptions: eng.preemptions, outputTokens: eng.output_tokens,
    tokensPerSecond: eng.throughput_tok_per_s,
  },
  hardware: cmp.arms.self_hosted_sampled.self_hosted.hardware,
  servedConfig: (() => {
    const c = json(`${R}/qwen2.5-coder-1.5b/run.json`).engine_version.config;
    return { clientWorkers: json(`${R}/qwen2.5-coder-1.5b/run.json`).workers, model: c.model, backend: c.backend, batching: c.batching, maxBatch: c.max_batch, blockSize: c.block_size, kvBudgetMib: c.kv_budget_mib, maxContext: c.max_context, numThreads: c.num_threads, preemption: c.preemption };
  })(),
};

// ---------------------------------------------------------------- gate
const gate = json("bench/results/humaneval_gate.json");
must(gate.gate_ok === true, "the harness gate file reports gate_ok=false");

// ---------------------------------------------------------------- facts that live only in prose (regex-verified)
// The docs are prose that people keep editing (the task board changes with every commit). They feed the site only through the
// regex-verified facts below, so provenance is a hash of the PARSED FACTS, not of the whole document.
const step7 = read("bench/docs/step7-real-model-run.md", false, false);
const board = read("docs/agentforge-task-board.md", false, false);
const total647 = match1(step7, /resolved (\d+) of (\d+) attempts/, "ts-bench local-model total");
const tallySection = step7.slice(step7.indexOf("**Final tallies at the stopping point**"));
const models = [...tallySection.matchAll(/\| `([^`]+)` \| (\d+)\/(\d+) \((complete|partial)\) \| (\d+) \|/g)].map((m) => ({
  model: m[1], attempts: Number(m[2]), planned: Number(m[3]), status: m[4], resolved: Number(m[5]),
}));
const gbByModel = Object.fromEntries([...step7.matchAll(/^\| `([^`]+)` \| (\d+(?:\.\d+)?)GB \|/gm)].map((m) => [m[1], Number(m[2])]));
for (const m of models) {
  must(gbByModel[m.model] !== undefined, `no download size for ${m.model} in the ts-bench hardware table`);
  m.downloadGB = gbByModel[m.model];
}
must(models.length === 3, `expected 3 completed/partial local models in the ts-bench tally, found ${models.length}`);
must(models.reduce((s, m) => s + m.attempts, 0) === Number(total647[2]), "per-model attempts do not sum to the stated total");
must(models.every((m) => m.resolved === 0) && Number(total647[1]) === 0, "ts-bench local models are no longer all-zero; the story text would be wrong");
const caveat = match1(step7, /It does not show (these models cannot solve such tasks under a different scaffold)/, "the narrow-claim caveat")[1];
const paramsB = (name) => {
  const m = name.match(/[:-](\d+(?:\.\d+)?)b$/);
  return m ? Number(m[1]) : null;
};
const servedParamsB = paramsB(engine.model);
must(servedParamsB !== null, "could not read the served model size from its name");
const firstGate = match1(board, /FIRST run failed (\d+)\/(\d+)/, "the first (failed) run of the harness gate");
const haiku = match1(step7, /Final running total: (\d+) attempts, (\d+) resolved \((\d+)%\)/, "ts-bench Haiku 4.5 result");
const narrow = match1(step7, /The claim this data supports is narrow: \*([^*]+)\*/, "the narrow-claim sentence")[1];
const goldenTests = Number(match1(board, /(\d+) passed in 1313\.19s/, "golden test count")[1]);
const engineTests = Number(match1(board, /(\d+) passed, 1 deselected in 1559\.58s/, "engine suite count")[1]);
const benchTests = Number(match1(board, /(\d+) passed in 37\.75 s/, "bench suite count")[1]);
const phaseTable = board.slice(0, board.indexOf("\n---\n"));
const phases = [...phaseTable.matchAll(/^\| (\d) \| (.+?) \| (.+?) \|$/gm)].map((m) => ({ n: Number(m[1]), title: m[2], status: m[3] }));
must(phases.length === 7, `expected 7 phases in the board table, found ${phases.length}`);
const facts = {
  tsBench: {
    totalAttempts: Number(total647[2]), totalResolved: Number(total647[1]),
    models: models.map((m) => ({ ...m, paramsB: paramsB(m.model) })), servedParamsB,
    haiku: { attempts: Number(haiku[1]), resolved: Number(haiku[2]), percent: Number(haiku[3]) },
    narrowClaim: narrow, narrowClaimCaveat: caveat,
  },
  gateFirstRun: { passed: Number(firstGate[1]), of: Number(firstGate[2]) },
  tests: { golden: goldenTests, engine: engineTests, bench: benchTests },
  phases,
};

sources["(facts parsed from bench/docs/step7-real-model-run.md and docs/agentforge-task-board.md)"] = createHash("sha256").update(JSON.stringify(facts)).digest("hex");

// ---------------------------------------------------------------- how a sample is executed and scored (parsed from the code)
const sbSrc = read("bench/humaneval/sandbox.py");
const repSrc = read("bench/humaneval/report.py");
const sandbox = {
  image: match1(sbSrc, /IMAGE = "([^"]+)"/, "sandbox image")[1],
  network: match1(sbSrc, /"--network", "(\w+)"/, "sandbox --network")[1],
  user: match1(sbSrc, /"--user", "([^"]+)"/, "sandbox --user")[1],
  capDrop: match1(sbSrc, /"--cap-drop", "(\w+)"/, "sandbox --cap-drop")[1],
  tmpfs: match1(sbSrc, /"--tmpfs", "([^"]+)"/, "sandbox --tmpfs")[1],
  memory: match1(sbSrc, /memory: str = "([^"]+)"/, "sandbox memory")[1],
  pids: Number(match1(sbSrc, /pids: int = (\d+)/, "sandbox pids")[1]),
  cpus: match1(sbSrc, /cpus: str = "([^"]+)"/, "sandbox cpus")[1],
  fsizeBytes: Number(match1(sbSrc, /"fsize=(\d+)"/, "sandbox fsize")[1]),
  readOnlyRootfs: /"--read-only"/.test(sbSrc),
  noNewPrivileges: /"no-new-privileges"/.test(sbSrc),
};
must(sandbox.readOnlyRootfs && sandbox.noNewPrivileges, "sandbox source no longer sets --read-only / no-new-privileges");
const bootstrap = {
  draws: Number(match1(repSrc, /def bootstrap_ci\(values: list\[float\], b: int = ([\d_]+)/, "bootstrap draws")[1].replace(/_/g, "")),
  seed: Number(match1(repSrc, /def bootstrap_ci\(values: list\[float\], b: int = [\d_]+, seed: int = (\d+)/, "bootstrap seed")[1]),
};

// ---------------------------------------------------------------- one real request, for the eval-architecture diagram
const ex = json(`${R}/site_examples.json`);
const slimSample = (s) => ({
  sampleIdx: s.sample_idx, seed: s.seed, reply: s.reply, assemblyMode: s.assembly_mode, finishReason: s.finish_reason,
  promptTokens: s.prompt_tokens, completionTokens: s.completion_tokens, clientLatencySeconds: s.client_latency_seconds,
  execSeconds: s.exec_seconds, passed: s.passed, outcome: s.outcome, detail: s.detail, programLines: s.program.split("\n").length,
});
const example = {
  rule: ex.rule, run: ex.run,
  problem: { taskId: ex.problem.task_id, entryPoint: ex.problem.entry_point, prompt: ex.problem.prompt, test: ex.problem.test, passes: ex.problem.passes_out_of_n, n: ex.problem.n },
  userMessage: ex.user_message, passing: slimSample(ex.passing), failing: slimSample(ex.failing),
  passAt1: ex.problem.passes_out_of_n / ex.problem.n, passAt3: passAtK(ex.problem.n, ex.problem.passes_out_of_n, 3),
};
const selfEngine = { ttftP50: eng.ttft_p50_s, tpotP50: eng.tpot_p50_s, e2eP50: eng.e2e_p50_s, e2eP99: eng.e2e_p99_s };

// ---------------------------------------------------------------- optional: recorded scheduler trace (W2)
const trace = json("engine/results/scheduler_trace.json", true);

const repoUrl = execFileSync("git", ["-C", ROOT, "remote", "get-url", "origin"], { encoding: "utf8" }).trim().replace(/\.git$/, "");

const data = {
  meta: { generator: "web/scripts/build-data.mjs", repoUrl, dataset: cmp.arms.self_hosted_sampled.dataset_sha256, promptTemplateSha256: cmp.arms.self_hosted_sampled.prompt_template_sha256, sources },
  task: { problems: 164, harnessGate: { goldPassed: gate.gold_passed, emptyPassed: gate.empty_passed, problems: gate.n_problems, seconds: gate.seconds } },
  arms, strips, costModel, spend, engine, engineRequestStats: selfEngine, facts, example, sandbox, bootstrap,
  caveats: cmp.caveats,
  trace,
};
mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(data, null, 1) + "\n");
console.log(`wrote ${OUT} (${Object.keys(sources).length} source files hashed${trace ? "" : "; scheduler trace not present yet"})`);
