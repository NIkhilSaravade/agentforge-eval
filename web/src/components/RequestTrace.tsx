// One real request's path through the harness: prompt -> engine -> reply -> assemble -> sandbox -> score -> aggregate.
// Every value shown is read from data.example (a real sample from the self-hosted run, chosen by a fixed rule) or from
// data.sandbox / data.bootstrap / data.arms (parsed from the code and result files). Scroll drives the stage; the buttons are alternatives.
import { motion, useMotionValueEvent, useReducedMotion, useScroll } from "motion/react";
import { useRef, useState, type ReactNode } from "react";
import { arm, data } from "../lib/data";
import { ci, dec, int, pct, seconds } from "../lib/fmt";

type Which = "passing" | "failing";
const NAMES = ["Prompt", "Engine", "Reply", "Assemble", "Sandbox", "Score", "Aggregate"] as const;
const N = NAMES.length;

const RAIL_W = 900;
const nodeX = (i: number) => 30 + (i * (RAIL_W - 60)) / (N - 1);

function firstLines(s: string, n: number): string {
  return s.split("\n").slice(0, n).join("\n");
}

export function RequestTrace() {
  const reduce = useReducedMotion() ?? false;
  const ref = useRef<HTMLDivElement>(null);
  const [stage, setStage] = useState(0);
  const [which, setWhich] = useState<Which>("passing");
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end end"] });
  useMotionValueEvent(scrollYProgress, "change", (v) => setStage(Math.min(N - 1, Math.max(0, Math.floor(v * N)))));

  const ex = data.example;
  const sample = ex[which];
  const self = arm("self");
  const failed = !sample.passed;
  const gate = data.task.harnessGate;

  const outcomes = Object.entries(self.outcomes).sort((a, b) => b[1] - a[1]);
  const stagePanel: Record<number, { says: ReactNode; real: string; extra?: ReactNode }> = {
    0: {
      says: <>Each problem is wrapped in one fixed prompt, identical for every model. Nothing is tuned per model, ever. This is <strong>{ex.problem.taskId}</strong>, <code>{ex.problem.entryPoint}</code>.</>,
      real: ex.userMessage,
    },
    1: {
      says: <>The prompt goes to the engine&rsquo;s OpenAI-compatible endpoint. The per-sample seed makes the draw replayable. The wait is mostly queueing: the run kept {int(data.engine.servedConfig.clientWorkers)} requests in flight against a batch of {int(data.engine.servedConfig.maxBatch)}.</>,
      real:
        `POST /v1/chat/completions\n` +
        `{ model: "${data.engine.servedConfig.model}", temperature: ${self.protocol.temperature}, top_p: ${self.protocol.top_p},\n` +
        `  max_tokens: ${self.maxTokens}, seed: ${sample.seed} }\n\n` +
        `prompt tokens      ${int(sample.promptTokens)}\n` +
        `completion tokens  ${int(sample.completionTokens)}\n` +
        `finish reason      ${sample.finishReason}\n` +
        `client latency     ${seconds(sample.clientLatencySeconds, 2)}\n\n` +
        `Across the whole run the median request waited ${seconds(data.engineRequestStats.ttftP50, 2)} for its first token\n` +
        `and ${seconds(data.engineRequestStats.e2eP50, 2)} in total (${seconds(data.engineRequestStats.e2eP99, 2)} at the 99th percentile).`,
    },
    2: {
      says: <>The reply is the model&rsquo;s raw text. The harness never edits it. This is exactly what came back for sample {int(sample.sampleIdx)}.</>,
      real: sample.reply,
    },
    3: {
      says: <>The first fenced block that defines the function is appended to the original prompt, so helpers the tests rely on stay defined. Program and tests share one namespace, as in the original human-eval. Assembly mode: <code>{sample.assemblyMode}</code>, {int(sample.programLines)} lines.</>,
      real: `# the dataset's hidden test, appended to the program\n${ex.problem.test.trim()}\n\ndef test_humaneval():\n    check(${ex.problem.entryPoint})`,
    },
    4: {
      says: <>Every sample runs in a throwaway container: no network, non-root, no capabilities, read-only root filesystem, capped memory, processes and file size. Generated code is untrusted.</>,
      extra: (
        <ul className="bullets">
          <li>A test program tries to reach the network from inside the sandbox and must fail.</li>
          <li>It tries to write to the root filesystem and must fail, and it must not be root.</li>
          <li>Infinite loops are killed and reported as timeouts; a memory bomb is reported as a resource kill.</li>
        </ul>
      ),
      real:
        `docker run --rm --network ${data.sandbox.network} --user ${data.sandbox.user} \\\n` +
        `  --cap-drop ${data.sandbox.capDrop} --security-opt no-new-privileges --read-only \\\n` +
        `  --tmpfs ${data.sandbox.tmpfs} --memory ${data.sandbox.memory} --pids-limit ${data.sandbox.pids} \\\n` +
        `  --cpus ${data.sandbox.cpus} --ulimit fsize=${data.sandbox.fsizeBytes} \\\n` +
        `  ${data.sandbox.image} pytest --json-report test_solution.py\n\n` +
        `ran in ${seconds(sample.execSeconds, 3)}`,
    },
    5: {
      says: <>The dataset&rsquo;s own <code>check</code> decides. Assertion failures, exceptions, syntax errors, timeouts and out-of-memory kills are all failures, each with its own label.</>,
      extra: (
        <table className="mini">
          <caption className="label">Outcomes across all {int(self.nSamples)} self-hosted samples</caption>
          <tbody>
            {outcomes.map(([k, v]) => (
              <tr key={k}><td>{k.replace("_", " ")}</td><td>{int(v)}</td></tr>
            ))}
          </tbody>
        </table>
      ),
      real: `outcome: ${sample.outcome}\n${sample.detail ? `\n${firstLines(sample.detail, 6)}` : "\nno failing assertions"}`,
    },
    6: {
      says: <>Per problem, the unbiased pass@k estimator over that problem&rsquo;s samples. Across problems, a mean; the 95% interval resamples the problems {int(data.bootstrap.draws)} times with a fixed seed.</>,
      real:
        `${ex.problem.taskId}: ${int(ex.problem.passes)} of ${int(ex.problem.n)} samples passed\n` +
        `  pass@1 = ${dec(ex.passAt1, 2)}   pass@3 = ${dec(ex.passAt3, 4)}\n\n` +
        `all ${int(data.task.problems)} problems, self-hosted, sampled\n` +
        `  pass@1 = ${pct(self.pass1.value)}  [${ci(self.pass1, 2)}]\n` +
        `  pass@3 = ${self.pass3 ? pct(self.pass3.value) : ""}  [${self.pass3 ? ci(self.pass3, 2) : ""}]\n\n` +
        `harness gate, before any of this counts:\n` +
        `  canonical solutions ${int(gate.goldPassed)}/${int(gate.problems)} pass, empty answers ${int(gate.emptyPassed)}/${int(gate.problems)} pass\n` +
        `  (its first run passed only ${int(data.facts.gateFirstRun.passed)}/${int(data.facts.gateFirstRun.of)}: a harness bug the gate caught)`,
    },
  };
  const cur = stagePanel[stage] ?? stagePanel[0];
  if (!cur) return null;
  const packetX = nodeX(stage);
  const dur = reduce ? 0 : 0.45;

  return (
    <div ref={ref} className="scrolly" style={{ height: `${N * 42 + 100}vh` }}>
      <div className="sticky-fig trace">
        <figure className="figure" style={{ margin: 0 }}>
          <div className="controls" style={{ marginTop: 0, marginBottom: "0.7rem" }}>
            <span className="label">a real sample from the self-hosted run</span>
            <button className="btn" aria-pressed={which === "passing"} onClick={() => setWhich("passing")}>Sample {int(ex.passing.sampleIdx)}: passes</button>
            <button className="btn" aria-pressed={which === "failing"} onClick={() => setWhich("failing")}>Sample {int(ex.failing.sampleIdx)}: fails</button>
          </div>
          <div className="well">
            <svg viewBox={`0 0 ${RAIL_W} 84`} width="100%" role="img" aria-label="Seven stages a sample passes through, from prompt to aggregate score.">
              <line x1={nodeX(0)} y1={30} x2={nodeX(N - 1)} y2={30} stroke="var(--ink)" strokeWidth={1} />
              {NAMES.map((n, i) => (
                <g key={n}>
                  <rect x={nodeX(i) - 8} y={22} width={16} height={16} fill={i <= stage ? "var(--ink)" : "var(--paper-2)"} stroke="var(--ink)" strokeWidth={1.5} />
                  <text x={nodeX(i)} y={62} textAnchor="middle" className={i === stage ? "rail-n on" : "rail-n"}>{n.toUpperCase()}</text>
                </g>
              ))}
              {failed && stage >= 5 && <rect x={nodeX(5) - 12} y={18} width={24} height={24} fill="none" stroke="var(--self)" strokeWidth={3} />}
              <motion.rect width={22} height={22} fill="var(--self)" initial={false} animate={{ x: packetX - 11, y: 19 }} transition={{ duration: dur, ease: [0.22, 0.8, 0.2, 1] }} />
            </svg>
            <div className="stage">
              <div className="stage-says">
                <div className="label">stage {int(stage + 1)} of {int(N)}: {NAMES[stage]}</div>
                <p>{cur.says}</p>
                {cur.extra}
              </div>
              <pre className="stage-data" tabIndex={0} aria-label={`Real data for the ${NAMES[stage]} stage`}>{cur.real}</pre>
            </div>
          </div>
          <div className="controls">
            <button className="btn" onClick={() => setStage((s) => Math.max(0, s - 1))} disabled={stage === 0}>Prev</button>
            <button className="btn" onClick={() => setStage((s) => Math.min(N - 1, s + 1))} disabled={stage === N - 1}>Next</button>
            <span className="label" style={{ marginLeft: "auto" }}>scroll drives this figure</span>
          </div>
        </figure>
      </div>
    </div>
  );
}
