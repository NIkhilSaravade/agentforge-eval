import { EngineTrace } from "../components/EngineTrace";
import { Reveal, SectionHead } from "../components/Chrome";
import { data } from "../lib/data";
import { dec, int, pct, seconds, times } from "../lib/fmt";
import { trace } from "../lib/trace";

const MODE_LABEL: Record<string, string> = {
  main: "loaded and run on the main thread",
  thread: "loaded on the main thread, run on another",
  thread_all: "loaded and run on the same other thread",
};

export function Engine() {
  const e = data.engine;
  const b1 = e.decodeSweep.find((r) => r.batch === 1);
  const b32 = e.decodeSweep.find((r) => r.batch === 32);
  if (!b1 || !b32) throw new Error("decode sweep is missing batch 1 or 32");
  const real = e.realWorkload;
  const served = e.servedConfig;
  return (
    <section className="section" aria-labelledby="engine">
      <SectionHead id="engine" no="01" title={<>Many requests, one model, <em>one pool of memory</em>.</>} />

      <div className="cols">
        <div className="prose">
          <p className="lede">
            A language model produces one token per step, and every step has to stream all of its weights through the processor. That makes the
            cost of a step almost independent of how many sequences share it.
          </p>
          <p>
            Measured on this CPU with the 1.5-billion-parameter model: one decode step for a single sequence took {seconds(b1.decode_step_seconds, 4)};
            for {int(b32.batch)} sequences at once it took {seconds(b32.decode_step_seconds, 4)}. That is {times(e.batchingGain.stepTimeRatio)} the time
            for {times(e.batchingGain.tokensPerSecondRatio)} the tokens per second.
          </p>
          <p>
            <strong>Continuous batching</strong> is the scheduler&rsquo;s way of cashing that in. It does not wait for a full batch to finish. Between
            any two steps, finished requests leave and waiting ones join, so the batch stays full while requests of different lengths come and go.
          </p>
          <p>
            Each running request also owns a <strong>KV cache</strong>: the keys and values of every token it has seen. Reserving the worst-case length
            for each request strands most of the memory. A <strong>paged</strong> cache hands out small fixed-size blocks as a request grows, and a
            per-request block table maps its logical blocks to wherever free blocks happen to be. In the figure below, <code>C2</code> means request
            C&rsquo;s second block. It does not sit next to <code>C1</code>.
          </p>
          <p>
            When the pool runs dry, something has to give. This scheduler <strong>evicts</strong> a request, throws its cache away, and queues it again;
            on readmission it re-runs the prompt plus everything it had already generated. That is wasteful, and it is correct: in the recording below,
            all {int(trace.requests.length)} requests produced exactly the tokens a lone, unbatched run produces.
          </p>
        </div>
        <div className="margin">
          <span className="label">Source</span>
          Decode timings: <code>engine/results/decode_sweep_qwen2.5-coder-1.5b.json</code>, run on {e.hardware.cpu}, {e.threads} threads, {int(e.contextTokens)}-token context.
          <br />
          <br />
          <span className="label">Noise</span>
          One run per batch size. An earlier run of the same script put batch 8 well above where this one did.
        </div>
      </div>

      <Reveal>
        <p className="label" style={{ margin: "2.4rem 0 0.3rem" }}>Scroll to run the recording. Or use the controls.</p>
      </Reveal>
      <EngineTrace />

      <div className="cols">
        <div className="prose">
          <h3 style={{ marginBottom: "0.6rem" }}>What you just watched</h3>
          <p>
            It is a recording, not an illustration: {int(trace.totals.steps)} steps of the real scheduler and block manager serving a small model with
            a deliberately tiny pool of {int(trace.config.num_blocks)} blocks of {int(trace.config.block_size)} tokens, so that eviction happens.
            {" "}{int(trace.totals.preemptions)} evictions occurred. The recorder refuses to write the file unless at least one eviction happened, no
            block leaked, and every output matches an unbatched greedy run.
          </p>
          <p>
            The HumanEval runs used the same code with a {int(served.kvBudgetMib)} MiB pool of {int(served.blockSize)}-token blocks and a batch of up to{" "}
            {int(served.maxBatch)}. That pool never came close to filling: mean utilisation was {pct(real.kvUtilisation)} and there were{" "}
            {int(real.preemptions)} evictions. In that run paging bought capacity headroom, not eviction. The eviction behaviour above is real
            code, exercised on purpose.
          </p>
        </div>
        <div className="margin">
          <span className="label">Recorded by</span>
          <code>engine/scripts/record_scheduler_trace.py</code>. It wraps the live scheduler to observe allocate, grow, free, prefill, preempt and
          token events. No engine code is changed.
        </div>
      </div>

      <div className="cols" style={{ marginTop: "1.6rem" }}>
        <div className="prose">
          <h3 style={{ marginBottom: "0.6rem" }}>Measured, and not always as hoped</h3>
          <p>
            In the real HumanEval run the batch averaged {dec(real.averageBatch, 1)} of {int(served.maxBatch)} rows, and a decode step took {seconds(real.decodeStepSeconds, 2)}.
            The uniform benchmark above measured {seconds(b32.decode_step_seconds, 2)} for the same batch size. About {pct(real.attnPaddingWaste, 0)} of attention
            work was spent on padding, because sequences of very different lengths are padded to the longest. That is a suspect for the gap, and it was not
            verified.
          </p>
          <p>
            One measured surprise was found and fixed along the way: loading the model on one thread and serving it from another made decoding{" "}
            <strong>{pct(e.threadPenalty.slowdown, 0)} slower</strong>. The mechanism was not established; raw matrix multiplies run at the same speed on
            either thread. The server now builds the engine on the thread that serves.
          </p>
        </div>
      </div>

      <div className="figure" style={{ marginTop: "0.6rem" }}>
        <table>
          <caption className="label" style={{ textAlign: "left", paddingBottom: "0.5rem" }}>
            Same load, three arrangements: {int(e.threadPenalty.concurrentRequests)} concurrent requests, {int(e.threadPenalty.promptTokens)} prompt and{" "}
            {int(e.threadPenalty.newTokens)} new tokens each, no HTTP
          </caption>
          <thead>
            <tr><th>Arrangement</th><th>Wall-clock</th><th>Throughput</th></tr>
          </thead>
          <tbody>
            {e.threadPenalty.modes.map((m) => (
              <tr key={m.mode}>
                <td className="name">{MODE_LABEL[m.mode] ?? m.mode}</td>
                <td>{seconds(m.wall_seconds)}</td>
                <td>{dec(m.tokens_per_second)} tok/s</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="figcaption-note mono">
          Source: <code>engine/results/thread_penalty_qwen2.5-coder-1.5b.json</code>. Each arrangement ran in a fresh process.
        </div>
      </div>
    </section>
  );
}
