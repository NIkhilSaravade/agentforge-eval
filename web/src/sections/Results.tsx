import type { ReactNode } from "react";
import { Reveal, SectionHead } from "../components/Chrome";
import { BreakEven, IntervalChart, Strips } from "../components/ResultsCharts";
import { arm, CONF, data, SERIES, type ArmId } from "../lib/data";
import { ci, clock, gap, int, pct, usd } from "../lib/fmt";
import { HEX } from "../lib/palette";

const ORDER: ArmId[] = ["self", "greedy", "haiku", "opus"];
const SHORT: Partial<Record<ArmId, string>> = { self: "Self-hosted", haiku: "Haiku 4.5", opus: "Opus 5" };

/** The caveats as readable prose, with every number interpolated from data.json. The results file
 *  (phase6_comparison.json) is the source of truth for WHICH caveats exist: if it gains or loses one, the page refuses to render. */
function caveats(): ReactNode[] {
  const self = arm("self"), haiku = arm("haiku"), opus = arm("opus");
  const items: ReactNode[] = [
    <>
      <strong>Sampling is not identical.</strong> The self-hosted model sampled at temperature {self.protocol.temperature} and top-p {self.protocol.top_p} with fixed seeds. The hosted arms used
      the API&rsquo;s own default sampling, because Anthropic&rsquo;s API reference says Opus 5 rejects both parameters (this project followed that and never sent them). Hosted runs cannot be regenerated identically; scoring is exactly reproducible from the saved completions.
    </>,
    <>
      <strong>Sample counts differ.</strong> The self-hosted model and Haiku 4.5 have {int(self.samplesPerProblem[0] ?? 0)} samples per problem and Opus 5 has {int(opus.samplesPerProblem[0] ?? 0)}, its budget. Compare at
      k of {int(opus.samplesPerProblem[0] ?? 0)} or less. From {int(self.samplesPerProblem[0] ?? 0)} samples pass@k is the unbiased estimator; from {int(opus.samplesPerProblem[0] ?? 0)} it is the plain fraction of problems solved in
      that many draws.
    </>,
    <>
      <strong>The output cap cut replies off.</strong> Every arm was limited to {int(self.maxTokens)} tokens. That cut off {int(self.truncated)} self-hosted, {int(haiku.truncated)} Haiku 4.5 and {int(opus.truncated)} Opus 5 replies. The
      table beside this shows the most it could matter.
    </>,
    <>
      <strong>Opus 5 ran with thinking off,</strong> to fit the budget. Default thinking was measured on only two calls and projected over the cap, so how Opus 5 does with it on is not measured here.
    </>,
    <>
      <strong>HumanEval is public,</strong> and very likely in every model&rsquo;s training data. Absolute scores overstate real-world ability for all arms.
    </>,
    <>
      <strong>The intervals resample problems.</strong> They do not include seed-to-seed or run-to-run variance.
    </>,
    <>
      <strong>The self-hosted machine has no dollar cost here.</strong> No hourly rate was supplied, so the break-even chart is the rate-free comparison.
    </>,
  ];
  if (items.length !== data.caveats.length) {
    throw new Error(`the results file lists ${data.caveats.length} caveats but the page writes ${items.length}; update the page so none is dropped`);
  }
  return items;
}

function Cell({ v }: { v: { value: number; lo: number; hi: number } | null }) {
  if (!v) return <td className="na">n/a</td>;
  return (
    <td>
      {pct(v.value)}
      <span className="ci">{ci(v, 1)}</span>
    </td>
  );
}

function ResultsTable() {
  return (
    <div className="figure table-wrap" style={{ marginTop: "1.4rem" }}>
      <table>
        <caption className="label" style={{ textAlign: "left", paddingBottom: "0.5rem" }}>
          Same {int(data.task.problems)} problems, same fixed prompt, same sandbox and scorer. Intervals are {CONF}, resampling problems.
        </caption>
        <thead>
          <tr>
            <th>Arm</th><th>Samples</th><th>pass@1</th><th>pass@3</th><th>pass@10</th><th>Cost</th><th>Wall-clock</th>
          </tr>
        </thead>
        <tbody>
          {ORDER.map((id) => {
            const a = arm(id);
            return (
              <tr key={id}>
                <td className="name">
                  <i className={`swatch${a.marker === "hollow" ? " hollow" : ""}`} style={{ background: SERIES[id], color: HEX[id] }} />
                  {a.short}
                  <span className="ci" style={{ fontFamily: "var(--mono)" }}>{a.decoding}</span>
                </td>
                <td>{int(a.nSamples)}<span className="ci">{a.samplesPerProblem.map((n) => `${int(n)} per problem`).join(", ")}</span></td>
                <Cell v={a.pass1} />
                <Cell v={a.pass3} />
                {id === "opus" ? <td className="na">not measured</td> : <Cell v={a.pass10} />}
                {a.cost ? (
                  <td>{usd(a.cost.usdTotal, 2)}<span className="ci">{usd(a.cost.usdPerSample, 4)} per sample</span></td>
                ) : (
                  <td className="na">no $ figure<span className="ci">no hourly rate supplied</span></td>
                )}
                <td>{clock(a.wallSeconds)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Truncation() {
  return (
    <table className="compact">
      <caption className="label" style={{ textAlign: "left", paddingBottom: "0.5rem" }}>
        Upper bound: the pass rate if every cut-off sample had passed. Every arm got the same {int(arm("self").maxTokens)}-token cap.
      </caption>
      <thead>
        <tr><th>Arm</th><th>Cut off</th><th>Pass rate</th><th>Upper bound</th></tr>
      </thead>
      <tbody>
        {(["self", "haiku", "opus"] as ArmId[]).map((id) => {
          const a = arm(id);
          const rate = a.passedSamples / a.nSamples;
          return (
            <tr key={id}>
              <td className="name"><i className="swatch" style={{ background: SERIES[id] }} />{SHORT[id]}</td>
              <td>{int(a.truncated)} of {int(a.nSamples)}</td>
              <td>{pct(rate)}</td>
              <td>{pct(a.truncationUpperBound)}<span className="ci">up to {gap(a.truncationUpperBound, rate)} points higher</span></td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export function Results() {
  const self = arm("self"), haiku = arm("haiku"), opus = arm("opus");
  const strips = data.strips as Record<string, { c: number; n: number }[]>;
  const never = (id: string) => (strips[id] ?? []).filter((e) => e.c === 0).length;
  const always = (id: string) => (strips[id] ?? []).filter((e) => e.c === e.n).length;
  const s = data.spend;
  return (
    <section className="section" aria-labelledby="results">
      <SectionHead id="results" no="04" title={<>What the run <em>measured</em>.</>} />

      <div className="cols">
        <div className="prose">
          <p className="lede">
            Every arm faced the same {int(data.task.problems)} problems, the same fixed prompt, the same sandbox and the same scorer. The table is the result. What follows says what it
            supports and what it does not, at the same size.
          </p>
        </div>
        <div className="margin">
          <span className="label">Source</span>
          <code>bench/results/humaneval/phase6_comparison.json</code>, rebuilt from the saved per-sample results. Nothing on this page is typed by hand.
        </div>
      </div>

      <Reveal><ResultsTable /></Reveal>

      <div className="cols">
        <div className="prose">
          <h3>The gap</h3>
          <p>
            At pass@1 the self-hosted model is <strong>{gap(haiku.pass1.value, self.pass1.value)} points behind</strong> Haiku 4.5 and <strong>{gap(opus.pass1.value, self.pass1.value)} points behind</strong>{" "}
            Opus 5. Its interval does not touch Haiku&rsquo;s. Greedy decoding helps it ({pct(arm("greedy").pass1.value)}), and it still trails.
          </p>
        </div>
      </div>
      <IntervalChart />

      <div className="cols">
        <div className="prose">
          <h3>Every problem, every arm</h3>
          <p>
            The self-hosted model never solved <strong>{int(never("self"))}</strong> of {int(data.task.problems)} problems in {int(self.samplesPerProblem[0] ?? 0)} tries and solved{" "}
            <strong>{int(always("self"))}</strong> every time. Haiku 4.5 never solved {int(never("haiku"))} and solved {int(always("haiku"))} every time; Opus 5 never solved {int(never("opus"))} in {int(opus.samplesPerProblem[0] ?? 0)}.
          </p>
          <p>
            The misses do not line up neatly. Of the {int(data.overlap.selfNever)} problems the self-hosted model never solved, Opus 5 solved <strong>{int(data.overlap.selfNeverSolvedEveryTimeByOpus)}</strong>{" "}
            every time and Haiku 4.5 solved {int(data.overlap.selfNeverSolvedEveryTimeByHaiku)} every time; only {int(data.overlap.selfNeverAlsoNeverHaiku)} are also ones Haiku never solved. Going the other way, the{" "}
            {int(data.overlap.opusMissed)} problems Opus 5 missed were each solved at least once by the self-hosted model ({int(data.overlap.opusMissedSelfSolvedAtLeastOnce)} of {int(data.overlap.opusMissed)}), and{" "}
            {int(data.overlap.opusMissedWithTruncatedReply)} of those {int(data.overlap.opusMissed)} are problems where an Opus 5 reply was cut off by the shared output cap. That is a real point for the small model,
            and mostly an artifact of the cap.
          </p>
        </div>
      </div>
      <Strips />

      <div className="cols">
        <div className="prose">
          <h3>Cost and time</h3>
          <p>
            The hosted runs were paid for with real money: <strong>{usd(haiku.cost?.usdTotal ?? 0)}</strong> for Haiku 4.5 and <strong>{usd(opus.cost?.usdTotal ?? 0)}</strong> for Opus 5, {usd(s.pilotsUsd)} of pilots, {usd(s.totalUsd)} in total
            against a hard cap of {usd(s.capUsd)}. The self-hosted run took {clock(self.wallSeconds)}. The hosted arms finished in {clock(haiku.wallSeconds)} and {clock(opus.wallSeconds)}.
          </p>
          <p>
            A dollar figure for self-hosting needs an hourly cost for the machine, and none was supplied. So the question is asked the other way round: below what hourly cost does self-hosting
            the same samples come out cheaper? Slide it.
          </p>
        </div>
      </div>
      <BreakEven />

      <div className="notes-grid">
        <div className="prose">
          <h3 className="tri">What the numbers do not say</h3>
          <p className="lede" style={{ fontSize: "1.3rem" }}>
            These belong to the result. They sit here, at the same size, because leaving them out would change what the numbers mean.
          </p>
          <ol className="caveats">
            {caveats().map((c, i) => <li key={i}>{c}</li>)}
          </ol>
        </div>
        <div className="prose">
          <h3 className="tri">How much the output cap matters</h3>
          <p>
            Hosted models write longer answers, so the shared cap cut more of their replies off. It cannot change the ranking: even if every cut-off sample had passed, the self-hosted
            model would stay far behind. But it does mean the hosted scores above are understated.
          </p>
          <div className="figure table-wrap" style={{ margin: "1.2rem 0 0" }}>
            <Truncation />
          </div>
          <h3 className="tri" style={{ marginTop: "2rem" }}>Where self-hosting still helps</h3>
          <p>
            There is no per-token bill, and no data leaves the machine. The engine&rsquo;s continuous batching is what made {clock(self.wallSeconds)} possible at all. None of that closes a quality gap of this size.
          </p>
        </div>
      </div>
    </section>
  );
}
