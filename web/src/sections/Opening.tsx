import { Reveal } from "../components/Chrome";
import { arm, data, need, SERIES, type ArmId } from "../lib/data";
import { ci, gap, int, pct, usd } from "../lib/fmt";

function Score({ id, name }: { id: ArmId; name: string }) {
  const a = arm(id);
  return (
    <div className="score" style={{ borderTopColor: SERIES[id] }}>
      <div className="label">{name}</div>
      <div className="big num">{pct(a.pass1.value)}</div>
      <div className="mono small">
        pass@1 &middot; 95% interval {ci(a.pass1)}
        <br />
        {int(a.nSamples)} samples on {int(a.pass1.problems)} problems
      </div>
    </div>
  );
}

export function Opening() {
  const self = arm("self");
  const haiku = arm("haiku");
  const opus = arm("opus");
  const be = data.costModel.breakEven.haiku.perSample;
  return (
    <header className="opening">
      <div className="label" style={{ paddingTop: "1.6rem" }}>
        AgentForge Eval &middot; HumanEval, {data.task.problems} problems &middot; dataset sha256 {data.meta.dataset.slice(0, 12)}
      </div>
      <Reveal>
        <h1 className="display" id="result">
          A CPU inference engine, put through an <em>honest</em> test.
        </h1>
      </Reveal>
      <div className="cols" style={{ marginTop: "2.2rem" }}>
        <div className="prose">
          <p className="lede">
            The thesis: serve a small model yourself, on an engine built to run many requests at once, and code generation gets cheaper
            without giving up much. The project built that engine from scratch, measured it, and spent real money comparing it with hosted
            models. The thesis mostly did not survive.
          </p>
        </div>
        <div className="margin">
          <span className="label">How to read this page</span>
          The result is first, on purpose. The rest is how it was found, and whether it can be trusted. Every number is generated from the
          repository&rsquo;s result files.
        </div>
      </div>

      <div className="scores" role="group" aria-label="pass@1 on HumanEval">
        <Score id="self" name="Self-hosted Qwen2.5-Coder-1.5B, on this CPU" />
        <Score id="haiku" name="Claude Haiku 4.5, hosted" />
        <Score id="opus" name="Claude Opus 5, hosted" />
      </div>

      <div className="cols" style={{ marginTop: "2rem" }}>
        <div className="prose">
          <p>
            The self-hosted model is <strong>{gap(haiku.pass1.value, self.pass1.value)} points behind</strong> Haiku 4.5 and{" "}
            <strong>{gap(opus.pass1.value, self.pass1.value)} points behind</strong> Opus 5 at pass@1, and its 95% interval (
            {ci(self.pass1)}) does not overlap Haiku&rsquo;s ({ci(need(haiku.pass1, "haiku pass@1"))}). On cost it comes out cheaper than
            Haiku 4.5 only if the machine costs less than <strong>{usd(be)} an hour</strong>, before counting the quality gap.
          </p>
          <p>
            That is the outcome of a plan that changed twice, and of an engine that works. Both are told below, with the negative result kept at
            the same size as everything else.
          </p>
        </div>
      </div>
    </header>
  );
}
