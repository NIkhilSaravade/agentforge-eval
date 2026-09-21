import { Reveal, SectionHead } from "../components/Chrome";
import { RequestTrace } from "../components/RequestTrace";
import { arm, data } from "../lib/data";
import { int } from "../lib/fmt";

export function Harness() {
  const self = arm("self");
  const haiku = arm("haiku");
  const opus = arm("opus");
  return (
    <section className="section" aria-labelledby="harness">
      <SectionHead id="harness" no="03" title={<>How a number gets made: <em>follow one request</em>.</>} />

      <div className="cols">
        <div className="prose">
          <p className="lede">
            A pass@k score is only as trustworthy as the path a completion takes to earn it. So here is that path for one real sample, end to end, with the real data at every
            stage.
          </p>
          <p>
            The harness asks a model for several completions of each of the {int(data.task.problems)} problems, runs every one against the problem&rsquo;s own unit tests in
            an isolated container, and scores the results. Three models went through it: the self-hosted engine ({int(self.nSamples)} samples), Claude Haiku 4.5 (
            {int(haiku.nSamples)}) and Claude Opus 5 ({int(opus.nSamples)}). The generation call differs by provider, the engine through LiteLLM and the hosted models through
            Anthropic&rsquo;s SDK with no sampling parameters sent. The prompt, the sandbox and the scorer are shared.
          </p>
          <p>
            The sample below was picked by a rule fixed in advance, not by hand (see the margin). It landed on {data.example.problem.taskId}, where {int(data.example.problem.passes)} of{" "}
            {int(data.example.problem.n)} samples passed, so the trace shows a problem the model sometimes gets right, and the toggle shows a pass and a fail side by side.
          </p>
        </div>
        <div className="margin">
          <span className="label">Selection rule</span>
          {data.example.rule}.
          <br />
          <br />
          <span className="label">Scoring is replayable</span>
          Regenerating hosted completions gives different samples, but every completion is saved, so scoring them again gives identical results.
        </div>
      </div>

      <Reveal>
        <p className="label" style={{ margin: "2.4rem 0 0.3rem" }}>Scroll to follow the request. Or use the controls.</p>
      </Reveal>
      <RequestTrace />
    </section>
  );
}
