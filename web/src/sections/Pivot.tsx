import { AttemptsGrid } from "../components/AttemptsGrid";
import { Reveal, SectionHead } from "../components/Chrome";
import { data } from "../lib/data";
import { int, times } from "../lib/fmt";

export function Pivot() {
  const t = data.facts.tsBench;
  // only models whose size is stated in their name are used; nothing is assumed about the others
  const sized = t.models.flatMap((m) => (m.paramsB === null ? [] : [{ ...m, paramsB: m.paramsB }]));
  const smallB = Math.min(...sized.map((m) => m.paramsB));
  const nSmall = sized.filter((m) => m.paramsB === smallB).length;
  const sizeRatio = smallB / t.servedParamsB;
  const unsized = t.models.find((m) => m.paramsB === null);
  const sizedRef = t.models.find((m) => m.paramsB === smallB);
  return (
    <section className="section" aria-labelledby="pivot">
      <SectionHead id="pivot" no="02" title={<>The plan changed, <em>and why</em>.</>} />

      <div className="cols">
        <div className="prose">
          <p className="beat label">The problem</p>
          <p className="lede">
            The project&rsquo;s thesis was about money. On hosted APIs a small budget buys a handful of agent trials, nowhere near enough for a
            statistically meaningful pass@k. A self-hosted engine that runs many generations at once was supposed to buy the missing trials.
          </p>
          <p>
            The benchmark was ts-bench: SWE-bench-style tasks where an agent fixes a real bug in a TypeScript, Python or Java repository and the
            repository&rsquo;s own tests decide. Real API money would be spent once, on one frontier model, as an anchor.
          </p>
          <p>
            But ts-bench had already run local open-weight models under its plain bash-loop scaffold, and the evidence was in the repository before the first line of
            the new code was written.
          </p>
        </div>
        <div className="margin">
          <span className="label">Where this comes from</span>
          The final tallies in ts-bench&rsquo;s own write-up of its first real-model run. The run was stopped early by choice after it stopped producing
          information; it is not presented there as a complete comparison, and it is not here.
        </div>
      </div>

      <Reveal>
        <AttemptsGrid />
      </Reveal>

      <div className="cols">
        <div className="prose">
          <p>
            Three local models produced <strong>{int(t.totalAttempts)} attempts and {int(t.totalResolved)} resolutions</strong>. Claude Haiku 4.5, under the same
            scaffold, resolved <strong>{int(t.haiku.resolved)} of {int(t.haiku.attempts)}</strong>.
          </p>
          <blockquote>
            The claim this data supports is narrow: <em>{t.narrowClaim}</em> It does not show {t.narrowClaimCaveat}.
          </blockquote>
          <p className="small">The quote is from the write-up that recorded the runs. It is the limit of what the evidence supports, and this page keeps to it.</p>

          <p className="beat label" style={{ marginTop: "2.4rem" }}>The reasoning</p>
          <p>
            {int(nSmall)} of the {int(t.models.length)} local models are {int(smallB)} billion parameter models, about {times(sizeRatio)} the size of the {t.servedParamsB} billion
            parameter model this CPU can serve{unsized && sizedRef ? `, and the third, ${unsized.model}, is a ${unsized.downloadGB} GB download against ${sizedRef.downloadGB} GB for a ${smallB}B model` : ""}. A model that solves nothing anywhere tells you nothing, and more
            trials of it only produce more zeros. The scarce resource was not dollars per rollout. It was capability, and a cheaper serving engine does not change
            that.
          </p>
          <p>
            Spending the budget to confirm it would have meant paying to learn what the repository already said. So the work stopped, and the question changed.
          </p>

          <p className="beat label" style={{ marginTop: "2.4rem" }}>The decision</p>
          <p>
            Leave ts-bench exactly as it was and stop pointing the self-hosted model at it. Evaluate on <strong>HumanEval</strong> instead: {int(data.task.problems)}{" "}
            short Python functions, each with hidden unit tests, a task class small code models are built for and one where a model could land anywhere between zero
            and all of them. It is not a lowered bar. The hosted models get the same problems, the same fixed prompt and the same scorer, so the comparison stays like
            for like.
          </p>
          <p>
            It also reversed an earlier decision. HumanEval&rsquo;s standard protocol is pass@k with real sampling, so seeded temperature and top-p sampling were added to
            the engine, on short independent generations where that is safe, with the golden-tested greedy path left untouched and a test proving it.
          </p>
          <p>
            Two reframes preceded this, both approved and both written on the task board before code changed: first, measure the engine&rsquo;s throughput under agent
            traffic instead of solve rate; then the better fix, change the task class.
          </p>
        </div>
        <div className="margin">
          <span className="label">The price of the pivot</span>
          HumanEval is public and very likely in these models&rsquo; training data, so every absolute score here overstates real ability. The comparison between
          models on the same problems is less affected than the absolute numbers.
        </div>
      </div>
    </section>
  );
}
