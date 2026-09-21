import { SectionHead } from "../components/Chrome";
import { data } from "../lib/data";
import { int } from "../lib/fmt";

const contact = data.contact as { label: string; url: string }[];

export function Close() {
  const { stack, facts } = data;
  return (
    <section className="section" aria-labelledby="close">
      <SectionHead id="close" no="05" title={<>What was built, and <em>where to find it</em>.</>} />

      <div className="cols">
        <div className="prose">
          <p className="lede">
            An inference engine written from scratch, a benchmark harness with an isolated sandbox and a spend limit, and a comparison run with real money and reported as it
            came out.
          </p>

          <dl className="deflist">
            <div>
              <dt className="label">Engine</dt>
              <dd>
                Python {stack.engine.python}, PyTorch {stack.engine.torch} on the CPU, FastAPI and uvicorn for the API. The forward passes for GPT-2 and Qwen2 are written by hand; there is no
                <code>generate()</code> in the serving path. A continuous-batching scheduler, a paged KV cache, and seeded sampling.
              </dd>
            </div>
            <div>
              <dt className="label">Harness</dt>
              <dd>
                Python, LiteLLM {stack.harness.litellm} and Anthropic&rsquo;s SDK {stack.harness.anthropic}, pytest {stack.harness.pytest} inside a Docker sandbox on{" "}
                <code>{stack.harness.sandboxBase}</code>, and a spend ledger with a circuit breaker.
              </dd>
            </div>
            <div>
              <dt className="label">Verification</dt>
              <dd>
                The engine has {int(facts.tests.engine)} tests, {int(facts.tests.golden)} of them golden tests that compare against the HuggingFace reference token for token. The
                benchmark harness has {int(facts.tests.bench)}.
              </dd>
            </div>
            <div>
              <dt className="label">This page</dt>
              <dd>
                React {stack.web.react}, TypeScript {stack.web.typescript}, Vite {stack.web.vite}, Motion {stack.web.motion}. Newsreader and IBM Plex Mono, self-hosted. Every number is generated
                from the repository&rsquo;s result files by <code>web/scripts/build-data.mjs</code>, and the two diagrams replay a recorded scheduler trace and a real sample.
              </dd>
            </div>
          </dl>

          <p>
            Built with Claude Code as a pairing tool. The task board in the repository records every decision, every mistake and every check, including the ones this page is about.
          </p>
        </div>

        <div className="margin">
          <span className="label">Repository</span>
          <a href={data.meta.repoUrl}>{data.meta.repoUrl.replace(/^https?:\/\//, "")}</a>
          <br />
          <br />
          {contact.length > 0 ? (
            <>
              <span className="label">Contact</span>
              {contact.map((c) => (
                <span key={c.url}>
                  <a href={c.url}>{c.label}</a>
                  <br />
                </span>
              ))}
              <br />
            </>
          ) : null}
          <span className="label">Corrections</span>
          If a number here is wrong, that is a bug. Open an issue on the repository.
          <br />
          <br />
          <span className="label">Provenance</span>
          dataset sha256 <code>{data.meta.dataset.slice(0, 12)}</code>
          <br />
          prompt sha256 <code>{data.meta.promptTemplateSha256.slice(0, 12)}</code>
          <br />
          {int(Object.keys(data.meta.sources).length)} source files hashed in <code>data.json</code>
        </div>
      </div>
    </section>
  );
}
