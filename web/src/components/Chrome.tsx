// Page chrome: the sticky section index (left rail), a section header, and a restrained scroll reveal.
import { motion, useReducedMotion } from "motion/react";
import { useEffect, useState, type ReactNode } from "react";
import { data } from "../lib/data";

export const SECTIONS = [
  { id: "result", no: "00", label: "The result" },
  { id: "engine", no: "01", label: "The engine" },
  { id: "pivot", no: "02", label: "The pivot" },
  { id: "harness", no: "03", label: "The harness" },
  { id: "results", no: "04", label: "The numbers" },
  { id: "close", no: "05", label: "Close" },
] as const;

function useActive(ids: readonly string[]): string {
  const [active, setActive] = useState(ids[0] ?? "");
  useEffect(() => {
    const els = ids.map((id) => document.getElementById(id)).filter((e): e is HTMLElement => e !== null);
    const io = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        const first = visible[0];
        if (first) setActive(first.target.id);
      },
      { rootMargin: "-20% 0px -65% 0px" },
    );
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, [ids]);
  return active;
}

export function Rail() {
  const active = useActive(SECTIONS.map((s) => s.id));
  return (
    <aside className="rail">
      <div>
        <div className="brand">
          AgentForge Eval
          <span>self-hosted inference vs hosted Claude, on HumanEval</span>
        </div>
        <nav aria-label="Sections">
          <ol>
            {SECTIONS.map((s) => (
              <li key={s.id}>
                <a href={`#${s.id}`} aria-current={active === s.id ? "true" : undefined}>
                  <span className="no">{s.no}</span>
                  <span>{s.label}</span>
                </a>
              </li>
            ))}
          </ol>
        </nav>
      </div>
      <div className="foot">
        <a href={data.meta.repoUrl}>source repository</a>
        <br />
        every number on this page is generated from the repo&rsquo;s result files
      </div>
    </aside>
  );
}

export function SectionHead({ id, no, title }: { id: string; no: string; title: ReactNode }) {
  return (
    <div className="sec-head" id={id}>
      <span className="no">{no}</span>
      <h2>{title}</h2>
    </div>
  );
}

/** Fades content in once as it scrolls into view. Used on headings and figures only, never on every element. */
export function Reveal({ children }: { children: ReactNode }) {
  const reduce = useReducedMotion();
  if (reduce) return <>{children}</>;
  return (
    <motion.div data-reveal initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-8% 0px" }} transition={{ duration: 0.5, ease: [0.22, 0.8, 0.2, 1] }}>
      {children}
    </motion.div>
  );
}
