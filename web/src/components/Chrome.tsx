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

/** The active section is the last one whose heading has scrolled above 40% of the viewport. */
function useActive(ids: readonly string[]): string {
  const [active, setActive] = useState(ids[0] ?? "");
  useEffect(() => {
    let raf = 0;
    const compute = () => {
      raf = 0;
      let cur = ids[0] ?? "";
      for (const id of ids) {
        const el = document.getElementById(id);
        if (el && el.getBoundingClientRect().top <= window.innerHeight * 0.4) cur = id;
      }
      setActive(cur);
    };
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(compute);
    };
    compute();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
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
