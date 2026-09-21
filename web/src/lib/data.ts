// The one place the site touches data. Every component reads from here; nothing else holds a fact.
import raw from "../data.json";

export const data = raw;
export type Data = typeof raw;
export type ArmId = "self" | "greedy" | "haiku" | "opus";
export type Arm = Data["arms"][number];

export const arm = (id: ArmId): Arm => {
  const a = data.arms.find((x) => x.id === id);
  if (!a) throw new Error(`data.json has no arm "${id}"`);
  return a;
};

export const SERIES: Record<ArmId, string> = {
  self: "var(--self)",
  greedy: "var(--self)", // same entity as "self": same hue, hollow marker
  haiku: "var(--haiku)",
  opus: "var(--opus)",
};

/** A pass@k estimate. pass1 always exists; pass3 exists for every arm except greedy; pass10 for three of four. */
export const need = <T,>(v: T | null | undefined, what: string): T => {
  if (v === null || v === undefined) throw new Error(`data.json is missing ${what}`);
  return v;
};

/** The confidence level of every interval on the page, read from the code that computes them (alpha in report.py). */
export const CONF = `${Math.round(raw.bootstrap.confidence * 100)}%`;

export const trace = need(data.trace, "the scheduler trace (run engine/scripts/record_scheduler_trace.py)");
