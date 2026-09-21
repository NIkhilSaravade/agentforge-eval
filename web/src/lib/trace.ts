// Turns the recorded scheduler trace (engine/results/scheduler_trace.json) into what the diagram needs.
// Everything is derived from the trace's own events and snapshots; nothing is invented here.
import { trace as rawTrace } from "./data";

export interface RunningReq { id: string; seq_len: number; generated: number; max_new: number; blocks: number[]; preempted: number }
export interface WaitingReq { id: string; prompt: number; generated: number; max_new: number; preempted: number }
export interface StepState { running: RunningReq[]; waiting: WaitingReq[]; free_blocks: number[]; used_blocks: number; preemptions_total: number }
export interface TraceEvent { step: number; type: string; id?: string; blocks?: number[]; block?: number; recompute?: boolean; tokens?: number; generated?: number; kv_tokens_discarded?: number; prompt?: number; new_tokens?: number; n?: number; finished?: boolean; reason?: string }
export interface TraceReq { id: string; prompt_tokens: number; new_tokens: number; arrives_at_step: number }
export interface Trace {
  config: { model: string; backend: string; batching: string; max_batch: number; block_size: number; num_blocks: number; kv_budget_mib: number; preemption: boolean };
  requests: TraceReq[];
  steps: { step: number; arrived: string[]; state: StepState }[];
  events: TraceEvent[];
  totals: { steps: number; preemptions: number; tokens_generated: number };
  provenance: Record<string, string>;
}
export const trace = rawTrace as unknown as Trace;

export type Where = "hidden" | "waiting" | "running" | "done";

export interface Model {
  steps: number;
  rowOf: Map<string, number>[];               // per step: request id -> batch row while running
  where: Map<string, Where>[];                // per step: request id -> where it is after that step
  eventsAt: TraceEvent[][];                   // per step
  evictedBlocks: Map<string, number[]>[];     // per step: request id -> blocks discarded by a preemption in that step
  touchedBlocks: Set<number>[];               // per step: blocks newly allocated or grown into
  preemptSteps: number[];
  finishSteps: number[];
  arriveSteps: number[];
}

export function buildModel(t: Trace): Model {
  const n = t.steps.length;
  const eventsAt: TraceEvent[][] = Array.from({ length: n }, () => []);
  for (const e of t.events) eventsAt[e.step]?.push(e);

  const rowOf: Map<string, number>[] = [];
  const where: Map<string, Where>[] = [];
  let rows = new Map<string, number>();
  const seen = new Set<string>();
  for (let s = 0; s < n; s++) {
    const st = t.steps[s]?.state;
    if (!st) throw new Error("trace has a hole");
    const next = new Map<string, number>();
    for (const r of st.running) if (rows.has(r.id)) next.set(r.id, rows.get(r.id) as number);
    for (const r of st.running) {
      if (next.has(r.id)) continue;
      let row = 0;
      while ([...next.values()].includes(row)) row++;
      next.set(r.id, row);
    }
    rows = next;
    rowOf.push(new Map(next));
    const w = new Map<string, Where>();
    for (const q of t.requests) {
      const arrived = q.arrives_at_step <= s;
      if (arrived) seen.add(q.id);
      if (st.running.some((r) => r.id === q.id)) w.set(q.id, "running");
      else if (st.waiting.some((r) => r.id === q.id)) w.set(q.id, "waiting");
      else w.set(q.id, arrived ? "done" : "hidden");
    }
    where.push(w);
  }

  const evictedBlocks = eventsAt.map((evs) => {
    const m = new Map<string, number[]>();
    for (const e of evs) if (e.type === "preempt" && e.id && e.blocks) m.set(e.id, e.blocks);
    return m;
  });
  const touchedBlocks = eventsAt.map((evs) => {
    const s = new Set<number>();
    for (const e of evs) {
      if (e.type === "allocate") for (const b of e.blocks ?? []) s.add(b);
      if (e.type === "grow" && e.block !== undefined) s.add(e.block);
    }
    return s;
  });
  const stepsWith = (pred: (e: TraceEvent) => boolean) => eventsAt.flatMap((evs, s) => (evs.some(pred) ? [s] : []));
  return {
    steps: n, rowOf, where, eventsAt, evictedBlocks, touchedBlocks,
    preemptSteps: stepsWith((e) => e.type === "preempt"),
    finishSteps: stepsWith((e) => e.type === "free" && e.reason === "finished"),
    arriveSteps: stepsWith((e) => e.type === "arrive"),
  };
}

const plural = (n: number, one: string, many: string) => (n === 1 ? one : many);

/** Plain-language account of one scheduler step, built only from that step's recorded events. */
export function describeStep(t: Trace, m: Model, s: number): string[] {
  const out: string[] = [];
  const evs = m.eventsAt[s] ?? [];
  for (const e of evs) {
    if (e.type === "arrive") out.push(`${e.id} arrives: ${e.prompt} prompt ${plural(e.prompt ?? 0, "token", "tokens")}, ${e.new_tokens} to generate.`);
  }
  for (const e of evs) {
    if (e.type === "preempt") {
      out.push(
        `The pool is full and ${e.id} cannot get the next block it needs. ${e.id} is evicted: its ${e.kv_tokens_discarded} tokens of cache are discarded and it goes back to the queue with ${e.generated} tokens already generated.`,
      );
    }
  }
  for (const e of evs) {
    if (e.type === "allocate") {
      const nb = e.blocks?.length ?? 0;
      out.push(`${e.id} is admitted and given ${nb} ${plural(nb, "block", "blocks")}.${e.recompute ? " It was evicted earlier, so nothing was kept." : ""}`);
    }
    if (e.type === "prefill") {
      out.push(e.recompute ? `${e.id} recomputes its cache: ${e.tokens} tokens (prompt plus what it had generated) go back through the model.` : `${e.id} prefills its ${e.tokens}-token prompt.`);
    }
  }
  const grows = evs.filter((e) => e.type === "grow");
  if (grows.length) out.push(`${grows.map((g) => g.id).join(", ")} ${plural(grows.length, "crosses", "cross")} a block boundary and ${plural(grows.length, "is", "are")} given a new block.`);
  for (const e of evs) {
    if (e.type === "free" && e.reason === "finished") out.push(`${e.id} finishes and frees ${e.blocks?.length} ${plural(e.blocks?.length ?? 0, "block", "blocks")}.`);
  }
  const toks = evs.filter((e) => e.type === "token").length;
  const st = t.steps[s]?.state;
  if (st) out.push(`${st.running.length} of ${t.config.max_batch} batch rows running; ${toks} ${plural(toks, "token", "tokens")} produced this step.`);
  return out;
}
