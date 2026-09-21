// The engine diagram. It renders a REAL recorded trace of engine/scheduler.py + block_manager.py (see
// engine/scripts/record_scheduler_trace.py): every chip position, block owner, token cell and eviction below is read from
// the trace's snapshots and events for the current step. Nothing is scripted or simulated here.
import { motion, useMotionValueEvent, useReducedMotion, useScroll } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { buildModel, describeStep, trace, type Where } from "../lib/trace";

const W = 960;
const H = 470;
const Q_X = 0, Q_Y = 34, Q_PITCH = 50;
const R_X = 250, R_Y = 34, R_PITCH = 96, R_W = 392, R_H = 88;
const P_X = 676, P_Y = 34, B_W = 62, B_H = 72, B_GAP_X = 8, B_GAP_Y = 8, P_COLS = 4;
const CHIP = 38;

const model = buildModel(trace);
const N = model.steps;
const B = trace.config.block_size;
const requestById = new Map(trace.requests.map((r) => [r.id, r]));
const cumulativeTokens = (() => {
  const out: number[] = [];
  let c = 0;
  for (let s = 0; s < N; s++) {
    c += (model.eventsAt[s] ?? []).filter((e) => e.type === "token").length;
    out.push(c);
  }
  return out;
})();
const cumulativeEvictions = (() => {
  const out: number[] = [];
  let c = 0;
  for (let s = 0; s < N; s++) {
    c += (model.eventsAt[s] ?? []).filter((e) => e.type === "preempt").length;
    out.push(c);
  }
  return out;
})();

// where each request's chip should be at each step (carried forward while finished, so it fades out where it stood)
interface Pos { x: number; y: number; opacity: number; kind: Where }
const chipPositions: Map<string, Pos[]> = (() => {
  const all = new Map<string, Pos[]>();
  for (const q of trace.requests) {
    const list: Pos[] = [];
    let last: Pos = { x: -70, y: Q_Y, opacity: 0, kind: "hidden" };
    for (let s = 0; s < N; s++) {
      const w = model.where[s]?.get(q.id) ?? "hidden";
      const st = trace.steps[s]?.state;
      if (w === "running") {
        last = { x: R_X + 8, y: R_Y + (model.rowOf[s]?.get(q.id) ?? 0) * R_PITCH + 8, opacity: 1, kind: w };
      } else if (w === "waiting") {
        const qi = st?.waiting.findIndex((r) => r.id === q.id) ?? 0;
        last = { x: Q_X, y: Q_Y + qi * Q_PITCH + 2, opacity: 1, kind: w };
      } else if (w === "done") last = { ...last, opacity: 0, kind: w };
      else last = { x: Q_X - 70, y: Q_Y, opacity: 0, kind: w };
      list.push(last);
    }
    all.set(q.id, list);
  }
  return all;
})();

const blockXY = (b: number) => ({ x: P_X + (b % P_COLS) * (B_W + B_GAP_X), y: P_Y + Math.floor(b / P_COLS) * (B_H + B_GAP_Y) });
const stepFromProgress = (p: number) => Math.min(N - 1, Math.max(0, Math.round(p * (N - 1))));

export function EngineTrace() {
  const reduce = useReducedMotion() ?? false;
  const wrapRef = useRef<HTMLDivElement>(null);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const { scrollYProgress } = useScroll({ target: wrapRef, offset: ["start start", "end end"] });
  // Scrolling takes over from play/manual control; the last input wins.
  useMotionValueEvent(scrollYProgress, "change", (v) => {
    setPlaying(false);
    setStep(stepFromProgress(v));
  });
  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => {
      setStep((s) => {
        if (s >= N - 1) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, 650);
    return () => clearInterval(t);
  }, [playing]);

  const go = useCallback((s: number) => {
    setPlaying(false);
    setStep(Math.min(N - 1, Math.max(0, s)));
  }, []);
  const nextEviction = () => {
    const nxt = model.preemptSteps.find((s) => s > step) ?? model.preemptSteps[0];
    if (nxt !== undefined) go(nxt);
  };

  const state = trace.steps[step]?.state;
  if (!state) return null;
  const dur = reduce ? 0 : 0.5;
  const evicted = model.evictedBlocks[step] ?? new Map<string, number[]>();
  const touched = model.touchedBlocks[step] ?? new Set<number>();
  const owner = new Map<number, { id: string; idx: number; kv: number }>();
  for (const r of state.running) r.blocks.forEach((b, idx) => owner.set(b, { id: r.id, idx, kv: r.seq_len - 1 }));
  const lines = describeStep(trace, model, step);

  return (
    <div ref={wrapRef} className="scrolly" style={{ height: `${N * 5 + 100}vh` }}>
      <div className="sticky-fig">
        <figure className="figure" style={{ margin: 0 }}>
          <div className="readout mono" aria-hidden="true">
            <span>STEP <b className="num">{step + 1}</b> / <span className="num">{N}</span></span>
            <span>BATCH <b className="num">{state.running.length}</b> / <span className="num">{trace.config.max_batch}</span></span>
            <span>WAITING <b className="num">{state.waiting.length}</b></span>
            <span>BLOCKS USED <b className="num">{state.used_blocks}</b> / <span className="num">{trace.config.num_blocks}</span></span>
            <span className="evict">EVICTIONS <b className="num">{cumulativeEvictions[step]}</b></span>
            <span>TOKENS <b className="num">{cumulativeTokens[step]}</b></span>
          </div>
          <div className="well">
            <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Animated diagram of the scheduler: a waiting queue, a running batch of up to four requests, and a pool of KV-cache blocks. Step-by-step text is below.">
              <defs>
                <pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <line x1="0" y1="0" x2="0" y2="7" stroke="var(--rule)" strokeWidth="1.4" />
                </pattern>
              </defs>

              {/* lane headers */}
              <text x={Q_X} y={14} className="lane-h">WAITING</text>
              <text x={R_X} y={14} className="lane-h">RUNNING BATCH: one decode step for all rows</text>
              <text x={P_X} y={14} className="lane-h">KV BLOCK POOL</text>
              <text x={P_X} y={P_Y + 5 * (B_H + B_GAP_Y) + 10} className="lane-sub">each block holds {B} tokens of keys and values</text>
              <text x={P_X} y={P_Y + 5 * (B_H + B_GAP_Y) + 26} className="lane-sub">label = request + its logical block number</text>

              {/* batch rows */}
              {Array.from({ length: trace.config.max_batch }, (_, r) => (
                <rect key={r} x={R_X} y={R_Y + r * R_PITCH} width={R_W} height={R_H} fill="none" stroke="var(--rule)" />
              ))}
              {state.running.map((r) => {
                const row = model.rowOf[step]?.get(r.id) ?? 0;
                const y0 = R_Y + row * R_PITCH;
                const req = requestById.get(r.id);
                return (
                  <motion.g key={`row-${r.id}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: dur }}>
                    <text x={R_X + 58} y={y0 + 26} className="row-t">{r.id}: prompt {req?.prompt_tokens}, generated {r.generated} of {r.max_new}</text>
                    <rect x={R_X + 58} y={y0 + 38} width={250} height={6} fill="none" stroke="var(--ink)" strokeWidth={1} />
                    <motion.rect x={R_X + 58} y={y0 + 38} height={6} fill="var(--ink)" initial={false} animate={{ width: (250 * r.generated) / r.max_new }} transition={{ duration: dur * 0.6 }} />
                    <text x={R_X + 58} y={y0 + 62} className="row-s">block table: {r.blocks.join(", ")}</text>
                    {r.preempted > 0 && <text x={R_X + 58} y={y0 + 79} className="tag">EVICTED {r.preempted}x EARLIER, RECOMPUTED</text>}
                  </motion.g>
                );
              })}

              {/* waiting details */}
              {state.waiting.map((q, i) => (
                <motion.g key={`wait-${q.id}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: dur }}>
                  <text x={Q_X + 50} y={Q_Y + i * Q_PITCH + 16} className="row-t">{q.id}: prompt {q.prompt}</text>
                  <text x={Q_X + 50} y={Q_Y + i * Q_PITCH + 32} className={q.preempted ? "tag" : "row-s"}>
                    {q.preempted ? `RECOMPUTE ${q.prompt + q.generated} tokens` : "not yet admitted"}
                  </text>
                </motion.g>
              ))}

              {/* KV pool */}
              {Array.from({ length: trace.config.num_blocks }, (_, b) => {
                const { x, y } = blockXY(b);
                const o = owner.get(b);
                const filled = o ? Math.max(0, Math.min(B, o.kv - o.idx * B)) : 0;
                return (
                  <g key={b}>
                    <motion.rect
                      x={x} y={y} width={B_W} height={B_H}
                      initial={false}
                      animate={{ fill: o ? "#f2eee4" : "url(#hatch)", strokeWidth: touched.has(b) ? 3 : 1 }}
                      stroke={o ? "var(--ink)" : "var(--rule)"}
                      transition={{ duration: dur * 0.7 }}
                    />
                    {!o && <rect x={x} y={y} width={B_W} height={B_H} fill="url(#hatch)" />}
                    {o && Array.from({ length: B }, (_, i) => (
                      <motion.rect
                        key={i}
                        x={x + 6 + (i % 2) * 28} y={y + 6 + Math.floor(i / 2) * 28} width={22} height={22}
                        initial={false}
                        animate={{ opacity: i < filled ? 1 : 0.16 }}
                        fill="var(--self)"
                        transition={{ duration: dur * 0.5 }}
                      />
                    ))}
                    <text x={x + 5} y={y + B_H - 6} className="blk-l">{o ? `${o.id}${o.idx + 1}` : ""}</text>
                    <text x={x + B_W - 4} y={y + B_H - 6} textAnchor="end" className="blk-n">{b}</text>
                  </g>
                );
              })}

              {/* eviction ghosts: the blocks the scheduler just threw away, marked once and fading */}
              {[...evicted.entries()].flatMap(([id, bl]) =>
                bl.map((b) => {
                  const { x, y } = blockXY(b);
                  return (
                    <motion.g key={`ghost-${step}-${id}-${b}`} initial={{ opacity: 1 }} animate={{ opacity: 0 }} transition={{ duration: reduce ? 0 : 0.9, delay: reduce ? 0 : 0.35 }}>
                      <rect x={x - 1} y={y - 1} width={B_W + 2} height={B_H + 2} fill="none" stroke="var(--self)" strokeWidth={3} />
                      <line x1={x + 4} y1={y + 4} x2={x + B_W - 4} y2={y + B_H - 4} stroke="var(--self)" strokeWidth={2} />
                      <line x1={x + B_W - 4} y1={y + 4} x2={x + 4} y2={y + B_H - 4} stroke="var(--self)" strokeWidth={2} />
                    </motion.g>
                  );
                }),
              )}

              {/* request chips: these travel between the queue, the batch rows and back after an eviction */}
              {trace.requests.map((q) => {
                const p = chipPositions.get(q.id)?.[step];
                if (!p) return null;
                const running = p.kind === "running";
                const reQueued = p.kind === "waiting" && (state.waiting.find((w) => w.id === q.id)?.preempted ?? 0) > 0;
                return (
                  <motion.g key={q.id} initial={false} animate={{ x: p.x, y: p.y, opacity: p.opacity }} transition={{ duration: dur, ease: [0.22, 0.8, 0.2, 1] }}>
                    <rect width={CHIP} height={CHIP} fill={running ? "var(--ink)" : "var(--paper)"} stroke={reQueued ? "var(--self)" : "var(--ink)"} strokeWidth={reQueued ? 3 : 1.5} />
                    <text x={CHIP / 2} y={CHIP / 2 + 8} textAnchor="middle" className="chip-t" fill={running ? "var(--paper)" : "var(--ink)"}>{q.id}</text>
                  </motion.g>
                );
              })}
            </svg>

            <Timeline step={step} onScrub={go} />
          </div>

          <div className="controls">
            <button className="btn" onClick={() => go(step - 1)} disabled={step === 0}>Prev</button>
            <button className="btn" onClick={() => go(step + 1)} disabled={step === N - 1}>Next</button>
            <button className="btn" aria-pressed={playing} onClick={() => (step >= N - 1 ? (setStep(0), setPlaying(true)) : setPlaying((p) => !p))}>{playing ? "Pause" : "Play"}</button>
            <button className="btn" onClick={nextEviction}>Next eviction</button>
            <span className="label" style={{ marginLeft: "auto" }}>scroll drives this figure</span>
          </div>

          <div className="step-text" role="status" aria-live="off">
            {lines.map((l, i) => <p key={i} className={l.includes("evicted") ? "evict" : undefined}>{l}</p>)}
          </div>
        </figure>
      </div>
    </div>
  );
}

function Timeline({ step, onScrub }: { step: number; onScrub: (s: number) => void }) {
  const pos = (s: number) => `${(s / (N - 1)) * 100}%`;
  const ticks = useMemo(() => ({ arrive: model.arriveSteps, finish: model.finishSteps, evict: model.preemptSteps }), []);
  return (
    <div className="timeline">
      <div className="ticks" aria-hidden="true">
        {ticks.arrive.map((s) => <i key={`a${s}`} className="t-arrive" style={{ left: pos(s) }} />)}
        {ticks.finish.map((s) => <i key={`f${s}`} className="t-finish" style={{ left: pos(s) }} />)}
        {ticks.evict.map((s) => <i key={`e${s}`} className="t-evict" style={{ left: pos(s) }} />)}
        <i className="t-now" style={{ left: pos(step) }} />
      </div>
      <input type="range" min={0} max={N - 1} value={step} onChange={(e) => onScrub(Number(e.target.value))} aria-label="Scheduler step" />
      <div className="tick-key mono">
        <span><i className="k-arrive" /> arrival</span>
        <span><i className="k-finish" /> finish</span>
        <span className="evict"><i className="k-evict" /> eviction</span>
      </div>
    </div>
  );
}
