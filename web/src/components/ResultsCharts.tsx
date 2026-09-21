// The results visuals. Every plotted value comes from data.json. Design follows the dataviz rules: one axis, thin marks,
// solid hairline grid, direct labels only on the values that matter, a legend whenever there is more than one series,
// a table view of the same numbers next to it, and series color never used for text.
import { useState } from "react";
import { arm, CONF, data, need, SERIES, type ArmId } from "../lib/data";
import { dec, int, pct, usd } from "../lib/fmt";
import { HEX, RAMP, rampStep } from "../lib/palette";

const ORDER: ArmId[] = ["self", "greedy", "haiku", "opus"];

export function Legend() {
  return (
    <div className="legend mono" role="list" aria-label="Series">
      {ORDER.map((id) => {
        const a = arm(id);
        return (
          <span key={id} role="listitem">
            <i className={`swatch${a.marker === "hollow" ? " hollow" : ""}`} style={{ background: SERIES[id], color: HEX[id] }} />
            {a.short}
          </span>
        );
      })}
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ interval chart
const W = 940, ROW = 50, TOP = 46, LBL = 190, GAP = 40;
const PW = (W - LBL - GAP - 96) / 2; // leaves room on the right for the value labels
const px = (panel: 0 | 1, v: number) => LBL + panel * (PW + GAP) + v * PW;

export function IntervalChart() {
  const [hover, setHover] = useState<string | null>(null);
  const H = TOP + ORDER.length * ROW + 34;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return (
    <figure className="figure">
      <Legend />
      <div className="well">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`pass@1 and pass@3 with ${CONF} intervals for each arm. The same numbers are in the table above.`}>
          {([0, 1] as const).map((panel) => (
            <g key={panel}>
              <text x={px(panel, 0)} y={18} className="lane-h">{panel === 0 ? "PASS@1" : "PASS@3"} &middot; {CONF} INTERVAL</text>
              {ticks.map((t) => (
                <g key={t}>
                  <line x1={px(panel, t)} y1={TOP - 10} x2={px(panel, t)} y2={TOP + ORDER.length * ROW - 6} stroke="var(--rule)" strokeWidth={1} />
                  <text x={px(panel, t)} y={TOP + ORDER.length * ROW + 12} textAnchor="middle" className="lane-sub">{pct(t, 0)}</text>
                </g>
              ))}
            </g>
          ))}
          {ORDER.map((id, i) => {
            const a = arm(id);
            const y = TOP + i * ROW + 14;
            const hollow = a.marker === "hollow";
            return (
              <g key={id} onMouseEnter={() => setHover(id)} onMouseLeave={() => setHover(null)} opacity={hover && hover !== id ? 0.45 : 1}>
                <text x={0} y={y + 4} className="row-t">{a.short}</text>
                {([0, 1] as const).map((panel) => {
                  const v = panel === 0 ? a.pass1 : a.pass3;
                  if (!v) return <text key={panel} x={px(panel, 0)} y={y + 4} className="lane-sub">not measured (one sample per problem)</text>;
                  const stroke = SERIES[id];
                  return (
                    <g key={panel}>
                      <title>{`${a.short}, pass@${panel === 0 ? 1 : 3}: ${pct(v.value)} (${CONF} interval ${pct(v.lo)} to ${pct(v.hi)})`}</title>
                      <line x1={px(panel, v.lo)} y1={y} x2={px(panel, v.hi)} y2={y} stroke={stroke} strokeWidth={2} />
                      <line x1={px(panel, v.lo)} y1={y - 5} x2={px(panel, v.lo)} y2={y + 5} stroke={stroke} strokeWidth={2} />
                      <line x1={px(panel, v.hi)} y1={y - 5} x2={px(panel, v.hi)} y2={y + 5} stroke={stroke} strokeWidth={2} />
                      <circle cx={px(panel, v.value)} cy={y} r={hover === id ? 8 : 6.5} fill={hollow ? "var(--paper-2)" : stroke} stroke={hollow ? stroke : "var(--paper-2)"} strokeWidth={hollow ? 2.5 : 2} />
                      <text x={px(panel, v.hi) + 9} y={y + 4} className="val">{pct(v.value)}</text>
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>
      <figcaption>
        <b>Full axes, {pct(0, 0)} to {pct(1, 0)}.</b> A dot is the estimate, the bar is the {CONF} interval over problems (the sampling unit for &ldquo;a different set of problems&rdquo;); it does not
        include seed-to-seed variance. Opus 5 has {int(arm("opus").samplesPerProblem[0] ?? 0)} samples per problem, and its failures are all-or-nothing per problem, so its pass@1 and pass@3
        are identical. The hollow marker is the same self-hosted model under greedy decoding.
      </figcaption>
    </figure>
  );
}

// ------------------------------------------------------------------------------------------------ per-problem strips
const COLS = 41;
export function Strips() {
  const strips = data.strips as Record<string, { task_id: string; n: number; c: number }[]>;
  return (
    <figure className="figure">
      <Legend />
      <div className="well strips">
        {ORDER.map((id) => {
          const a = arm(id);
          const s = strips[id];
          if (!s) return null;
          const never = s.filter((e) => e.c === 0).length;
          const always = s.filter((e) => e.c === e.n).length;
          const some = s.length - never - always;
          return (
            <div className="strip" key={id}>
              <div className="strip-h">
                <span className="row-name"><i className={`swatch${a.marker === "hollow" ? " hollow" : ""}`} style={{ background: SERIES[id], color: HEX[id] }} />{a.short}</span>
                <span className="mono small">
                  never solved <b>{int(never)}</b> &middot; sometimes <b>{int(some)}</b> &middot; every sample <b>{int(always)}</b>
                </span>
              </div>
              <div className="cells" style={{ gridTemplateColumns: `repeat(${COLS}, 1fr)` }}>
                {s.map((e) => {
                  const step = rampStep(e.c, e.n);
                  return (
                    <i
                      key={e.task_id}
                      className={step === null ? "cell hollow" : "cell"}
                      style={step === null ? undefined : { background: RAMP[id][step] }}
                      title={`${e.task_id}: ${e.c} of ${e.n} samples passed`}
                    />
                  );
                })}
              </div>
            </div>
          );
        })}
        <div className="ramp-key mono">
          <span><i className="cell hollow" /> no sample passed</span>
          <span><i className="cell" style={{ background: RAMP.self[0] }} /> under half</span>
          <span><i className="cell" style={{ background: RAMP.self[1] }} /> half or more</span>
          <span><i className="cell" style={{ background: RAMP.self[2] }} /> every sample</span>
        </div>
      </div>
      <figcaption>
        <b>One cell per problem, in problem order.</b> Shade shows how many of that problem&rsquo;s samples passed, in each arm&rsquo;s own hue. Hover a cell for the problem and count.
      </figcaption>
    </figure>
  );
}

// ------------------------------------------------------------------------------------------------ break-even
type Mode = "sample" | "solved";
const BW = 940, BH = 380, BL = 64, BR = 150, BT = 24, BB = 46;

export function BreakEven() {
  const cm = data.costModel;
  const haiku = arm("haiku");
  const opus = arm("opus");
  const self = arm("self");
  const hours = cm.selfHostedHoursFor1640;
  const [mode, setMode] = useState<Mode>("sample");
  const [rate, setRate] = useState<number>(Math.round(cm.breakEven.haiku.perSample * 100) / 100);
  const xMax = Math.ceil(cm.breakEven.opus.perSample * 1.3);
  const passedSelf = self.passedSamples;
  // y in dollars: per the 1,640 samples compared, or per 1,000 solved samples
  const k = mode === "sample" ? 1 : 1000 / passedSelf;
  const hostedY = (id: "haiku" | "opus") =>
    mode === "sample" ? cm.hostedCostFor1640[id] : need(id === "haiku" ? haiku.cost : opus.cost, "hosted cost").usdPerSolved * 1000;
  const selfY = (r: number) => r * hours * k;
  const be = (id: "haiku" | "opus") => (mode === "sample" ? cm.breakEven[id].perSample : cm.breakEven[id].perSolved);
  const rawMax = Math.max(hostedY("opus"), hostedY("haiku")) * 1.15;
  const step = [1, 2, 2.5, 5, 10, 20].find((s) => s * 4 >= rawMax) ?? 20; // a "nice" tick step, so ticks are round numbers
  const yMax = step * 4;
  const sx = (v: number) => BL + (v / xMax) * (BW - BL - BR);
  const sy = (v: number) => BT + (1 - v / yMax) * (BH - BT - BB);
  const yTicks = [0, 1, 2, 3, 4].map((i) => i * step);
  const xTicks = Array.from({ length: xMax + 1 }, (_, i) => i);
  const unit = mode === "sample" ? `for the same ${int(cm.samplesCompared)} samples` : "per 1,000 solved samples";
  return (
    <figure className="figure">
      <div className="controls" style={{ marginTop: 0, marginBottom: "0.7rem" }}>
        <span className="label">cost of</span>
        <button className="btn" aria-pressed={mode === "sample"} onClick={() => setMode("sample")}>every sample</button>
        <button className="btn" aria-pressed={mode === "solved"} onClick={() => setMode("solved")}>solved samples</button>
      </div>
      <div className="well">
        <svg viewBox={`0 0 ${BW} ${BH}`} width="100%" role="img" aria-label="Cost of self-hosting as a function of the machine's hourly cost, against the hosted models' measured costs.">
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={BL} y1={sy(t)} x2={BW - BR} y2={sy(t)} stroke="var(--rule)" strokeWidth={1} />
              <text x={BL - 8} y={sy(t) + 4} textAnchor="end" className="lane-sub">{usd(t, Number.isInteger(t) ? 0 : 1)}</text>
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={t} x={sx(t)} y={BH - BB + 18} textAnchor="middle" className="lane-sub">{usd(t, 0)}</text>
          ))}
          <text x={(BL + BW - BR) / 2} y={BH - 6} textAnchor="middle" className="lane-h">ASSUMED HOURLY COST OF THE SELF-HOSTED MACHINE ($ PER HOUR)</text>
          <text x={BL} y={14} className="lane-h">{usd(0, 0)} = FREE &middot; {unit.toUpperCase()}</text>

          {(["haiku", "opus"] as const).map((id) => (
            <g key={id}>
              <line x1={BL} y1={sy(hostedY(id))} x2={BW - BR} y2={sy(hostedY(id))} stroke={SERIES[id]} strokeWidth={2} />
              <text x={BW - BR + 8} y={sy(hostedY(id)) + 4} className="row-t">{id === "haiku" ? "Haiku 4.5" : "Opus 5"} {usd(hostedY(id), mode === "sample" ? 2 : 2)}</text>
              <circle cx={sx(be(id))} cy={sy(hostedY(id))} r={6.5} fill="var(--paper-2)" stroke={SERIES.self} strokeWidth={3} />
              <text x={sx(be(id)) + 11} y={sy(hostedY(id)) + 22} className="val">{usd(be(id))}/h break-even</text>
            </g>
          ))}
          <line x1={sx(0)} y1={sy(0)} x2={sx(xMax)} y2={sy(selfY(xMax))} stroke={SERIES.self} strokeWidth={2.5} />
          <text x={BW - BR + 8} y={sy(selfY(xMax)) + 4} className="row-t">Self-hosted</text>
          <line x1={sx(rate)} y1={BT} x2={sx(rate)} y2={BH - BB} stroke="var(--ink)" strokeWidth={1.5} />
          <circle cx={sx(rate)} cy={sy(selfY(rate))} r={5} fill="var(--ink)" />
        </svg>
        <div className="be-slider">
          <label htmlFor="rate" className="label">If the machine costs</label>
          <input id="rate" type="range" min={0} max={xMax} step={0.01} value={rate} onChange={(e) => setRate(Number(e.target.value))} />
          <span className="mono be-read">
            <b>{usd(rate)}</b> an hour: self-hosted <b>{usd(selfY(rate))}</b> &middot; Haiku 4.5 <b>{usd(hostedY("haiku"))}</b> &middot; Opus 5 <b>{usd(hostedY("opus"))}</b>
          </span>
        </div>
      </div>
      <figcaption>
        <b>No dollar figure is given for the self-hosted run: no hourly rate was supplied, and none is assumed.</b> The line is {dec(hours)}&nbsp;h of measured wall-clock times the rate you choose. Circles mark the break-even:
        below that hourly cost, self-hosting the same samples is cheaper. Haiku 4.5 and Opus 5 costs are measured from token usage;{" "}
        {mode === "sample" ? `the Opus 5 line is extrapolated from its measured per-sample cost to the same ${int(cm.samplesCompared)} samples (it ran ${int(opus.nSamples)}).` : "the per-solved figures divide each arm’s cost by its own passing samples."}
        {" "}Solved samples: {int(self.passedSamples)} self-hosted, {int(haiku.passedSamples)} Haiku 4.5, {int(opus.passedSamples)} Opus 5.
      </figcaption>
    </figure>
  );
}
