// Every real attempt in ts-bench's earlier runs, one square each. Hollow = not resolved. Rendered from data.facts.tsBench.
import { data } from "../lib/data";
import { int } from "../lib/fmt";

const COLS = 60, CELL = 11, GAP = 3, PITCH = CELL + GAP;

interface RowSpec { label: string; sub: string; attempts: number; planned: number; resolved: number; color: string }

function Grid({ row }: { row: RowSpec }) {
  const rows = Math.ceil(row.planned / COLS);
  const w = COLS * PITCH - GAP;
  const h = rows * PITCH - GAP;
  return (
    <div className="attempts-row">
      <div className="attempts-label">
        <div className="name">{row.label}</div>
        <div className="mono small">
          {row.sub}
        </div>
        <div className="mono res"><b>{int(row.resolved)}</b> of {int(row.attempts)} resolved</div>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} width="100%" role="img" aria-label={`${row.label}: ${row.resolved} of ${row.attempts} attempts resolved`} style={{ maxWidth: w }}>
        {Array.from({ length: row.planned }, (_, i) => {
          const x = (i % COLS) * PITCH, y = Math.floor(i / COLS) * PITCH;
          if (i >= row.attempts) return <rect key={i} x={x} y={y} width={CELL} height={CELL} fill="var(--paper-2)" stroke="var(--rule)" strokeWidth={1} />; // planned, never run
          const solved = i < row.resolved;
          return <rect key={i} x={x + 0.5} y={y + 0.5} width={CELL - 1} height={CELL - 1} fill={solved ? row.color : "none"} stroke={solved ? row.color : "var(--ink-3)"} strokeWidth={1} />;
        })}
      </svg>
    </div>
  );
}

export function AttemptsGrid() {
  const t = data.facts.tsBench;
  const specs: RowSpec[] = [
    ...t.models.map((m) => ({
      label: m.model,
      sub: m.status === "complete" ? "complete" : `partial: ${int(m.attempts)} of ${int(m.planned)} planned`,
      attempts: m.attempts, planned: m.planned, resolved: m.resolved, color: "var(--ink)",
    })),
    { label: "claude-haiku-4.5 (hosted)", sub: "same scaffold, same task set", attempts: t.haiku.attempts, planned: t.haiku.attempts, resolved: t.haiku.resolved, color: "var(--haiku)" },
  ];
  return (
    <figure className="figure">
      <div className="well">
        {specs.map((r) => <Grid key={r.label} row={r} />)}
      </div>
      <figcaption>
        <b>{int(t.totalAttempts)} attempts by local open-weight models, {int(t.totalResolved)} resolved.</b> One square is one real attempt at fixing a real bug in a
        real repository under ts-bench&rsquo;s plain bash-loop scaffold. Hollow means not resolved; shaded grey squares were planned and never run. Source:{" "}
        <code>bench/docs/step7-real-model-run.md</code>, final tallies.
      </figcaption>
    </figure>
  );
}
