// The site's integrity claims, as tests. data.json must be (1) fresh, (2) consistent with the docs a human reads
// (the task board and the README), and (3) internally consistent.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import data from "../src/data.json";

const ROOT = resolve(__dirname, "../..");
const board = readFileSync(join(ROOT, "docs/agentforge-task-board.md"), "utf8");
const readme = readFileSync(join(ROOT, "README.md"), "utf8");
const arm = (id: string) => {
  const a = data.arms.find((x) => x.id === id);
  if (!a) throw new Error(`no arm ${id}`);
  return a;
};
const pct = (v: number, dp: number) => (v * 100).toFixed(dp);

describe("data.json is fresh", () => {
  it("regenerating from the result files reproduces the committed file byte for byte", () => {
    const out = join(mkdtempSync(join(tmpdir(), "web-data-")), "data.json");
    execFileSync("node", [resolve(__dirname, "../scripts/build-data.mjs"), "--out", out], { stdio: "pipe" });
    expect(readFileSync(out, "utf8")).toBe(readFileSync(resolve(__dirname, "../src/data.json"), "utf8"));
  });
});

describe("results match the task board (Phase 6 table)", () => {
  const rowFor = (prefix: string) => {
    // the board also has cost-estimate rows with the same arm names; the results row is the one with percentage cells
    const line = board.split("\n").find((l) => l.startsWith(`| ${prefix}`) && /\*\*\d+\.\d+%\*\*/.test(l));
    expect(line, `board results row starting "| ${prefix}"`).toBeTruthy();
    return line as string;
  };
  const cases: [string, string][] = [
    ["Self-hosted Qwen2.5-Coder-1.5B, sampled", "self"],
    ["Self-hosted Qwen2.5-Coder-1.5B, greedy", "greedy"],
    ["Claude Haiku 4.5", "haiku"],
    ["Claude Opus 5 (thinking off)", "opus"],
  ];
  for (const [prefix, id] of cases) {
    it(`${id}: pass@1 and its CI equal the board's`, () => {
      const row = rowFor(prefix);
      const p1 = row.match(/\*\*(\d+\.\d+)%\*\*\s*\[(\d+\.\d+), (\d+\.\d+)\]/);
      expect(p1, row).toBeTruthy();
      const a = arm(id);
      expect(p1?.[1]).toBe(pct(a.pass1.value, 2));
      expect(p1?.[2]).toBe(pct(a.pass1.lo, 1)); // the board prints CIs to 1 decimal
      expect(p1?.[3]).toBe(pct(a.pass1.hi, 1));
    });
  }
  it("pass@3 for the three multi-sample arms equals the board's", () => {
    for (const [prefix, id] of [cases[0], cases[2], cases[3]] as [string, string][]) {
      const row = rowFor(prefix);
      const all = [...row.matchAll(/\*\*(\d+\.\d+)%\*\*/g)].map((m) => m[1]);
      expect(all[1], `${id} pass@3 in board row`).toBe(pct(arm(id).pass3?.value ?? NaN, 2));
    }
  });
  it("Haiku and Opus cost lines equal the board's", () => {
    expect(board).toContain(`**$${arm("haiku").cost?.usdTotal.toFixed(4)}**`);
    expect(board).toContain(`**$${arm("opus").cost?.usdTotal.toFixed(4)}**`);
    expect(board).toContain(`$${data.spend.totalUsd.toFixed(4)}`);
  });
  it("break-even rates equal the board's", () => {
    const f = (n: number) => n.toFixed(2);
    expect(board).toContain(`$${f(data.costModel.breakEven.haiku.perSample)}/h per sample`);
    expect(board).toContain(`**$${f(data.costModel.breakEven.opus.perSample)}/h per sample**`);
  });
});

describe("results match the README table", () => {
  const cell = (v: { value: number; lo: number; hi: number }) => `${pct(v.value, 2)}% [${pct(v.lo, 2)}, ${pct(v.hi, 2)}]`;
  it("every pass@k cell in the README results table is data.json's value", () => {
    for (const id of ["self", "greedy", "haiku", "opus"]) {
      const a = arm(id);
      for (const v of [a.pass1, a.pass3, a.pass10]) if (v) expect(readme, `${id}`).toContain(cell(v));
    }
  });
  it("cost, wall-clock and total spend", () => {
    expect(readme).toContain(`$${arm("haiku").cost?.usdTotal.toFixed(6)}`);
    expect(readme).toContain(`$${arm("opus").cost?.usdTotal.toFixed(5)}`);
    expect(readme).toContain(`$${data.spend.totalUsd.toFixed(6)}`);
  });
});

describe("internal consistency", () => {
  it("every arm has all 164 problems and sample counts that add up", () => {
    for (const a of data.arms) {
      const strip = (data.strips as Record<string, { n: number; c: number }[]>)[a.id];
      expect(strip).toHaveLength(164);
      expect(strip.reduce((s, e) => s + e.n, 0)).toBe(a.nSamples);
      expect(strip.reduce((s, e) => s + e.c, 0)).toBe(a.passedSamples);
    }
  });
  it("the self-hosted arm is behind Haiku and Opus, and the data says so", () => {
    expect(arm("self").pass1.value).toBeLessThan(arm("haiku").pass1.value);
    expect(arm("haiku").pass1.value).toBeLessThan(arm("opus").pass1.value);
    expect(arm("self").pass1.hi).toBeLessThan(arm("haiku").pass1.lo); // the intervals do not overlap
  });
  it("the ts-bench evidence is what the doc says", () => {
    expect(data.facts.tsBench.totalResolved).toBe(0);
    expect(data.facts.tsBench.models.reduce((s, m) => s + m.attempts, 0)).toBe(data.facts.tsBench.totalAttempts);
  });
  it("hosted spend is under the cap, which is under the funded amount", () => {
    expect(data.spend.totalUsd).toBeLessThan(data.spend.capUsd);
    expect(data.spend.capUsd).toBeLessThan(data.spend.fundedUsd);
  });
});
