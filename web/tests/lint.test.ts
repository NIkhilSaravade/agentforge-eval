// The two lints must (a) pass on the real site and (b) actually catch what they claim to catch.
import { spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

const SCRIPTS = resolve(__dirname, "../scripts");
const run = (script: string, root?: string) =>
  spawnSync("node", [join(SCRIPTS, script), ...(root ? ["--root", root] : [])], { encoding: "utf8" });
const scratch = (files: Record<string, string>) => {
  const d = mkdtempSync(join(tmpdir(), "web-lint-"));
  for (const [n, c] of Object.entries(files)) writeFileSync(join(d, n), c);
  return d;
};

describe("the real site", () => {
  it("has none of the template-generated visual tells", () => {
    const r = run("lint-tells.mjs");
    expect(r.stderr).toBe("");
    expect(r.status).toBe(0);
  });
  it("has no number typed into a component", () => {
    const r = run("lint-numbers.mjs");
    expect(r.stderr).toBe("");
    expect(r.status).toBe(0);
  });
});

describe("lint-tells catches each pattern", () => {
  const bad: [string, string, string][] = [
    ["glassmorphism", "a { backdrop-filter: blur(8px); }", "blur"],
    ["Inter", "a { font-family: Inter, sans-serif; }", "Inter"],
    ["endless animation", "a { animation: pulse 2s infinite; }", "endless"],
    ["emoji", "const x = 'launch \u{1F680}';", "emoji"],
    ["sparkle icon", "const icon = 'sparkle';", "sparkle"],
  ];
  for (const [name, css, expected] of bad) {
    it(`flags ${name}`, () => {
      const isTs = css.startsWith("const");
      const dir = scratch({ [isTs ? "x.tsx" : "x.css"]: css });
      const r = run("lint-tells.mjs", dir);
      expect(r.status).toBe(1);
      expect(r.stderr).toContain(expected);
    });
  }
  // Rounded cards, shadows, and a blue/purple gradient are this site's design choice, not banned.
  it("accepts rounded corners, shadow, gradient and purple, since the design intentionally uses them", () => {
    const dir = scratch({
      "x.css": "a { border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,.2); background: linear-gradient(90deg, #7c3aed, #2b58a6); color: #7c3aed; }",
    });
    expect(run("lint-tells.mjs", dir).status).toBe(0);
  });
});

describe("lint-numbers catches each way of typing a fact", () => {
  const wrap = (jsx: string) => `export const X = () => (${jsx});`;
  it("flags digits in JSX text", () => {
    const r = run("lint-numbers.mjs", scratch({ "x.tsx": wrap("<p>The model scored 66 percent.</p>") }));
    expect(r.status).toBe(1);
    expect(r.stderr).toContain("digits in JSX text");
  });
  it("flags a numeric literal rendered as a child", () => {
    const r = run("lint-numbers.mjs", scratch({ "x.tsx": wrap("<b>{42}</b>") }));
    expect(r.status).toBe(1);
    expect(r.stderr).toContain("numeric literal");
  });
  it("flags digits in a static string rendered as a child", () => {
    const r = run("lint-numbers.mjs", scratch({ "x.tsx": wrap('<b>{"1,640 samples"}</b>') }));
    expect(r.status).toBe(1);
  });
  it("flags digits in an aria-label", () => {
    const r = run("lint-numbers.mjs", scratch({ "x.tsx": wrap('<svg aria-label="a chart of 164 problems" />') }));
    expect(r.status).toBe(1);
  });
  it("accepts numbers that arrive through expressions, and names that contain digits", () => {
    const ok = wrap("<p>Haiku 4.5 and Opus 5 scored {pct(a.value)} on pass@1 (see step7-real-model-run.md, block C2).</p>");
    expect(run("lint-numbers.mjs", scratch({ "x.tsx": ok })).status).toBe(0);
  });
});
