// Number-provenance lint: no fact may be typed into a component. Parses every .tsx with the TypeScript AST and fails on
//   - digits in JSX text                      <p>the model scored 66 percent</p>
//   - a numeric literal rendered as a child   <b>{42}</b>
//   - digits in a static string rendered as a child, or in aria-label / title / alt
// Numbers reach the page only through expressions over data.json (e.g. {pct(a.pass1.value)}).
// Allowed: names and labels that contain a digit but are not facts (model names, "pass@1", "GPT-2", "sha256").
//   node scripts/lint-numbers.mjs [--root <dir>]
import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "ts5";  // TypeScript 5 is installed under this alias only for its JS compiler API; the project type-checks with TypeScript 7

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const rootArg = process.argv.indexOf("--root");
const ROOT = rootArg > 0 ? resolve(process.argv[rootArg + 1]) : resolve(WEB, "src");

const ALLOWED = [
  /[\w.-]+\/[\w./-]+/g, /[\w-]+\.(?:md|json|mjs|py|tsx?)\b/g, /\b[A-H]\d\b/g, // file paths; diagram block labels such as C2 (names, not facts)
  /pass@\d+/gi, /Haiku 4\.5/g, /Opus 5/g, /Sonnet 5/g, /Qwen2\.5-Coder-1\.5B/gi, /Qwen2/g, /GPT-2/g, /sha256/gi, /qwen\d[\w.-]*/gi, /\bv\d+\b/g,
];
const hasFact = (text) => {
  let t = text;
  for (const re of ALLOWED) t = t.replace(re, " ");
  return /\d/.test(t);
};

const walk = (d) =>
  readdirSync(d).flatMap((n) => {
    const p = join(d, n);
    return statSync(p).isDirectory() ? walk(p) : extname(p) === ".tsx" ? [p] : [];
  });

const violations = [];
for (const file of walk(ROOT)) {
  const src = readFileSync(file, "utf8");
  const sf = ts.createSourceFile(file, src, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TSX);
  const at = (node) => sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1;
  const report = (node, msg) => violations.push(`${file}:${at(node)}  ${msg}`);
  const staticText = (e) => {
    if (ts.isStringLiteral(e) || ts.isNoSubstitutionTemplateLiteral(e)) return e.text;
    if (ts.isTemplateExpression(e)) return [e.head.text, ...e.templateSpans.map((s) => s.literal.text)].join(" ");
    return null;
  };
  const visit = (node) => {
    if (ts.isJsxText(node)) {
      const t = node.text.replace(/\s+/g, " ").trim();
      if (t && hasFact(t)) report(node, `digits in JSX text: "${t.slice(0, 60)}"`);
    } else if (ts.isJsxExpression(node) && node.expression && (ts.isJsxElement(node.parent) || ts.isJsxFragment(node.parent))) {
      const e = node.expression;
      if (ts.isNumericLiteral(e)) report(node, `numeric literal rendered as a child: {${e.text}}`);
      const st = staticText(e);
      if (st !== null && hasFact(st)) report(node, `digits in a static string rendered as a child: "${st.slice(0, 60)}"`);
    } else if (ts.isJsxAttribute(node) && node.initializer && /^(aria-label|title|alt|placeholder)$/.test(node.name.getText(sf))) {
      const init = node.initializer;
      const e = ts.isJsxExpression(init) ? init.expression : init;
      const st = e ? staticText(e) : null;
      if (st !== null && hasFact(st)) report(node, `digits in ${node.name.getText(sf)}: "${st.slice(0, 60)}"`);
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
}

if (violations.length) {
  console.error(`lint-numbers: ${violations.length} violation(s)\n` + violations.map((v) => "  " + v).join("\n"));
  process.exit(1);
}
console.log(`lint-numbers: clean (${walk(ROOT).length} components scanned)`);
