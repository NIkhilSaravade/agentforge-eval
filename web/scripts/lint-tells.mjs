// Mechanical check for the visual patterns that make a site read as template-generated. Fails (exit 1) on any hit.
//   node scripts/lint-tells.mjs [--root <dir>]      (default: ../src, plus ../index.html and ../package.json)
// Judgement calls that cannot be linted (a centred hero with two pill buttons, a repeated icon-heading-paragraph grid,
// a generic type hierarchy) are checked by looking at the screenshots; see docs/agentforge-task-board.md (W5).
import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const rootArg = process.argv.indexOf("--root");
const ROOT = rootArg > 0 ? resolve(process.argv[rootArg + 1]) : resolve(WEB, "src");
const EXTRA = rootArg > 0 ? [] : [resolve(WEB, "index.html")];

const walk = (d) =>
  readdirSync(d).flatMap((n) => {
    const p = join(d, n);
    if (statSync(p).isDirectory()) return walk(p);
    return [".css", ".tsx", ".ts", ".html"].includes(extname(p)) && !p.endsWith("data.json") ? [p] : [];
  });

// Rounded cards, soft shadows and a blue/purple accent gradient are an intentional design choice on
// this site (Clerk-style alternating light/dark sections with bordered cards) - not banned. Still banned:
// glassmorphism blur, sparkle/emoji iconography, non-brand typefaces, endless/bouncy animation, decorative
// canvas/WebGL/particle effects.
const RULES = [
  ["glass/blur", /backdrop-filter|filter\s*:\s*blur|\bblur\(/, "blur or glassmorphism"],
  ["sparkle", /sparkle|✨|magic[-_ ]?wand/i, "sparkle / AI-magic icon"],
  ["Inter", /font-family\s*:[^;{}]*\bInter\b/i, "Inter as a typeface"],
  ["endless animation", /animation[^;{}]*infinite|repeat\s*:\s*Infinity|\bbounce\b/i, "endless or bouncy animation"],
  ["webgl/canvas", /WebGL|getContext\(\s*["']webgl|<canvas\b|\bthree\b\s+from/i, "WebGL / canvas (only allowed when it is the content)"],
  ["lottie/particles", /lottie|particles/i, "Lottie or particle effects"],
];

const violations = [];
const lineOf = (text, idx) => text.slice(0, idx).split("\n").length;
for (const f of [...walk(ROOT), ...EXTRA]) {
  const text = readFileSync(f, "utf8");
  for (const [name, re, msg] of RULES) {
    for (const m of text.matchAll(new RegExp(re.source, re.flags.includes("g") ? re.flags : re.flags + "g"))) {
      violations.push(`${f}:${lineOf(text, m.index ?? 0)}  ${name}: ${msg}  ("${m[0].slice(0, 40)}")`);
    }
  }
  if (/\p{Extended_Pictographic}/u.test(text)) {
    const m = text.match(/\p{Extended_Pictographic}/u);
    violations.push(`${f}:${lineOf(text, m?.index ?? 0)}  emoji: "${m?.[0]}" (no emoji as icons or decoration)`);
  }
}

// icon and effect libraries in package.json
if (rootArg < 0) {
  const pkg = JSON.parse(readFileSync(resolve(WEB, "package.json"), "utf8"));
  const deps = Object.keys({ ...pkg.dependencies, ...pkg.devDependencies });
  const banned = deps.filter((d) => /lottie|three|particles|lucide|heroicons|react-icons|fontawesome|tsparticles|@react-three/.test(d));
  for (const d of banned) violations.push(`package.json  dependency "${d}" (decorative icon or effect library)`);
}

if (violations.length) {
  console.error(`lint-tells: ${violations.length} violation(s)\n` + violations.map((v) => "  " + v).join("\n"));
  process.exit(1);
}
console.log(`lint-tells: clean (${walk(ROOT).length} files scanned)`);
