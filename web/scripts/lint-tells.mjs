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

// hex -> HSL hue/saturation, to catch purple and violet without listing every shade
const hsl = (hex) => {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? [...h].map((c) => c + c).join("") : h.slice(0, 6);
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16) / 255);
  const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
  if (d === 0) return { hue: 0, sat: 0 };
  const l = (max + min) / 2;
  const sat = d / (1 - Math.abs(2 * l - 1));
  let hue = max === r ? ((g - b) / d) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
  hue = (hue * 60 + 360) % 360;
  return { hue, sat };
};

const RULES = [
  ["gradient", /(?:linear|radial|conic|repeating-linear)-gradient\(/, "gradient (no gradient hero, no gradient text or blobs)"],
  ["glass/blur", /backdrop-filter|filter\s*:\s*blur|\bblur\(/, "blur or glassmorphism"],
  ["shadow", /box-shadow|drop-shadow|text-shadow/, "soft shadow"],
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
  for (const m of text.matchAll(/border-radius\s*:\s*([^;}{]+)/g)) {
    const bad = (m[1].match(/[\d.]+(?:px|rem|em|%)?/g) ?? []).some((tok) => {
      const n = parseFloat(tok);
      if (!n) return false;
      if (tok.endsWith("%")) return true;
      const px = tok.endsWith("rem") || tok.endsWith("em") ? n * 16 : n;
      return px > 2;
    });
    if (bad) violations.push(`${f}:${lineOf(text, m.index ?? 0)}  rounded corners: border-radius "${m[1].trim()}" (cards and buttons here are square; max 2px)`);
  }
  if (/\p{Extended_Pictographic}/u.test(text)) {
    const m = text.match(/\p{Extended_Pictographic}/u);
    violations.push(`${f}:${lineOf(text, m?.index ?? 0)}  emoji: "${m?.[0]}" (no emoji as icons or decoration)`);
  }
  for (const m of text.matchAll(/#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b/g)) {
    const { hue, sat } = hsl(m[0]);
    if (sat > 0.2 && hue >= 255 && hue <= 320) violations.push(`${f}:${lineOf(text, m.index ?? 0)}  purple/violet hue: ${m[0]} (hue ${Math.round(hue)})`);
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
