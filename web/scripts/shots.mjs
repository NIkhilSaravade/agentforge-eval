// Screenshots of the built site, so the design is judged by looking at it, not by reading CSS.
//   npm run build && node scripts/shots.mjs            (writes web/shots/*.png, gitignored)
// The engine figure is captured at several scroll positions through its recording.
import { spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = resolve(WEB, "shots");
mkdirSync(OUT, { recursive: true });
const PORT = 4173;

const server = spawn("npx", ["vite", "preview", "--port", String(PORT), "--strictPort"], { cwd: WEB, stdio: "ignore" });
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
for (let i = 0; i < 40; i++) {
  try {
    if ((await fetch(`http://localhost:${PORT}/`)).ok) break;
  } catch { /* not up yet */ }
  await wait(250);
}

const browser = await chromium.launch();
try {
  const errors = [];
  for (const [name, viewport] of [["desktop", { width: 1440, height: 900 }], ["mobile", { width: 390, height: 844 }]]) {
    const page = await browser.newPage({ viewport });
    page.on("pageerror", (e) => errors.push(`${name}: ${e.message}`));
    page.on("console", (m) => { if (m.type() === "error") errors.push(`${name} console: ${m.text()}`); });
    await page.goto(`http://localhost:${PORT}/`, { waitUntil: "networkidle" });
    await page.evaluate(() => document.fonts.ready);
    await page.addStyleTag({ content: "html { scroll-behavior: auto !important; }" }); // captures should not wait out smooth scrolling
    await wait(600);
    await page.screenshot({ path: `${OUT}/${name}-top.png` });
    await page.screenshot({ path: `${OUT}/${name}-full.png`, fullPage: true });
    if (name === "desktop") {
      const box = await page.evaluate(() => {
        const el = document.querySelector(".scrolly");
        if (!el) return null;
        const r = el.getBoundingClientRect();
        return { top: r.top + window.scrollY, height: r.height };
      });
      if (box) {
        // an eviction moment: jump to the first eviction, capture mid-animation (ghost markers) and settled
        await page.evaluate(([t]) => window.scrollTo(0, t + 5), [box.top]);
        await wait(400);
        await page.getByRole("button", { name: "Next eviction" }).click();
        await wait(250);
        await page.screenshot({ path: `${OUT}/eviction-mid.png` });
        await wait(1500);
        await page.screenshot({ path: `${OUT}/eviction-settled.png` });
        for (const f of [0.02, 0.16, 0.3, 0.45, 0.6, 0.78, 0.95]) {
          await page.evaluate(([t, h, f]) => window.scrollTo(0, t + f * (h - window.innerHeight)), [box.top, box.height, f]);
          await wait(1300);
          await page.screenshot({ path: `${OUT}/engine-${String(Math.round(f * 100)).padStart(2, "0")}.png` });
        }
      }
    }
    if (name === "desktop") {
      const hb = await page.evaluate(() => {
        const el = document.querySelectorAll(".scrolly")[1];
        if (!el) return null;
        const r = el.getBoundingClientRect();
        return { top: r.top + window.scrollY, height: r.height };
      });
      if (hb) {
        for (const f of [0.03, 0.2, 0.36, 0.52, 0.68, 0.84, 0.97]) {
          await page.evaluate(([t, h, f]) => window.scrollTo(0, t + f * (h - window.innerHeight)), [hb.top, hb.height, f]);
          await wait(900);
          await page.screenshot({ path: `${OUT}/harness-${String(Math.round(f * 100)).padStart(2, "0")}.png` });
        }
        await page.evaluate(([t, h]) => window.scrollTo(0, t + 0.8 * (h - window.innerHeight)), [hb.top, hb.height]);
        await wait(500);
        await page.screenshot({ path: `${OUT}/harness-pass-score.png` });
        await page.getByRole("button", { name: /fails/ }).click();
        await page.evaluate(([t, h]) => window.scrollTo(0, t + 0.8 * (h - window.innerHeight)), [hb.top, hb.height]);
        await wait(900);
        await page.screenshot({ path: `${OUT}/harness-fail-score.png` });
      }
      const pv = await page.evaluate(() => {
        const el = document.getElementById("pivot");
        return el ? el.getBoundingClientRect().top + window.scrollY : null;
      });
      if (pv !== null) {
        for (const [i, off] of [[1, -20], [2, 700], [3, 1500]]) {
          await page.evaluate((y) => window.scrollTo(0, y), pv + off);
          await wait(700);
          await page.screenshot({ path: `${OUT}/pivot-${i}.png` });
        }
      }
    }
    if (name === "desktop") {
      const rt = await page.evaluate(() => {
        const el = document.getElementById("results");
        return el ? el.getBoundingClientRect().top + window.scrollY : null;
      });
      if (rt !== null) {
        for (let i = 0; i < 9; i++) {
          await page.evaluate((y) => window.scrollTo(0, y), rt - 20 + i * 780);
          await wait(600);
          await page.screenshot({ path: `${OUT}/results-${i}.png` });
        }
      }
    }
    await page.close();
  }
  console.log(errors.length ? `PAGE ERRORS:\n${errors.join("\n")}` : "no page errors");
} finally {
  await browser.close();
  server.kill();
}
