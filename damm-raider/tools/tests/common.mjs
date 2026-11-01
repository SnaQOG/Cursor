// Gemeinsame Hilfen für die Browser-Tests (Playwright, headless Chromium).
// Vorher bauen und einen lokalen Server im Ordner damm-raider starten:
//   node tools/build.mjs && npx --yes http-server -c-1 -p 8790 .
// Andere Adresse: GAME_URL=http://localhost:8080/dist/damm-raider.html node tools/tests/stats.mjs
import { chromium } from 'playwright';
export const SHOTS = new URL('./shots/', import.meta.url).pathname;
export async function open(url = process.env.GAME_URL || 'http://localhost:8790/dist/damm-raider.html', { w = 1280, h = 720 } = {}) {
  // Ohne Grafikkarte (Server, CI): SWIFTSHADER=1 setzen
  const args = process.env.SWIFTSHADER ? ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] : ['--ignore-gpu-blocklist'];
  const browser = await chromium.launch({ args });
  const page = await browser.newPage({ viewport: { width: w, height: h } });
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') errors.push(m.type() + ': ' + m.text()); });
  await page.goto(url);
  try {
    await page.waitForFunction(() => window.__game && window.__game.state === 'hub', null, { timeout: 120000 });
  } catch (e) {
    console.log('TIMEOUT. boot:', await page.evaluate(() => document.getElementById('boot-text')?.textContent), errors.slice(0, 20));
    await browser.close();
    process.exit(1);
  }
  return { browser, page, errors };
}
