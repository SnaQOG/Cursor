// Echte Bildrate je Grafikstufe: läuft im normalen Spieltakt (mit Gegnern und HUD), ohne Vsync-Grenze.
// Braucht eine Grafikeinheit und das installierte Chrome:
//   GAME_URL=file://$PWD/dist/damm-raider.html node tools/tests/fps.mjs [BxH] [Sekunden je Ort]
import { chromium } from 'playwright';
const url = process.env.GAME_URL || 'http://localhost:8790/dist/damm-raider.html';
const [W, H] = (process.argv[2] || '1440x900').split('x').map(Number);
const SECS = Number(process.argv[3] || 7);
const DSF = Number(process.env.DSF || 2);
const QUALITIES = (process.env.QUALITY || 'niedrig,mittel,hoch').split(',');
const SAVE = 'damm-raider.save.v1';

// Ort: x, z, Blick (Yaw), Höhe über Grund. Blick über die Karte ist der schwerste Fall.
const VIEWS = {
  damm: [['Kraftwerk Vorplatz', 60, -22, 0], ['Halle', 60, -44, 0], ['Talblick Süd', 0, 60, 0], ['Aufbereitung', 150, 40, 0]],
  weltenbrueck: [['Kreuzmarkt', 0, 96, 0], ['Markt Richtung Dom', 0, 118, 0], ['Gasse Süd', -60, 150, 0], ['Waldterrasse', 60, -30, 0]],
};

const browser = await chromium.launch({
  channel: 'chrome',
  args: ['--ignore-gpu-blocklist', '--disable-gpu-vsync', '--disable-frame-rate-limit', '--disable-renderer-backgrounding', '--disable-background-timer-throttling'],
});
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: DSF });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
const ready = (id) => page.waitForFunction((m) => window.__game && window.__game.state === 'hub' && window.__game.mapId === m && !window.__game.loadingMap && !window.__game.building, id, { timeout: 300000 });

await page.goto(url);
await page.waitForFunction(() => window.__game && window.__game.state === 'hub', null, { timeout: 300000 });
const rows = [];
for (const q of QUALITIES) {
  for (const map of Object.keys(VIEWS)) {
    // Gras und Schatten werden beim Kartenaufbau festgelegt: Einstellung speichern und neu laden
    await page.evaluate(([q, map, key]) => {
      const g = window.__game;
      localStorage.setItem(key, JSON.stringify({ ...g.save, map, settings: { ...g.save.settings, quality: q } }));
    }, [q, map, SAVE]);
    await page.reload();
    await ready(map);
    await page.evaluate(() => {
      const g = window.__game;
      g.startRaid();
      g.player.hp = 1e9; g.player.maxHp = 1e9;
      g.world.applyCondition('tag');
    });
    for (const [name, x, z, yaw] of VIEWS[map]) {
      const r = await page.evaluate(([x, z, yaw, secs]) => new Promise((done) => {
        const g = window.__game, p = g.player;
        p.pos.set(x, g.col.groundAt(x, z, 200, 999, 0) + 0.05, z);
        p.vel.set(0, 0, 0);
        p.camYaw = yaw; p.yaw = yaw; p.camPitch = -0.02;
        const times = [];
        let last = performance.now(), t0 = last;
        const tick = (now) => {
          if (now - t0 > 1000) times.push(now - last);
          last = now;
          p.hp = 1e9;
          if (now - t0 < secs * 1000 + 1000) requestAnimationFrame(tick);
          else {
            times.sort((a, b) => a - b);
            const n = times.length, sum = times.reduce((a, b) => a + b, 0);
            const i = g.renderer.info;
            done({ fps: +(1000 * n / sum).toFixed(1), p50: +times[Math.floor(n * 0.5)].toFixed(1), p95: +times[Math.floor(n * 0.95)].toFixed(1), p99: +times[Math.floor(n * 0.99)].toFixed(1), frames: n, state: g.state, pr: g.renderer.getPixelRatio() });
          }
        };
        requestAnimationFrame(tick);
      }), [x, z, yaw, SECS]);
      rows.push({ q, map, name, ...r });
      console.log(q.padEnd(8), map.padEnd(13), name.padEnd(20), `${String(r.fps).padStart(6)} fps  p50 ${r.p50}  p95 ${r.p95}  p99 ${r.p99} ms  pr ${r.pr}  ${r.state}`);
    }
    await page.evaluate(() => { const g = window.__game; g.toHub(); });
  }
}
const gpu = await page.evaluate(() => {
  const gl = window.__game.renderer.getContext(), d = gl.getExtension('WEBGL_debug_renderer_info');
  return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : '?';
});
console.log('GPU:', gpu, `| ${W}x${H} @${DSF}x | Fehler:`, errors.slice(0, 5));
await browser.close();
