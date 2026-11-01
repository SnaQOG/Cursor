// Raid auf beiden Karten starten und simulieren (Gegner, Spieler, Fehler)
import { open, SHOTS } from './common.mjs';
const { browser, page, errors } = await open();
// 1) Damm: Raid starten, 20 s simulieren
const damm = await page.evaluate(() => {
  const g = window.__game;
  g.startRaid();
  g.player.hp = 1e9; g.player.maxHp = 1e9;
  const n0 = g.enemies.length;
  g.debugStep(20);
  return { map: g.mapId, enemies: n0, alive: g.enemies.filter((e) => e.alive).length, y: +g.player.pos.y.toFixed(2), state: g.state, shapes: g.col.shapes.length };
});
console.log('damm', JSON.stringify(damm));
await page.evaluate(() => { const g = window.__game; g.building = true; g.player.updateCamera?.(0.016); g.world.update(0.016, g.player.pos, null); (g.renderFrame ? g.renderFrame(0.016) : g.renderer.render(g.scene, g.camera)); });
await page.screenshot({ path: SHOTS + 'damm-raid.png', timeout: 180000, timeout: 180000 });
await page.evaluate(() => { const g = window.__game; g.building = false; g.toHub(); });
// 2) Stadt: Raid mit Gegnern 30 s
await page.evaluate(() => window.__game.selectMap('weltenbrueck'));
await page.waitForFunction(() => window.__game.mapId === 'weltenbrueck' && !window.__game.loadingMap, null, { timeout: 300000 });
const city = await page.evaluate(() => {
  const g = window.__game;
  g.startRaid();
  g.player.hp = 1e9; g.player.maxHp = 1e9;
  const n0 = g.enemies.length;
  const t = performance.now();
  g.debugStep(30);
  const types = {};
  for (const e of g.enemies) types[e.type] = (types[e.type] || 0) + 1;
  return { enemies: n0, alive: g.enemies.filter((e) => e.alive).length, types, simMs: Math.round(performance.now() - t), hp: g.player.hp > 0 };
});
console.log('city', JSON.stringify(city));
// 3) Innenraum: Spieler in ein Haus im 1. Stock stellen
await page.evaluate(() => {
  const g = window.__game, p = g.player;
  const l = g.world.mapData.lots.find((l) => l.stair0 && l.style === 'fachwerk' && l.floors >= 3 && l.district === 'markt');
  const s = l.stair0;
  p.pos.set(s.x, s.y + 0.05, s.z); p.vel.set(0, 0, 0);
  p.camYaw = Math.atan2(-s.dx, -s.dz); p.yaw = p.camYaw;
  g.input.keys.clear(); g.input.keys.add('KeyW'); g.debugStep(3); g.input.keys.clear();
  p.camYaw += Math.PI * 0.9; p.yaw = p.camYaw; p.camPitch = -0.05;
  g.debugStep(0.5);
  g.building = true;
  g.world.update(0.016, p.pos, null);
  (g.renderFrame ? g.renderFrame(0.016) : g.renderer.render(g.scene, g.camera));
  window.__lot = { id: l.id, floor: +(p.pos.y - l.level).toFixed(2) };
});
await page.screenshot({ path: SHOTS + 'innen.png', timeout: 180000, timeout: 180000 });
console.log('lot', JSON.stringify(await page.evaluate(() => window.__lot)));
console.log('errors:', errors.slice(0, 10));
await browser.close();
