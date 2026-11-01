// Spieleransichten Weltenbrück: Innenraum, Waldgasse, Riesenbaum, Markt, Gesichter, Dom-Spitze
import { open, SHOTS } from './common.mjs';
const { browser, page, errors } = await open();
await page.evaluate(() => window.__game.selectMap('weltenbrueck'));
await page.waitForFunction(() => window.__game.mapId === 'weltenbrueck' && !window.__game.loadingMap, null, { timeout: 300000 });
await page.evaluate(() => { const g = window.__game; g.startRaid(); for (const e of g.enemies) { e.alive = false; e.root.visible = false; } g.player.hp = 100; g.condition = 'tag'; g.world.applyCondition('tag'); });
const shoot = async (name, fn) => {
  await page.evaluate(fn);
  await page.evaluate(() => { const g = window.__game; g.building = true; g.player.updateCamera(0.016); g.world.update(0.016, g.player.pos, null); g.renderer.render(g.scene, g.camera); });
  await page.screenshot({ path: SHOTS + name + '.png', timeout: 240000, timeout: 180000 });
  await page.evaluate(() => { window.__game.building = false; });
};
// Innenraum im 1. OG eines Fachwerkhauses
await shoot('innen', () => {
  const g = window.__game, p = g.player;
  const l = g.world.mapData.lots.find((l) => l.stair0 && l.style === 'fachwerk' && l.floors >= 3 && l.district === 'markt' && l.x0 > -150 && l.w > 6);
  const s = l.stair0;
  p.pos.set(s.x, s.y + 0.05, s.z); p.vel.set(0, 0, 0); p.camYaw = Math.atan2(-s.dx, -s.dz); p.yaw = p.camYaw;
  g.input.keys.clear(); g.input.keys.add('KeyW'); g.debugStep(3); g.input.keys.clear();
  p.camYaw += Math.PI; p.yaw = p.camYaw; p.camPitch = -0.1; g.debugStep(0.3);
});
// Gasse in der Waldsiedlung (Promenade der mittleren Terrasse)
await shoot('wald-gasse', () => {
  const g = window.__game, p = g.player;
  p.spawn(-60, -43.5); p.camYaw = -Math.PI / 2 - 0.25; p.yaw = p.camYaw; p.camPitch = 0.05; g.debugStep(0.3);
});
// Plattform am Alten Riesen
await shoot('baum', () => {
  const g = window.__game, p = g.player;
  const t = g.world.mapData.trees.find((t) => t.id === 'riese');
  p.pos.set(t.x - 4, t.deckY + 0.1, t.z + 5); p.vel.set(0, 0, 0); p.camYaw = Math.PI * 0.75; p.yaw = p.camYaw; p.camPitch = -0.15; g.debugStep(0.3);
});
// Kreuzmarkt aus Spielersicht
await shoot('markt', () => {
  const g = window.__game, p = g.player;
  p.spawn(-20, 118); p.camYaw = -0.35; p.yaw = p.camYaw; p.camPitch = 0.12; g.debugStep(0.3);
});
// Gründergesichter vom Felsband aus
await shoot('gesichter', () => {
  const g = window.__game, p = g.player;
  p.pos.set(-70, 30.2, -169.5); p.vel.set(0, 0, 0); p.camYaw = Math.PI * 0.62; p.yaw = p.camYaw; p.camPitch = 0.22; g.debugStep(0.3);
});
// Dom-Spitze von der Galerie
await shoot('spitze', () => {
  const g = window.__game, p = g.player;
  p.pos.set(-6.5, 27.4, 70.5); p.vel.set(0, 0, 0); p.camYaw = Math.PI * 0.85; p.yaw = p.camYaw; p.camPitch = 0.35; g.debugStep(0.3);
});
console.log('errors:', errors.slice(0, 8));
await browser.close();
