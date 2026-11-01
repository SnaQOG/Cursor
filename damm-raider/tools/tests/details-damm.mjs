// Detailbilder Dammkarte: Parkplatz mit Autos, Fichtenwald (Bilder in shots/)
import { open, SHOTS } from './common.mjs';
const { browser, page, errors } = await open();
await page.evaluate(() => { const g = window.__game; g.startRaid(); for (const e of g.enemies) { e.alive = false; e.root.visible = false; } g.world.applyCondition('tag'); g.building = true; g.player.model.root.visible = false; });
const cam = async (name, fn, arg) => {
  const info = await page.evaluate(fn, arg);
  await page.evaluate(() => { const g = window.__game; g.world.update(0.016, g.camera.position, null); g.renderer.render(g.scene, g.camera); });
  await page.screenshot({ path: SHOTS + name + '.png', timeout: 240000 });
  console.log(name, JSON.stringify(info));
};
// Parkplatz am Forschungsgelände (fx=-150, fz=5): Autos in Reihe bei z=21
await cam('damm-autos', () => {
  const g = window.__game, c = g.camera; c.fov = 60; c.updateProjectionMatrix();
  const h = g.world.terrain.height(-157, 28);
  c.position.set(-157, h + 2.6, 29.5); c.lookAt(-152, h + 0.6, 21);
  return h;
});
await cam('damm-wald', () => {
  const g = window.__game, c = g.camera; c.fov = 65; c.updateProjectionMatrix();
  let im = null; g.scene.traverse((o) => { if (o.isInstancedMesh && o.name === 'v_pine' && (!im || o.count > im.count)) im = o; });
  const m = new im.matrixWorld.constructor(); im.getMatrixAt(0, m);
  const x = m.elements[12], y = m.elements[13], z = m.elements[14];
  c.position.set(x + 9, y + 3, z + 9); c.lookAt(x, y + 3.5, z);
  return { x, z };
});
console.log('errors:', errors.slice(0, 8));
await browser.close();
