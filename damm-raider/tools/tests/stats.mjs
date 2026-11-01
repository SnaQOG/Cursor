// Ladezeit beider Karten, Draw-Calls, Dreiecke, Häuser, Beuteplätze, Gegner
import { open, SHOTS } from './common.mjs';
const t0 = Date.now();
const { browser, page, errors } = await open();
console.log('hub after', (Date.now() - t0) / 1000, 's');
await page.waitForTimeout(500);
await page.screenshot({ path: SHOTS + 'hub-damm.png', timeout: 180000 });
const t1 = Date.now();
await page.evaluate(() => window.__game.selectMap('weltenbrueck'));
await page.waitForFunction(() => window.__game.mapId === 'weltenbrueck' && !window.__game.loadingMap, null, { timeout: 300000 });
console.log('city load', (Date.now() - t1) / 1000, 's');
await page.waitForTimeout(800);
await page.screenshot({ path: SHOTS + 'hub-city.png', timeout: 180000 });
const info = await page.evaluate(() => {
  const g = window.__game;
  let verts = 0, meshes = 0, inter = 0;
  g.scene.traverse((o) => { if (o.isMesh && o.geometry?.attributes?.position) { meshes++; verts += o.geometry.attributes.position.count * (o.count || 1); } });
  return {
    verts, meshes, interior: g.world.interiorMeshes.length, shapes: g.col.shapes.length,
    stats: g.world.mapData.stats, spots: g.world.mapData.spots.length, containers: g.loot.containers.length,
    enemiesDefs: g.world.mapData.enemySpawns.length, calls: g.renderer.info.render.calls, tris: g.renderer.info.render.triangles,
  };
});
console.log(JSON.stringify(info));
console.log('errors:', errors.slice(0, 15));
await browser.close();
