// Prüfbilder der Ladenschilder, Wegweiser und Tafeln
//   node tools/tests/schilder.mjs <version> [damm]
import { open, SHOTS } from './common.mjs';
const version = process.argv[2] || 'v1', damm = process.argv[3] === 'damm';
const { browser, page, errors } = await open();
if (!damm) {
  await page.evaluate(() => window.__game.selectMap('weltenbrueck'));
  await page.waitForFunction(() => window.__game.mapId === 'weltenbrueck' && !window.__game.loadingMap, null, { timeout: 300000 });
}
const info = await page.evaluate((damm) => {
  const g = window.__game;
  g.startRaid();
  for (const e of g.enemies) { e.alive = false; e.root.visible = false; }
  g.building = true; g.player.model.root.visible = false;
  document.getElementById('hud').style.display = 'none';
  let verts = 0, meshes = 0;
  g.scene.traverse((o) => { if (o.isMesh && o.material === g.world.batcher?.materials?.sign) { meshes++; verts += o.geometry.attributes.position.count; } });
  const OUT = { s: [0, 1], n: [0, -1], e: [1, 0], w: [-1, 0] };
  const views = [];
  if (!damm) {
    const shops = g.world.mapData.lots.filter((l) => l.shop && l.doors?.[0]);
    for (const l of [shops[0], shops[Math.floor(shops.length / 3)], shops[Math.floor(shops.length * 2 / 3)], shops[shops.length - 1]]) {
      const d = l.doors[0], [ox, oz] = OUT[d.side];
      const fx = d.side === 'e' ? l.x1 : d.side === 'w' ? l.x0 : d.a, fz = d.side === 's' ? l.z1 : d.side === 'n' ? l.z0 : d.a;
      views.push([`laden${l.id}`, [fx + ox * 3.4, l.level + 1.7, fz + oz * 3.4], [fx, l.level + 2.9, fz]]);
    }
  }
  // Wegweiser und Tafeln stehen frei: über die Höhe ihrer Fläche finden
  const found = { 0.24: 0, 0.42: 0 };
  g.scene.traverse((o) => {
    if (!o.isMesh || o.material !== g.world.batcher?.materials?.sign) return;
    const P = o.geometry.attributes.position, N = o.geometry.attributes.normal;
    for (let i = 0; i + 5 < P.count; i += 6) {
      const h = +(P.getY(2 + i) - P.getY(i)).toFixed(2);
      if (found[h] == null || found[h] >= 2) continue;
      found[h]++;
      const x = (P.getX(i) + P.getX(i + 2)) / 2, y = P.getY(i), z = (P.getZ(i) + P.getZ(i + 2)) / 2;
      views.push([`tafel${h}-${found[h]}`, [x + N.getX(i) * 2.6, y + 0.1, z + N.getZ(i) * 2.6], [x, y + h / 2, z]]);
    }
  });
  return { verts, meshes, views };
}, damm);
console.log(JSON.stringify({ verts: info.verts, meshes: info.meshes }));
for (const [name, pos, tgt] of [...info.views, ...JSON.parse(process.env.EXTRA || '[]')]) {
  await page.evaluate(([pos, tgt]) => {
    const g = window.__game, c = g.camera;
    g.world.applyCondition('tag');
    c.fov = 62; c.updateProjectionMatrix(); c.position.set(...pos); c.lookAt(...tgt);
    for (let i = 0; i < 2; i++) { g.world.update(0.016, c.position, null); g.renderFrame(0.016); }
  }, [pos, tgt]);
  await page.screenshot({ path: `${SHOTS}schild-${version}-${name}.png`, timeout: 240000 });
}
console.log('Fehler:', errors.filter((e) => !/fonts|ERR_|GPU stall|GL Driver/.test(e)).slice(0, 8));
await browser.close();
