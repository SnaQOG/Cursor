// Prüfbilder von Wiese und Boden: die größten Höfe, Trampelpfade und eine Pflastergasse aus der Nähe.
//   node tools/tests/boden.mjs g2 [damm]     Bilder: tools/tests/shots/boden-<version>-<karte>-<ansicht>.png
import { open, SHOTS } from './common.mjs';
const version = process.argv[2] || 'g0';
const { browser, page, errors } = await open();
const map = process.argv[3] || 'weltenbrueck';
if (map !== 'damm') {
  await page.evaluate((m) => window.__game.selectMap(m), map);
  await page.waitForFunction((m) => window.__game.mapId === m && !window.__game.loadingMap, map, { timeout: 300000 });
}
const info = await page.evaluate(() => {
  const g = window.__game;
  g.startRaid();
  for (const e of g.enemies) { e.alive = false; e.root.visible = false; }
  g.building = true; g.player.model.root.visible = false; document.getElementById('hud').style.display = 'none';
  const T = g.world.terrain, L = g.world.mapData.lots || [];
  // Grüne Flächen aus der Kollision und dem Boden ablesen: Stellen, an denen Bodenpflanzen liegen, aus der Geometrie
  const pts = [];
  for (const m of g.world.detailMeshes) {
    if (!m.userData.cover) continue;
    const P = m.geometry.attributes.position, n = P.count;
    if (n < 400 * 12) continue;
    pts.push([m.userData.center.x, m.userData.center.z, n / 12]);
  }
  pts.sort((a, b) => b[2] - a[2]);
  const views = [];
  const clear = (x, z) => !L.some((l) => x > l.x0 - 0.2 && x < l.x1 + 0.2 && z > l.z0 - 0.2 && z < l.z1 + 0.2);
  for (const [cx, cz] of pts.slice(0, 40).filter((_, i) => i % 7 === 0)) {
    // Blick vom Rand der Kachel zur Mitte, mit freiem Standort
    for (const [dx, dz] of [[-9, 0], [9, 0], [0, -9], [0, 9]]) {
      const x = cx + dx, z = cz + dz;
      if (!clear(x, z)) continue;
      views.push([[x, T.height(x, z) + 1.65, z], [cx, T.height(cx, cz) + 0.9, cz]]);
      break;
    }
  }
  const h = (x, z) => T.height(x, z);
  return { stats: g.world.coverStats, chunks: pts.length, views, probe: [[-100, -44], [-110, -48], [-120, -58]].map(([x, z]) => [x, z, h(x, z), L.find((l) => x > l.x0 && x < l.x1 && z > l.z0 && z < l.z1)?.style]) };
});
console.log(JSON.stringify(info.stats), 'Kacheln', info.chunks, JSON.stringify(info.probe));
const VIEWS = [...info.views.map((v, i) => ['hof' + i, ...v]), ...(map === 'damm' ? [['talblick', [20, 30, 60], [90, 2, 150]], ['strasse', [0, 3.5, 70], [40, 1, 110]]] : [['gasse', [-60, 4.7, 96.5], [-74, 3, 96.5]], ['strasse', [0, 4.2, 170], [0, 3, 180]]])];
for (const [name, pos, tgt] of VIEWS) {
  await page.evaluate(([pos, tgt]) => {
    const g = window.__game, c = g.camera;
    g.world.applyCondition('tag');
    c.fov = 66; c.updateProjectionMatrix();
    c.position.set(...pos); c.lookAt(...tgt);
    for (let i = 0; i < 2; i++) { g.world.update(0.016, c.position, null); g.renderFrame(0.016); }
  }, [pos, tgt]);
  await page.screenshot({ path: `${SHOTS}boden-${version}-${map}-${name}.png`, timeout: 240000 });
}
console.log('Fehler:', errors.filter((e) => !/fonts|ERR_|GPU stall|GL Driver/.test(e)).slice(0, 5));
await browser.close();
