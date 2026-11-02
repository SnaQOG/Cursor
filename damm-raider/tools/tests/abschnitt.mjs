// Prüfansichten eines Stadtabschnitts nach festem Ablauf: Straße auf Augenhöhe, Luft, Fassade, Rückseite.
//   node tools/tests/abschnitt.mjs v00 [grau]     Bilder: tools/tests/shots/abschnitt-<version>[-grau]-<ansicht>.png
// grau: alle Materialien neutral, damit nur Form und Proportion zu sehen sind.
import { open, SHOTS } from './common.mjs';
const version = process.argv[2] || 'v00';
const grey = process.argv[3] === 'grau';
const ROW = { x0: -292, x1: -211, z0: -160, z1: -94 }; // Weiße Stadt Nord

const { browser, page, errors } = await open();
// QUALITY=hoch prüft die Ansichten auf einer anderen Grafikstufe
if (process.env.QUALITY) {
  await page.evaluate((q) => { const g = window.__game; localStorage.setItem('damm-raider.save.v1', JSON.stringify({ ...g.save, settings: { ...g.save.settings, quality: q } })); }, process.env.QUALITY);
  await page.reload();
  await page.waitForFunction(() => window.__game && window.__game.state === 'hub', null, { timeout: 300000 });
}
await page.evaluate(() => window.__game.selectMap('weltenbrueck'));
await page.waitForFunction(() => window.__game.mapId === 'weltenbrueck' && !window.__game.loadingMap, null, { timeout: 300000 });
const info = await page.evaluate(([R, grey]) => {
  const g = window.__game;
  g.startRaid();
  for (const e of g.enemies) { e.alive = false; e.root.visible = false; }
  g.building = true;
  g.player.model.root.visible = false;
  document.getElementById('hud').style.display = 'none';
  if (grey) {
    const seen = new Set();
    g.scene.traverse((o) => {
      if (!o.isMesh || !o.material || o.material.fog === false) return;
      for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
        if (seen.has(m) || m.transparent) continue;
        seen.add(m);
        m.map = null; m.normalMap = null; m.roughnessMap = null; m.metalnessMap = null;
        m.vertexColors = false; m.color?.setRGB(0.55, 0.55, 0.55); m.metalness = 0; m.roughness = 0.9; m.emissive?.setRGB(0, 0, 0);
        m.needsUpdate = true;
      }
      if (o.isInstancedMesh && o.instanceColor) { o.instanceColor.array.fill(1); o.instanceColor.needsUpdate = true; }
    });
  }
  const lots = g.world.mapData.lots.filter((l) => l.kind === 'house' && l.x0 >= R.x0 && l.x1 <= R.x1 && l.z0 >= R.z0 && l.z1 <= R.z1);
  return { n: lots.length, lots: lots.map((l) => [l.id, l.x0, l.x1, l.z0, l.z1, l.floors, l.doors?.[0]?.side]), oriels: lots.filter((l) => l.orielAt).map((l) => [l.id, ...l.orielAt.map((v) => +v.toFixed(1)), (l.x0 + l.x1) / 2, (l.z0 + l.z1) / 2]) };
}, [ROW, grey]);
console.log('Häuser im Abschnitt:', info.n);
if (process.env.LOTS) console.log(JSON.stringify(info.lots));
console.log('Erker:', JSON.stringify(info.oriels));

const VIEWS = [
  ['strasse', [-252.5, 4.7, -103], [-252.5, 7, -150], 62],
  ['querstrasse', [-222, 4.7, -97], [-285, 7, -97], 62],
  ['luft', [-205, 42, -70], [-255, 4, -130], 58],
  ['fassade', [-246, 4.7, -128], [-256, 7, -134], 62],
  ['ecke', [-214, 4.7, -96.5], [-226, 9, -108], 66],
  ['hof', [-232, 4.7, -118], [-226, 8, -127], 70],
  ['erker', [-205.6, 4.7, -121], [-212, 9.5, -130], 72],
  ['ringstrasse', [-207, 4.7, -96], [-209, 8, -150], 62],
  ['felsen', [-233, 4.8, -163.5], [-225, 24, -190], 74],
  ['markt-gasse', [-60, 4.7, 96.5], [-90, 5, 96], 68],
  ['wiese1', [-168, 4.7, 120], [-150, 3.6, 124], 66],
  ['wiese2', [120, 4.7, 165], [136, 3.6, 170], 66],
  ['wiese3', [-100, 9.7, -28], [-84, 8.2, -26], 66],
  ['wiese4', [-232, 4.7, -118], [-226, 3, -130], 70],
  ['pflaster', [0, 4.2, 170], [0, 3, 180], 60],
  ['altstadt-haus', [-72, 4.7, 98], [-80, 6.5, 90], 66],
  ['boden', [-40, 4.6, 96], [-48, 3, 96], 66],
  ['backstein', [40, 4.7, 96], [50, 6.5, 91], 66],
  ['waldweg', [-100, 9.7, -44], [-130, 9, -44], 68],
  ['turmstadt', [252.5, 4.7, 70], [252.5, 8, 110], 68],
  ['turm-hof', [262, 4.8, 22], [262, 3, 40], 72],
  ['kai', [120, 4.8, 236], [60, 4, 233], 66],
  ['weisse-gasse', [-252.5, 4.7, 60], [-252.5, 6, 90], 66],
  ['lager', [-262, 5.2, 24], [-262, 3, 36], 66],
  ['baeume', [-232, 4.7, -118], [-222, 7, -118], 70],
];
for (const [name, pos, tgt, fov] of VIEWS) {
  await page.evaluate(([pos, tgt, fov]) => {
    const g = window.__game, c = g.camera;
    g.world.applyCondition('tag');
    c.fov = fov; c.updateProjectionMatrix();
    c.position.set(...pos); c.lookAt(...tgt);
    for (let i = 0; i < 2; i++) { g.world.update(0.016, c.position, null); g.renderFrame(0.016); }
  }, [pos, tgt, fov]);
  await page.screenshot({ path: `${SHOTS}abschnitt-${version}${grey ? '-grau' : ''}-${name}.png`, timeout: 240000 });
}
console.log('Fehler:', errors.filter((e) => !/fonts|ERR_|GPU stall|GL Driver/.test(e)).slice(0, 8));
await browser.close();
