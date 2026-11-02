// Woher kommen die Draw-Calls? Sichtbare Meshes je Material und Layer, dazu der Anteil des Schattenpasses.
//   node tools/tests/drawcalls.mjs
import { open } from './common.mjs';
const { browser, page, errors } = await open();
await page.evaluate(() => window.__game.selectMap('weltenbrueck'));
await page.waitForFunction(() => window.__game.mapId === 'weltenbrueck' && !window.__game.loadingMap, null, { timeout: 300000 });
const res = await page.evaluate(() => {
  const g = window.__game, r = g.renderer, cam = g.camera;
  g.startRaid(); g.building = true; g.world.applyCondition('tag');
  const out = [];
  for (const [name, x, z] of [['Kreuzmarkt', 0, 96], ['Gasse Sued', -60, 150], ['Kai', 40, 236], ['Turmstadt', 252, 96]]) {
    const p = g.player;
    p.pos.set(x, g.col.groundAt(x, z, 0.4, 999, 0) + 0.05, z); p.vel.set(0, 0, 0); p.camYaw = 0; p.yaw = 0; p.camPitch = -0.02;
    g.updateRaid(1 / 60); g.input.endFrame();
    const calls = () => { r.info.autoReset = false; r.info.reset(); g.renderFrame(1 / 60); const c = r.info.render.calls, t = r.info.render.triangles; r.info.autoReset = true; return [c, +(t / 1e6).toFixed(2)]; };
    const all = calls();
    // Schatten werden nur jedes zweite Bild gezeichnet: zwei Bilder zeigen beide Fälle
    const second = calls();
    // Sichtbare Meshes im Kamerakegel
    cam.updateMatrixWorld();
    const f = new (cam.constructor.prototype.constructor === Object ? Object : Object)();
    const M = cam.projectionMatrix.clone().multiply(cam.matrixWorldInverse);
    const E = M.elements;
    const planes = [[E[3]+E[0],E[7]+E[4],E[11]+E[8],E[15]+E[12]],[E[3]-E[0],E[7]-E[4],E[11]-E[8],E[15]-E[12]],[E[3]+E[1],E[7]+E[5],E[11]+E[9],E[15]+E[13]],[E[3]-E[1],E[7]-E[5],E[11]-E[9],E[15]-E[13]],[E[3]+E[2],E[7]+E[6],E[11]+E[10],E[15]+E[14]],[E[3]-E[2],E[7]-E[6],E[11]-E[10],E[15]-E[14]]].map((q) => { const l = Math.hypot(q[0],q[1],q[2]); return q.map((v) => v / l); });
    const groups = {}; let total = 0, shadowCast = 0;
    const matName = new Map();
    for (const [k, v] of Object.entries(g.world.batcher?.materials || {})) matName.set(v, k);
    g.scene.traverseVisible((o) => {
      if (!o.isMesh) return;
      if (o.isInstancedMesh && !o.boundingSphere) o.computeBoundingSphere?.();
      const bs = (o.boundingSphere || o.geometry.boundingSphere || (o.geometry.computeBoundingSphere(), o.geometry.boundingSphere)).clone();
      bs.applyMatrix4(o.matrixWorld);
      const inside = o.frustumCulled === false || planes.every((q) => q[0]*bs.center.x + q[1]*bs.center.y + q[2]*bs.center.z + q[3] > -bs.radius);
      if (o.castShadow) shadowCast++;
      if (!inside) return;
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      const key = (o.isInstancedMesh ? 'inst:' : 'batch:') + (o.isInstancedMesh ? (o.name || '?').replace(/\d+$/, '') : (matName.get(mats[0]) || mats[0]?.name || '?')) + (o.userData.layer ? '/' + o.userData.layer : '');
      groups[key] = (groups[key] || 0) + mats.length; total += mats.length;
    });
    const top = Object.entries(groups).sort((a, b) => b[1] - a[1]);
    let inst = 0, batch = 0; for (const [k, v] of top) { if (k.startsWith('inst')) inst += v; else batch += v; }
    out.push({ name, frames: [all, second], total, inst, batch, shadowCast, top: top.slice(0, 26).map(([k, v]) => `${k} ${v}`).join(', ') });
  }
  g.building = false;
  return out;
});
for (const o of res) console.log(JSON.stringify(o));
console.log('Fehler:', errors.filter((e) => !/fonts|ERR_|GPU stall|GL Driver/.test(e)).slice(0, 5));
await browser.close();
