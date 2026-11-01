// Detailbilder Weltenbrück: Fassaden je Hausstil, Stube mit Eckbank, Wiese, Autos (Bilder in shots/)
import { open, SHOTS } from './common.mjs';
const { browser, page, errors } = await open();
await page.evaluate(() => window.__game.selectMap('weltenbrueck'));
await page.waitForFunction(() => window.__game.mapId === 'weltenbrueck' && !window.__game.loadingMap, null, { timeout: 300000 });
await page.evaluate(() => { const g = window.__game; g.startRaid(); for (const e of g.enemies) { e.alive = false; e.root.visible = false; } g.world.applyCondition('tag'); g.building = true; g.player.model.root.visible = false; });
const cam = async (name, fn, arg) => {
  const info = await page.evaluate(fn, arg);
  await page.evaluate(() => { const g = window.__game; g.world.update(0.016, g.camera.position, null); g.renderer.render(g.scene, g.camera); });
  await page.screenshot({ path: SHOTS + name + '.png', timeout: 240000 });
  console.log(name, JSON.stringify(info));
};
const OUT = { s: [0, 1], n: [0, -1], e: [1, 0], w: [-1, 0] };
// Fassaden: Haus von der Straße aus
for (const [name, style, k] of [['d-fachwerk', 'fachwerk', 0], ['d-fachwerk2', 'fachwerk', 3], ['d-stein', 'stein', 0], ['d-backstein', 'backstein', 1], ['d-schwarzwald', 'schwarzwald', 0]]) {
  await cam(name, ([style, k, OUT]) => {
    const g = window.__game, c = g.camera;
    c.fov = 72; c.updateProjectionMatrix();
    const lots = g.world.mapData.lots.filter((l) => l.kind === 'house' && l.style === style && l.floors >= 2 && l.doors?.length && l.x0 > -190 && l.x1 < 190 && l.z1 < 190);
    const l = lots[Math.min(k, lots.length - 1)];
    const d = l.doors[0], o = OUT[d.side];
    const fx = d.side === 'e' ? l.x1 : d.side === 'w' ? l.x0 : d.a, fz = d.side === 's' ? l.z1 : d.side === 'n' ? l.z0 : d.a;
    // so weit zurück wie frei (nicht im Nachbarhaus)
    const cy = l.level + 1.8;
    let best = 1.5;
    for (let dd = 1.5; dd <= 9; dd += 0.25) {
      const x = fx + o[0] * dd + o[1] * dd * 0.35, z = fz + o[1] * dd - o[0] * dd * 0.35;
      if (g.col.query(x - 0.4, z - 0.4, x + 0.4, z + 0.4).some((sh) => sh.minY < cy + 0.3 && sh.maxY > cy - 0.3)) break;
      best = dd;
    }
    c.position.set(fx + o[0] * best + o[1] * best * 0.35, cy, fz + o[1] * best - o[0] * best * 0.35);
    c.lookAt(fx, l.level + 4.0, fz);
    return { id: l.id, side: d.side, floors: l.floors, district: l.district, dist: best };
  }, [style, k, OUT]);
}
// Stube mit Eckbank
await cam('d-stube', () => {
  const g = window.__game, c = g.camera;
  const l = g.world.mapData.lots.find((l) => l.decor?.some((d) => d.kind === 'eckbank'));
  if (!l) return 'keine Eckbank';
  const d = l.decor.find((d) => d.kind === 'eckbank');
  c.fov = 80; c.updateProjectionMatrix();
  const cx = (l.x0 + l.x1) / 2, cz = (l.z0 + l.z1) / 2;
  const vx = d.x - cx, vz = d.z - cz, n = Math.hypot(vx, vz) || 1;
  c.position.set(d.x - vx / n * 2.4, d.y + 1.6, d.z - vz / n * 2.4);
  c.lookAt(d.x, d.y + 0.6, d.z);
  return { id: l.id, style: l.style, n: g.world.mapData.lots.filter((l) => l.decor).length };
});
// Wiese (eine v_meadow-Instanz)
await cam('d-wiese', () => {
  const g = window.__game, c = g.camera;
  c.fov = 60; c.updateProjectionMatrix();
  let im = null; g.scene.traverse((o) => { if (o.isInstancedMesh && o.name === 'v_meadow' && (!im || o.count > im.count)) im = o; });
  if (!im) return 'keine Wiese';
  im.visible = true;
  const m = new im.matrixWorld.constructor(); im.getMatrixAt(Math.floor(im.count / 2), m);
  const x = m.elements[12], y = m.elements[13], z = m.elements[14];
  c.position.set(x - 2.2, y + 1.3, z + 2.2); c.lookAt(x + 0.5, y, z - 0.5);
  return { count: im.count, x, z };
});
// Autos (Vorschau auf dem Kreuzmarkt)
await cam('d-autos', () => {
  const g = window.__game, c = g.camera;
  let proto = null; g.scene.traverse((o) => { if (!proto && o.type === 'Mesh' && o.geometry.attributes.color) proto = o; });
  const add = (name, x, z, ry, col) => {
    const r = ((col >> 16) & 255) / 255, gg = ((col >> 8) & 255) / 255, b = (col & 255) / 255;
    for (const m of g.world.assets.meshes(name)) {
      const geo = m.geometry.clone(); geo.applyMatrix4(m.matrix);
      const ca = geo.attributes.color;
      if (ca) for (let i = 0; i < ca.count; i++) { if (ca.getX(i) > 0.8 && ca.getY(i) < 0.2 && ca.getZ(i) > 0.8) ca.setXYZ(i, r, gg, b); else if (ca.getX(i) > 0.15 && ca.getY(i) < 0.05 && ca.getZ(i) > 0.15) ca.setXYZ(i, r * 0.6, gg * 0.6, b * 0.6); }
      const o = new proto.constructor(geo, m.material);
      o.position.set(x, g.world.terrain.height(x, z), z); o.rotation.y = ry; o.castShadow = true;
      g.scene.add(o);
    }
  };
  add('p_car_a', -8, 100, 0.6, 0x8a3b30); add('p_car_b', 0, 101, -0.5, 0x566b7a);
  c.fov = 60; c.updateProjectionMatrix();
  const h = g.world.terrain.height(-4, 108);
  c.position.set(-4, h + 2.4, 109); c.lookAt(-4, h + 0.8, 100);
  return 'ok';
});
console.log('errors:', errors.slice(0, 8));
await browser.close();
