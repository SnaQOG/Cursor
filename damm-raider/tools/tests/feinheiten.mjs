// Feuertreppen vom Boden aufs Dach und Prüfbilder der Schilder
//   node tools/tests/feinheiten.mjs <version>   Bilder: tools/tests/shots/fein-<version>-<ansicht>.png
import { open, SHOTS } from './common.mjs';
const version = process.argv[2] || 'v1';
const { browser, page, errors } = await open();
await page.evaluate(() => window.__game.selectMap('weltenbrueck'));
await page.waitForFunction(() => window.__game.mapId === 'weltenbrueck' && !window.__game.loadingMap, null, { timeout: 300000 });
const info = await page.evaluate(() => {
  const g = window.__game, p = g.player, lots = g.world.mapData.lots;
  g.startRaid(); p.hp = 1e9; p.maxHp = 1e9;
  for (const e of g.enemies) { e.root.visible = false; e.alive = false; }
  g.enemies.length = 0;
  const OUT = { s: [0, 1], n: [0, -1], e: [1, 0], w: [-1, 0] };
  const yawFor = (dx, dz) => Math.atan2(-dx, -dz);
  // Wegpunkte ablaufen: je Punkt aus der aktuellen Lage auf den Punkt zu
  const go = (a, b, secs) => {
    const dx = a - p.pos.x, dz = b - p.pos.z, y = yawFor(dx, dz);
    p.camYaw = y; p.yaw = y; g.input.keys.clear(); g.input.keys.add('KeyW');
    let t = 0;
    while (t < secs && Math.hypot(a - p.pos.x, b - p.pos.z) > 0.25) { g.debugStep(0.05); t += 0.05; }
    g.input.keys.clear();
  };
  const out = [], views = [];
  for (const l of lots) {
    if (!l.escape) continue;
    const e = l.escape, [ox, oz] = OUT[e.side], al = e.side === 'n' || e.side === 's';
    const fixed = e.side === 's' ? l.z1 : e.side === 'n' ? l.z0 : e.side === 'e' ? l.x1 : l.x0;
    const P = (a, d) => (al ? [a, fixed + oz * d] : [fixed + ox * d, a]); // d = Abstand von der Wand
    const fh = (l.roofY - e.y) / l.floors, as = e.a - 1.8, ae = e.a + 1.8;
    const [sx, sz] = P(as - 0.9, 1.48);
    p.pos.set(sx, e.y + 0.1, sz); p.vel.set(0, 0, 0); p.alive = true;
    g.debugStep(0.3);
    const y0 = p.pos.y;
    for (let f = 0; f < l.floors; f++) {
      const [a1, b1] = P(ae + 0.35, 1.48); go(a1, b1, 5); // Lauf hinauf
      if (f === l.floors - 1) break;
      const [a2, b2] = P(ae + 0.35, 0.5); go(a2, b2, 1.5); // auf den Laufsteg
      const [a3, b3] = P(as - 0.35, 0.5); go(a3, b3, 3.5); // zurück
      const [a4, b4] = P(as - 0.35, 1.48); go(a4, b4, 1.5); // zum nächsten Lauf
    }
    const top = p.pos.y;
    // Durch die Lücke aufs Dach
    const [a5, b5] = P(ae + 0.35, 0.5); go(a5, b5, 1.5);
    const [a6, b6] = P(ae + 0.35, -1.5); go(a6, b6, 2);
    const inside = p.pos.x > l.x0 && p.pos.x < l.x1 && p.pos.z > l.z0 && p.pos.z < l.z1;
    out.push({ id: l.id, side: e.side, floors: l.floors, y0: +y0.toFixed(2), lvl: l.level, top: +top.toFixed(2), roof: +l.roofY.toFixed(2), onRoof: inside && Math.abs(p.pos.y - l.roofY) < 0.4, end: [+p.pos.x.toFixed(1), +p.pos.y.toFixed(1), +p.pos.z.toFixed(1)] });
    const [vx, vz] = P(e.a + 7, 8), [tx, tz] = P(e.a, 0);
    views.push([`feuer${l.id}`, [vx, e.y + 3, vz], [tx, e.y + 7, tz]]);
  }
  // Läden ansehen
  const shops = lots.filter((l) => l.shop && l.doors?.[0]).slice(0, 40);
  const pickS = [shops[1], shops[12], shops[24]].filter(Boolean);
  for (const l of pickS) {
    const d = l.doors[0], [ox, oz] = OUT[d.side];
    const fx = d.side === 'e' ? l.x1 : d.side === 'w' ? l.x0 : d.a, fz = d.side === 's' ? l.z1 : d.side === 'n' ? l.z0 : d.a;
    views.push([`laden${l.id}`, [fx + ox * 5.5 + oz * 1.5, l.level + 1.7, fz + oz * 5.5 + ox * 1.5], [fx, l.level + 2.6, fz]]);
  }
  return { out, views, nShops: lots.filter((l) => l.shop).length };
});
console.log('Läden', info.nShops, 'Feuertreppen', info.out.length);
for (const o of info.out) if (!o.onRoof || process.env.ALL) console.log(JSON.stringify(o));
await page.evaluate(() => { const g = window.__game; g.building = true; g.player.model.root.visible = false; document.getElementById('hud').style.display = 'none'; });
const VIEWS = [...info.views.slice(0, 2), ...info.views.slice(-3), ...JSON.parse(process.env.EXTRA || '[]')];
for (const [name, pos, tgt] of VIEWS) {
  await page.evaluate(([pos, tgt]) => {
    const g = window.__game, c = g.camera;
    g.world.applyCondition('tag');
    c.fov = 62; c.updateProjectionMatrix(); c.position.set(...pos); c.lookAt(...tgt);
    for (let i = 0; i < 2; i++) { g.world.update(0.016, c.position, null); g.renderFrame(0.016); }
  }, [pos, tgt]);
  await page.screenshot({ path: `${SHOTS}fein-${version}-${name}.png`, timeout: 240000 });
  console.log(name);
}
console.log('Fehler:', errors.filter((e) => !/fonts|ERR_|GPU stall|GL Driver/.test(e)).slice(0, 8));
await browser.close();
