// Bewegungstests: Türen, Treppen, Brücken, Spawns
import { open } from './common.mjs';
const { browser, page, errors } = await open();
await page.evaluate(() => window.__game.selectMap('weltenbrueck'));
await page.waitForFunction(() => window.__game.mapId === 'weltenbrueck' && !window.__game.loadingMap, null, { timeout: 300000 });
await page.evaluate(() => { const g = window.__game; g.startRaid(); g.player.hp = 1e9; g.player.maxHp = 1e9; for (const e of g.enemies) { e.root.visible = false; e.alive = false; } g.enemies.length = 0; });
const res = await page.evaluate(() => {
  const g = window.__game, p = g.player, lots = g.world.mapData.lots;
  const yawFor = (dx, dz) => Math.atan2(-dx, -dz);
  const walk = (x, y, z, yaw, secs) => {
    p.pos.set(x, y, z); p.vel.set(0, 0, 0); p.camYaw = yaw; p.yaw = yaw; p.alive = true;
    g.input.keys.clear(); g.input.keys.add('KeyW');
    g.debugStep(secs);
    g.input.keys.clear();
    return { x: +p.pos.x.toFixed(2), y: +p.pos.y.toFixed(2), z: +p.pos.z.toFixed(2) };
  };
  const out = { doors: [], stairs: [], spawns: [], bridges: [] };
  // Türen: von draußen hineinlaufen
  const OUT = { s: [0, 1], n: [0, -1], e: [1, 0], w: [-1, 0] };
  let n = 0;
  for (const l of lots) {
    if (l.kind !== 'house' || !l.doors?.length || n >= 60 || l.id % 3 !== 1) continue;
    const d = l.doors[0];
    const [ox, oz] = OUT[d.side];
    const fx = d.side === 'e' || d.side === 'w' ? (d.side === 'e' ? l.x1 : l.x0) : d.a;
    const fz = d.side === 'n' || d.side === 's' ? (d.side === 's' ? l.z1 : l.z0) : d.a;
    const r = walk(fx + ox * 1.6, l.level + 0.05, fz + oz * 1.6, yawFor(-ox, -oz), 2.2);
    const inside = r.x > l.x0 + 0.3 && r.x < l.x1 - 0.3 && r.z > l.z0 + 0.3 && r.z < l.z1 - 0.3;
    out.doors.push([l.id, l.style, d.side, inside]);
    n++;
  }
  // Treppen hinauf
  n = 0;
  for (const l of lots) {
    if (!l.stair0 || n >= 60 || l.id % 2 === 0) continue;
    const s = l.stair0;
    const r = walk(s.x, s.y + 0.05, s.z, yawFor(s.dx, s.dz), 3.2);
    out.stairs.push([l.id, l.style, +(r.y - s.y).toFixed(2), s.fh]);
    n++;
  }
  // Brücken: vom Turmdach zur Plattform
  for (const t of g.world.mapData.trees) {
    for (const bsd of t.bridges) {
      const D = 7;
      const dir = { n: [0, -1], s: [0, 1], w: [-1, 0], e: [1, 0] }[bsd.side];
      const sx = dir[0] ? t.x + dir[0] * (D + 9) : bsd.c, sz = dir[1] ? t.z + dir[1] * (D + 9) : bsd.c;
      // auf dem Dach des Hauses starten (Richtung Baum laufen)
      const r = walk(dir[0] ? t.x + dir[0] * (D + 2.5) : bsd.c, t.deckY + 0.3, dir[1] ? t.z + dir[1] * (D + 2.5) : bsd.c, yawFor(-dir[0], -dir[1]), 1.5);
      out.bridges.push([t.id, bsd.side, +(r.y - t.deckY).toFixed(2), Math.abs(r.x - t.x) < D && Math.abs(r.z - t.z) < D]);
      void sx; void sz;
    }
  }
  // Spawns: stehen bleiben
  for (const [x, z] of g.map.spawns) {
    p.spawn(x, z); g.input.keys.clear(); g.debugStep(1);
    out.spawns.push([x, z, +p.pos.y.toFixed(2), +g.terrain.height(x, z).toFixed(2)]);
  }
  return out;
});
console.log('doors', JSON.stringify(res.doors));
console.log('stairs', JSON.stringify(res.stairs));
console.log('bridges', JSON.stringify(res.bridges));
console.log('spawns', JSON.stringify(res.spawns));
console.log('errors:', errors.slice(0, 10));
await browser.close();
