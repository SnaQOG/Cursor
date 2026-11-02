// Gesamte Stadt: Ladezeit, Häuser, Tore, Türen und Treppen der äußeren Viertel, Himmelsbrücken, Kai, Galeone
import { open, SHOTS } from './common.mjs';
const { browser, page, errors } = await open();
const t0 = Date.now();
await page.evaluate(() => window.__game.selectMap('weltenbrueck'));
await page.waitForFunction(() => window.__game.mapId === 'weltenbrueck' && !window.__game.loadingMap, null, { timeout: 300000 });
console.log('Ladezeit Stadt', (Date.now() - t0) / 1000, 's');
const res = await page.evaluate(() => {
  const g = window.__game;
  g.startRaid(); g.player.hp = 1e9; g.player.maxHp = 1e9;
  for (const e of g.enemies) { e.alive = false; e.root.visible = false; }
  const p = g.player, d = g.world.mapData, lots = d.lots;
  const yawFor = (dx, dz) => Math.atan2(-dx, -dz);
  const walk = (x, y, z, yaw, secs) => {
    p.pos.set(x, y, z); p.vel.set(0, 0, 0); p.camYaw = yaw; p.yaw = yaw; p.alive = true;
    g.input.keys.clear(); g.input.keys.add('KeyW');
    for (let t = 0; t < secs; t += 1 / 30) { p.hp = 1e9; p.update(1 / 30); g.input.endFrame(); }
    g.input.keys.clear();
    return { x: p.pos.x, y: p.pos.y, z: p.pos.z };
  };
  const out = { stats: d.stats, spots: d.spots.length, containers: g.loot.containers.length, enemies: g.enemies.length, barrels: g.barrels.length, turrets: d.turrets.length };
  const by = {};
  for (const l of lots) { const k = `${l.district}/${l.kind}`; by[k] = (by[k] || 0) + 1; }
  out.lots = by;
  // Tore: von innen nach außen laufen
  out.gates = {
    west: walk(-190, 3.05, 96, yawFor(-1, 0), 6).x, ost: walk(190, 3.05, 96, yawFor(1, 0), 6).x, hafen: walk(0, 3.05, 190, yawFor(0, 1), 6).z,
  };
  // Türen und Treppen in den neuen Vierteln
  const OUT = { s: [0, 1], n: [0, -1], e: [1, 0], w: [-1, 0] };
  for (const dist of ['weiss', 'turm', 'hafen']) {
    const list = lots.filter((l) => l.district === dist && l.kind === 'house' && l.doors?.length);
    let doors = 0, dn = 0, stairs = 0, sn = 0;
    const failed = [];
    for (const l of list.filter((_, i) => i % 3 === 0).slice(0, 30)) {
      const door = l.doors[0], [ox, oz] = OUT[door.side];
      const fx = door.side === 'e' || door.side === 'w' ? (door.side === 'e' ? l.x1 : l.x0) : door.a;
      const fz = door.side === 'n' || door.side === 's' ? (door.side === 's' ? l.z1 : l.z0) : door.a;
      const r = walk(fx + ox * 1.6, l.level + 0.05, fz + oz * 1.6, yawFor(-ox, -oz), 2.2);
      dn++;
      if (r.x > l.x0 + 0.3 && r.x < l.x1 - 0.3 && r.z > l.z0 + 0.3 && r.z < l.z1 - 0.3) doors++; else failed.push(l.id);
      if (l.stair0) {
        const s = l.stair0, t = walk(s.x, s.y + 0.05, s.z, yawFor(s.dx, s.dz), 3.2);
        sn++;
        if (t.y - s.y > s.fh - 0.1) stairs++;
      }
    }
    out[dist] = { houses: list.length, doors: `${doors}/${dn}`, stairs: `${stairs}/${sn}`, failed };
  }
  // Himmelsbrücke: von einem Dach zum anderen
  const pairs = lots.filter((l) => l.district === 'turm' && l.bridged && l.gaps?.length);
  let ok = 0, n = 0;
  const seen = new Set();
  for (const a of pairs) {
    if (seen.has(a.id)) continue;
    const gap = a.gaps[0];
    const [ox, oz] = OUT[gap.side];
    const m = (gap.a0 + gap.a1) / 2;
    const fx = gap.side === 'e' ? a.x1 : gap.side === 'w' ? a.x0 : m, fz = gap.side === 's' ? a.z1 : gap.side === 'n' ? a.z0 : m;
    const r = walk(fx - ox * 1.2, a.roofY + 0.05, fz - oz * 1.2, yawFor(ox, oz), 5);
    const c = pairs.find((q) => q !== a && !seen.has(q.id) && r.x > q.x0 && r.x < q.x1 && r.z > q.z0 && r.z < q.z1);
    n++;
    if (c && r.y > a.roofY - 0.6) { ok++; seen.add(c.id); }
    seen.add(a.id);
  }
  out.bridges = `${ok}/${n}`;
  // Kai, Steg zur Galeone, Stufen aufs Deck
  const deck = walk(70, 3.05, 236, yawFor(0, 1), 8);
  out.deck = { z: +deck.z.toFixed(1), y: +deck.y.toFixed(2) };
  // Strand: aus dem Wasser wieder an Land
  const shore = walk(-90, 0.8, 262, yawFor(0, -1), 12);
  out.shore = { z: +shore.z.toFixed(1), y: +shore.y.toFixed(2), swim: p.swimming };
  // Spawns
  out.spawns = g.map.spawns.map(([x, z]) => [x, z, +g.col.groundAt(x, z, 0.4, 999, 0).toFixed(1)]);
  return out;
});
console.log(JSON.stringify(res, null, 1).replace(/\n\s+/g, ' '));
console.log('errors:', errors.filter((e) => !/fonts|ERR_|GPU stall|GL Driver/.test(e)).slice(0, 12));
await browser.close();
