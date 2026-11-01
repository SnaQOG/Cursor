// Baut Weltenbrück aus dem Stadtplan: Häuser, Dom, Rathaus, Markt,
// Terrassen, Stadtmauer, Riesenbäume mit Hängebrücken, Gründerfelsen und
// die Kulisse der künftigen Viertel.
import * as THREE from 'three';
import { Builder } from '../../structures.js';
import { HouseKit, PAL, STYLE, mix } from './house.js';
import {
  CITY, STAIRS, STAIR_RUN, PLAZA, DOM, RATHAUS, STREETS, GIANT_TREES, FACES, STEIG, Occupancy, levelAt, overlap,
} from './layout.js';
import { mulberry32 } from '../../util.js';

const STONE_LIGHT = 0xb7ab92;
const STONE_WALL = 0x9a917f;
const METAL = 0x5c5650;
const COPPER = 0x8a5a3a;

export function buildCity({ batcher, col, terrain, assets, world, layout, instancer }) {
  const B = new Builder(batcher, col, terrain, assets);
  const b = batcher;
  const rnd = mulberry32(4711);
  const R = (a, c) => a + rnd() * (c - a);
  const pick = (arr) => arr[Math.floor(rnd() * arr.length) % arr.length];
  const enemySpawns = [], turrets = [], containers = [], barrels = [], labels = [];
  const glowMats = [];
  const lots = layout.lots.map((l) => ({ ...l }));

  // ---------------------------------------------------------------- Sonderbauten als Grundstücke
  const addLot = (l) => { l.id = lots.length; lots.push(l); return l; };
  const rathaus = addLot({ ...RATHAUS, kind: 'house', style: 'backstein', floors: 3, shop: true, tier: 3, poi: 'rathaus', level: 3, district: 'markt' });
  const domTower = addLot({ ...DOM.tower, kind: 'house', style: 'dom', floors: 8, tier: 3, poi: 'dom', level: 3, district: 'markt', noTank: true, forceDoors: [{ side: 'n', a: 0 }] });
  const steig = addLot({ ...STEIG.tower, kind: 'house', style: 'turm', floors: 4, tier: 3, poi: 'felsen', level: 18, district: 'wald', noTank: true, gaps: [{ side: 'n', a0: -169, a1: -167 }] });

  // Häuser an den Riesenbäumen werden zu Wohntürmen mit Brücke zur Plattform
  const DECK = 7;
  const bridges = [];
  for (const t of GIANT_TREES) {
    const lvl = levelAt(t.z);
    t.deckY = lvl + 0.15 + 3 * STYLE.turm.fh;
    t.level = lvl;
    t.bridges = [];
    const dirs = [
      { side: 'n', axis: 'z', sign: -1 }, { side: 's', axis: 'z', sign: 1 },
      { side: 'w', axis: 'x', sign: -1 }, { side: 'e', axis: 'x', sign: 1 },
    ];
    for (const d of dirs) {
      let best = null;
      for (const l of lots) {
        if (l.district !== 'wald' || l.level !== lvl) continue;
        if (d.axis === 'z') {
          const ov = Math.min(l.x1, t.x + DECK - 1.5) - Math.max(l.x0, t.x - DECK + 1.5);
          if (ov < 2.2) continue;
          const dist = d.sign < 0 ? (t.z - DECK) - l.z1 : l.z0 - (t.z + DECK);
          if (dist < 0.5 || dist > 14) continue;
          if (!best || dist < best.dist) best = { l, dist, c: (Math.max(l.x0, t.x - DECK + 1.5) + Math.min(l.x1, t.x + DECK - 1.5)) / 2 };
        } else {
          const ov = Math.min(l.z1, t.z + DECK - 1.5) - Math.max(l.z0, t.z - DECK + 1.5);
          if (ov < 2.2) continue;
          const dist = d.sign < 0 ? (t.x - DECK) - l.x1 : l.x0 - (t.x + DECK);
          if (dist < 0.5 || dist > 14) continue;
          if (!best || dist < best.dist) best = { l, dist, c: (Math.max(l.z0, t.z - DECK + 1.5) + Math.min(l.z1, t.z + DECK - 1.5)) / 2 };
        }
      }
      if (!best || best.l.bridged) continue;
      const l = best.l;
      l.bridged = true;
      l.kind = 'house';
      l.style = 'turm';
      l.floors = 3;
      l.noTank = true;
      // Lücke in der Brüstung des Hauses (gegenüberliegende Seite)
      const opp = { n: 's', s: 'n', w: 'e', e: 'w' }[d.side];
      l.gaps = [...(l.gaps || []), { side: opp, a0: best.c - 0.9, a1: best.c + 0.9 }];
      const from = d.axis === 'z' ? { x: best.c, z: t.z + d.sign * DECK } : { x: t.x + d.sign * DECK, z: best.c };
      const to = d.axis === 'z' ? { x: best.c, z: d.sign < 0 ? l.z1 : l.z0 } : { x: d.sign < 0 ? l.x1 : l.x0, z: best.c };
      t.bridges.push({ side: d.side, c: best.c });
      bridges.push({ from, to, y: t.deckY, axis: d.axis });
    }
  }

  // ---------------------------------------------------------------- Belegung
  const occ = new Occupancy();
  for (const l of lots) occ.mark(l, l.id);
  occ.mark(DOM.nave, -3);
  occ.mark(DOM.choir, -3);
  for (const s of STAIRS) occ.mark({ x0: -197, x1: 197, z0: s.z - 0.2, z1: s.z + 2 }, -4);
  occ.mark({ x0: -200, x1: -197, z0: -200, z1: 200 }, -4);
  occ.mark({ x0: 197, x1: 200, z0: -200, z1: 200 }, -4);
  occ.mark({ x0: -200, x1: 200, z0: 197, z1: 200 }, -4);
  for (const t of GIANT_TREES) occ.mark({ x0: t.x - 2.6, x1: t.x + 2.6, z0: t.z - 2.6, z1: t.z + 2.6 }, -3);

  // ---------------------------------------------------------------- Häuser
  const kit = new HouseKit(B, occ, lots, instancer);
  const inst = (name, x, y, z, ry = 0, scale = 1, opts = {}) => instancer.add(name, x, y, z, ry, scale, opts);
  STYLE.dom = { fh: 3.0, roof: 'flat', pitch: 0, roofMat: 'stone', spacing: 3.2, winW: 0.8, winH: 1.8 };
  kit.styleHook = (lot) => (lot.style === 'dom' ? { wall: STONE_LIGHT, base: STONE_LIGHT, mat: 'stone', frame: STONE_WALL } : null);
  for (const l of lots) kit.house(l, rnd);

  // ---------------------------------------------------------------- Terrassen: Stützmauern und Treppen
  for (const s of STAIRS) {
    const lower = levelAt(s.z + 1), upper = levelAt(s.z - 1);
    const edges = [];
    for (const [cx, w] of s.list) edges.push([cx - w / 2, cx + w / 2]);
    edges.sort((a, c) => a[0] - c[0]);
    let x = -197;
    const wallSeg = (a0, a1) => {
      if (a1 - a0 < 0.1) return;
      B.solid((a0 + a1) / 2, lower - 0.6, s.z + 0.9, a1 - a0, upper + 0.9 - (lower - 0.6), 2.2, STONE_WALL, { mat: 'stone' });
      // Abdeckplatte
      b.box((a0 + a1) / 2, upper + 0.9, s.z + 0.9, a1 - a0, 0.12, 2.4, STONE_LIGHT, { mat: 'stone' });
      // Moos-Streifen
      for (let k = 0; k < (a1 - a0) / 9; k++) {
        const mx = a0 + rnd() * (a1 - a0), mh = R(1, 3.5);
        b.addQuad([mx, upper + 0.9 - mh, s.z + 2.03], [mx + R(1, 3), upper + 0.9 - mh * 0.7, s.z + 2.03], [mx + R(1, 3), upper + 0.85, s.z + 2.03], [mx, upper + 0.85, s.z + 2.03], PAL.moss, { shade: false });
      }
    };
    for (const [e0, e1] of edges) {
      wallSeg(x, e0);
      x = e1;
      const w = e1 - e0, cx = (e0 + e1) / 2;
      B.stairs(cx, s.z - STAIR_RUN / 2, lower, upper, 'n', w, STONE_LIGHT);
      // Seitenwände der Treppengasse auf der oberen Terrasse
      for (const [sx0, sx1] of [[e0 - 2, e0], [e1, e1 + 2]]) {
        B.solid((sx0 + sx1) / 2, lower - 0.6, s.z - STAIR_RUN / 2 - 0.15, sx1 - sx0, upper + 0.9 - (lower - 0.6), STAIR_RUN + 0.3, STONE_WALL, { mat: 'stone' });
      }
      // Laternen am Treppenkopf
      lantern(cx - w / 2 - 1, s.z - STAIR_RUN - 0.6, upper);
      lantern(cx + w / 2 + 1, s.z - STAIR_RUN - 0.6, upper);
    }
    wallSeg(x, 197);
    // Bänke und Laternen auf der Promenade der oberen Terrasse
    for (let px = -185; px < 190; px += R(18, 28)) {
      if (edges.some(([e0, e1]) => px > e0 - 4 && px < e1 + 4)) continue;
      if (rnd() < 0.5) bench(px, s.z - 1.2, upper, 0);
      else lantern(px, s.z - 0.9, upper);
    }
  }

  // ---------------------------------------------------------------- Stadtmauer mit Türmen und Toren
  const WALL_H = 11;
  const wallBands = [[-172, -120, 18], [-120, -80, 13], [-80, -40, 8], [-40, 200, 3]];
  const gates = [
    { x: -198.5, z: 96, axis: 'z', name: 'TOR ZUR WEISSEN STADT' },
    { x: 198.5, z: 96, axis: 'z', name: 'TOR ZUR TURMSTADT' },
    { x: 0, z: 198.5, axis: 'x', name: 'HAFENTOR' },
  ];
  const inGate = (x, z) => gates.some((g) => (g.axis === 'z' ? Math.abs(x - g.x) < 3 && Math.abs(z - g.z) < 7 : Math.abs(z - g.z) < 3 && Math.abs(x - g.x) < 7));
  const wallPiece = (x0, x1, z0, z1, lvl) => {
    B.solid((x0 + x1) / 2, lvl - 1, (z0 + z1) / 2, x1 - x0, WALL_H + 1, z1 - z0, STONE_WALL, { mat: 'stone' });
  };
  for (const side of [-1, 1]) {
    for (const [za, zb, lvl] of wallBands) {
      const x0 = side < 0 ? -200 : 197, x1 = x0 + 3;
      if (za < 96 && zb > 96) {
        wallPiece(x0, x1, za, 89, lvl);
        wallPiece(x0, x1, 103, zb, lvl);
      } else wallPiece(x0, x1, za, zb, lvl);
      // Zinnen außen
      for (let z = za + 1; z < zb; z += 2.2) {
        if (inGate(x0 + 1.5, z)) continue;
        b.box(side < 0 ? x0 + 0.4 : x1 - 0.4, lvl + WALL_H, z, 0.8, 1.1, 1.2, STONE_WALL, { mat: 'stone' });
      }
    }
  }
  wallPiece(-200, -7, 197, 200, 3);
  wallPiece(7, 200, 197, 200, 3);
  for (let x = -198; x < 198; x += 2.2) if (!inGate(x, 198.5)) b.box(x, 3 + WALL_H, 199.6, 1.2, 1.1, 0.8, STONE_WALL, { mat: 'stone' });
  // Mauertürme
  const tower = (x, z, lvl, h = 17) => {
    B.solid(x, lvl - 1, z, 7, h + 1, 7, STONE_WALL, { mat: 'stone' });
    b.addPrim('cone4', x, lvl + h + 3, z, 5.6, 6, 5.6, pick(PAL.slate), { ry: Math.PI / 4, mat: 'slate' });
    for (const [dx, dz] of [[0, 3.52], [0, -3.52], [3.52, 0], [-3.52, 0]]) {
      b.addQuad(...slit(x + dx, lvl + h - 4, z + dz, dx, dz), 0x1d1a17, { shade: false });
    }
  };
  for (const side of [-1, 1]) {
    for (let z = -150; z <= 190; z += 42) {
      if (Math.abs(z - 96) < 14) continue;
      tower(side * 201.5, z, levelAt(z));
    }
  }
  for (let x = -170; x <= 170; x += 42) if (Math.abs(x) > 14) tower(x, 201.5, 3);
  // Tore (geschlossen, Fallgitter unten)
  for (const g of gates) {
    const lvl = 3;
    const along = g.axis === 'z';
    for (const s of [-1, 1]) {
      const tx = along ? g.x + Math.sign(g.x) * 3.6 : g.x + s * 9.5, tz = along ? g.z + s * 9.5 : g.z + 3.6;
      B.solid(tx, lvl - 1, tz, along ? 9 : 7, 21, along ? 7 : 9, STONE_WALL, { mat: 'stone' });
      b.addPrim('cone4', tx, lvl + 23, tz, 6.5, 7, 6.5, pick(PAL.slate), { ry: Math.PI / 4, mat: 'slate' });
    }
    // Torbogen und Fallgitter
    B.solid(g.x, lvl + 7, g.z, along ? 3 : 12, WALL_H - 6, along ? 12 : 3, STONE_WALL, { mat: 'stone' });
    for (let k = -5; k <= 5; k += 0.7) {
      const px = along ? g.x - 0.9 * Math.sign(g.x) : g.x + k, pz = along ? g.z + k : g.z - 0.9;
      b.box(px, lvl, pz, along ? 0.12 : 0.1, 7, along ? 0.1 : 0.12, 0x2b2724, { mat: 'metal' });
    }
    for (let yy = 0.6; yy < 7; yy += 0.8) b.box(along ? g.x - 0.9 * Math.sign(g.x) : g.x, lvl + yy, along ? g.z : g.z - 0.9, along ? 0.14 : 11, 0.1, along ? 11 : 0.14, 0x2b2724, { mat: 'metal' });
    const px0 = along ? g.x - 0.9 * Math.sign(g.x) : g.x, pz0 = along ? g.z : g.z - 0.9;
    col.addBox(px0 - (along ? 0.2 : 6), lvl, pz0 - (along ? 6 : 0.2), px0 + (along ? 0.2 : 6), lvl + 7, pz0 + (along ? 6 : 0.2));
    const inward = along ? -Math.sign(g.x) : -1;
    labels.push({
      text: g.name, x: along ? g.x + inward * 1.7 : g.x, y: lvl + 8.2, z: along ? g.z : g.z + inward * 1.7,
      ry: along ? (inward > 0 ? Math.PI / 2 : -Math.PI / 2) : (inward > 0 ? 0 : Math.PI), w: 8, h: 1.4,
    });
    // Schutt vor dem Tor
    for (let i = 0; i < 6; i++) {
      const s = R(0.6, 1.6);
      const rx = along ? g.x + inward * R(2, 6) : g.x + R(-5, 5), rz = along ? g.z + R(-5, 5) : g.z + inward * R(2, 6);
      b.addPrim('ico', rx, lvl + s * 0.3, rz, s * 1.3, s * 0.7, s, pick(PAL.rubble), { mat: 'stone', ry: rnd() * 6 });
    }
  }

  // ---------------------------------------------------------------- Dom mit Erfinderturm
  buildDom();

  // ---------------------------------------------------------------- Kreuzmarkt
  buildPlaza();

  // ---------------------------------------------------------------- Riesenbäume, Plattformen, Brücken
  for (const t of GIANT_TREES) giantTree(t);
  for (const br of bridges) ropeBridge(br.from, br.to, br.y, br.axis);

  // ---------------------------------------------------------------- Gründerfelsen
  buildCliff();

  // ---------------------------------------------------------------- Plätze, Laternen, Kram in den Gassen
  buildStreets();

  // ---------------------------------------------------------------- Kulisse der künftigen Viertel
  buildBackdrop();

  // ---------------------------------------------------------------- Gegner
  const E = (type, x, z, n = 1, spread = 6, extra = {}) => {
    for (let i = 0; i < n; i++) enemySpawns.push({ type, x: x + R(-spread, spread), z: z + R(-spread, spread), ...extra });
  };
  // Gassen-ARC: Zecken und Knaller in Gassen und Höfen
  const alleys = layout.alleys.filter((a) => Math.min(a.x1 - a.x0, a.z1 - a.z0) >= 2);
  for (let i = 0; i < 26; i++) {
    const a = alleys[Math.floor(rnd() * alleys.length)];
    const x = (a.x0 + a.x1) / 2 + (rnd() - 0.5) * (a.x1 - a.x0 - 1), z = (a.z0 + a.z1) / 2 + (rnd() - 0.5) * (a.z1 - a.z0 - 1);
    enemySpawns.push({ type: rnd() < 0.6 ? 'tick' : 'pop', x, z, roam: rnd() < 0.4 });
  }
  E('rocketeer', 0, 96, 1, 10);
  E('bastion', 0, 102, 1, 4);
  E('leaper', -60, 2, 1, 10);
  E('leaper', 60, -60, 1, 10);
  E('rocketeer', -48, -120, 1, 8);
  E('snitch', 0, 40, 1, 6, { y: 34, roam: true });
  E('snitch', 20, -100, 1, 10, { y: 30, roam: true });
  E('hornet', 0, 140, 1, 20, { roam: true });
  E('hornet', -100, -40, 1, 20, { roam: true });
  for (let i = 0; i < 8; i++) enemySpawns.push({ type: 'wasp', x: R(-180, 180), z: R(-150, 180), roam: true });
  for (const t of GIANT_TREES) E('wasp', t.x, t.z, 1, 4, { y: t.deckY + 8, roam: true });

  // ---------------------------------------------------------------- Ergebnis
  return {
    builder: B, spots: B.spots, enemySpawns, turrets, containers, barrels, labels, glowMats,
    spotChance: 0.24, stats: kit.stats, lots, trees: GIANT_TREES,
    // Beutebehälter im Stadtstil
    containerSkin: { crate: 'truhe', locker: 'schrank', drawer: 'kommode' },
    containerNames: { crate: 'Truhe', locker: 'Schrank', drawer: 'Kommode' },
  };

  // ================================================================ Hilfen
  function slit(x, y, z, dx, dz) {
    const w = 0.35, h = 1.6;
    if (dz !== 0) {
      const zz = z + Math.sign(dz) * 0.03;
      return dz > 0 ? [[x - w / 2, y, zz], [x + w / 2, y, zz], [x + w / 2, y + h, zz], [x - w / 2, y + h, zz]] : [[x + w / 2, y, zz], [x - w / 2, y, zz], [x - w / 2, y + h, zz], [x + w / 2, y + h, zz]];
    }
    const xx = x + Math.sign(dx) * 0.03;
    return dx > 0 ? [[xx, y, z + w / 2], [xx, y, z - w / 2], [xx, y + h, z - w / 2], [xx, y + h, z + w / 2]] : [[xx, y, z - w / 2], [xx, y, z + w / 2], [xx, y + h, z + w / 2], [xx, y + h, z - w / 2]];
  }

  function lantern(x, z, y) {
    if (inst('kc_lantern', x, y, z, rnd() * 6)) {
      col.addCyl(x, z, 0.14, y, y + 3.6);
      return;
    }
    b.box(x, y, z, 0.14, 3.6, 0.14, 0x26221f, { mat: 'metal' });
    b.box(x, y + 3.6, z, 0.36, 0.08, 0.36, 0x26221f, { mat: 'metal' });
    b.box(x, y + 3.1, z, 0.26, 0.42, 0.26, 0xffcf7a, { mat: 'glow' });
    b.addPrim('cone4', x, y + 3.82, z, 0.3, 0.36, 0.3, 0x26221f, { mat: 'metal', ry: Math.PI / 4 });
    col.addCyl(x, z, 0.12, y, y + 3.6);
  }

  function bench(x, z, y, ry) {
    const w = 1.8;
    const sw = Math.abs(Math.sin(ry)) > 0.5;
    if (inst('kc_bench', x, y, z, ry)) {
      col.addBox(x - (sw ? 0.22 : w / 2), y, z - (sw ? w / 2 : 0.22), x + (sw ? 0.22 : w / 2), y + 0.49, z + (sw ? w / 2 : 0.22));
      return;
    }
    b.box(x, y + 0.42, z, sw ? 0.45 : w, 0.07, sw ? w : 0.45, 0x6a4a30, { mat: 'wood' });
    b.box(x, y, z, sw ? 0.4 : w * 0.9, 0.42, sw ? w * 0.9 : 0.4, 0x3a3532, { mat: 'metal' });
    col.addBox(x - (sw ? 0.22 : w / 2), y, z - (sw ? w / 2 : 0.22), x + (sw ? 0.22 : w / 2), y + 0.49, z + (sw ? w / 2 : 0.22));
  }

  function rubble(x, z, y, n = 4, spread = 1.5) {
    for (let i = 0; i < n; i++) {
      const s = R(0.4, 1.2);
      b.addPrim('ico', x + R(-spread, spread), y + s * 0.3, z + R(-spread, spread), s * 1.3, s * 0.6, s, pick(PAL.rubble), { mat: 'stone', ry: rnd() * 6 });
    }
  }

  function cart(x, z, y, ry) {
    const ca = Math.cos(ry), sa = Math.sin(ry);
    if (inst('kc_cart', x, y, z, ry)) {
      const hw = Math.abs(ca) * 1.2 + Math.abs(sa) * 0.65, hd = Math.abs(sa) * 1.2 + Math.abs(ca) * 0.65;
      col.addBox(x - hw, y, z - hd, x + hw, y + 1.1, z + hd);
      return;
    }
    b.box(x, y + 0.55, z, 2.4, 0.5, 1.3, 0x6a4a30, { ry, mat: 'wood' });
    for (const [lx, lz] of [[0.7, 0.72], [0.7, -0.72], [-0.7, 0.72], [-0.7, -0.72]]) {
      b.addPrim('torus', x + lx * ca + lz * sa, y + 0.5, z - lx * sa + lz * ca, 0.5, 0.5, 1.6, 0x3a2a1c, { ry: ry + Math.PI / 2, mat: 'wood' });
    }
    // Deichsel
    b.box(x + 1.9 * ca, y + 0.45, z - 1.9 * sa, 1.6, 0.08, 0.08, 0x5a3e2a, { ry, rz: 0.15, mat: 'wood' });
    const hw = Math.abs(ca) * 1.2 + Math.abs(sa) * 0.65, hd = Math.abs(sa) * 1.2 + Math.abs(ca) * 0.65;
    col.addBox(x - hw, y, z - hd, x + hw, y + 0.8, z + hd);
  }

  function stall(x, z, y, ry, color) {
    const sw = Math.abs(Math.sin(ry)) > 0.5;
    const w = 3.6, d = 2.2;
    const W = sw ? d : w, D = sw ? w : d;
    if (inst('kc_stall', x, y, z, ry, 1, { tint: color })) {
      col.addBox(x - (W - 0.3) / 2, y, z - (D - 0.6) / 2, x + (W - 0.3) / 2, y + 0.95, z + (D - 0.6) / 2);
      return;
    }
    for (const [dx, dz] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) b.box(x + dx * (W / 2 - 0.1), y, z + dz * (D / 2 - 0.1), 0.1, 2.4, 0.1, 0x5a3e2a, { mat: 'wood' });
    B.solid(x, y, z, W - 0.3, 0.9, D - 0.6, 0x6e4e32, { mat: 'wood' });
    // Ware
    for (let i = 0; i < 5; i++) b.box(x + R(-W / 2 + 0.5, W / 2 - 0.5), y + 0.9, z + R(-D / 2 + 0.4, D / 2 - 0.4), R(0.25, 0.5), R(0.15, 0.35), R(0.25, 0.5), pick([0xa83a2a, 0xd9a33a, 0x6a8a3a, 0xc9b28a, 0x7a4a8a]));
    // Markise (schräg)
    const h0 = y + 2.5, h1 = y + 2.1;
    if (sw) b.addQuad([x - W / 2 - 0.2, h1, z - D / 2 - 0.2], [x - W / 2 - 0.2, h1, z + D / 2 + 0.2], [x + W / 2 + 0.2, h0, z + D / 2 + 0.2], [x + W / 2 + 0.2, h0, z - D / 2 - 0.2], color, { double: true });
    else b.addQuad([x - W / 2 - 0.2, h0, z - D / 2 - 0.2], [x + W / 2 + 0.2, h0, z - D / 2 - 0.2], [x + W / 2 + 0.2, h1, z + D / 2 + 0.2], [x - W / 2 - 0.2, h1, z + D / 2 + 0.2], color, { double: true });
  }

  // ---------------------------------------------------------------- Dom
  function buildDom() {
    const y = 3;
    const N = DOM.nave;
    const wallH = 18;
    const inner = { x0: N.x0 + 0.3 - 0.02, x1: N.x1 - 0.3 + 0.02, z0: N.z0 + 0.3 - 0.02, z1: N.z1 - 0.3 + 0.02, k: 0.45 };
    const sides = {
      e: { side: 'e', fixed: N.x1, a0: N.z0, a1: N.z1, ia0: N.z0 + 0.3, ia1: N.z1 - 0.3 },
      w: { side: 'w', fixed: N.x0, a0: N.z0, a1: N.z1, ia0: N.z0 + 0.3, ia1: N.z1 - 0.3 },
      n: { side: 'n', fixed: N.z0, a0: N.x0, a1: N.x1, ia0: N.x0 + 0.3, ia1: N.x1 - 0.3 },
      s: { side: 's', fixed: N.z1, a0: N.x0, a1: N.x1, ia0: N.x0 + 0.3, ia1: N.x1 - 0.3 },
    };
    const glassCols = [0x3a6ab0, 0xb03a3a, 0xd9a33a, 0x3a9a6a, 0x7a4ab0];
    for (const k of ['e', 'w']) {
      const w = sides[k];
      const ops = [];
      for (let z = N.z0 + 3.5; z < N.z1 - 2; z += 5) {
        if (Math.abs(z - 41) < 2) ops.push({ a0: z - 0.9, a1: z + 0.9, y0: 0.15, y1: 2.9, kind: 'door', frame: true });
        else ops.push({ a0: z - 1.0, a1: z + 1.0, y0: 4.5, y1: 14, kind: 'win', frame: true });
      }
      kit.wall(w, y, wallH, ops, STONE_LIGHT, { mat: 'stone', inner, frameCol: STONE_WALL });
      // Bleiglas
      for (const o of ops) {
        if (o.kind !== 'win') continue;
        const xx = k === 'e' ? N.x1 - 0.15 : N.x0 + 0.15;
        b.addQuad([xx, y + o.y0, o.a0], [xx, y + o.y0, o.a1], [xx, y + o.y1, o.a1], [xx, y + o.y1, o.a0], pick(glassCols), { mat: 'glass', double: true });
        // Spitzbogen angedeutet
        b.addTri([xx, y + o.y1, o.a0], [xx, y + o.y1, o.a1], [xx, y + o.y1 + 1.2, (o.a0 + o.a1) / 2], pick(glassCols), { mat: 'glass', double: true });
      }
      // Strebepfeiler
      for (let z = N.z0 + 1; z <= N.z1 - 1; z += 5) {
        const sx = k === 'e' ? N.x1 + 0.7 : N.x0 - 0.7;
        B.solid(sx, y, z, 1.4, wallH - 2, 1.4, STONE_WALL, { mat: 'stone' });
        if (!inst('kc_pinnacle', sx, y + wallH - 2, z, 0)) b.addPrim('cone4', sx, y + wallH - 0.5, z, 0.9, 2.2, 0.9, STONE_WALL, { mat: 'stone', ry: Math.PI / 4 });
      }
    }
    // Nordwand mit großem Bogen zum Chor, Südwand mit Tür zum Turm
    kit.wall(sides.n, y, wallH, [{ a0: -6, a1: 6, y0: 0, y1: 12, kind: 'door' }], STONE_LIGHT, { mat: 'stone', inner });
    kit.wall(sides.s, y, wallH, [{ a0: -0.62, a1: 0.62, y0: 0.15, y1: 2.35, kind: 'door' }], STONE_LIGHT, { mat: 'stone', inner });
    kit.slab({ x0: N.x0, x1: N.x1, z0: N.z0, z1: N.z1 }, y, y + 0.15, null, 0x8f8676, inner, true);
    kit.slab({ x0: N.x0 + 0.3, x1: N.x1 - 0.3, z0: N.z0 + 0.3, z1: N.z1 - 0.3 }, y + wallH - 0.4, y + wallH, null, 0x3a2e24, inner);
    kit.gableRoof({ ...N, style: 'stein' }, false, y + wallH, 1.25, pick(PAL.slate), 'slate', STONE_LIGHT, 0, rnd, false);
    // Säulen und Bänke
    for (let z = N.z0 + 6; z < N.z1 - 2; z += 5) {
      for (const sx of [-6, 6]) {
        b.addPrim('cyl8', sx, y + 0.15 + 8.5, z, 0.6, 17, 0.6, STONE_LIGHT, { mat: 'stone', layer: 'in', inner });
        col.addCyl(sx, z, 0.6, y, y + wallH);
      }
      for (const bx of [-3, 3]) {
        b.box(bx, y + 0.15, z + 2.5, 4, 0.45, 0.6, 0x5a3e2a, { mat: 'wood', layer: 'in', inner });
        col.addBox(bx - 2, y + 0.15, z + 2.2, bx + 2, y + 0.6, z + 2.8);
      }
    }
    // Chor
    const Cr = DOM.choir;
    const cIn = { x0: Cr.x0 + 0.28, x1: Cr.x1 - 0.28, z0: Cr.z0 + 0.28, z1: N.z0 + 0.5, k: 0.45 };
    const cw = (side, fixed, a0, a1) => ({ side, fixed, a0, a1, ia0: a0 + 0.3, ia1: a1 - 0.3 });
    const chH = 14;
    for (const [w, n] of [[cw('e', Cr.x1, Cr.z0, Cr.z1), 2], [cw('w', Cr.x0, Cr.z0, Cr.z1), 2], [cw('n', Cr.z0, Cr.x0, Cr.x1), 3]]) {
      const ops = [];
      for (let i = 0; i < n; i++) {
        const c = w.a0 + ((w.a1 - w.a0) / n) * (i + 0.5);
        ops.push({ a0: c - 0.8, a1: c + 0.8, y0: 4, y1: 11, kind: 'win', frame: true });
      }
      kit.wall(w, y, chH, ops, STONE_LIGHT, { mat: 'stone', inner: cIn, frameCol: STONE_WALL });
      for (const o of ops) {
        const p = (a, yy) => (w.side === 'n' ? [a, yy, Cr.z0 + 0.15] : [w.side === 'e' ? Cr.x1 - 0.15 : Cr.x0 + 0.15, yy, a]);
        b.addQuad(p(o.a0, y + o.y0), p(o.a1, y + o.y0), p(o.a1, y + o.y1), p(o.a0, y + o.y1), pick(glassCols), { mat: 'glass', double: true });
      }
    }
    kit.slab({ x0: Cr.x0, x1: Cr.x1, z0: Cr.z0, z1: Cr.z1 }, y, y + 0.15, null, 0x8f8676, cIn, true);
    kit.slab({ x0: Cr.x0 + 0.3, x1: Cr.x1 - 0.3, z0: Cr.z0 + 0.3, z1: Cr.z1 }, y + chH - 0.4, y + chH, null, 0x3a2e24, cIn);
    kit.gableRoof({ ...Cr, style: 'stein' }, false, y + chH, 1.1, pick(PAL.slate), 'slate', STONE_LIGHT, 0, rnd, false);
    // Altar mit ARC-Kristall
    B.solid(0, y + 0.15, Cr.z0 + 4, 3.2, 1.1, 1.4, STONE_LIGHT, { mat: 'stone' });
    b.addPrim('ico', 0, y + 2.2, Cr.z0 + 4, 0.5, 1.1, 0.5, 0x7ff6ff, { mat: 'glow' });
    containers.push({ type: 'weapon', x: -3.5, y: y + 0.15, z: N.z0 + 3, tier: 3, ry: 0 });
    containers.push({ type: 'medical', x: 3.5, y: y + 0.15, z: N.z0 + 3, tier: 3, ry: 0 });
    containers.push({ type: 'arcprobe', x: 0, y: y + 0.15, z: Cr.z0 + 6.5, tier: 3, ry: 0 });
    // Erfinderturm über dem Turmsockel
    const ty = domTower.roofY;
    const tx = 0, tz = (DOM.tower.z0 + DOM.tower.z1) / 2;
    const spire = B.model('l_spire', tx, ty, tz, 0);
    if (!spire) {
      b.addPrim('cyl8', tx, ty + 7, tz, 4.6, 14, 4.6, STONE_LIGHT, { mat: 'stone', ry: Math.PI / 8 });
      // Schallöffnungen (dunkel)
      for (let k = 0; k < 8; k++) {
        const a = (k / 8) * Math.PI * 2 + Math.PI / 8;
        const r = 4.45;
        b.box(tx + Math.cos(a) * r, ty + 6, tz + Math.sin(a) * r, 1.2, 3.4, 0.12, 0x15171a, { ry: -a + Math.PI / 2 });
      }
      b.addPrim('cone8', tx, ty + 14 + 13, tz, 4.9, 26, 4.9, pick(PAL.slate), { mat: 'slate', ry: Math.PI / 8 });
    }
    col.addCyl(tx, tz, 4.6, ty, ty + 14);
    col.addCyl(tx, tz, 3.2, ty + 14, ty + 30);
    // Kupferringe, Rohre, Zahnrad und Antenne
    if (!spire) for (const yy of [ty + 2.2, ty + 12.6]) b.addPrim('torus', tx, yy, tz, 5.1, 5.1, 6, COPPER, { mat: 'metal', rx: Math.PI / 2 });
    for (let k = 0; k < 4; k++) {
      const a = (k / 4) * Math.PI * 2;
      b.box(tx + Math.cos(a) * 4.9, ty - 22, tz + Math.sin(a) * 4.9 * 0.85, 0.35, 34, 0.35, COPPER, { mat: 'metal' });
    }
    b.addPrim('torus', tx, 3 + 17, DOM.tower.z1 + 0.25, 2.6, 2.6, 4, METAL, { mat: 'metal' });
    for (let k = 0; k < 6; k++) b.box(tx, 3 + 17, DOM.tower.z1 + 0.25, 5.0, 0.3, 0.2, METAL, { rz: (k / 6) * Math.PI, mat: 'metal' });
    if (!spire) b.addPrim('cyl8', tx, ty + 44, tz, 0.25, 8, 0.25, METAL, { mat: 'metal' });
    // Energiekern (pulsierend, eigene Materialien)
    const coreMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const ringMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    glowMats.push(coreMat, ringMat);
    const core = new THREE.Mesh(new THREE.IcosahedronGeometry(1.5, 1), coreMat);
    core.material.color.set(0x9ff8ff);
    core.position.set(tx, ty + 6, tz);
    const ringGeo = new THREE.TorusGeometry(5.3, 0.22, 6, 40);
    for (const yy of [ty + 4.2, ty + 9.5]) {
      const r = new THREE.Mesh(ringGeo, ringMat);
      r.rotation.x = Math.PI / 2;
      r.position.set(tx, yy, tz);
      world.scene.add(r);
    }
    const tip = new THREE.Mesh(new THREE.SphereGeometry(0.6, 10, 8), ringMat);
    tip.position.set(tx, ty + 48.4, tz);
    world.scene.add(core, tip);
    // Licht im Inneren des Doms (Kristall)
    turrets.push({ x: DOM.tower.x0 + 1.6, y: domTower.roofY, z: DOM.tower.z1 - 1.6 });
    labels.push({ text: 'DOM ZU WELTENBRÜCK', x: 0, y: 3 + 3.5, z: DOM.tower.z1 + 0.06, ry: 0, w: 7, h: 1.3 });
    labels.push({ text: 'RATHAUS', x: (RATHAUS.x0 + RATHAUS.x1) / 2, y: 3 + 5.2, z: RATHAUS.z0 - 0.06, ry: Math.PI, w: 5, h: 1.2 });
  }

  // ---------------------------------------------------------------- Kreuzmarkt
  function buildPlaza() {
    const y = 3;
    const fx = 0, fz = 96;
    // Brunnen: achteckiger Rand, Säule, Figur
    const fountain = B.model('kc_fountain', fx, y, fz, 0);
    for (let k = 0; k < 8 && !fountain; k++) {
      const a = (k / 8) * Math.PI * 2;
      const r = 4.8;
      const px = fx + Math.cos(a) * r, pz = fz + Math.sin(a) * r;
      b.box(px, y, pz, 3.9, 0.75, 0.55, STONE_LIGHT, { ry: -a + Math.PI / 2, mat: 'stone' });
    }
    col.addCyl(fx, fz, 5.1, y, y + 0.75);
    if (!fountain) {
      b.addPrim('cyl8', fx, y + 1.8, fz, 0.6, 3.6, 0.6, STONE_LIGHT, { mat: 'stone' });
      b.addPrim('cyl8', fx, y + 3.7, fz, 1.4, 0.3, 1.4, STONE_LIGHT, { mat: 'stone' });
      b.addPrim('sphere', fx, y + 4.6, fz, 0.55, 0.65, 0.5, 0x8a8a82, { mat: 'metal' });
      b.addPrim('cone8', fx, y + 5.8, fz, 0.35, 1.4, 0.35, 0x8a8a82, { mat: 'metal' });
    }
    col.addCyl(fx, fz, 0.7, y, y + 6);
    // Marktstände in zwei Reihen
    const cols = PAL.awning;
    for (const z of [80, 112]) {
      for (const x of [-25, -17.5, 17.5, 25]) {
        if (rnd() < 0.15) { rubble(x, z, y, 5, 1.6); continue; }
        stall(x, z, y, 0, pick(cols));
      }
    }
    cart(-10, 86, y, 0.4);
    cart(12, 108, y, 2.6);
    for (const [x, z] of [[-30, 76], [30, 76], [-30, 116], [30, 116], [-12, 74], [12, 74], [-12, 118], [12, 118]]) lantern(x, z, y);
    barrels.push({ x: -6, z: 111 }, { x: 22, z: 88 }, { x: -22, z: 104 });
    containers.push({ type: 'arcwreck', x: 14, y, z: 98, tier: 2, ry: 0.6 });
    containers.push({ type: 'crate', x: -14, y, z: 104, tier: 2, ry: 0 });
    // Uhrtürmchen auf dem Rathaus
    const rx = (RATHAUS.x0 + RATHAUS.x1) / 2, rz = (RATHAUS.z0 + RATHAUS.z1) / 2;
    const ry0 = rathaus.roofY + (RATHAUS.x1 - RATHAUS.x0) / 2 * 1.2 * 0.55;
    if (!B.model('l_clock', rx, ry0, rz, 0)) {
      b.box(rx, ry0, rz, 3, 5, 3, pick(PAL.brick), { mat: 'brick' });
      b.addPrim('cone4', rx, ry0 + 7, rz, 2.6, 4, 2.6, pick(PAL.slate), { mat: 'slate', ry: Math.PI / 4 });
      b.addPrim('cyl16', rx, ry0 + 3.2, rz - 1.52, 0.95, 0.05, 0.95, 0xe8e2d4, { rx: Math.PI / 2 });
      b.box(rx, ry0 + 3.2, rz - 1.56, 0.06, 0.7, 0.04, 0x1d1a17);
    }
    turrets.push({ x: rx, y: ry0 + 5, z: rz });
  }

  // ---------------------------------------------------------------- Riesenbaum
  function giantTree(t) {
    const y = t.level;
    const bark = 0x4a3626, barkDark = 0x3a2a1e;
    const model = B.model('v_giant', t.x, y, t.z, rnd() * 6, { scale: t.h / 44 });
    if (!model) {
      // Stamm in Abschnitten
      let r = 2.4, yy = y;
      for (let k = 0; k < 5; k++) {
        const h = t.h * 0.14;
        b.addPrim('cyl16', t.x, yy + h / 2, t.z, r, h, r, k % 2 ? bark : barkDark, { mat: 'wood' });
        yy += h;
        r *= 0.86;
      }
      // Wurzeln
      for (let k = 0; k < 7; k++) {
        const a = (k / 7) * Math.PI * 2 + rnd();
        b.box(t.x + Math.cos(a) * 2.6, y, t.z + Math.sin(a) * 2.6, 3.6, 1.4, 1.1, barkDark, { ry: -a, rz: 0.25, mat: 'wood' });
      }
      // Äste
      for (let k = 0; k < 8; k++) {
        const a = (k / 8) * Math.PI * 2 + rnd() * 0.5;
        const ay = y + t.h * (0.5 + rnd() * 0.25), len = 6 + rnd() * 5;
        b.box(t.x + Math.cos(a) * len / 2, ay, t.z + Math.sin(a) * len / 2, len, 0.7, 0.7, bark, { ry: -a, rz: 0.35, mat: 'wood' });
      }
      // Krone aus vielen Laubballen
      for (let k = 0; k < 22; k++) {
        const a = rnd() * Math.PI * 2, d = rnd() * 9;
        const s = 4 + rnd() * 4;
        b.addPrim('ico1', t.x + Math.cos(a) * d, y + t.h * (0.62 + rnd() * 0.35), t.z + Math.sin(a) * d, s, s * 0.75, s, mix(0x3e5a2c, 0x56702f, rnd()), { ry: rnd() * 6 });
      }
    }
    col.addCyl(t.x, t.z, 2.3, y - 1, y + t.h * 0.7);
    // Plattform auf Höhe der Turmdächer
    const dy = t.deckY;
    const D = DECK;
    for (const [x0, x1, z0, z1] of [[-D, D, -D, -2.4], [-D, D, 2.4, D], [-D, -2.4, -2.4, 2.4], [2.4, D, -2.4, 2.4]]) {
      B.solid(t.x + (x0 + x1) / 2, dy - 0.3, t.z + (z0 + z1) / 2, x1 - x0, 0.3, z1 - z0, 0x6a4a30, { mat: 'wood' });
    }
    // Stützbalken
    for (const [dx, dz] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
      b.box(t.x + dx * (D - 2), dy - 3.5, t.z + dz * (D - 2), 0.3, 3.2, 0.3, 0x4e3624, { mat: 'wood', rx: dz * 0.5, rz: -dx * 0.5 });
    }
    // Geländer mit Lücken für die Brücken
    const gapAt = (side, a) => t.bridges.some((g) => g.side === side && Math.abs(g.c - a) < 1.1);
    for (const side of ['n', 's', 'w', 'e']) {
      for (let a = -D + 0.5; a < D; a += 1) {
        const along = side === 'n' || side === 's';
        const cx = along ? t.x + a : t.x + (side === 'w' ? -D : D) + (side === 'w' ? 0.1 : -0.1);
        const cz = along ? t.z + (side === 'n' ? -D : D) + (side === 'n' ? 0.1 : -0.1) : t.z + a;
        if (gapAt(side, along ? cx : cz)) continue;
        b.box(cx, dy, cz, along ? 1.0 : 0.08, 1.0, along ? 0.08 : 1.0, 0x5a3e2a, { mat: 'wood' });
        col.addBox(cx - (along ? 0.5 : 0.06), dy, cz - (along ? 0.06 : 0.5), cx + (along ? 0.5 : 0.06), dy + 1.0, cz + (along ? 0.06 : 0.5));
      }
    }
    // Hütte auf der Plattform
    const hut = { x0: t.x + 2.8, x1: t.x + 6.6, z0: t.z + 2.8, z1: t.z + 6.6, kind: 'house', style: 'schwarzwald', floors: 1, tier: 2, poi: t.id, level: dy, district: 'wald', id: -10 };
    kit.house(hut, rnd);
    containers.push({ type: 'crate', x: t.x - 4.5, y: dy, z: t.z - 4.5, tier: 2, ry: 0 });
  }

  // Hängebrücke zwischen zwei Punkten gleicher Höhe (achsparallel)
  function ropeBridge(from, to, y, axis) {
    const len = axis === 'z' ? Math.abs(to.z - from.z) : Math.abs(to.x - from.x);
    if (len < 0.3) return;
    const n = Math.max(2, Math.ceil(len / 0.5));
    const sag = Math.min(0.55, len * 0.05);
    const w = 1.6;
    for (let i = 0; i < n; i++) {
      const t0 = (i + 0.5) / n;
      const px = from.x + (to.x - from.x) * t0, pz = from.z + (to.z - from.z) * t0;
      const py = y - sag * 4 * t0 * (1 - t0);
      const sx = axis === 'z' ? w : len / n * 0.9, sz = axis === 'z' ? len / n * 0.9 : w;
      b.box(px, py - 0.08, pz, sx, 0.08, sz, 0x6a4a30, { mat: 'wood' });
      col.addBox(px - sx / 2 - 0.03, py - 0.4, pz - sz / 2 - 0.03, px + sx / 2 + 0.03, py, pz + sz / 2 + 0.03);
    }
    // Seile
    for (const s of [-1, 1]) {
      const off = (w / 2) * s;
      for (let i = 0; i < 6; i++) {
        const a0 = i / 6, a1 = (i + 1) / 6;
        const p0 = [from.x + (to.x - from.x) * a0, y + 1.0 - sag * 4 * a0 * (1 - a0) * 0.7, from.z + (to.z - from.z) * a0];
        const p1 = [from.x + (to.x - from.x) * a1, y + 1.0 - sag * 4 * a1 * (1 - a1) * 0.7, from.z + (to.z - from.z) * a1];
        const mx = (p0[0] + p1[0]) / 2 + (axis === 'z' ? off : 0), mz = (p0[2] + p1[2]) / 2 + (axis === 'z' ? 0 : off);
        const sl = Math.atan2(p1[1] - p0[1], len / 6);
        b.box(mx, (p0[1] + p1[1]) / 2 - 0.02, mz, axis === 'z' ? 0.05 : len / 6, 0.05, axis === 'z' ? len / 6 : 0.05, 0x8a7a5a, axis === 'z' ? { rx: -sl } : { rz: sl });
      }
    }
  }

  // ---------------------------------------------------------------- Gründerfelsen
  function buildCliff() {
    // Felswand mit großen Brocken verkleiden
    for (let x = -296; x < 296; x += R(5, 9)) {
      // vor den Gesichtern frei lassen
      if (FACES.some((f) => Math.abs(x - f.x) < 12)) continue;
      const z = CITY.CLIFF_Z - R(3, 14);
      const s = R(4, 9);
      const yy = terrain.height(x, z);
      if (!inst('v_rock' + (Math.floor(rnd() * 3) % 3), x, yy - s * 0.25, z, rnd() * 6, [s * 1.3, s * 1.1, s * 1.2])) {
        b.addPrim('ico', x, yy + s * 0.2, z, s * 1.2, s * 0.9, s, mix(0x6f685c, 0x8a8274, rnd()), { mat: 'stone', ry: rnd() * 6, rx: rnd() * 0.5 });
      }
    }
    // Gesichter der Gründer
    for (const f of FACES) founderFace(f);
    // Felsband vor den Gesichtern (vom Felsensteig erreichbar)
    const L = STEIG.ledge;
    B.solid((L.x0 + L.x1) / 2, 17, (L.z0 + L.z1) / 2, L.x1 - L.x0, L.y - 17, L.z1 - L.z0, 0x8a8274, { mat: 'stone' });
    for (let x = L.x0 + 0.5; x < L.x1; x += 1.0) {
      if (x > STEIG.tower.x0 - 0.5 && x < STEIG.tower.x1 + 0.5 && Math.abs(x + 168) < 1.2) continue;
      b.box(x, L.y, L.z1 - 0.1, 1.0, 1.0, 0.1, 0x4e3624, { mat: 'wood' });
    }
    col.addBox(L.x0, L.y, L.z1 - 0.18, -169.2, L.y + 1, L.z1);
    col.addBox(-166.8, L.y, L.z1 - 0.18, L.x1, L.y + 1, L.z1);
    for (let x = L.x0 + 3; x < L.x1; x += R(6, 12)) rubble(x, L.z0 + 1.5, L.y, 2, 0.8);
    // Brücke vom Steigturm zum Felsband
    ropeBridge({ x: -168, z: STEIG.tower.z0 }, { x: -168, z: L.z1 }, steig.roofY, 'z');
    turrets.push({ x: -100, y: L.y, z: (L.z0 + L.z1) / 2 });
    containers.push({ type: 'arcprobe', x: L.x1 - 4, y: L.y, z: (L.z0 + L.z1) / 2, tier: 3, ry: 0 });
    containers.push({ type: 'weapon', x: -130, y: L.y, z: L.z0 + 1.6, tier: 3, ry: 0 });
    labels.push({ text: 'FELSENSTEIG', x: (STEIG.tower.x0 + STEIG.tower.x1) / 2, y: 18 + 3.3, z: STEIG.tower.z1 + 0.06, ry: 0, w: 4, h: 0.9 });
  }

  function founderFace(f) {
    const cx = f.x, cy = f.y + f.h / 2, cz = f.z;
    const s = f.h / 20;
    if (B.model(`f_${f.id}`, cx, f.y, cz, 0, { scale: s })) {
      col.addBox(cx - 8 * s, f.y, cz - 4 * s, cx + 8 * s, f.y + f.h, cz + 7 * s);
      return;
    }
    const stone = 0x9c9586, dark = 0x2a2724, moss = PAL.moss;
    const P = (dx, dy, dz) => [cx + dx * s, cy + dy * s, cz + dz * s];
    const add = (prim, d, sc, color, opts = {}) => { const p = P(...d); b.addPrim(prim, p[0], p[1], p[2], sc[0] * s, sc[1] * s, sc[2] * s, color, { mat: 'stone', ...opts }); };
    add('sphere', [0, 0, 0], [7, 9.5, 6.5], stone);
    add('box', [0, 3.2, 5.6], [10, 1.6, 2.2], stone, { rx: -0.2 });
    add('sphere', [-2.6, 1.4, 5.6], [1.5, 1.1, 0.9], dark);
    add('sphere', [2.6, 1.4, 5.6], [1.5, 1.1, 0.9], dark);
    add('cone4', [0, -0.6, 6.6], [1.4, 4, 1.4], stone, { rx: -0.25, ry: Math.PI / 4 });
    add('box', [0, -4, 5.6], [4, 0.6, 1.2], dark);
    add('sphere', [0, -6.4, 4.2], [4, 3, 3.2], stone);
    add('sphere', [-6.8, 0, 0], [1.2, 2.4, 1.4], stone);
    add('sphere', [6.8, 0, 0], [1.2, 2.4, 1.4], stone);
    if (f.id === 'seefahrer') {
      add('cyl16', [0, 8.2, 0], [9.5, 1.0, 7.5], stone);
      add('cone8', [0, 10.6, 0], [5.5, 4.4, 5.0], stone);
      for (let k = 0; k < 6; k++) add('ico', [(k - 2.5) * 1.6, -7.5 - (k % 2), 4.6], [1.6, 2.4, 1.4], mix(stone, 0x7a7466, 0.3));
    } else if (f.id === 'schatten') {
      add('sphere', [0, 1, -1.2], [8.6, 10.6, 7.4], mix(stone, 0x6c665a, 0.3));
      add('box', [0, -3.4, 6.2], [9.6, 3.6, 1.2], mix(stone, 0x6c665a, 0.2));
    } else if (f.id === 'erfinder') {
      add('torus', [-2.6, 5.4, 5.2], [1.8, 1.8, 2.5], 0x6f675a);
      add('torus', [2.6, 5.4, 5.2], [1.8, 1.8, 2.5], 0x6f675a);
      add('box', [0, 5.4, 5.0], [9, 0.5, 0.5], 0x6f675a);
      add('box', [0, -2.8, 6.1], [5, 0.8, 1.2], stone, { rz: 0.05 });
    } else {
      add('dome', [0, 4.2, 0], [7.6, 6, 7], mix(stone, 0x7a7466, 0.2));
      add('box', [0, 10.4, -0.5], [0.9, 2.6, 9], stone);
      add('box', [-5.6, -2.5, 3.5], [1.2, 6, 4], stone);
      add('box', [5.6, -2.5, 3.5], [1.2, 6, 4], stone);
    }
    // Moos und Risse
    for (let k = 0; k < 6; k++) add('ico', [R(-6, 6), R(-8, 8), R(3, 6)], [R(0.8, 1.6), R(0.5, 1), R(0.5, 1)], moss);
    col.addBox(cx - 7 * s, f.y, cz - 6 * s, cx + 7 * s, f.y + f.h, cz + 7 * s);
  }

  // ---------------------------------------------------------------- Gassen und Plätze
  function buildStreets() {
    const y = 3;
    // Laternen entlang der Hauptstraßen
    for (const s of STREETS) {
      if (!s.main) continue;
      const alongX = s.x1 - s.x0 > s.z1 - s.z0;
      const len = alongX ? s.x1 - s.x0 : s.z1 - s.z0;
      for (let k = 6; k < len - 4; k += 17) {
        const side = (k / 17) % 2 < 1 ? 0 : 1;
        const x = alongX ? s.x0 + k : side ? s.x1 - 0.5 : s.x0 + 0.5;
        const z = alongX ? (side ? s.z1 - 0.5 : s.z0 + 0.5) : s.z0 + k;
        if (overlap({ x0: x - 1, x1: x + 1, z0: z - 1, z1: z + 1 }, PLAZA)) continue;
        lantern(x, z, levelAt(z));
      }
    }
    // Zunftbaum auf dem Zunftplatz
    const zp = layout.squares.find((q) => q.name === 'Zunftplatz');
    if (zp) {
      const zx = (zp.x0 + zp.x1) / 2, zz = (zp.z0 + zp.z1) / 2;
      const pole = B.model('kc_maypole', zx, y, zz, 0);
      if (!pole) b.addPrim('cyl8', zx, y + 10, zz, 0.28, 20, 0.28, 0xd8d0bc, { mat: 'wood' });
      if (!pole) b.addPrim('torus', zx, y + 19, zz, 1.4, 1.4, 3, 0x3e5a2c, { rx: Math.PI / 2 });
      for (let k = 0; k < 5 && !pole; k++) {
        const yy = y + 6 + k * 2.4, a = k * 1.3;
        b.box(zx + Math.cos(a) * 1.0, yy, zz + Math.sin(a) * 1.0, 1.4, 0.9, 0.06, pick(PAL.awning), { ry: -a });
      }
      col.addCyl(zx, zz, 0.3, y, y + 20);
      for (let i = 0; i < 4; i++) bench(zx + (i < 2 ? -6 : 6), zz + (i % 2 ? -5 : 5), y, 0);
      containers.push({ type: 'toolbox', x: zp.x0 + 2, y, z: zp.z0 + 2, tier: 2, ry: 0 });
    }
    // Brunnenplatz: kleiner Ziehbrunnen
    const bp = layout.squares.find((q) => q.name === 'Brunnenplatz');
    if (bp) {
      const bx = (bp.x0 + bp.x1) / 2, bz = (bp.z0 + bp.z1) / 2;
      if (!B.model('kc_well', bx, y, bz, 0)) {
        b.addPrim('cyl16', bx, y + 0.45, bz, 1.4, 0.9, 1.4, STONE_LIGHT, { mat: 'stone' });
        for (const s of [-1, 1]) b.box(bx + s * 1.2, y + 0.9, bz, 0.18, 2.2, 0.18, 0x5a3e2a, { mat: 'wood' });
        b.box(bx, y + 3.1, bz, 2.8, 0.12, 1.6, 0x7a4430, { mat: 'tile', rz: 0 });
      }
      col.addCyl(bx, bz, 1.45, y, y + 0.9);
      cart(bx - 6, bz + 4, y, 1.2);
      containers.push({ type: 'crate', x: bx + 5, y, z: bz - 4, tier: 1, ry: 0.3 });
    }
    // Gerberhof: Gestelle und Bottiche
    const gh = layout.squares.find((q) => q.name === 'Gerberhof');
    if (gh) {
      for (let i = 0; i < 4; i++) {
        const gx = gh.x0 + 4 + i * 5, gz = gh.z0 + 6;
        b.box(gx, y, gz, 0.15, 2.4, 0.15, 0x5a3e2a, { mat: 'wood' });
        b.box(gx, y, gz + 6, 0.15, 2.4, 0.15, 0x5a3e2a, { mat: 'wood' });
        b.box(gx, y + 2.3, gz + 3, 0.12, 0.12, 6.2, 0x5a3e2a, { mat: 'wood' });
        b.addQuad([gx, y + 0.8, gz + 0.5], [gx, y + 0.8, gz + 5.5], [gx, y + 2.25, gz + 5.5], [gx, y + 2.25, gz + 0.5], pick([0x8a6a4a, 0x6a4a3a, 0x9a8060]), { double: true });
      }
      for (let i = 0; i < 3; i++) {
        const vx = gh.x0 + 5 + i * 6, vz = gh.z1 - 5;
        b.addPrim('cyl16', vx, y + 0.6, vz, 1.3, 1.2, 1.3, 0x5a3e2a, { mat: 'wood' });
        col.addCyl(vx, vz, 1.3, y, y + 1.2);
      }
      containers.push({ type: 'medical', x: gh.x1 - 3, y, z: gh.z0 + 3, tier: 1, ry: 0 });
    }
    // Höfe: Karren, Kisten, Schutt, Fässer
    let nb = 0;
    for (const yard of layout.yards) {
      const w = yard.x1 - yard.x0, d = yard.z1 - yard.z0;
      if (w < 4 || d < 4 || yard.passage) continue;
      const lvl = levelAt((yard.z0 + yard.z1) / 2);
      const cx = (yard.x0 + yard.x1) / 2, cz = (yard.z0 + yard.z1) / 2;
      const r = rnd();
      if (r < 0.2) cart(cx + R(-w / 4, w / 4), cz + R(-d / 4, d / 4), lvl, rnd() * 3);
      else if (r < 0.45) rubble(cx, cz, lvl, 5, Math.min(w, d) / 3);
      else if (r < 0.6) containers.push({ type: 'crate', x: cx, y: lvl, z: cz, tier: 1, ry: rnd() < 0.5 ? 0 : Math.PI / 2 });
      if (rnd() < 0.12 && nb < 14) { barrels.push({ x: cx + R(-1, 1), z: cz + R(-1, 1) }); nb++; }
    }
  }

  // ---------------------------------------------------------------- Kulisse
  function buildBackdrop() {
    const simpleHouse = (x, z, w, d, h, wall, roof, roofMat = 'tile', pitch = 1.1) => {
      b.box(x, 0, z, w, h, d, wall);
      const lot = { x0: x - w / 2, x1: x + w / 2, z0: z - d / 2, z1: z + d / 2 };
      const alongX = w >= d;
      const P0 = alongX ? lot.x0 : lot.z0, P1 = alongX ? lot.x1 : lot.z1, Q0 = alongX ? lot.z0 : lot.x0, Q1 = alongX ? lot.z1 : lot.x1;
      const M = (p, yy, q) => (alongX ? [p, yy, q] : [q, yy, p]);
      const qc = (Q0 + Q1) / 2, yr = h + ((Q1 - Q0) / 2) * pitch;
      kit.quad([M(P0, h, Q0), M(P1, h, Q0), M(P1, yr, qc), M(P0, yr, qc)], alongX ? [0, 1, -1] : [-1, 1, 0], roof, { mat: roofMat });
      kit.quad([M(P1, h, Q1), M(P0, h, Q1), M(P0, yr, qc), M(P1, yr, qc)], alongX ? [0, 1, 1] : [1, 1, 0], roof, { mat: roofMat });
      kit.tri([M(P0, h, Q0), M(P0, h, Q1), M(P0, yr, qc)], alongX ? [-1, 0, 0] : [0, 0, -1], wall);
      kit.tri([M(P1, h, Q0), M(P1, h, Q1), M(P1, yr, qc)], alongX ? [1, 0, 0] : [0, 0, 1], wall);
    };
    // Weiße Stadt (Westen): weiße Häuser, rote Dächer, weißer Turm
    for (let x = -292; x < -206; x += R(9, 14)) {
      for (let z = -150; z < 290; z += R(9, 14)) {
        if (rnd() < 0.12) continue;
        const h = R(6, 13);
        simpleHouse(x + R(-1, 1), z, R(6, 10), R(6, 10), terrain.height(x, z) + h, pick([0xeeeae0, 0xe6e1d4, 0xf2efe6]), pick(PAL.tile));
      }
    }
    b.addPrim('cyl16', -250, 24, 40, 6, 48, 6, 0xf0ece2);
    b.addPrim('cone8', -250, 54, 40, 7, 12, 7, 0xa8452e, { mat: 'tile' });
    for (let z = -140; z < 280; z += 60) b.box(-226, 0, z, 2, 9, 40, 0xece8dd);
    // Turmstadt (Osten): Hochhäuser mit Himmelsbrücken
    const towers = [];
    for (let x = 212; x < 296; x += R(16, 24)) {
      for (let z = -140; z < 290; z += R(18, 26)) {
        const h = R(25, 95) * (1 - Math.abs(z - 60) / 600);
        const w = R(10, 16), d = R(10, 16);
        b.box(x, 0, z, w, h, d, pick([0x56606a, 0x6a737a, 0x4a525a, 0x7a6a5a]), { mat: rnd() < 0.3 ? 'brick' : 'world' });
        // Fensterbänder (leuchtend)
        for (let yy = 8; yy < h - 4; yy += R(9, 16)) if (rnd() < 0.35) b.box(x - w / 2 - 0.05, yy, z, 0.1, 0.6, d * 0.8, 0xffd59a, { mat: 'glow' });
        towers.push({ x, z, h });
      }
    }
    for (let i = 0; i < towers.length - 1; i++) {
      const a = towers[i], c = towers[i + 1];
      if (Math.abs(a.x - c.x) < 2 && Math.abs(c.z - a.z) < 30) {
        const yy = Math.min(a.h, c.h) * 0.6;
        b.box(a.x, yy, (a.z + c.z) / 2, 3, 2.5, Math.abs(c.z - a.z), 0x8a8f94, { mat: 'metal' });
      }
    }
    b.addPrim('cyl16', 252, 75, 70, 9, 150, 9, 0x5c646c, { mat: 'metal' });
    for (const yy of [60, 100, 140]) b.addPrim('torus', 252, yy, 70, 11, 11, 10, 0x9ff8ff, { mat: 'glow', rx: Math.PI / 2 });
    b.addPrim('cone8', 252, 158, 70, 6, 16, 6, 0x5c646c, { mat: 'metal' });
    // Hafenviertel (Süden): Speicher, Kräne, gestrandete Galeone, Leuchtturm
    for (let x = -190; x < 190; x += R(12, 18)) {
      for (let z = 206; z < CITY.SEA_Z - 14; z += R(12, 16)) {
        if (Math.abs(x) < 8) continue;
        simpleHouse(x, z, R(9, 14), R(10, 16), terrain.height(x, z) + R(8, 16), pick(PAL.brick), pick(PAL.tile), 'tile', 1.0);
      }
    }
    for (let x = -160; x < 170; x += 55) {
      b.box(x, 0, CITY.SEA_Z - 6, 1.2, 18, 1.2, 0x5a4a3a, { mat: 'wood' });
      b.box(x + 5, 17, CITY.SEA_Z - 6, 11, 0.8, 0.8, 0x5a4a3a, { mat: 'wood' });
    }
    // Galeone (schräg gestrandet)
    const gx = 70, gz = CITY.SEA_Z + 14;
    b.box(gx, -2, gz, 46, 8, 12, 0x4a3424, { mat: 'wood', rz: 0.06, ry: 0.15 });
    b.box(gx - 17, 4, gz + 2.5, 12, 7, 10, 0x5a3e2a, { mat: 'wood', ry: 0.15 });
    for (const [mx, mh] of [[-6, 34], [8, 40], [20, 28]]) {
      b.addPrim('cyl8', gx + mx, 4 + mh / 2, gz + mx * 0.15, 0.5, mh, 0.5, 0x3a2a1e, { mat: 'wood', rz: 0.08 });
      b.box(gx + mx, 4 + mh * 0.7, gz + mx * 0.15, 0.4, 0.4, 13, 0x3a2a1e, { mat: 'wood', ry: 0.15 });
      b.addQuad([gx + mx + 0.6, 4 + mh * 0.35, gz + mx * 0.15 - 5.5], [gx + mx + 0.6, 4 + mh * 0.35, gz + mx * 0.15 + 5.5], [gx + mx + 0.6, 4 + mh * 0.68, gz + mx * 0.15 + 6], [gx + mx + 0.6, 4 + mh * 0.68, gz + mx * 0.15 - 6], 0xd8ceb6, { double: true });
    }
    // Leuchtturm
    for (let k = 0; k < 6; k++) b.addPrim('cyl16', -150, 1 + k * 5 + 2.5, CITY.SEA_Z + 4, 3.6 - k * 0.25, 5, 3.6 - k * 0.25, k % 2 ? 0xa8392e : 0xece8dd);
    b.addPrim('cyl16', -150, 33.5, CITY.SEA_Z + 4, 2.0, 3, 2.0, 0xffe7a0, { mat: 'glow' });
    b.addPrim('cone8', -150, 36.5, CITY.SEA_Z + 4, 2.4, 3, 2.4, 0x2a2a2a, { mat: 'metal' });
  }
}
