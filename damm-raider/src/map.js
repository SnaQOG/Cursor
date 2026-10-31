// Baut die Karte "Der Damm": Staumauer, Kraftwerk, Kontrollturm und alle
// weiteren Orte. Liefert Beute-Spots, Gegner-Spawns und Wächter-Positionen.
import { Builder, COL } from './structures.js';
import { MAP } from './config.js';
import { mulberry32 } from './util.js';

export function buildMap(batcher, col, terrain) {
  const B = new Builder(batcher, col, terrain);
  const rnd = mulberry32(99);
  const R = (a, b) => a + rnd() * (b - a);
  const enemySpawns = []; // {type, x, z, y?, poi, patrol}
  const turrets = []; // {x,y,z}
  const containers = []; // explizite Behälter {type, x, y, z, ry, tier}
  const barrels = []; // Sprengfässer
  const labels = [];

  const E = (type, x, z, n = 1, spread = 8, extra = {}) => {
    for (let i = 0; i < n; i++) enemySpawns.push({ type, x: x + R(-spread, spread), z: z + R(-spread, spread), ...extra });
  };
  const C = (type, x, y, z, tier = 1, ry = 0) => containers.push({ type, x, y, z, tier, ry });

  // ------------------------------------------------------------------ Damm
  const dz = MAP.DAM_Z, dt = MAP.DAM_THICK, top = MAP.DAM_TOP, hx = MAP.DAM_HALF_X;
  const dz0 = dz - dt / 2, dz1 = dz + dt / 2;
  B.solid(0, -4, dz, hx * 2, top + 4, dt, COL.concrete);
  // Pfeiler an der Luftseite
  for (let x = -hx + 12; x <= hx - 12; x += 22) {
    if (Math.abs(x) < 20) continue;
    const gy = Math.min(B.ground(x, dz1 + 3), 6);
    B.solid(x, gy - 2, dz1 + 1.6, 4, top - gy + 1, 3.2, COL.concreteDark);
  }
  // Wasserseite: Fugen
  for (let x = -hx; x <= hx; x += 11) B.deco('box', x, top / 2 + 6, dz0 - 0.05, 0.3, top - 10, 0.2, COL.concreteDark);
  // Brüstungen mit Lücken (Brücke zum Turm, Aufzug)
  const parapet = (zz) => {
    for (let x = -hx; x < hx; x += 8) {
      if (zz > dz && Math.abs(x + 4 + 95) < 5.2) continue;
      B.solid(x + 4, top, zz, 7.4, 1.1, 0.5, COL.concreteLight);
    }
  };
  parapet(dz0 + 0.3);
  parapet(dz1 - 0.3);
  // Straßenmarkierung
  for (let x = -hx + 4; x < hx; x += 6) B.deco('box', x, top + 0.01, dz, 3, 0.02, 0.25, COL.yellow, { shade: false });
  for (let x = -120; x <= 120; x += 30) B.lamp(x, dz1 - 1.2, top, 5);
  // Überlauf mit Schützen und Wasserfall-Kanal
  for (let x = -15; x <= 15; x += 7.5) {
    B.solid(x, top, dz0 + 3, 1.6, 6, 3, COL.concreteDark);
  }
  B.deco('box', 0, top + 5.6, dz0 + 3, 32, 0.8, 3.4, COL.teal);
  for (let x = -11.25; x <= 11.25; x += 7.5) B.deco('box', x, top - 4, dz0 - 0.2, 5.6, 8, 0.6, COL.rust);
  B.deco('box', 0, 14, dz1 + 0.1, 24, 30, 0.4, COL.concreteLight);
  // Portalkran über dem Überlauf
  for (const sx of [-18, 18]) {
    B.solid(sx, top, dz0 + 2, 1.2, 12, 1.2, COL.orange);
    B.solid(sx, top, dz1 - 2, 1.2, 12, 1.2, COL.orange);
    B.deco('box', sx, top + 12, dz, 1.4, 1.2, dt, COL.orange);
  }
  B.deco('box', 0, top + 12.6, dz, 38, 1.4, 1.6, COL.orange);
  B.deco('box', 4, top + 11.2, dz, 3, 1.6, 2.4, COL.trim);
  // Torhäuser auf der Krone
  for (const gx of [-70, 70]) {
    const g = B.building({ x: gx, z: dz, w: 9, d: 8, floors: 2, fh: 3.3, baseY: top, wall: COL.paleDark, doors: { e: [0.5], w: [0.5] }, tier: 2, poi: 'krone' });
    turrets.push({ x: gx, y: g.roofY, z: dz });
  }
  C('weapon', 0, top, dz - 3, 2);
  C('crate', -40, top, dz + 5, 2);
  C('crate', 40, top, dz - 5, 2);
  C('toolbox', 110, top, dz + 4, 2);
  E('rocketeer', 0, dz, 1, 20, { y: top + 14, poi: 'krone' });
  E('wasp', -40, dz, 2, 15, { poi: 'krone' });

  // ------------------------------------------------------------------ Kraftwerk
  {
    const kx = 60, kz = -50, w = 64, d = 24;
    const gy = B.ground(kx, kz);
    const hall = B.building({
      x: kx, z: kz, w, d, floors: 1, fh: 13, wall: COL.concreteLight, trim: COL.teal,
      doors: { s: [{ c: 0.5, w: 8, h: 7 }, { c: 0.15, w: 2, h: 2.6 }], e: [0.5], w: [0.5] },
      windows: true, winY0: 8.5, winH: 3, winEvery: 6, roofAccess: false, tier: 3, poi: 'kraftwerk',
    });
    // Turbinen
    for (const tx of [42, 60, 78]) {
      B.tank(tx, kz - 2, 3.4, 4.5, COL.teal, gy + 0.15);
      B.deco('cyl16', tx, gy + 5.2, kz - 2, 2, 1.4, 2, COL.metalDark);
      B.deco('box', tx, gy + 0.9, kz + 4.2, 1.2, 1.5, 1.2, COL.yellow);
      // Druckrohre vom Damm
      B.pipe(tx, 16, dz1 - 0.5, tx, 8, kz - d / 2 + 0.2, 1.5, COL.rust, true);
    }
    // Galerie an der Nordwand
    const my = gy + 0.15 + 6;
    B.solid(kx, my - 0.3, kz - d / 2 + 2.6, w - 2, 0.3, 4.4, COL.metal, {});
    for (let x = kx - w / 2 + 6; x < kx + w / 2 - 2; x += 4) B.deco('box', x + 2, my + 0.5, kz - d / 2 + 4.7, 3.8, 0.08, 0.08, COL.yellow);
    B.stairs(kx - w / 2 + 4, kz - d / 2 + 4.8 + 3.4, gy + 0.15, my, 'n', 1.5, COL.metal);
    for (const sx of [kx - 20, kx, kx + 20]) {
      B.spots.push({ x: sx, y: my, z: kz - d / 2 + 1.8, tier: 3, poi: 'kraftwerk', indoor: true });
    }
    // Leitstand
    B.building({ x: kx + 24, z: kz + 6, w: 10, d: 7, floors: 1, fh: 3, baseY: gy + 0.15, wall: COL.teal, doors: { w: [0.5] }, roofAccess: false, parapet: 0, tier: 3, poi: 'kraftwerk' });
    C('weapon', kx - 10, gy + 0.15, kz + 8, 3);
    C('toolbox', kx + 12, gy + 0.15, kz + 9, 3);
    C('medical', kx - 26, gy + 0.15, kz + 9, 3);
    C('arcprobe', kx, gy + 0.15, kz + 6, 3);
    for (let i = 0; i < 4; i++) B.container(kx - 18 + i * 12, kz + 22 + R(-2, 2), R(-0.2, 0.2), [COL.teal, COL.rust, COL.orange, COL.red][i]);
    barrels.push({ x: kx - 5, z: kz + 16 }, { x: kx + 30, z: kz + 15 });
    E('tick', kx, kz, 3, 18, { poi: 'kraftwerk', indoor: true });
    E('hornet', kx, kz + 20, 2, 16, { poi: 'kraftwerk' });
    E('wasp', kx + 25, kz + 25, 2, 12, { poi: 'kraftwerk' });
    E('leaper', kx - 20, kz + 30, 1, 5, { poi: 'kraftwerk' });
    labels.push({ text: 'KRAFTWERK 3', x: kx, y: hall.roofY - 3, z: hall.z1 + 0.2, ry: 0 });
  }

  // ------------------------------------------------------------------ Kontrollturm
  {
    const tx = -95, tz = -58;
    const gy = B.ground(tx, tz);
    const fh = (top - gy - 0.15) / 7;
    const t = B.building({
      x: tx, z: tz, w: 10, d: 10, floors: 9, fh, wall: COL.pale, trim: COL.orange,
      doors: { s: [0.5], 'n@7': [0.5] }, winEvery: 3.2, tier: 2, poi: 'turm',
    });
    // Brücke
    B.solid(tx, top - 0.4, (t.z0 + dz1) / 2, 2.8, 0.4, t.z0 - dz1 + 0.4, COL.metal);
    B.solid(tx - 1.45, top, (t.z0 + dz1) / 2, 0.1, 1.0, t.z0 - dz1, COL.orange);
    B.solid(tx + 1.45, top, (t.z0 + dz1) / 2, 0.1, 1.0, t.z0 - dz1, COL.orange);
    turrets.push({ x: tx, y: t.roofY, z: tz });
    E('snitch', tx, tz, 1, 10, { y: top + 20, poi: 'turm' });
    E('tick', tx, tz, 2, 3, { poi: 'turm', indoor: true });
    labels.push({ text: 'KONTROLLE', x: tx, y: gy + 6, z: t.z1 + 0.2, ry: 0 });
  }

  // ------------------------------------------------------------------ Alte Brücke
  {
    const bx = 5, bz = -24;
    const y = 3.2;
    B.solid(bx, y - 0.6, bz, 44, 0.6, 7, COL.concrete);
    B.solid(bx, y, bz - 3.3, 44, 1.0, 0.35, COL.concreteLight);
    B.solid(bx, y, bz + 3.3, 44, 1.0, 0.35, COL.concreteLight);
    for (const px of [-8, 5, 18]) B.solid(bx + px, -2, bz, 2.4, y - 0.6 + 2, 5, COL.concreteDark);
    B.car(bx - 6, bz - 1, 0.2, 0x7a3b2e);
    B.car(bx + 10, bz + 1.2, Math.PI + 0.1, 0x3d5a6b);
    C('crate', bx + 2, y, bz + 1.5, 1);
    E('pop', bx, bz + 20, 2, 10, { poi: 'bruecke' });
  }

  // ------------------------------------------------------------------ Forschungsanlage
  {
    const fx = -150, fz = 5;
    B.building({ x: fx, z: fz, w: 28, d: 16, floors: 3, fh: 3.5, wall: COL.pale, trim: COL.teal, doors: { s: [0.3, 0.7], e: [0.5] }, tier: 3, poi: 'forschung' });
    B.building({ x: fx + 22, z: fz + 26, w: 14, d: 10, floors: 1, fh: 3.6, wall: COL.paleDark, trim: COL.orange, doors: { w: [0.5], n: [0.5] }, tier: 2, poi: 'forschung' });
    B.building({ x: fx - 24, z: fz + 24, w: 7, d: 6, floors: 1, fh: 3, wall: COL.teal, doors: { e: [0.5] }, roofAccess: false, tier: 1, poi: 'forschung' });
    // Radarschüssel auf dem Dach
    const ry = B.ground(fx, fz) + 0.15 + 3 * 3.5;
    B.solid(fx - 8, ry, fz - 3, 1.2, 3, 1.2, COL.metalDark);
    B.deco('dome', fx - 8, ry + 3.6, fz - 3, 3.2, 1.2, 3.2, COL.concreteLight, { rx: 2.2 });
    // Parkplatz
    for (let i = 0; i < 5; i++) B.car(fx - 12 + i * 5.5, fz + 16, Math.PI / 2 + R(-0.15, 0.15), [0x8a8f7a, 0x6d4a3a, 0x496070, 0x8c7a4a, 0x5a5a5a][i]);
    B.fence(fx - 30, fz - 22, fx + 30, fz - 22);
    B.fence(fx - 30, fz - 22, fx - 30, fz + 40);
    B.lamp(fx + 16, fz + 12);
    B.lamp(fx - 16, fz + 12);
    C('weapon', fx + 20, B.ground(fx + 20, fz + 26) + 0.15, fz + 28, 3);
    C('arcprobe', fx + 5, B.ground(fx, fz - 14), fz - 14, 3);
    barrels.push({ x: fx + 16, z: fz - 12 }, { x: fx - 18, z: fz + 20 });
    E('wasp', fx, fz, 3, 20, { poi: 'forschung' });
    E('snitch', fx + 10, fz + 10, 1, 10, { y: 22, poi: 'forschung' });
    E('tick', fx, fz, 2, 8, { poi: 'forschung', indoor: true });
    E('leaper', fx + 30, fz - 15, 1, 5, { poi: 'forschung' });
    labels.push({ text: 'FORSCHUNG', x: fx, y: B.ground(fx, fz) + 2.9, z: fz + 8.2, ry: 0 });
  }

  // ------------------------------------------------------------------ Wasseraufbereitung
  {
    const wx = 150, wz = 20;
    const gy = B.ground(wx, wz);
    B.building({ x: wx, z: wz, w: 16, d: 12, floors: 2, fh: 3.6, wall: COL.concreteLight, trim: COL.teal, doors: { s: [0.5], n: [0.5] }, tier: 2, poi: 'wasser' });
    for (const [tx, tz] of [[122, -4], [178, -4], [122, 44], [178, 44]]) {
      B.tank(tx, tz, 9, 2.6, COL.concrete, gy, 0x3b6b6a);
      B.deco('box', tx, gy + 3.2, tz, 18, 0.25, 0.5, COL.metalDark);
      B.deco('box', tx, gy + 2.8, tz, 0.5, 1.0, 0.5, COL.metalDark);
    }
    for (const [cx, cz] of [[166, 20], [170, 25], [166, 30]]) B.tank(cx, cz, 1.8, 6, COL.pale, gy);
    B.pipe(122, gy + 1, 5, 142, gy + 1, 16, 0.6, COL.teal, false);
    B.pipe(178, gy + 1, 5, 158, gy + 1, 16, 0.6, COL.teal, false);
    B.building({ x: wx, z: wz + 34, w: 7, d: 6, floors: 1, fh: 3, wall: COL.teal, doors: { s: [0.5] }, roofAccess: false, tier: 1, poi: 'wasser' });
    B.container(wx - 12, wz + 25, 0, COL.orange);
    B.container(wx + 14, wz - 20, Math.PI / 2, COL.teal);
    C('medical', wx + 6, gy + 0.15 + 3.6, wz - 3, 2);
    C('toolbox', 166, gy, 14, 2);
    barrels.push({ x: 160, z: 34 }, { x: 138, z: 22 }, { x: 170, z: 12 });
    E('pop', wx, wz, 3, 25, { poi: 'wasser' });
    E('tick', wx, wz, 2, 5, { poi: 'wasser', indoor: true });
    E('wasp', wx, wz + 30, 2, 15, { poi: 'wasser' });
    labels.push({ text: 'AUFBEREITUNG', x: wx, y: gy + 5.5, z: wz + 6.2, ry: 0 });
  }

  // ------------------------------------------------------------------ Umspannwerk
  {
    const ux = -60, uz = 110;
    const gy = B.ground(ux, uz);
    B.fence(ux - 20, uz - 15, ux + 20, uz - 15, gy);
    B.fence(ux - 20, uz + 15, ux + 8, uz + 15, gy);
    B.fence(ux - 20, uz - 15, ux - 20, uz + 15, gy);
    B.fence(ux + 20, uz - 15, ux + 20, uz + 15, gy);
    for (let i = 0; i < 3; i++) {
      const tx = ux - 12 + i * 9, tz = uz - 5;
      B.solid(tx, gy, tz, 3.4, 3, 2.6, COL.metalDark);
      for (let k = -1; k <= 1; k++) B.deco('box', tx + k * 1.1, gy + 1.5, tz + 1.5, 0.3, 2.6, 0.4, COL.metal);
      for (let k = -1; k <= 1; k++) B.deco('cyl8', tx + k * 1, gy + 3.7, tz, 0.2, 1.4, 0.2, 0xd8d0c0);
    }
    B.building({ x: ux + 10, z: uz + 7, w: 9, d: 7, floors: 1, fh: 3.2, wall: COL.concrete, trim: COL.yellow, doors: { w: [0.5] }, tier: 1, poi: 'umspann' });
    C('toolbox', ux - 14, gy, uz + 8, 1);
    C('crate', ux, gy, uz + 10, 1);
    E('wasp', ux, uz, 2, 12, { poi: 'umspann' });
    E('pop', ux, uz + 25, 1, 8, { poi: 'umspann' });
  }
  // Strommasten quer durchs Tal
  for (const [px, pz] of [[-240, 120], [-190, 118], [-120, 114], [-10, 105], [40, 100], [100, 96], [230, 90]]) B.pylon(px, pz, 24, 0.05);

  // ------------------------------------------------------------------ Blasse Wohnblöcke
  {
    const ax = 70, az = 168;
    B.building({ x: ax - 22, z: az - 16, w: 20, d: 11, floors: 4, fh: 3.1, wall: COL.pale, trim: COL.rust, doors: { s: [0.3], n: [0.7] }, tier: 2, poi: 'wohn' });
    B.building({ x: ax + 22, z: az - 12, w: 11, d: 20, floors: 3, fh: 3.1, wall: COL.paleDark, trim: COL.teal, doors: { w: [0.5], e: [0.3] }, tier: 2, poi: 'wohn' });
    B.building({ x: ax - 4, z: az + 22, w: 24, d: 11, floors: 4, fh: 3.1, wall: COL.pale, trim: COL.orange, doors: { n: [0.25, 0.75] }, tier: 2, poi: 'wohn' });
    for (let i = 0; i < 6; i++) B.car(ax - 30 + i * 11 + R(-2, 2), az + 5 + R(-2, 2), R(-0.3, 0.3) + (i % 2 ? Math.PI : 0), [0x9a6b3a, 0x566b7a, 0x7a7a70, 0x8a3b30, 0x4a6040, 0xb0a890][i]);
    // Bushaltestelle
    B.solid(ax + 30, B.ground(ax + 30, az + 10), az + 10, 4, 0.1, 1.6, COL.teal, { noCol: true });
    B.deco('box', ax + 30, B.ground(ax + 30, az + 10) + 2.5, az + 10, 4.2, 0.12, 1.8, COL.teal);
    B.deco('box', ax + 30, B.ground(ax + 30, az + 10) + 1.25, az + 10.8, 4, 2.5, 0.08, COL.glass, { mat: 'glass' });
    for (let i = 0; i < 4; i++) B.lamp(ax - 30 + i * 18, az + 12);
    for (const [bx, bz] of [[ax - 10, az + 6], [ax + 8, az + 3], [ax + 34, az - 30]]) B.barrier(bx, bz, R(0, Math.PI));
    barrels.push({ x: ax + 5, z: az + 9 }, { x: ax - 34, z: az - 4 });
    E('tick', ax, az, 4, 25, { poi: 'wohn', indoor: true });
    E('wasp', ax, az, 2, 20, { poi: 'wohn' });
    E('hornet', ax + 10, az - 20, 1, 5, { poi: 'wohn' });
    E('snitch', ax, az, 1, 10, { y: 24, poi: 'wohn' });
  }

  // ------------------------------------------------------------------ Giftsumpf mit Außenposten
  {
    const sx = -165, sz = 178;
    const huts = [[-178, 162], [-150, 186], [-188, 196], [-158, 150], [-140, 205]];
    const deckY = 2.2;
    for (const [hx2, hz2] of huts) {
      for (const [ox, oz] of [[-2.6, -2.2], [2.6, -2.2], [-2.6, 2.2], [2.6, 2.2]]) B.solid(hx2 + ox, -1, hz2 + oz, 0.3, deckY + 1, 0.3, COL.woodDark);
      B.solid(hx2, deckY - 0.25, hz2, 7, 0.25, 6, COL.wood);
      B.building({ x: hx2, z: hz2, w: 5.4, d: 4.6, floors: 1, fh: 2.8, baseY: deckY - 0.15, foundation: false, wall: COL.wood, trim: COL.woodDark, floorCol: COL.wood, roofCol: COL.rustDark, doors: { s: [0.5], n: [0.5] }, winEvery: 2.6, roofAccess: false, parapet: 0, tier: 2, poi: 'sumpf' });
    }
    // Stege
    const walk = (x0, z0, x1, z1) => {
      const len = Math.hypot(x1 - x0, z1 - z0);
      const n = Math.ceil(len / 2);
      for (let i = 0; i < n; i++) {
        const t = (i + 0.5) / n;
        const x = x0 + (x1 - x0) * t, z = z0 + (z1 - z0) * t;
        B.solid(x, deckY - 0.45, z, 1.5, 0.2, 1.5, COL.wood);
        if (i % 2 === 0) B.solid(x + 0.6, -1, z, 0.2, deckY, 0.2, COL.woodDark, { noCol: true });
      }
    };
    walk(-178, 165.5, -158, 146.5);
    walk(-174, 162, -153, 183);
    walk(-150, 189.5, -142, 202);
    walk(-184, 192.5, -176, 167);
    walk(-158, 147, -128, 130);
    walk(-188, 199.5, -215, 222);
    C('arcprobe', sx, deckY - 0.2, sz - 3, 3);
    E('pop', sx, sz, 3, 30, { poi: 'sumpf' });
    E('tick', sx, sz, 3, 20, { poi: 'sumpf', indoor: true });
    E('leaper', sx + 10, sz + 10, 1, 10, { poi: 'sumpf' });
    E('wasp', sx, sz, 1, 20, { poi: 'sumpf' });
  }

  // ------------------------------------------------------------------ Altes Schlachtfeld mit ARC-Wrack
  {
    const ox = 195, oz = 158;
    const gy = B.ground(ox, oz);
    // Riesiges Wrack einer Königin: Wirbelsäule, Rippen, Kopf
    const dark = 0x3d3f42, plate = 0x6c6a64;
    for (let i = 0; i < 10; i++) {
      const x = ox - 22 + i * 4.6, y = gy + 4 + Math.sin(i * 0.4) * 3;
      B.deco('sphere', x, y, oz, 2.4, 2.1, 2.4, dark);
      if (i > 1 && i < 9) {
        for (const side of [-1, 1]) {
          B.deco('box', x, y - 2 + 3, oz + side * 5, 1.1, 9, 1.2, plate, { rx: side * 0.9 });
          B.deco('box', x, gy + 0.5, oz + side * 9.5, 1.1, 3.5, 1.2, plate, { rx: side * 0.2 });
        }
      }
    }
    B.col.addBox(ox - 25, gy - 1, oz - 3, ox + 23, gy + 6.5, oz + 3);
    // Kopf
    B.deco('sphere', ox + 27, gy + 5, oz + 2, 5, 4, 4.5, dark);
    B.deco('sphere', ox + 31, gy + 5.5, oz + 2, 1.4, 1.4, 1.4, 0x5a2a20);
    B.col.addCyl(ox + 27, oz + 2, 5, gy - 1, gy + 8.5);
    // Beine
    for (const [lx, lz, rx2, rz2] of [[-10, -10, 0.6, 0.4], [4, 11, -0.5, -0.3], [14, -12, 0.7, -0.2], [-18, 9, -0.6, 0.5]]) {
      B.deco('box', ox + lx, gy + 6, oz + lz, 1.6, 16, 1.6, dark, { rx: rx2, rz: rz2 });
    }
    // Schützengräben & Sandsäcke
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      B.sandbags(ox + Math.cos(a) * 32, oz + Math.sin(a) * 32, 5 + R(0, 3), a + Math.PI / 2);
    }
    // Panzerwracks
    for (const [px, pz, r2] of [[ox - 30, oz - 20, 0.5], [ox + 20, oz + 30, 2.1], [ox - 12, oz + 34, -0.4]]) {
      const pg = B.ground(px, pz);
      B.solid(px, pg, pz, 7, 2, 3.6, 0x4f5540, { ry: r2, shrink: 0.9 });
      B.deco('box', px, pg + 2.5, pz, 3, 1.2, 2.6, 0x464b39, { ry: r2 + 0.3 });
      B.deco('box', px + Math.cos(r2 + 0.3) * 3, pg + 2.6, pz - Math.sin(r2 + 0.3) * 3, 4.5, 0.35, 0.35, 0x3a3e30, { ry: r2 + 0.3 });
    }
    // Bunker
    B.building({ x: ox - 34, z: oz + 8, w: 9, d: 7, floors: 1, fh: 2.8, t: 0.7, wall: COL.concreteDark, doors: { e: [0.5] }, winY0: 1.6, winH: 0.5, roofAccess: false, parapet: 0, tier: 3, poi: 'schlacht' });
    C('arcwreck', ox - 6, gy, oz + 6, 3);
    C('arcwreck', ox + 12, gy, oz - 5.5, 3);
    C('arcwreck', ox + 25, gy, oz + 7.5, 3);
    C('weapon', ox - 34, B.ground(ox - 34, oz + 8) + 0.15, oz + 10, 3);
    E('bastion', ox, oz + 20, 1, 5, { poi: 'schlacht' });
    E('wasp', ox, oz, 3, 25, { poi: 'schlacht' });
    E('tick', ox, oz, 2, 12, { poi: 'schlacht' });
  }

  // ------------------------------------------------------------------ Hydrokuppeln (Plateau Ost)
  {
    const hx2 = 234, hz2 = -165;
    const gy = B.ground(hx2, hz2);
    for (const [cx, cz, r] of [[226, -136, 11], [252, -168, 10], [228, -196, 11]]) {
      B.deco('cyl24open', cx, gy + 0.6, cz, r + 0.2, 1.2, r + 0.2, COL.concreteLight);
      B.deco('dome', cx, gy + 1.1, cz, r, r * 0.85, r, COL.glass, { mat: 'glass' });
      B.deco('dome', cx, gy + 1.1, cz, r + 0.06, r * 0.85 + 0.06, r + 0.06, COL.metalDark, { mat: 'frame' });
      for (let k = -2; k <= 2; k++) B.deco('box', cx + k * 3.5, gy + 0.9, cz, 1.8, 0.8, r * 1.2, COL.green);
      B.col.addCyl(cx, cz, r * 0.96, gy - 1, gy + r * 0.6);
      B.col.addCyl(cx, cz, r * 0.6, gy + r * 0.6, gy + r * 0.85);
    }
    B.building({ x: 214, z: -168, w: 12, d: 10, floors: 2, fh: 3.4, wall: COL.pale, trim: COL.teal, doors: { w: [0.5], e: [0.5] }, tier: 3, poi: 'kuppeln' });
    B.building({ x: 256, z: -208, w: 8, d: 7, floors: 1, fh: 3.2, wall: COL.teal, doors: { n: [0.5] }, roofAccess: false, tier: 2, poi: 'kuppeln' });
    C('arcprobe', 240, gy, -182, 3);
    C('medical', 214, gy + 0.15 + 3.4, -165, 3);
    turrets.push({ x: 214, y: gy + 0.15 + 2 * 3.4, z: -171 });
    E('hornet', hx2, hz2, 2, 20, { poi: 'kuppeln' });
    E('rocketeer', hx2, hz2 - 20, 1, 5, { y: gy + 16, poi: 'kuppeln' });
    E('snitch', hx2, hz2, 1, 10, { y: gy + 20, poi: 'kuppeln' });
    E('tick', 214, -168, 2, 3, { poi: 'kuppeln', indoor: true });
  }

  // ------------------------------------------------------------------ Pumpstation (Plateau West)
  {
    const px = -228, pz = -162;
    const gy = B.ground(px, pz);
    B.building({ x: px, z: pz, w: 14, d: 10, floors: 2, fh: 3.6, wall: COL.concrete, trim: COL.rust, doors: { e: [0.5], s: [0.5] }, tier: 2, poi: 'pumpe' });
    // Steg in den See
    const py = MAP.LAKE_LEVEL + 1.4;
    for (let x = -218, i = 0; x < -120; x += 4, i++) {
      B.solid(x, py - 0.3, pz, 4, 0.3, 3.2, COL.wood);
      if (i % 2 === 0) {
        B.solid(x, MAP.LAKE_FLOOR - 1, pz - 1.4, 0.3, py - MAP.LAKE_FLOOR + 0.7, 0.3, COL.woodDark, { noCol: true });
        B.solid(x, MAP.LAKE_FLOOR - 1, pz + 1.4, 0.3, py - MAP.LAKE_FLOOR + 0.7, 0.3, COL.woodDark, { noCol: true });
      }
    }
    B.solid(-124, py, pz - 3, 3, 4, 3, COL.rust);
    B.deco('box', -124, py + 6, pz + 2, 0.5, 4, 0.5, COL.yellow);
    B.deco('box', -124, py + 8, pz + 5, 0.5, 0.5, 7, COL.yellow);
    B.pipe(-221, gy + 0.8, pz + 2.5, -140, MAP.LAKE_LEVEL - 1, pz + 2.5, 0.8, COL.rust, false);
    C('crate', -126, py, pz + 0.5, 2);
    C('toolbox', px - 3, gy + 0.15, pz + 9, 2);
    E('wasp', px, pz, 2, 15, { poi: 'pumpe' });
    E('pop', px + 10, pz + 10, 2, 10, { poi: 'pumpe' });
    E('hornet', -150, pz, 1, 5, { poi: 'pumpe' });
  }

  // ------------------------------------------------------------------ Verstreute Deckung und Beute
  const scatter = [
    [-110, -20], [-40, 40], [30, 60], [110, -10], [200, -20], [-210, 80], [-100, 180], [0, 200], [120, 240], [220, 230],
    [-230, 240], [-250, 140], [250, 120], [-20, -50], [90, 60], [-200, -60], [160, -130], [-170, -120], [240, -90], [-80, 240],
  ];
  const colors = [COL.teal, COL.rust, COL.orange, COL.red, COL.tealDark, 0x6b6f4a];
  for (const [x, z] of scatter) {
    const gy = B.ground(x, z);
    if (terrain.waterAt(x, z) && gy < terrain.waterAt(x, z).level + 0.3) continue;
    const kind = Math.floor(rnd() * 4);
    if (kind === 0) {
      const t2 = B.container(x, z, rnd() * Math.PI, colors[Math.floor(rnd() * colors.length)]);
      if (rnd() < 0.5) B.container(x + 1, z + R(-1, 1), rnd() * Math.PI, colors[Math.floor(rnd() * colors.length)], t2);
      C('crate', x + 4.2, gy, z + 2, 1);
    } else if (kind === 1) {
      B.car(x, z, rnd() * Math.PI * 2, [0x8a3b30, 0x566b7a, 0x9a8a60][Math.floor(rnd() * 3)]);
      B.barrier(x + 5, z - 3, rnd() * Math.PI);
      C('locker', x - 3, gy, z + 3, 1);
    } else if (kind === 2) {
      for (let i = 0; i < 3; i++) B.crate(x + R(-3, 3), z + R(-3, 3), R(0.9, 1.4));
      C('crate', x, gy, z, 1);
      barrels.push({ x: x + 3.5, z: z - 2.5 });
    } else {
      // kleine Hütte
      B.building({ x, z, w: 6, d: 5, floors: 1, fh: 3, wall: [COL.pale, COL.concrete, COL.teal][Math.floor(rnd() * 3)], doors: { s: [0.5] }, roofAccess: false, parapet: 0, tier: 1, poi: null });
    }
  }
  // Umherstreifende ARC zwischen den Orten
  for (const [x, z] of [[0, 60], [-100, 60], [100, 100], [-200, 100], [200, 80], [0, 150], [-60, -30], [150, -120], [-150, -110]]) {
    enemySpawns.push({ type: rnd() < 0.7 ? 'wasp' : 'pop', x, z, roam: true });
  }
  enemySpawns.push({ type: 'snitch', x: 40, z: 60, y: 24, roam: true });
  enemySpawns.push({ type: 'hornet', x: -40, z: 170, roam: true });

  return { builder: B, spots: B.spots, enemySpawns, turrets, containers, barrels, labels };
}
