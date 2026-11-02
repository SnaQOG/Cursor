// Die Viertel vor der Stadtmauer: Weiße Stadt, Turmstadt und Hafenviertel.
// Die Häuser kommen aus dem Stadtplan (layout.js) und dem Hausgenerator (house.js),
// hier stehen Sonderbauten, Himmelsbrücken, Kai, Stege, Galeone und Leuchtturm.
import { CITY, OUTER_STREETS, WHITE_TOWER, ENERGY_TOWER } from './layout.js';
import { STYLE, PAL } from './house.js';
import { mulberry32 } from '../../util.js';

const WOOD = 0x6a4a30, WOOD_DARK = 0x3a2a1e, STONE = 0x8a8274;
const GALLEON = { x: 70, z: 274, deck: 5 };
const LIGHTHOUSE = { x: -150, z: 259 };
const PIERS = [[-200, 262], [LIGHTHOUSE.x, 252], [GALLEON.x, 265.6], [170, 262]];

// Vor dem Hausbau: Sonderbauten als Grundstücke, Hochhauspaare mit gleicher Höhe für die Brücken
export function prepareOuter(lots, addLot) {
  addLot({ ...WHITE_TOWER, kind: 'house', style: 'weissturm', floors: 12, tier: 3, poi: 'weissturm', level: CITY.GROUND, district: 'weiss', noTank: true });
  const fh = STYLE.hochhaus.fh;
  const tall = lots.filter((l) => l.style === 'hochhaus' && l.kind === 'house' && Math.min(l.x1 - l.x0, l.z1 - l.z0) >= 7);
  const bridges = [];
  for (const a of tall) {
    if (a.bridged || a.id % 2 || bridges.length >= 18) continue;
    for (const c of tall) {
      if (c === a || c.bridged) continue;
      const gapX = c.x0 - a.x1, gapZ = c.z0 - a.z1;
      const ovZ = Math.min(a.z1, c.z1) - Math.max(a.z0, c.z0), ovX = Math.min(a.x1, c.x1) - Math.max(a.x0, c.x0);
      const alongX = gapX >= 1.5 && gapX <= 8 && ovZ >= 3;
      const alongZ = gapZ >= 1.5 && gapZ <= 8 && ovX >= 3;
      if (!alongX && !alongZ) continue;
      const floors = Math.min(a.floors, c.floors);
      const mid = alongX ? (Math.max(a.z0, c.z0) + Math.min(a.z1, c.z1)) / 2 : (Math.max(a.x0, c.x0) + Math.min(a.x1, c.x1)) / 2;
      for (const [l, side] of [[a, alongX ? 'e' : 's'], [c, alongX ? 'w' : 'n']]) {
        l.bridged = true;
        l.floors = floors;
        l.noTank = true;
        l.gaps = [...(l.gaps || []), { side, a0: mid - 0.9, a1: mid + 0.9 }];
      }
      bridges.push({
        from: alongX ? { x: a.x1, z: mid } : { x: mid, z: a.z1 },
        to: alongX ? { x: c.x0, z: mid } : { x: mid, z: c.z0 },
        y: CITY.GROUND + 0.15 + floors * fh,
        axis: alongX ? 'x' : 'z',
      });
      break;
    }
  }
  return bridges;
}

// Nach dem Hausbau: alles, was nicht aus dem Hausgenerator kommt
export function buildOuter({ B, lots, barrels, containers, turrets, enemySpawns, labels, lantern, bench, stall, ropeBridge, bridges, E }) {
  const b = B.b, col = B.col, y = CITY.GROUND;
  const rnd = mulberry32(3301);
  const R = (a, c) => a + rnd() * (c - a);
  const clear = (x, z, r = 0.7) => !col.query(x - r, z - r, x + r, z + r).some((s) => s.minY < y + 4 && s.maxY > y + 0.2);

  for (const br of bridges) ropeBridge(br.from, br.to, br.y, br.axis);

  // ---------------------------------------------------------------- Außenmauern und Straßenlaternen
  for (const side of [-1, 1]) {
    const color = side < 0 ? 0xe6e1d4 : 0x5c6066;
    for (let z = CITY.CLIFF_Z; z < CITY.QUAY - 4; z += 40) {
      const len = Math.min(40, CITY.QUAY - 4 - z);
      B.solid(side * (CITY.OUTER + 0.6), y - 1, z + len / 2, 1.2, 8, len, color, { mat: 'stone' });
      B.solid(side * (CITY.OUTER + 0.6), y - 1, z, 3.4, 11, 3.4, color, { mat: 'stone' });
    }
  }
  for (const s of OUTER_STREETS) {
    if (!s.main) continue;
    const alongX = s.x1 - s.x0 > s.z1 - s.z0;
    const a0 = alongX ? s.x0 : s.z0, a1 = alongX ? s.x1 : s.z1;
    for (let a = a0 + 10; a < a1 - 6; a += 27) {
      const lx = alongX ? a : s.x0 + 0.7, lz = alongX ? s.z0 + 0.7 : a;
      if (clear(lx, lz)) lantern(lx, lz, y);
    }
  }

  // ---------------------------------------------------------------- Weiße Stadt
  const wx = (WHITE_TOWER.x0 + WHITE_TOWER.x1) / 2, wz = (WHITE_TOWER.z0 + WHITE_TOWER.z1) / 2;
  const wRoof = y + 0.15 + 12 * STYLE.weissturm.fh;
  containers.push({ type: 'weapon', x: wx - 4, y: wRoof, z: wz - 4, tier: 3, ry: 0.4 });
  for (const [x, z, ry] of [[wx, wz - 11, 0], [wx, wz + 11, 0], [wx - 11, wz, Math.PI / 2], [wx + 11, wz, Math.PI / 2]]) bench(x, z, y, ry);
  for (const [x, z] of [[wx - 9, wz - 9], [wx + 9, wz - 9], [wx - 9, wz + 9], [wx + 9, wz + 9]]) lantern(x, z, y);
  labels.push({ text: 'WEISSER TURM', x: wx, y: y + 5.6, z: WHITE_TOWER.z1 + 0.05, ry: 0, w: 6, h: 1.1 });

  // ---------------------------------------------------------------- Turmstadt
  const { x: ex, z: ez, r: er } = ENERGY_TOWER;
  b.addPrim('cyl16', ex, 80, ez, er, 160, er, 0x5c646c, { mat: 'metal' });
  for (const yy of [24, 60, 100, 140]) b.addPrim('torus', ex, yy, ez, er + 2, er + 2, 10, 0x9ff8ff, { mat: 'glow', rx: Math.PI / 2 });
  b.addPrim('cone8', ex, 168, ez, er * 0.7, 16, er * 0.7, 0x5c646c, { mat: 'metal' });
  col.addCyl(ex, ez, er, 0, 176);
  for (const [dx, dz, type] of [[12, 0, 'arcprobe'], [-12, 2, 'weapon'], [0, 12, 'toolbox']]) containers.push({ type, x: ex + dx, y, z: ez + dz, tier: 3, ry: 0.3 });
  const roofs = lots.filter((l) => l.style === 'hochhaus' && l.kind === 'house' && l.flat && !l.bridged);
  roofs.sort((p, q) => q.roofY - p.roofY);
  for (const l of roofs.slice(0, 4)) turrets.push({ x: (l.x0 + l.x1) / 2, y: l.roofY, z: (l.z0 + l.z1) / 2 });

  // ---------------------------------------------------------------- Hafenviertel: Kai
  const kai = (x) => Math.abs(x) > 8 && !(x > -138 && x < -42) && !PIERS.some(([px]) => Math.abs(x - px) < 5);
  for (let x = -285; x < 285; x += 17) {
    if (!kai(x)) continue;
    const z = 233.6 + rnd() * 0.8, k = rnd();
    if (k < 0.2) B.crate(x, z, R(0.9, 1.3), y);
    else if (k < 0.34) B.prop('sr_pallet_load', x, z, R(-0.2, 0.2));
    else if (k < 0.46) B.prop('sr_crate_long', x, z, R(-0.1, 0.1));
    else if (k < 0.58) { B.prop('sr_slat_crate', x, z, 0.3, { y, scale: 1.7 }); B.prop('sr_basket_tall', x + 1.2, z, 1, { y, scale: 3, collide: 'cyl' }); } else if (k < 0.66) barrels.push({ x, z });
    else if (k < 0.74) containers.push({ type: 'crate', x, y, z, tier: 1, ry: 0 });
  }
  // Fischmarkt
  for (const [x, z] of [[-18, 217], [18, 217], [-18, 226], [18, 226]]) stall(x, z, y, 0, PAL.awning[Math.floor(rnd() * PAL.awning.length)]);
  // Kräne
  for (const x of [-262, -178, -30, 30, 118, 242]) {
    B.solid(x, y, 238, 1.2, 18, 1.2, WOOD, { mat: 'wood' });
    b.box(x + 4, y + 17, 238, 12, 0.8, 0.8, WOOD, { mat: 'wood' });
    b.box(x + 9, y + 9, 238, 0.08, 8, 0.08, 0x2b2724, { mat: 'metal' });
  }
  // Stege
  for (const [x, end] of PIERS) {
    B.solid(x, y - 0.4, (CITY.QUAY + end) / 2, 4, 0.4, end - CITY.QUAY, WOOD, { mat: 'wood' });
    for (let z = CITY.QUAY + 2; z < end; z += 6) {
      for (const s of [-1, 1]) b.box(x + s * 1.7, -3, z, 0.3, 6, 0.3, WOOD_DARK, { mat: 'wood' });
    }
  }

  // ---------------------------------------------------------------- Gestrandete Galeone (das Deck ist begehbar)
  const { x: gx, z: gz, deck } = GALLEON;
  B.solid(gx, -2, gz, 46, deck + 2, 12, 0x4a3424, { mat: 'wood' });
  B.solid(gx - 17, deck, gz, 12, 5, 12, 0x5a3e2a, { mat: 'wood' });
  b.box(gx + 27, 0, gz, 8, deck, 8, 0x4a3424, { mat: 'wood' });
  B.stairs(gx, 266.79, y, deck, 's', 3, WOOD);
  for (const s of [-1, 1]) b.box(gx + 2, deck, gz + s * 5.9, 34, 0.9, 0.2, 0x3a2a1e, { mat: 'wood' });
  for (const [mx, mh] of [[-6, 34], [8, 40], [20, 28]]) {
    b.addPrim('cyl8', gx + mx, deck + mh / 2, gz, 0.5, mh, 0.5, WOOD_DARK, { mat: 'wood', rz: 0.04 });
    col.addCyl(gx + mx, gz, 0.5, deck, deck + mh);
    b.box(gx + mx, deck + mh * 0.7, gz, 0.4, 0.4, 13, WOOD_DARK, { mat: 'wood' });
    const sx = gx + mx + 0.6, top = deck + mh * 0.68, low = deck + mh * 0.35;
    b.addQuad([sx, low, gz - 5.5], [sx, low, gz + 5.5], [sx, top, gz + 5.5], [sx, top, gz - 5.5], 0xbdb196, { mat: 'cloth', double: true });
  }
  containers.push({ type: 'weapon', x: gx + 12, y: deck, z: gz + 2, tier: 3, ry: 0.2 });
  containers.push({ type: 'arcprobe', x: gx - 2, y: deck, z: gz - 3, tier: 3, ry: 1.1 });
  containers.push({ type: 'toolbox', x: gx + 16, y: deck, z: gz - 3, tier: 2, ry: 2.6 });

  // ---------------------------------------------------------------- Leuchtturm
  const { x: lx, z: lz } = LIGHTHOUSE;
  B.solid(lx, -2, lz, 14, y + 2, 14, STONE, { mat: 'stone' });
  for (let k = 0; k < 6; k++) b.addPrim('cyl16', lx, y + k * 5 + 2.5, lz, 3.6 - k * 0.25, 5, 3.6 - k * 0.25, k % 2 ? 0xa8392e : 0xece8dd);
  b.addPrim('cyl16', lx, y + 31.5, lz, 2, 3, 2, 0xffe7a0, { mat: 'glow' });
  b.addPrim('cone8', lx, y + 34.5, lz, 2.4, 3, 2.4, 0x2a2a2a, { mat: 'metal' });
  col.addCyl(lx, lz, 3.6, y, y + 36);
  containers.push({ type: 'medical', x: lx + 5, y, z: lz + 3, tier: 2, ry: 0.5 });

  // ---------------------------------------------------------------- Gegner
  const roam = { roam: true };
  E('rocketeer', wx, wz, 1, 14);
  E('leaper', -270, -60, 1, 12);
  E('hornet', -250, 130, 1, 25, roam);
  E('snitch', -250, 60, 1, 8, { y: 44, roam: true });
  E('bastion', ex + 14, ez + 14, 1, 2);
  E('rocketeer', 270, 120, 1, 12);
  E('hornet', 255, -60, 1, 25, roam);
  E('snitch', 250, 110, 1, 8, { y: 56, roam: true });
  E('leaper', 118, 222, 1, 10);
  E('hornet', -60, 236, 1, 30, roam);
  E('snitch', 60, 240, 1, 8, { y: 30, roam: true });
  E('wasp', gx, gz, 2, 10, { y: 16, roam: true });
  for (let i = 0; i < 8; i++) enemySpawns.push({ type: 'wasp', x: (rnd() < 0.5 ? -1 : 1) * R(212, 286), z: R(-150, 225), roam: true });
  for (let i = 0; i < 4; i++) enemySpawns.push({ type: 'wasp', x: R(-280, 280), z: R(212, 236), roam: true });
}
