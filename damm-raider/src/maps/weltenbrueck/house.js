// Begehbare Stadthäuser für Weltenbrück: Treppenhaus, Zimmer, Türen,
// echte Fensteröffnungen, Fachwerk, Läden, Möbel, Beuteplätze und Dächer.
// Wände bestehen aus Flächen (sparsamer als Boxen) plus Kollisionsboxen.
import { mulberry32 } from '../../util.js';

export const PAL = {
  plaster: [0xe6dcc4, 0xdcc79c, 0xd8ab90, 0xcbbda4, 0xadc0c8, 0xbcc6a4, 0xebe3d3, 0xd6b68c, 0xcaa18a, 0xe0cfa6],
  infill: [0xe9e1cf, 0xe4d9be, 0xdfd3b8, 0xeadfc8],
  timber: [0x4a3424, 0x3b2a1e, 0x5a3a26, 0x6b3426, 0x2e2420],
  tile: [0x8a4a32, 0x9a5638, 0x7d4430, 0x864f3c, 0x94603f, 0x7a3e2c],
  slate: [0x4c5056, 0x575a5f, 0x44474c, 0x50535a],
  stone: [0xa79f8e, 0x9e9584, 0xb2a996, 0x958d7e],
  brick: [0x8e4a36, 0x9a553c, 0x7f4030, 0x87503b],
  wood: [0x5a3e2a, 0x4e3624, 0x664630, 0x47321f],
  shutter: [0x3f6b4a, 0x6b2f2a, 0x3d5a73, 0x6a5232, 0x2f4f45, 0x7a6a3a],
  awning: [0x9b3a30, 0x2f5e55, 0xb48a2c, 0x5a4a7a, 0x8a5a2a],
  door: [0x6a4a30, 0x5a3e2a, 0x3f5a46, 0x6b2f2a, 0x34495e, 0x7a5a3a],
  frameLight: 0xe8e2d4,
  frameDark: 0x2c2420,
  moss: 0x4f6234,
  ivy: [0x34502a, 0x3e5c2e, 0x2c4424],
  shingle: [0x6e6258, 0x5d5249, 0x7a6a58, 0x665a4e, 0x857664],
  zinc: [0x6e6a62, 0x7a776f, 0x8a5a3a],
  bloom: [0xc8322a, 0xd8507a, 0xe8e2d8, 0x9a4ac0, 0xe07a2a, 0xc8322a],
  rug: [0x7a2e26, 0x2e4a6a, 0x6a5a2a, 0x4a3a5a, 0x3a5a3e],
  rubble: [0x8f877a, 0x7d766a, 0x9c9384],
  white: [0xf1ede3, 0xe9e4d7, 0xf5f2ea, 0xe3ded0],
  concrete: [0x9a9ea3, 0x8a8f95, 0xa8a9a4, 0x7f858b, 0xa89a88],
};

// Stilparameter: Geschosshöhe, Dach, Materialien
export const STYLE = {
  fachwerk: { fh: 3.0, roof: 'gable', pitch: 1.35, roofMat: 'tile', spacing: 2.3, winW: 0.95, winH: 1.3 },
  stein: { fh: 3.3, roof: 'gable', pitch: 1.0, roofMat: 'slate', spacing: 2.5, winW: 1.1, winH: 1.6 },
  backstein: { fh: 3.2, roof: 'gable', pitch: 1.2, roofMat: 'tile', spacing: 2.4, winW: 1.0, winH: 1.5, stepped: true },
  schwarzwald: { fh: 3.0, roof: 'hip', pitch: 1.15, roofMat: 'wood', spacing: 2.2, winW: 1.0, winH: 1.15 },
  turm: { fh: 3.0, roof: 'flat', pitch: 0, roofMat: 'world', spacing: 2.4, winW: 0.9, winH: 1.2 },
  weiss: { fh: 3.2, roof: 'gable', pitch: 1.05, roofMat: 'tile', spacing: 2.6, winW: 1.1, winH: 1.6 },
  weissturm: { fh: 3.0, roof: 'flat', pitch: 0, roofMat: 'world', spacing: 2.8, winW: 0.8, winH: 1.5 },
  hochhaus: { fh: 3.2, roof: 'flat', pitch: 0, roofMat: 'world', spacing: 2.7, winW: 1.2, winH: 1.7 },
  speicher: { fh: 3.6, roof: 'gable', pitch: 0.85, roofMat: 'tile', spacing: 3.0, winW: 1.0, winH: 1.3 },
};

const T = 0.3; // Wandstärke
const DOOR_W = 1.15, DOOR_H = 2.2;
const STEP_D = 0.34;

const pick = (rnd, arr) => arr[Math.floor(rnd() * arr.length) % arr.length];

// Punkt auf einer Wandseite: a = Lage entlang der Wand, y = Höhe, d = Tiefe ab Außenfläche (nach innen positiv)
function wp(w, a, y, d) {
  switch (w.side) {
    case 's': return [a, y, w.fixed - d];
    case 'n': return [a, y, w.fixed + d];
    case 'e': return [w.fixed - d, y, a];
    default: return [w.fixed + d, y, a];
  }
}
const OUTN = { s: [0, 0, 1], n: [0, 0, -1], e: [1, 0, 0], w: [-1, 0, 0] };
const AXIS = { s: [1, 0, 0], n: [1, 0, 0], e: [0, 0, 1], w: [0, 0, 1] };
const neg = (v) => [-v[0], -v[1], -v[2]];
// Drehung, damit die Modell-Vorderseite (+Z) bzw. die Modell-X-Achse in Richtung (dx, dz) zeigt
const yawFront = (dx, dz) => Math.atan2(dx, dz);
const yawX = (dx, dz) => Math.atan2(-dz, dx);

export class HouseKit {
  constructor(B, occ, lots, inst = null) {
    this.B = B;
    this.b = B.b;
    this.col = B.col;
    this.occ = occ;
    this.lots = lots;
    this.inst = inst;
    this.stats = { houses: 0, ruins: 0 };
  }

  // Blender-Modell als Instanz (false, wenn es fehlt)
  model(name, pos, ry, scale = 1, opts = {}) {
    return !!this.inst && this.inst.add(name, pos[0], pos[1], pos[2], ry, scale, opts);
  }

  // Viereck mit gewünschter Normalen (Reihenfolge wird bei Bedarf umgedreht)
  quad(p, n, color, opts = {}) {
    const ux = p[1][0] - p[0][0], uy = p[1][1] - p[0][1], uz = p[1][2] - p[0][2];
    const vx = p[2][0] - p[0][0], vy = p[2][1] - p[0][1], vz = p[2][2] - p[0][2];
    const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
    if (cx * n[0] + cy * n[1] + cz * n[2] < 0) this.b.addQuad(p[0], p[3], p[2], p[1], color, opts);
    else this.b.addQuad(p[0], p[1], p[2], p[3], color, opts);
  }

  tri(p, n, color, opts = {}) {
    const ux = p[1][0] - p[0][0], uy = p[1][1] - p[0][1], uz = p[1][2] - p[0][2];
    const vx = p[2][0] - p[0][0], vy = p[2][1] - p[0][1], vz = p[2][2] - p[0][2];
    const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
    if (cx * n[0] + cy * n[1] + cz * n[2] < 0) this.b.addTri(p[0], p[2], p[1], color, opts);
    else this.b.addTri(p[0], p[1], p[2], color, opts);
  }

  // Box für Optik und Kollision (Unterkante y)
  box(x, y, z, w, h, d, color, opts = {}) {
    this.b.box(x, y, z, w, h, d, color, opts);
    if (!opts.noCol) this.col.addBox(x - w / 2, y, z - d / 2, x + w / 2, y + h, z + d / 2);
  }

  // Wandstück mit Öffnungen. w: {side, fixed, a0, a1 (außen), ia0, ia1 (innen)}
  // ops: [{a0, a1, y0, y1, frame, shutter, kind}] (y relativ zu y)
  wall(w, y, h, ops, color, { mat = 'world', inner = null, frameCol = PAL.frameLight, shutterCol = null, rnd, kf = null, inset = 0 } = {}) {
    const out = OUTN[w.side], ax = AXIS[w.side];
    const segs = [];
    const sorted = ops.slice().sort((p, q) => p.a0 - q.a0);
    let cur = w.a0;
    for (const o of sorted) {
      if (o.a0 > cur) segs.push([cur, o.a0, 0, h]);
      if (o.y0 > 0.01) segs.push([o.a0, o.a1, 0, o.y0]);
      if (o.y1 < h - 0.01) segs.push([o.a0, o.a1, o.y1, h]);
      cur = Math.max(cur, o.a1);
    }
    if (cur < w.a1) segs.push([cur, w.a1, 0, h]);
    for (const [a0, a1, s0, s1] of segs) {
      if (a1 - a0 < 0.01) continue;
      // Kollision
      const p0 = wp(w, a0, y + s0, 0), p1 = wp(w, a1, y + s1, T);
      this.col.addBox(Math.min(p0[0], p1[0]), y + s0, Math.min(p0[2], p1[2]), Math.max(p0[0], p1[0]), y + s1, Math.max(p0[2], p1[2]));
      // Außenfläche
      this.quad([wp(w, a0, y + s0, 0), wp(w, a1, y + s0, 0), wp(w, a1, y + s1, 0), wp(w, a0, y + s1, 0)], out, color, { mat, kf });
      // Innenfläche (an den Ecken gekürzt)
      const i0 = Math.max(a0, w.ia0), i1 = Math.min(a1, w.ia1);
      if (i1 - i0 > 0.01) this.quad([wp(w, i0, y + s0, T), wp(w, i1, y + s0, T), wp(w, i1, y + s1, T), wp(w, i0, y + s1, T)], neg(out), color, { mat, inner });
    }
    // Laibungen und Rahmen
    for (const o of sorted) {
      const Y0 = y + o.y0, Y1 = y + o.y1;
      this.quad([wp(w, o.a0, Y0, 0), wp(w, o.a0, Y0, T), wp(w, o.a0, Y1, T), wp(w, o.a0, Y1, 0)], ax, color, { mat, inner });
      this.quad([wp(w, o.a1, Y0, 0), wp(w, o.a1, Y0, T), wp(w, o.a1, Y1, T), wp(w, o.a1, Y1, 0)], neg(ax), color, { mat, inner });
      if (o.y1 < h - 0.01) this.quad([wp(w, o.a0, Y1, 0), wp(w, o.a1, Y1, 0), wp(w, o.a1, Y1, T), wp(w, o.a0, Y1, T)], [0, -1, 0], color, { mat, inner });
      if (o.y0 > 0.01) {
        // Fensterbank etwas heller
        this.quad([wp(w, o.a0 - 0.06, Y0, -0.06), wp(w, o.a1 + 0.06, Y0, -0.06), wp(w, o.a1, Y0, T), wp(w, o.a0, Y0, T)], [0, 1, 0], o.frame ? frameCol : color, { inner, mat });
      }
      const out2 = OUTN[w.side];
      if (o.frame && o.kind === 'win' && o.a1 - o.a0 <= 2.2 && o.y1 - o.y0 <= 2.6
        && this.model('kc_window', wp(w, (o.a0 + o.a1) / 2, Y0, inset), yawFront(out2[0], out2[2]), [o.a1 - o.a0, Y1 - Y0, 1], { tint: frameCol, layer: 'det', double: true })) {
        // Rahmen aus Blender
      } else if (o.frame) {
        const f = 0.09, d = -0.025;
        const det = { layer: 'det' };
        this.quad([wp(w, o.a0 - f, Y0, d), wp(w, o.a0, Y0, d), wp(w, o.a0, Y1 + f, d), wp(w, o.a0 - f, Y1 + f, d)], out, frameCol, det);
        this.quad([wp(w, o.a1, Y0, d), wp(w, o.a1 + f, Y0, d), wp(w, o.a1 + f, Y1 + f, d), wp(w, o.a1, Y1 + f, d)], out, frameCol, det);
        this.quad([wp(w, o.a0, Y1, d), wp(w, o.a1, Y1, d), wp(w, o.a1, Y1 + f, d), wp(w, o.a0, Y1 + f, d)], out, frameCol, det);
        // Fensterkreuz bei größeren Fenstern (nur Optik)
        if (o.kind === 'win' && o.a1 - o.a0 > 0.8 && rnd && rnd() < 0.45) {
          const m = (o.a0 + o.a1) / 2, ym = Y0 + (Y1 - Y0) * 0.62;
          this.quad([wp(w, m - 0.03, Y0, 0.08), wp(w, m + 0.03, Y0, 0.08), wp(w, m + 0.03, Y1, 0.08), wp(w, m - 0.03, Y1, 0.08)], out, frameCol, det);
          this.quad([wp(w, o.a0, ym - 0.03, 0.08), wp(w, o.a1, ym - 0.03, 0.08), wp(w, o.a1, ym + 0.03, 0.08), wp(w, o.a0, ym + 0.03, 0.08)], out, frameCol, det);
        }
      }
      if (o.shutter && shutterCol != null && this.inst && this.inst.has('kc_shutter')) {
        // Fensterläden aus Blender: Flügel an beiden Seiten, Scharnier an der Öffnung
        const hw = (o.a1 - o.a0) / 2;
        for (const [edge, dir] of [[o.a1, 1], [o.a0, -1]]) {
          if (dir > 0 ? edge + hw > w.a1 - 0.05 : edge - hw < w.a0 + 0.05) continue;
          const ry = yawX(ax[0] * dir, ax[2] * dir);
          const front = [Math.sin(ry), Math.cos(ry)];
          const inward = front[0] * out2[0] + front[1] * out2[2] < 0;
          // Ein loser Laden hängt nur auf einer Seite schief
          const tilt = o.shutter === 'hang' && dir > 0 ? 0.16 : 0;
          this.model('kc_shutter', wp(w, edge, Y0, inward ? -0.12 : -0.01), ry, [hw / 0.5, Y1 - Y0, 1], { tint: shutterCol, layer: 'det', rz: -tilt, double: true });
        }
      } else if (o.shutter && shutterCol != null) {
        const sw = (o.a1 - o.a0) / 2, d = -0.05;
        const hang = o.shutter === 'hang';
        if (o.a0 - sw > w.a0 + 0.05) this.quad([wp(w, o.a0 - sw, Y0, d), wp(w, o.a0, Y0, d), wp(w, o.a0, Y1, d), wp(w, o.a0 - sw, Y1, d)], out, shutterCol, { layer: 'det' });
        if (o.a1 + sw < w.a1 - 0.05) {
          if (hang) this.quad([wp(w, o.a1, Y0 - 0.3, d - 0.2), wp(w, o.a1 + sw * 0.9, Y0 - 0.1, d - 0.25), wp(w, o.a1 + sw, Y1 - 0.2, d - 0.1), wp(w, o.a1, Y1 - 0.4, d)], out, shutterCol, { double: true, layer: 'det' });
          else this.quad([wp(w, o.a1, Y0, d), wp(w, o.a1 + sw, Y0, d), wp(w, o.a1 + sw, Y1, d), wp(w, o.a1, Y1, d)], out, shutterCol, { layer: 'det' });
        }
      }
    }
  }

  // Balkenstreifen auf der Außenfläche (Fachwerk). p, q: [a, y]
  beam(w, p, q, width, color) {
    const d = -0.055;
    const dx = q[0] - p[0], dy = q[1] - p[1];
    const l = Math.hypot(dx, dy) || 1;
    const nx = (-dy / l) * width / 2, ny = (dx / l) * width / 2;
    this.quad([wp(w, p[0] - nx, p[1] - ny, d), wp(w, q[0] - nx, q[1] - ny, d), wp(w, q[0] + nx, q[1] + ny, d), wp(w, p[0] + nx, p[1] + ny, d)], OUTN[w.side], color, { shade: false, mat: 'timber' });
    // Seitenflächen, damit der Balken aus der Wand steht
    const ax = AXIS[w.side], mx = -dy / l, my = dx / l;
    for (const s of [1, -1]) {
      const n = [ax[0] * mx * s, my * s, ax[2] * mx * s], ex = nx * s, ey = ny * s;
      this.quad([wp(w, p[0] + ex, p[1] + ey, d), wp(w, q[0] + ex, q[1] + ey, d), wp(w, q[0] + ex, q[1] + ey, 0), wp(w, p[0] + ex, p[1] + ey, 0)], n, darker(color, 0.72), { mat: 'timber', layer: 'det' });
    }
  }

  // Fassadenleiste: Vorderfläche plus Ober- und Unterseite (für Gesimse, Schwellen, Sockel)
  ledge(w, a0, a1, y0, y1, depth, color, mat = 'world', layer = '') {
    const o = OUTN[w.side];
    this.quad([wp(w, a0, y0, -depth), wp(w, a1, y0, -depth), wp(w, a1, y1, -depth), wp(w, a0, y1, -depth)], o, color, { mat, layer });
    this.quad([wp(w, a0, y1, -depth), wp(w, a1, y1, -depth), wp(w, a1, y1, 0), wp(w, a0, y1, 0)], [0, 1, 0], color, { mat, layer });
    this.quad([wp(w, a0, y0, -depth), wp(w, a1, y0, -depth), wp(w, a1, y0, 0), wp(w, a0, y0, 0)], [0, -1, 0], darker(color, 0.7), { mat, layer });
  }

  // Details einer Fassadenseite (nach der Wand gebaut)
  facade(lot, w, f, y, h, fy, ops, runs, c) {
    if (!runs.length) return;
    const k = w.side, out = OUTN[k];
    const ry = yawFront(out[0], out[2]);
    const rnd = c.rnd;
    const stone = lot.style === 'stein' || lot.style === 'backstein' || lot.style === 'weiss' || lot.style === 'speicher' || lot.style === 'hochhaus';
    // Sockel aus Naturstein (Putz- und Backsteinhäuser)
    if (f === 0 && stone) {
      for (const [r0, r1] of runs) {
        let cur = r0;
        const cuts = ops.filter((o) => o.kind === 'door' && o.a1 > r0 && o.a0 < r1).sort((p, q) => p.a0 - q.a0);
        for (const o of [...cuts, { a0: r1, a1: r1 }]) {
          if (o.a0 - cur > 0.1) this.ledge(w, cur, o.a0, y, y + (lot.pilot ? 0.85 : 0.6), lot.pilot ? 0.08 : 0.04, c.plinth, 'stone');
          cur = Math.max(cur, o.a1);
        }
      }
    }
    // Eckquader
    if (lot.style === 'stein' || (lot.style === 'weiss' && (!lot.pilot || lot.id % 3 === 0))) {
      for (const [r0, r1] of runs) {
        for (const [edge, dir] of [[w.a0, 1], [w.a1, -1]]) {
          if (Math.abs((dir > 0 ? r0 : r1) - edge) > 0.3) continue;
          for (let yy = y + (f === 0 ? (lot.pilot ? 0.85 : 0.6) : 0), i = 0; yy < y + h - 0.05; yy += 0.36, i++) {
            const wdt = i % 2 ? 0.3 : 0.5, hh = Math.min(0.33, y + h - yy);
            const a0 = dir > 0 ? edge : edge - wdt, a1 = dir > 0 ? edge + wdt : edge;
            this.quad([wp(w, a0, yy, -0.03), wp(w, a1, yy, -0.03), wp(w, a1, yy + hh, -0.03), wp(w, a0, yy + hh, -0.03)], out, c.quoin, { mat: 'stone', layer: 'det' });
          }
        }
      }
    }
    for (const o of ops) {
      const Y0 = y + o.y0, Y1 = y + o.y1, m = (o.a0 + o.a1) / 2, ow = o.a1 - o.a0;
      if (o.kind === 'win' && (lot.style === 'stein' || lot.pilot) && ow <= 2.2) {
        // Faschen und Verdachung
        const b = 0.14, d = -0.03, fc = c.fasche, det = { layer: 'det', mat: 'wplaster' };
        // Fensterbank mit Tiefe
        this.ledge(w, o.a0 - b - 0.05, o.a1 + b + 0.05, Y0 - 0.14, Y0 - 0.05, 0.13, c.step, 'stone', 'det');
        this.quad([wp(w, o.a0 - b, Y0 - 0.06, d), wp(w, o.a0, Y0 - 0.06, d), wp(w, o.a0, Y1 + b, d), wp(w, o.a0 - b, Y1 + b, d)], out, fc, det);
        this.quad([wp(w, o.a1, Y0 - 0.06, d), wp(w, o.a1 + b, Y0 - 0.06, d), wp(w, o.a1 + b, Y1 + b, d), wp(w, o.a1, Y1 + b, d)], out, fc, det);
        this.quad([wp(w, o.a0, Y1, d), wp(w, o.a1, Y1, d), wp(w, o.a1, Y1 + b, d), wp(w, o.a0, Y1 + b, d)], out, fc, det);
        if (f > 0 && (!lot.pilot || lot.id % 3 !== 2)) this.ledge(w, o.a0 - b - 0.12, o.a1 + b + 0.12, Y1 + b + 0.04, Y1 + b + 0.18, 0.14, c.fasche, 'stone', 'det');
      }
      else if (o.kind === 'win' && ow <= 2.2) {
        // Fensterbank, im Backstein dazu ein Sturz aus hochkant gestellten Steinen
        const wood = lot.style === 'fachwerk' || lot.style === 'schwarzwald' || lot.style === 'turm';
        this.ledge(w, o.a0 - 0.08, o.a1 + 0.08, Y0 - 0.1, Y0 - 0.02, 0.1, wood ? c.timber : c.step, wood ? 'timber' : 'stone', 'det');
        if (lot.style === 'backstein' || lot.style === 'speicher') this.ledge(w, o.a0 - 0.1, o.a1 + 0.1, Y1 + 0.02, Y1 + 0.26, 0.025, 0x74402f, 'brick', 'det');
      }
      // Kellerfenster im Sockel
      if (f === 0 && stone && o.kind === 'win' && ow <= 2.2 && !lot.shop && rnd() < 0.28) {
        const pd = (lot.pilot ? 0.08 : 0.04) + 0.006, kw = 0.32, K0 = y + 0.13, K1 = y + 0.47, det = { layer: 'det', shade: false };
        this.quad([wp(w, m - kw - 0.05, K0 - 0.05, -pd), wp(w, m + kw + 0.05, K0 - 0.05, -pd), wp(w, m + kw + 0.05, K1 + 0.05, -pd), wp(w, m - kw - 0.05, K1 + 0.05, -pd)], out, c.step, { ...det, mat: 'stone' });
        this.quad([wp(w, m - kw, K0, -pd - 0.004), wp(w, m + kw, K0, -pd - 0.004), wp(w, m + kw, K1, -pd - 0.004), wp(w, m - kw, K1, -pd - 0.004)], out, 0x12110e, det);
        for (const t of [-0.6, -0.2, 0.2, 0.6]) this.quad([wp(w, m + kw * t - 0.012, K0, -pd - 0.012), wp(w, m + kw * t + 0.012, K0, -pd - 0.012), wp(w, m + kw * t + 0.012, K1, -pd - 0.012), wp(w, m + kw * t - 0.012, K1, -pd - 0.012)], out, 0x3a3835, { ...det, mat: 'metal' });
      }
      // Blumenkästen unter Fenstern der Obergeschosse
      if (o.kind === 'win' && f > 0 && ow <= 2.2 && !['turm', 'dom', 'hochhaus', 'speicher', 'weissturm'].includes(lot.style) && rnd() < 0.22) {
        this.model('kc_flowerbox', wp(w, m, Y0 - 0.02, 0), ry, [ow + 0.1, 1, 1], { tint: pick(rnd, PAL.bloom), layer: 'det' });
      }
      // Türgewände und Stufe
      if (o.kind === 'door') {
        const b = stone ? 0.2 : 0.12, d = -0.03, gc = stone ? c.fasche : c.timber;
        const gm = stone ? 'stone' : 'timber';
        this.quad([wp(w, o.a0 - b, Y0, d), wp(w, o.a0, Y0, d), wp(w, o.a0, Y1 + b, d), wp(w, o.a0 - b, Y1 + b, d)], out, gc, { mat: gm });
        this.quad([wp(w, o.a1, Y0, d), wp(w, o.a1 + b, Y0, d), wp(w, o.a1 + b, Y1 + b, d), wp(w, o.a1, Y1 + b, d)], out, gc, { mat: gm });
        this.ledge(w, o.a0 - b, o.a1 + b, Y1, Y1 + b + (stone ? 0.08 : 0), stone ? 0.08 : 0.05, gc, gm);
        if (stone) this.ledge(w, m - 0.13, m + 0.13, Y1 - 0.05, Y1 + b + 0.12, 0.12, darker(gc, 0.92), 'stone');
        if (f === 0) {
          // Emailschild mit Hausnummer neben der Tür
          const na = o.a1 + b + 0.2, ny = Y1 - 0.2, det2 = { layer: 'det', shade: false };
          if (na + 0.14 < w.a1) {
            this.quad([wp(w, na - 0.11, ny, -0.035), wp(w, na + 0.11, ny, -0.035), wp(w, na + 0.11, ny + 0.16, -0.035), wp(w, na - 0.11, ny + 0.16, -0.035)], out, 0x24466e, det2);
            this.quad([wp(w, na - 0.04, ny + 0.03, -0.04), wp(w, na + 0.04, ny + 0.03, -0.04), wp(w, na + 0.04, ny + 0.13, -0.04), wp(w, na - 0.04, ny + 0.13, -0.04)], out, 0xe8e4d8, det2);
          }
          // Turmstadt: Betonvordach mit Lichtleiste, liest sich auch in den dunklen Schluchten
          if (lot.style === 'hochhaus') {
            this.ledge(w, o.a0 - 0.35, o.a1 + 0.35, Y1 + 0.26, Y1 + 0.38, 0.95, c.fasche, 'wplaster', 'det');
            const lc = wp(w, m, Y1 + 0.2, -0.5);
            const al = k === 'n' || k === 's';
            this.b.box(lc[0], lc[1], lc[2], al ? ow + 0.2 : 0.07, 0.05, al ? 0.07 : ow + 0.2, 0x9ff8ff, { mat: 'glow', layer: 'det' });
          }
        }
        if (f === 0) {
          const p0 = wp(w, o.a0 - 0.15, y, -0.45), p1 = wp(w, o.a1 + 0.15, y, 0);
          const bx0 = Math.min(p0[0], p1[0]), bx1 = Math.max(p0[0], p1[0]), bz0 = Math.min(p0[2], p1[2]), bz1 = Math.max(p0[2], p1[2]);
          this.box((bx0 + bx1) / 2, y, (bz0 + bz1) / 2, bx1 - bx0, 0.15, bz1 - bz0, c.step, { mat: 'stone' });
        }
      }
    }
    // Fachwerk: vorstehende Schwelle mit Balkenköpfen an jeder Geschossgrenze
    if (lot.style === 'fachwerk' && f > 0) {
      for (const [r0, r1] of runs) {
        this.ledge(w, r0, r1, y - 0.04, y + 0.2, 0.1, c.timber, 'timber');
        for (let a = r0 + 0.4; a < r1 - 0.3; a += 0.8) this.model('kc_beamhead', wp(w, a, y - 0.24, 0), ry, 1, { tint: c.timber, layer: 'det' });
      }
    }
  }

  // Ist die Außenseite an Stelle (x,z) für ein Geschoss mit Unterkante y offen?
  openAt(x, z, yFloor, selfId) {
    const id = this.occ.at(x, z);
    if (id === -1) return true;
    if (id < 0 || id === selfId) return false;
    const n = this.lots[id];
    if (!n || n.kind === 'ruin') return n ? yFloor > n.level + 3.2 : false;
    const top = n.level + 0.15 + n.floors * STYLE[n.style].fh;
    return top < yFloor + 1.0;
  }

  // Offene Abschnitte einer Wandseite für ein Geschoss
  openRuns(w, yFloor, selfId) {
    const runs = [];
    const step = 0.5;
    let start = null;
    const off = 0.7;
    const o = OUTN[w.side];
    for (let a = w.a0 + step / 2; a < w.a1; a += step) {
      const p = wp(w, a, 0, -off);
      const open = this.openAt(p[0] + o[0] * 0, p[2], yFloor, selfId);
      if (open && start == null) start = a - step / 2;
      if (!open && start != null) { runs.push([start, a - step / 2]); start = null; }
    }
    if (start != null) runs.push([start, w.a1]);
    return runs;
  }

  // ---------------------------------------------------------------- Haus
  house(lot, rnd) {
    if (lot.kind === 'ruin') return this.ruin(lot, rnd);
    const st = STYLE[lot.style];
    const { x0, x1, z0, z1 } = lot;
    const y = lot.level;
    const fh = st.fh;
    const W = x1 - x0, D = z1 - z0;
    const alongX = W >= D;
    const L = alongX ? W : D, S = alongX ? D : W;
    const nSteps = Math.round(fh / 0.3);
    const run = nSteps * STEP_D;
    let floors = lot.floors;
    const canStairs = L >= run + 2 * T + 1.7 && S >= 3.9;
    if (!canStairs) floors = 1;
    const roofAccess = st.roof === 'flat' && canStairs;
    lot.floors = floors;

    // Wandseiten
    const sides = {
      s: { side: 's', fixed: z1, a0: x0, a1: x1, ia0: x0 + T, ia1: x1 - T },
      n: { side: 'n', fixed: z0, a0: x0, a1: x1, ia0: x0 + T, ia1: x1 - T },
      e: { side: 'e', fixed: x1, a0: z0, a1: z1, ia0: z0 + T, ia1: z1 - T },
      w: { side: 'w', fixed: x0, a0: z0, a1: z1, ia0: z0 + T, ia1: z1 - T },
    };
    const openLen = (k, yF) => this.openRuns(sides[k], yF, lot.id).reduce((s, r) => s + r[1] - r[0], 0);
    const shortSides = alongX ? ['w', 'e'] : ['n', 's'];
    const longSides = alongX ? ['n', 's'] : ['w', 'e'];
    const yG = y + 0.15;
    const front = openLen(shortSides[0], yG) >= openLen(shortSides[1], yG) ? shortSides[0] : shortSides[1];
    const back = front === shortSides[0] ? shortSides[1] : shortSides[0];
    // Treppen an der weniger offenen Längsseite
    const sideA = openLen(longSides[0], yG) <= openLen(longSides[1], yG) ? longSides[0] : longSides[1];
    const sideB = sideA === longSides[0] ? longSides[1] : longSides[0];

    // (u, v) -> Welt: u entlang L ab der Vorderseite, v quer ab Seite A
    const toX = (u, v) => (alongX ? (front === 'w' ? x0 + u : x1 - u) : (sideA === 'w' ? x0 + v : x1 - v));
    const toZ = (u, v) => (alongX ? (sideA === 'n' ? z0 + v : z1 - v) : (front === 'n' ? z0 + u : z1 - u));
    const rectUV = (u0, u1, v0, v1) => {
      const xa = toX(u0, v0), xb = toX(u1, v1), za = toZ(u0, v0), zb = toZ(u1, v1);
      return { x0: Math.min(xa, xb), x1: Math.max(xa, xb), z0: Math.min(za, zb), z1: Math.max(za, zb) };
    };
    // Lage entlang einer Wand (a) für u bzw. v
    const aOfU = (u) => (alongX ? toX(u, 0) : toZ(u, 0));
    const aOfV = (v) => (alongX ? toZ(0, v) : toX(0, v));

    const sw = S >= 5.5 ? 1.2 : 1.05;
    const uS1 = L - T - 0.1, uS0 = uS1 - run;
    // Zwischenwand mit 1 m Treppenabsatz davor
    const uPart = uS0 - 1.3; // Mitte der Zwischenwand
    const hasPart = canStairs && uPart - T >= 2.6;
    const frontRoomEnd = hasPart ? uPart - 0.08 : canStairs ? uS0 - 0.1 : L - T;

    // Farben
    const wallCol = lot.style === 'backstein' || lot.style === 'speicher' ? pick(rnd, PAL.brick)
      : lot.style === 'fachwerk' ? pick(rnd, PAL.infill)
        : lot.style === 'weiss' || lot.style === 'weissturm' ? pick(rnd, PAL.white)
          : lot.style === 'hochhaus' ? pick(rnd, PAL.concrete) : pick(rnd, PAL.plaster);
    const baseCol = lot.style === 'fachwerk' || lot.style === 'schwarzwald' || lot.style === 'turm' ? pick(rnd, PAL.stone) : wallCol;
    const upperCol = lot.style === 'schwarzwald' || lot.style === 'turm' ? pick(rnd, PAL.wood) : wallCol;
    const timber = pick(rnd, PAL.timber);
    const frameCol = lot.style === 'schwarzwald' || lot.style === 'turm' ? pick(rnd, PAL.wood) : PAL.frameLight;
    const shutterCol = rnd() < 0.6 ? mix(pick(rnd, PAL.shutter), 0x6a6458, 0.3) : null;
    const wallMat = lot.style === 'backstein' || lot.style === 'speicher' ? 'brick' : 'wplaster';
    let baseMat = lot.style === 'fachwerk' || lot.style === 'schwarzwald' || lot.style === 'turm' ? 'stone' : wallMat;
    let upperMat = lot.style === 'schwarzwald' ? 'wood' : lot.style === 'turm' ? 'wood' : wallMat;
    let baseColor = baseCol, upperColor = upperCol, frameColor = lot.style === 'hochhaus' ? PAL.frameDark : frameCol;
    let shutters = lot.style === 'hochhaus' || lot.style === 'speicher' ? null : shutterCol;
    // Schwarzwald: Hälfte der Häuser mit Schindelfassade
    if (lot.style === 'schwarzwald' && rnd() < 0.5) { upperMat = 'shingle'; upperColor = pick(rnd, PAL.shingle); }
    const facadeCtx = {
      rnd, timber, plinth: lot.pilot ? pick(rnd, PAL.stone) : darker(pick(rnd, PAL.stone), 0.85), quoin: pick(rnd, PAL.stone),
      fasche: lot.style === 'backstein' ? PAL.frameLight : mix(pick(rnd, PAL.stone), 0xffffff, 0.25), step: pick(rnd, PAL.stone),
      wall: wallCol,
    };
    const hook = this.styleHook && this.styleHook(lot);
    if (hook) {
      baseColor = hook.base; upperColor = hook.wall; baseMat = upperMat = hook.mat; frameColor = hook.frame; shutters = null;
    }
    const innerRect = { x0: x0 + T - 0.02, x1: x1 - T + 0.02, z0: z0 + T - 0.02, z1: z1 - T + 0.02, k: 0.62 };

    // Türen im Erdgeschoss
    const doors = []; // {side, a}
    const runsG = {};
    for (const k of ['n', 's', 'e', 'w']) runsG[k] = this.openRuns(sides[k], yG, lot.id);
    const fits = (k, a, half = DOOR_W / 2 + 0.15) => runsG[k].some(([r0, r1]) => a - half >= r0 && a + half <= r1);
    const vMid = S / 2;
    const tryDoor = (k, a) => { if (fits(k, a)) { doors.push({ side: k, a }); return true; } return false; };
    const frontRoomMidU = Math.max(T + 0.9, Math.min(frontRoomEnd - 0.9, (T + frontRoomEnd) / 2));
    const candidates = [
      () => tryDoor(front, aOfV(vMid)),
      () => tryDoor(sideB, aOfU(frontRoomMidU)),
      () => tryDoor(sideA, aOfU(frontRoomMidU)),
      () => canStairs && tryDoor(back, aOfV(vMid)),
      () => tryDoor(sideB, aOfU(Math.min(frontRoomEnd - 0.8, T + 1.0))),
    ];
    for (const c of candidates) if (c()) break;
    // zweite Tür (Hinterausgang, Hof)
    if (doors.length && rnd() < 0.75) {
      for (const k of [back, sideB, sideA]) {
        if (doors.some((d) => d.side === k)) continue;
        const a = k === back ? aOfV(vMid) : aOfU(frontRoomMidU);
        if (k === back && !canStairs) continue;
        if (tryDoor(k, a)) break;
      }
    }
    for (const d of lot.forceDoors || []) doors.push(d);
    // Notfalls: irgendein offener Abschnitt
    if (!doors.length) {
      for (const k of [front, sideB, back, sideA]) {
        const r = runsG[k].find(([r0, r1]) => r1 - r0 >= DOOR_W + 0.4);
        if (!r) continue;
        let a = (r[0] + r[1]) / 2;
        // nicht in die Treppe laufen
        if (k === sideA && canStairs) {
          const sa = aOfU(uS0), sb = aOfU(uS1);
          if (a > Math.min(sa, sb) - 0.8 && a < Math.max(sa, sb) + 0.3) continue;
        }
        doors.push({ side: k, a });
        break;
      }
    }

    // Schmutz am Sockel und Schatten unter der Traufe, je nach Alter des Hauses
    const yEaves = yG + floors * fh, age = lot.age = (lot.id * 0.618) % 1;
    // Hochhäuser stehen fast immer im Schatten der Nachbarn und bekommen etwas mehr Grundhelligkeit
    const foot = 0.8 - age * 0.14, top = 0.86 - age * 0.06, overall = (1 - age * 0.1) * (lot.style === 'hochhaus' ? 1.28 : 1);
    const wallShade = (yy) => overall * (yy < y + 1.3 ? foot + (1 - foot) * ((yy - y) / 1.3) : yy > yEaves - 0.8 ? 1 - (1 - top) * ((yy - (yEaves - 0.8)) / 0.8) : 1);
    if (lot.pilot) baseMat = upperMat = 'wplaster';
    const shopSide = lot.shop ? doors[0]?.side : null;
    lot.doors = doors;
    lot.sideA = sideA;
    if (canStairs && floors > 1) {
      const fu = uS0 - 0.55, fv = T + sw / 2;
      lot.stair0 = { x: toX(fu, fv), z: toZ(fu, fv), dx: toX(fu + 1, fv) - toX(fu, fv), dz: toZ(fu + 1, fv) - toZ(fu, fv), y: yG, fh };
    }
    const partA = hasPart ? aOfU(uPart) : null;

    // Geschosse
    for (let f = 0; f < floors; f++) {
      const fy = yG + f * fh; // Fußboden
      const yw0 = f === 0 ? y : fy;
      const hw = f === 0 ? fh + 0.15 : fh;
      const off = fy - yw0; // Öffnungen relativ zur Wandunterkante
      const isTop = f === floors - 1;
      const col = f === 0 ? baseColor : upperColor;
      const mat = f === 0 ? baseMat : upperMat;
      for (const k of ['n', 's', 'e', 'w']) {
        const w = sides[k];
        const runs = f === 0 ? runsG[k] : this.openRuns(w, fy, lot.id);
        const ops = [];
        if (f === 0) {
          for (const d of doors) {
            if (d.side !== k) continue;
            const shop = shopSide === k && d === doors[0];
            const dw = shop ? 1.6 : DOOR_W;
            ops.push({ a0: d.a - dw / 2, a1: d.a + dw / 2, y0: off, y1: off + DOOR_H + (shop ? 0.2 : 0), frame: true, kind: 'door' });
          }
        }
        const isShop = f === 0 && shopSide === k;
        const winW = isShop ? 2.0 : st.winW, winH = isShop ? 1.9 : Math.min(st.winH, fh - 1.35);
        const sill = isShop ? 0.45 : 0.9;
        const spacing = isShop ? 3.0 : st.spacing;
        for (const [r0, r1] of runs) {
          const len = r1 - r0 - 1.0;
          if (len < winW) continue;
          const n = Math.max(1, Math.floor(len / spacing));
          for (let i = 0; i < n; i++) {
            const c = r0 + 0.5 + (len / n) * (i + 0.5);
            const a0 = c - winW / 2, a1 = c + winW / 2;
            if (ops.some((o) => a0 < o.a1 + 0.35 && a1 > o.a0 - 0.35)) continue;
            if (partA != null && (k === sideA || k === sideB) && Math.abs(c - partA) < winW / 2 + 0.3) continue;
            if (a0 < w.a0 + 0.55 || a1 > w.a1 - 0.55) continue;
            const r = rnd();
            if (isShop) (lot.shopWins ||= []).push([a0, a1]);
            ops.push({
              a0, a1, y0: off + sill, y1: off + sill + winH, kind: 'win',
              frame: true,
              shutter: !isShop && shutters != null && r < 0.7 ? (r < 0.03 ? 'hang' : 'open') : null,
            });
          }
        }
        this.wall(w, yw0, hw, ops, col, { mat, inner: innerRect, frameCol: frameColor, shutterCol: shutters, rnd, kf: wallShade, inset: 0.13 });
        this.facade(lot, w, f, yw0, hw, fy, ops, runs, facadeCtx);
        const anyOpen = runs.length > 0;
        if (lot.pilot && anyOpen) this.pilotBands(lot, w, runs, f, isTop, yw0, hw, facadeCtx);
        if (lot.style === 'hochhaus' && anyOpen) this.towerFacade(lot, w, runs, ops, f, isTop, yw0, hw, facadeCtx);
        // Fachwerk auf offenen Seiten der Obergeschosse
        if (lot.style === 'fachwerk' && f > 0 && anyOpen) this.timberFrame(w, yw0, hw, ops, runs, timber, rnd);
        // Gesims bei Steinhäusern
        if ((lot.style === 'stein' || lot.style === 'backstein') && anyOpen && (f === 0 || isTop)) {
          const cy = isTop ? yw0 + hw - 0.18 : yw0 + hw - 0.05;
          for (const [r0, r1] of runs) {
            this.quad([wp(w, r0, cy, -0.12), wp(w, r1, cy, -0.12), wp(w, r1, cy + 0.2, -0.12), wp(w, r0, cy + 0.2, -0.12)], OUTN[k], lot.style === 'backstein' ? PAL.frameLight : pick(rnd, PAL.stone));
            this.quad([wp(w, r0, cy + 0.2, -0.12), wp(w, r1, cy + 0.2, -0.12), wp(w, r1, cy + 0.2, 0), wp(w, r0, cy + 0.2, 0)], [0, 1, 0], pick(rnd, PAL.stone));
          }
        }
        // Schwarzwald: Holzbalkon im Obergeschoss
        if (lot.style === 'schwarzwald' && f === 1 && anyOpen && k === front && rnd() < 0.5) this.balcony(w, fy, runs, pick(rnd, PAL.wood));
      }

      // Offene Türflügel innen an der Wand
      if (f === 0 && this.inst) {
        for (const d of doors) {
          if (rnd() < 0.25) continue;
          const w = sides[d.side];
          const shop = shopSide === d.side && d === doors[0];
          const dw = shop ? 1.6 : DOOR_W;
          this.doorLeaf(w, d.a - dw / 2, dw, fy, DOOR_H, pick(rnd, PAL.door));
          (lot.leaves ||= []).push({ side: d.side, a0: d.a - dw / 2 - 0.15, a1: d.a - dw / 2 + 0.8 });
        }
      }
      // Böden
      if (f === 0) this.slab(rectUV(0, L, 0, S), y, yG, null, 0x8a6a4a, innerRect, true);
      // Treppe nach oben
      const up = f < floors - 1 || roofAccess;
      if (up && canStairs) {
        const even = f % 2 === 0;
        const v0 = even ? T : S - T - sw, v1 = v0 + sw;
        this.stairsUV(toX, toZ, uS0, uS1, v0, v1, fy, fy + fh, even ? 1 : -1, even ? 1 : -1, innerRect, pick(rnd, PAL.wood));
      }
      // Decke über diesem Geschoss = Boden des nächsten
      const ceilY = fy + fh;
      let hole = null;
      if (up && canStairs) {
        const even = f % 2 === 0;
        const v0 = even ? T : S - T - sw - 0.05, v1 = even ? T + sw + 0.05 : S - T;
        hole = rectUV(uS0 - 0.05, L - T, v0, v1);
      }
      const inner = rectUV(T, L - T, T, S - T);
      if (isTop && st.roof === 'flat') this.slab(rectUV(0, L, 0, S), ceilY - 0.25, ceilY, hole, PAL.stone[1], innerRect, false, true);
      else this.slab(inner, ceilY - 0.2, ceilY, hole, 0x7d5f42, innerRect, false);

      // Zwischenwand mit Durchgang
      if (hasPart) this.partition(rectUV, uPart, S, fy, fh, upperColor, innerRect, lot.pilot ? 'wplaster' : 'world');
      // Möbel und Beuteplätze im vorderen Zimmer
      this.furnish(lot, rnd, rectUV, toX, toZ, frontRoomEnd, S, fy, f, sw, doors, aOfU, aOfV, sideB, front, innerRect, hasPart);
    }

    // Dach
    const yTop = yG + floors * fh;
    const roofCol = st.roofMat === 'slate' ? pick(rnd, PAL.slate) : st.roofMat === 'wood' ? pick(rnd, PAL.wood) : pick(rnd, PAL.tile);
    const mossy = rnd() < 0.45;
    const rc = mossy ? mix(roofCol, PAL.moss, 0.18 + rnd() * 0.2) : roofCol;
    if (lot.pilot && st.roof === 'gable') {
      // Giebelhaus: First quer zur langen Seite, wenn der Grundriss es erlaubt
      const giebel = lot.id % 3 === 1 && S / L > 0.68;
      const ridge = giebel ? !alongX : alongX;
      this.gableRoof(lot, ridge, yTop, giebel ? 1.25 : 1.0, mix(0xffffff, 0xb09a84, rnd() * 0.7), ridge ? 'wtile' : 'wtile90', wallCol, timber, rnd, false);
      if (lot.id % 3 === 2) this.oriel(lot, sides[front], yG, floors, fh, wallCol, frameColor);
      this.plinthGround(lot, sides, yG, y, doors, rnd);
    } else if (st.roof === 'gable') {
      // Ziegel als Foto, die Farbe aus dem Spiel nur als Tönung
      const photo = st.roofMat === 'tile';
      this.gableRoof(lot, alongX, yTop, st.pitch, photo ? mix(0xffffff, rc, 0.55) : rc, photo ? (alongX ? 'wtile' : 'wtile90') : st.roofMat, wallCol, timber, rnd, st.stepped && rnd() < 0.7);
    }
    else if (st.roof === 'hip') this.hipRoof(lot, alongX, yTop, st.pitch, rc);
    else this.flatRoof(lot, yTop, rnd, roofAccess ? rectUV(uS0 - 0.05, L - T, (floors - 1) % 2 === 0 ? T : S - T - sw - 0.05, (floors - 1) % 2 === 0 ? T + sw + 0.05 : S - T) : null);

    // Bodenanschluss und Ladenschilder
    if (!lot.pilot) this.plinthGround(lot, sides, yG, y, doors, rnd);
    if (lot.shop && doors[0]) this.shopFront(lot, sides[doors[0].side], doors[0].a, yG, rnd);
    if (lot.style === 'hochhaus' && floors >= 5 && lot.id % 2 === 0) this.fireEscape(lot, sides, doors, yG, floors, fh);
    this.stats.houses++;
    lot.roofY = yTop;
    lot.flat = st.roof === 'flat';
    return lot;
  }

  // Fachwerk: Schwelle, Rähm, Ständer und Streben auf offenen Abschnitten
  timberFrame(w, y, h, ops, runs, color, rnd) {
    const bw = 0.17;
    for (const [r0, r1] of runs) {
      this.beam(w, [r0, y + bw / 2], [r1, y + bw / 2], bw, color);
      this.beam(w, [r0, y + h - bw / 2], [r1, y + h - bw / 2], bw, color);
      // Ständer: Abschnittsenden und Öffnungskanten
      const posts = [r0 + bw / 2, r1 - bw / 2];
      for (const o of ops) {
        if (o.a1 < r0 || o.a0 > r1) continue;
        posts.push(o.a0 - bw / 2 - 0.02, o.a1 + bw / 2 + 0.02);
      }
      posts.sort((a, b) => a - b);
      const ps = [];
      for (const p of posts) if (!ps.length || p - ps[ps.length - 1] > 0.3) ps.push(p);
      for (const p of ps) this.beam(w, [p, y], [p, y + h], bw, color);
      // Felder zwischen den Ständern
      for (let i = 0; i + 1 < ps.length; i++) {
        const a = ps[i] + bw / 2, b = ps[i + 1] - bw / 2;
        if (b - a < 0.5) continue;
        const hasOp = ops.find((o) => o.a0 >= a - 0.05 && o.a1 <= b + 0.05);
        if (hasOp) {
          // Brüstung unter dem Fenster: Andreaskreuz oder Riegel
          const yb = y + hasOp.y0;
          if (hasOp.y0 > 0.5) {
            if (rnd() < 0.5) {
              this.beam(w, [a, y + bw], [b, yb], bw * 0.8, color);
              this.beam(w, [b, y + bw], [a, yb], bw * 0.8, color);
            } else this.beam(w, [a, yb - bw / 2], [b, yb - bw / 2], bw * 0.8, color);
          }
          continue;
        }
        // Riegel in halber Höhe und Strebe
        this.beam(w, [a, y + h * 0.5], [b, y + h * 0.5], bw * 0.85, color);
        if (b - a > 0.8) {
          if (rnd() < 0.5) this.beam(w, [a, y + bw], [b, y + h * 0.5], bw * 0.85, color);
          else this.beam(w, [b, y + bw], [a, y + h * 0.5], bw * 0.85, color);
        }
      }
    }
  }

  // Bodenplatte (Ober- und Unterseite, Lochränder), ground = Erdgeschossboden
  slab(r, yb, yt, hole, color, inner, ground = false, roof = false) {
    const parts = [];
    if (!hole) parts.push([r.x0, r.x1, r.z0, r.z1]);
    else {
      const h = { x0: Math.max(r.x0, hole.x0), x1: Math.min(r.x1, hole.x1), z0: Math.max(r.z0, hole.z0), z1: Math.min(r.z1, hole.z1) };
      parts.push([r.x0, r.x1, r.z0, h.z0], [r.x0, r.x1, h.z1, r.z1], [r.x0, h.x0, h.z0, h.z1], [h.x1, r.x1, h.z0, h.z1]);
      // Lochränder
      const e = [[h.x0, h.z0, h.x1, h.z0, [0, 0, 1]], [h.x1, h.z1, h.x0, h.z1, [0, 0, -1]], [h.x0, h.z1, h.x0, h.z0, [1, 0, 0]], [h.x1, h.z0, h.x1, h.z1, [-1, 0, 0]]];
      for (const [ax, az, bx, bz, n] of e) this.quad([[ax, yb, az], [bx, yb, bz], [bx, yt, bz], [ax, yt, az]], n, color, { inner, layer: 'in' });
    }
    for (const [a, b, c, d] of parts) {
      if (b - a < 0.02 || d - c < 0.02) continue;
      this.col.addBox(a, yb, c, b, yt, d);
      this.quad([[a, yt, c], [b, yt, c], [b, yt, d], [a, yt, d]], [0, 1, 0], color, { inner: roof ? null : inner, mat: roof ? 'world' : 'wood' });
      if (!ground) this.quad([[a, yb, c], [b, yb, c], [b, yb, d], [a, yb, d]], [0, -1, 0], 0xcfc6b2, { inner });
    }
  }

  // Treppe in (u,v): du = +1 steigt Richtung Rückseite, -1 Richtung Vorderseite
  stairsUV(toX, toZ, u0, u1, v0, v1, y0, y1, du, sideSign, inner, color) {
    const n = Math.round((y1 - y0) / 0.3);
    const hh = (y1 - y0) / n;
    for (let i = 0; i < n; i++) {
      const top = y0 + hh * (i + 1);
      const ua = du > 0 ? u0 + STEP_D * i : u1 - STEP_D * (i + 1);
      const ub = ua + STEP_D;
      const xa = toX(ua, v0), xb = toX(ub, v1), za = toZ(ua, v0), zb = toZ(ub, v1);
      const bx0 = Math.min(xa, xb), bx1 = Math.max(xa, xb), bz0 = Math.min(za, zb), bz1 = Math.max(za, zb);
      this.col.addBox(bx0, y0, bz0, bx1, top, bz1);
      const opts = { inner, layer: 'in' };
      // Trittfläche
      this.quad([[bx0, top, bz0], [bx1, top, bz0], [bx1, top, bz1], [bx0, top, bz1]], [0, 1, 0], color, opts);
      // Setzstufe (zur Laufrichtung hin)
      const uRise = du > 0 ? ua : ub;
      const rx0 = toX(uRise, v0), rx1 = toX(uRise, v1), rz0 = toZ(uRise, v0), rz1 = toZ(uRise, v1);
      const ux = toX(uRise - du, v0) - rx0, uz = toZ(uRise - du, v0) - rz0;
      this.quad([[rx0, top - hh, rz0], [rx1, top - hh, rz1], [rx1, top, rz1], [rx0, top, rz0]], [Math.sign(ux), 0, Math.sign(uz)], color, opts);
      // offene Seite zum Flur
      const vOpen = sideSign > 0 ? v1 : v0;
      const sx0 = toX(ua, vOpen), sx1 = toX(ub, vOpen), sz0 = toZ(ua, vOpen), sz1 = toZ(ub, vOpen);
      const vx = toX(ua, vOpen + (sideSign > 0 ? 1 : -1)) - sx0, vz = toZ(ua, vOpen + (sideSign > 0 ? 1 : -1)) - sz0;
      this.quad([[sx0, y0, sz0], [sx1, y0, sz1], [sx1, top, sz1], [sx0, top, sz0]], [Math.sign(vx), 0, Math.sign(vz)], darker(color, 0.85), opts);
    }
  }

  partition(rectUV, u, S, fy, fh, color, inner, mat = 'world') {
    const t = 0.14, dw = 1.0, dh = 2.15;
    const vm = S / 2;
    const segs = [[T, vm - dw / 2, 0, fh], [vm + dw / 2, S - T, 0, fh], [vm - dw / 2, vm + dw / 2, dh, fh]];
    for (const [v0, v1, h0, h1] of segs) {
      const r = rectUV(u - t / 2, u + t / 2, v0, v1);
      this.b.box((r.x0 + r.x1) / 2, fy + h0, (r.z0 + r.z1) / 2, r.x1 - r.x0, h1 - h0, r.z1 - r.z0, color, { inner, layer: 'in', mat });
      this.col.addBox(r.x0, fy + h0, r.z0, r.x1, fy + h1, r.z1);
    }
  }

  furnish(lot, rnd, rectUV, toX, toZ, uEnd, S, fy, f, sw, doors, aOfU, aOfV, sideB, front, inner, backWall = false) {
    const roomL = uEnd - T;
    if (roomL < 2.2) return;
    const wood = pick(rnd, PAL.wood);
    const opts = { inner, layer: 'in' };
    const nearDoor = (u, v) => doors.some((d) => {
      if (f !== 0) return false;
      if (d.side === front) return u < T + 1.3 && Math.abs(aOfV(v) - d.a) < 1.1;
      if (d.side === sideB) return v > S - T - 1.4 && Math.abs(aOfU(u) - d.a) < 1.1;
      return v < T + 1.4 && Math.abs(aOfU(u) - d.a) < 1.1;
    });
    // offener Türflügel an der Innenwand (Vorderseite oder Seite A) im Weg?
    const leafAt = (u, v) => f === 0 && (lot.leaves || []).some((l) => (l.side === front && u < T + 1.3 && aOfV(v) > l.a0 && aOfV(v) < l.a1)
      || (l.side === lot.sideA && v < T + 0.7 && aOfU(u) > l.a0 && aOfU(u) < l.a1));
    const du = [toX(1, 0) - toX(0, 0), toZ(1, 0) - toZ(0, 0)];
    const dv = [toX(0, 1) - toX(0, 0), toZ(0, 1) - toZ(0, 0)];
    const at = (u, v) => [toX(u, v), fy, toZ(u, v)];
    const mopts = { layer: 'in', shade: 0.62 };
    // Stube: Eckbank mit Tisch in der vorderen Ecke an Seite A
    let eck = false;
    const rustic = lot.style === 'fachwerk' || lot.style === 'schwarzwald';
    if (S >= 4.6 && roomL >= 3.8 && this.inst && this.inst.has('kc_eckbank') && rnd() < (rustic ? 0.6 : 0.3)
      && [[T + 0.3, T + 0.3], [T + 0.3, T + 1.9], [T + 2.0, T + 0.3], [T + 1.3, T + 1.0], [T + 1.3, T + 1.8]].every(([u, v]) => !nearDoor(u, v))
      && [0.1, 0.6, 1.1, 1.6, 2.0].every((k) => !leafAt(T + 0.3, T + k) && !leafAt(T + k, T + 0.3))) {
      // Modellarme liegen auf lokal +X und +Z; Drehung je nach Händigkeit von (u, v)
      const ax = du[0] * dv[1] - du[1] * dv[0] > 0 ? du : dv;
      this.model('kc_eckbank', at(T, T), yawX(ax[0], ax[1]), 1, { ...mopts, tint: pick(rnd, PAL.awning) });
      for (const r of [rectUV(T, T + 0.5, T, T + 1.9), rectUV(T, T + 1.9, T, T + 0.5)]) this.col.addBox(r.x0, fy, r.z0, r.x1, fy + 0.48, r.z1);
      const u = T + 1.3, v = T + 1.0;
      this.model('kc_table', at(u, v), yawX(du[0], du[1]), 1, mopts);
      const r = rectUV(u - 0.7, u + 0.7, v - 0.4, v + 0.4);
      this.col.addBox(r.x0, fy, r.z0, r.x1, fy + 0.76, r.z1);
      if (rnd() < 0.8) this.model('kc_chair', at(u + (rnd() - 0.5) * 0.4, v + 0.68), yawFront(-dv[0], -dv[1]) + (rnd() - 0.5) * 0.3, 1, mopts);
      eck = true;
      (lot.decor ||= []).push({ kind: 'eckbank', x: toX(T + 1.3, T + 1.0), y: fy, z: toZ(T + 1.3, T + 1.0) });
    }
    // sonst Tisch mit Stühlen an Seite A
    if (!eck && S >= 4.6 && roomL >= 3.0) {
      const u = Math.min(T + roomL * 0.5, uEnd - 1.9), v = T + 0.75;
      if (u > T + 0.9 && !nearDoor(u, v)) {
        const r = rectUV(u - 0.7, u + 0.7, v - 0.4, v + 0.4);
        if (this.model('kc_table', at(u, v), yawX(du[0], du[1]), 1, mopts)) {
          // Ein Teil der Stein- und Backsteinhäuser hat Sprossenstühle, die Bauernhäuser Brettstühle
          const chair = !rustic && lot.id % 3 === 0 && this.inst.has('sr_chair') ? 'sr_chair' : 'kc_chair';
          for (const k of [-0.35, 0.35]) if (rnd() < 0.8) this.model(chair, at(u + k, v + 0.62), yawFront(-dv[0], -dv[1]) + (rnd() - 0.5) * 0.4, 1, mopts);
        } else {
          this.b.box((r.x0 + r.x1) / 2, fy + 0.7, (r.z0 + r.z1) / 2, r.x1 - r.x0, 0.06, r.z1 - r.z0, wood, opts);
          this.b.box((r.x0 + r.x1) / 2, fy, (r.z0 + r.z1) / 2, (r.x1 - r.x0) * 0.8, 0.7, (r.z1 - r.z0) * 0.7, darker(wood, 0.8), opts);
        }
        this.col.addBox(r.x0, fy, r.z0, r.x1, fy + 0.76, r.z1);
      }
    }
    // Bett im Obergeschoss
    if (f > 0 && roomL >= 3.4 && rnd() < 0.6) {
      const u = T + 1.15, v = S - T - 0.55;
      const r = rectUV(u - 1.0, u + 1.0, v - 0.48, v + 0.48);
      if (!this.model('kc_bed', at(u, v), yawFront(du[0], du[1]), 1, { ...mopts, tint: pick(rnd, PAL.awning) })) {
        this.b.box((r.x0 + r.x1) / 2, fy, (r.z0 + r.z1) / 2, r.x1 - r.x0, 0.45, r.z1 - r.z0, 0xb9ae98, opts);
      }
      this.col.addBox(r.x0, fy, r.z0, r.x1, fy + 0.6, r.z1);
      // Fußbank am Bettende
      if (lot.id % 4 === 0 && u + 1.3 < uEnd - 1.4 && !nearDoor(u + 1.3, v)) this.model('sr_stool', at(u + 1.3, v), yawFront(du[0], du[1]), 1.3, mopts);
    } else {
      // Beuteplatz an Seite B vorn
      const u = T + 0.7, v = S - T - 0.6;
      if (!nearDoor(u, v)) this.B.spots.push({ x: toX(u, v), y: fy, z: toZ(u, v), tier: lot.tier, poi: lot.poi, indoor: true });
    }
    // zweiter Beuteplatz hinten im Zimmer
    if (roomL >= 3.6) {
      const u = uEnd - 0.75, v = S - T - 0.6;
      if (!nearDoor(u, v)) this.B.spots.push({ x: toX(u, v), y: fy, z: toZ(u, v), tier: lot.tier, poi: lot.poi, indoor: true });
    }
    // Teppich in der Raummitte
    if (!eck && rnd() < 0.55 && roomL >= 3.0 && S >= 4.2) {
      const hu = Math.min(1.3, roomL / 2 - 0.6), hv = Math.min(0.85, S / 2 - T - 0.5);
      const uc = T + roomL / 2, vc = S / 2, rc = pick(rnd, PAL.rug);
      const flat = (u0, u1, v0, v1, yy, c) => {
        const r = rectUV(u0, u1, v0, v1);
        this.quad([[r.x0, yy, r.z0], [r.x1, yy, r.z0], [r.x1, yy, r.z1], [r.x0, yy, r.z1]], [0, 1, 0], c, { inner, layer: 'in', mat: 'cloth' });
      };
      flat(uc - hu, uc + hu, vc - hv, vc + hv, fy + 0.012, darker(rc, 0.65));
      flat(uc - hu + 0.13, uc + hu - 0.13, vc - hv + 0.13, vc + hv - 0.13, fy + 0.018, rc);
    }
    // Kuckucksuhr an der Zwischenwand der Stube
    if (backWall && f === 0 && rnd() < (lot.style === 'schwarzwald' ? 0.75 : lot.style === 'fachwerk' ? 0.35 : 0.1)) {
      const v = S - T - 0.45;
      this.model('kc_cuckoo', [toX(uEnd + 0.01, v), fy + 1.75, toZ(uEnd + 0.01, v)], yawFront(-du[0], -du[1]), 1, mopts);
    }
    // Regal oder Kachelofen an der Vorderwand (Seite A)
    if (!eck && rnd() < 0.5 && S >= 5) {
      const stove = rnd() < 0.35;
      const dep = stove ? 0.42 : 0.21, v = T + (stove ? 1.0 : 0.85);
      const u = T + dep;
      if (!nearDoor(u + 0.4, v)) {
        const name = stove ? 'kc_stove' : 'kc_shelf';
        const r = stove ? rectUV(T, T + 1.5, v - 0.55, v + 0.55) : rectUV(T, T + 0.42, v - 0.6, v + 0.6);
        if (!this.model(name, at(u, v), yawFront(du[0], du[1]), 1, mopts)) {
          this.b.box((r.x0 + r.x1) / 2, fy, (r.z0 + r.z1) / 2, r.x1 - r.x0, 1.9, r.z1 - r.z0, wood, opts);
        }
        this.col.addBox(r.x0, fy, r.z0, r.x1, fy + (stove ? 1.6 : 1.9), r.z1);
      }
    }
  }

  balcony(w, fy, runs, color) {
    const r = runs.reduce((a, b) => (b[1] - b[0] > a[1] - a[0] ? b : a), runs[0]);
    if (!r || r[1] - r[0] < 3) return;
    const a0 = r[0] + 0.3, a1 = r[1] - 0.3;
    const p0 = wp(w, a0, fy - 0.12, -1.0), p1 = wp(w, a1, fy, 0);
    const bx0 = Math.min(p0[0], p1[0]), bx1 = Math.max(p0[0], p1[0]), bz0 = Math.min(p0[2], p1[2]), bz1 = Math.max(p0[2], p1[2]);
    this.b.box((bx0 + bx1) / 2, fy - 0.12, (bz0 + bz1) / 2, bx1 - bx0, 0.12, bz1 - bz0, color);
    // Brüstung (nur Optik, Bretter)
    const out = OUTN[w.side];
    this.quad([wp(w, a0, fy, -1.0), wp(w, a1, fy, -1.0), wp(w, a1, fy + 0.95, -1.0), wp(w, a0, fy + 0.95, -1.0)], out, color, { double: true });
  }

  gableRoof(lot, alongX, ye, pitch, color, mat, wallCol, timber, rnd, stepped) {
    const { x0, x1, z0, z1 } = lot;
    // lokale Koordinaten: p entlang First, q quer
    const P0 = alongX ? x0 : z0, P1 = alongX ? x1 : z1, Q0 = alongX ? z0 : x0, Q1 = alongX ? z1 : x1;
    const M = (p, yy, q) => (alongX ? [p, yy, q] : [q, yy, p]);
    const N = (np, ny, nq) => (alongX ? [np, ny, nq] : [nq, ny, np]);
    const hs = (Q1 - Q0) / 2, qc = (Q0 + Q1) / 2;
    const rise = hs * pitch;
    const yr = ye + rise;
    const e = 0.72, g = stepped ? -0.05 : 0.45;
    const yE = ye - e * pitch;
    const nl = Math.hypot(1, pitch);
    // Dachflächen
    this.quad([M(P0 - g, yE, Q0 - e), M(P1 + g, yE, Q0 - e), M(P1 + g, yr, qc), M(P0 - g, yr, qc)], N(0, 1 / nl, -pitch / nl), color, { mat });
    this.quad([M(P1 + g, yE, Q1 + e), M(P0 - g, yE, Q1 + e), M(P0 - g, yr, qc), M(P1 + g, yr, qc)], N(0, 1 / nl, pitch / nl), color, { mat });
    // Untersicht des Dachüberstands
    const under = darker(PAL.wood[1], 0.9);
    this.quad([M(P0 - g, yE, Q0 - e), M(P1 + g, yE, Q0 - e), M(P1 + g, ye, Q0), M(P0 - g, ye, Q0)], N(0, -1, 0), under);
    this.quad([M(P0 - g, yE, Q1 + e), M(P1 + g, yE, Q1 + e), M(P1 + g, ye, Q1), M(P0 - g, ye, Q1)], N(0, -1, 0), under);
    // Traufkante
    this.quad([M(P0 - g, yE - 0.14, Q0 - e), M(P1 + g, yE - 0.14, Q0 - e), M(P1 + g, yE, Q0 - e), M(P0 - g, yE, Q0 - e)], N(0, 0, -1), darker(color, 0.7));
    this.quad([M(P0 - g, yE - 0.14, Q1 + e), M(P1 + g, yE - 0.14, Q1 + e), M(P1 + g, yE, Q1 + e), M(P0 - g, yE, Q1 + e)], N(0, 0, 1), darker(color, 0.7));
    this.eaves(lot, alongX, M, P0, P1, Q0, Q1, e, g, yE, rnd);
    // Sparrenköpfe unter dem Überstand
    {
      for (const [qw, sq] of [[Q0, -1], [Q1, 1]]) {
        const out = alongX ? [0, sq] : [sq, 0];
        for (let p = P0 + 0.35; p < P1 - 0.2; p += 0.85) this.model('kc_beamhead', M(p, ye - 0.3, qw), yawFront(out[0], out[1]), [1, 1, 1.5], { tint: PAL.wood[1], layer: 'det' });
      }
    }
    // Ortgangbretter an den Giebelkanten
    if (!stepped) {
      const vb = lot.style === 'fachwerk' ? timber : lot.style === 'stein' ? 0x5d6266 : darker(pick(rnd, PAL.wood), 0.9);
      const vm = lot.style === 'stein' ? 'metal' : 'wood';
      for (const [p, s] of [[P0 - g - 0.02, -1], [P1 + g + 0.02, 1]]) {
        for (const qa of [Q0 - e, Q1 + e]) {
          this.quad([M(p, yE - 0.24, qa), M(p, yr - 0.24, qc), M(p, yr + 0.06, qc), M(p, yE + 0.06, qa)], N(s, 0, 0), vb, { mat: vm, layer: 'det' });
        }
      }
    }
    // Firstziegel
    const rp = M((P0 + P1) / 2, yr - 0.05, qc);
    this.b.box(rp[0], rp[1], rp[2], alongX ? P1 - P0 + 2 * g + 0.1 : 0.32, 0.22, alongX ? 0.32 : P1 - P0 + 2 * g + 0.1, darker(color, 0.75), { mat });
    // Giebel
    for (const [p, s] of [[P0, -1], [P1, 1]]) {
      if (stepped) {
        // Treppengiebel aus Backstein
        const nStep = 5;
        for (let k = 0; k < nStep; k++) {
          const q0 = Q0 + (hs * k) / nStep, q1 = Q1 - (hs * k) / nStep;
          const yy0 = ye + (rise * k) / nStep, yy1 = ye + (rise * (k + 1)) / nStep + 0.45;
          const c = M(p - (s * T) / 2, 0, (q0 + q1) / 2);
          this.b.box(c[0], yy0, c[2], alongX ? T : q1 - q0, yy1 - yy0, alongX ? q1 - q0 : T, wallCol, { mat: 'brick' });
        }
      } else {
        this.tri([M(p, ye, Q0), M(p, ye, Q1), M(p, yr, qc)], N(s, 0, 0), wallCol, { mat: lot.style === 'backstein' || lot.style === 'speicher' ? 'brick' : 'wplaster' });
        if (lot.style === 'fachwerk') {
          const ww = { side: alongX ? (s > 0 ? 'e' : 'w') : (s > 0 ? 's' : 'n'), fixed: p };
          this.beam(ww, [Q0 + 0.1, ye + 0.08], [Q1 - 0.1, ye + 0.08], 0.17, timber);
          this.beam(ww, [qc, ye], [qc, yr - 0.15], 0.17, timber);
          this.beam(ww, [Q0 + hs * 0.45, ye + rise * 0.45], [Q1 - hs * 0.45, ye + rise * 0.45], 0.15, timber);
          this.beam(ww, [Q0 + 0.2, ye + 0.1], [qc, ye + rise * 0.45], 0.14, timber);
          this.beam(ww, [Q1 - 0.2, ye + 0.1], [qc, ye + rise * 0.45], 0.14, timber);
        }
        // Dachbodenfenster (dunkel)
        if (rise > 2.4) {
          const ww = { side: alongX ? (s > 0 ? 'e' : 'w') : (s > 0 ? 's' : 'n'), fixed: p };
          const wy = ye + rise * 0.18;
          this.quad([wp(ww, qc - 0.35, wy, -0.035), wp(ww, qc + 0.35, wy, -0.035), wp(ww, qc + 0.35, wy + 0.8, -0.035), wp(ww, qc - 0.35, wy + 0.8, -0.035)], OUTN[ww.side], 0x2a3034, { shade: false });
          {
            const o = OUTN[ww.side];
            this.model('kc_window', wp(ww, qc, wy, -0.05), yawFront(o[0], o[2]), [0.7, 0.8, 1], { tint: PAL.frameLight, layer: 'det', double: true });
            this.ledge(ww, qc - 0.46, qc + 0.46, wy - 0.1, wy - 0.02, 0.1, 0xb9b09c, 'stone', 'det');
          }
        }
      }
    }
    // Schornstein
    if (rnd() < 0.7) {
      const cp = P0 + (P1 - P0) * (0.25 + rnd() * 0.5);
      const c = M(cp, 0, qc + hs * 0.3);
      const ch = rise * 0.35 + 1.0;
      if (!this.model('kc_chimney', [c[0], yr - rise * 0.35, c[2]], 0, [1, ch / 2.0, 1])) {
        this.b.box(c[0], yr - rise * 0.35, c[2], 0.6, ch, 0.6, pick(rnd, PAL.brick), { mat: 'brick' });
      }
    }
    // Gauben auf den Traufseiten
    if (this.inst && !stepped && P1 - P0 >= 7.5 && rise >= 3) {
      for (const side of [-1, 1]) {
        if (rnd() < 0.4) continue;
        const n = P1 - P0 >= 12 ? 2 : 1;
        const q = side < 0 ? Q0 + hs * 0.42 : Q1 - hs * 0.42;
        const yq = ye + hs * 0.42 * pitch;
        const out = alongX ? [0, side] : [side, 0];
        for (let i = 0; i < n; i++) {
          const p = P0 + ((P1 - P0) / (n + 1)) * (i + 1);
          this.model('kc_dormer', M(p, yq - 0.35, q), yawFront(out[0], out[1]), 0.9, { layer: 'det' });
        }
      }
    }
    // Kollision (Dachvolumen grob)
    const c1 = [M(P0, ye, Q0 + hs * 0.2), M(P1, ye + rise * 0.55, Q1 - hs * 0.2)];
    const c2 = [M(P0, ye + rise * 0.55, qc - hs * 0.35), M(P1, ye + rise * 0.9, qc + hs * 0.35)];
    for (const [a, b] of [c1, c2]) this.col.addBox(Math.min(a[0], b[0]), a[1], Math.min(a[2], b[2]), Math.max(a[0], b[0]), b[1], Math.max(a[2], b[2]));
  }

  // Musterabschnitt: Gurtgesims an der Geschossgrenze, Traufgesims oben, Lisenen an den Ecken
  pilotBands(lot, w, runs, f, isTop, y, h, c) {
    const fc = c.fasche;
    for (const [r0, r1] of runs) {
      if (f > 0) this.ledge(w, r0, r1, y - 0.03, y + 0.14, 0.07, fc, 'wplaster');
      if (isTop) {
        this.ledge(w, r0, r1, y + h - 0.34, y + h - 0.2, 0.1, fc, 'wplaster');
        this.ledge(w, r0, r1, y + h - 0.2, y + h, 0.2, fc, 'wplaster');
      }
      if (lot.id % 3 === 1) {
        for (const [edge, dir] of [[w.a0, 1], [w.a1, -1]]) {
          if (Math.abs((dir > 0 ? r0 : r1) - edge) > 0.3) continue;
          const a0 = dir > 0 ? edge : edge - 0.42, a1 = dir > 0 ? edge + 0.42 : edge;
          this.ledge(w, a0, a1, f === 0 ? y + 0.85 : y, y + h, 0.05, fc, 'wplaster');
        }
      }
    }
  }

  // Erker über dem Erdgeschoss
  oriel(lot, w, yG, floors, fh, color, frameCol) {
    if (floors < 2) return;
    const runs = this.openRuns(w, yG + fh, lot.id);
    const best = runs.slice().sort((p, q) => q[1] - q[0] - (p[1] - p[0]))[0];
    if (!best || best[1] - best[0] < 4) return;
    lot.orielAt = wp(w, (best[0] + best[1]) / 2, yG, -8);
    const m = (best[0] + best[1]) / 2, hw = 1.25, D = 0.62, y0 = yG + fh - 0.1, y1 = yG + floors * fh - 0.45;
    const o = OUTN[w.side], ax = AXIS[w.side];
    const mat = { mat: 'wplaster' };
    // Vorderwand mit echten Öffnungen, davor liegt nur die dunkle Rückwand
    const front = { side: w.side, fixed: w.side === 's' || w.side === 'e' ? w.fixed + D : w.fixed - D, a0: m - hw, a1: m + hw, ia0: m - hw, ia1: m + hw };
    const ops = [];
    for (let f = 1; f < floors; f++) ops.push({ a0: m - 0.72, a1: m + 0.72, y0: yG + f * fh + 0.85 - y0, y1: yG + f * fh + 2.4 - y0, frame: true, kind: 'win' });
    this.quad([wp(w, m - hw, y0, -0.02), wp(w, m + hw, y0, -0.02), wp(w, m + hw, y1, -0.02), wp(w, m - hw, y1, -0.02)], o, 0x1c2022, { shade: false });
    this.wall(front, y0, y1 - y0, ops, color, { mat: 'wplaster', frameCol, inset: 0.13 });
    for (const op of ops) this.ledge(front, op.a0 - 0.12, op.a1 + 0.12, y0 + op.y0 - 0.12, y0 + op.y0 - 0.03, 0.13, 0xb9b09c, 'stone', 'det');
    this.quad([wp(w, m - hw, y0, 0), wp(w, m - hw, y0, -D), wp(w, m - hw, y1, -D), wp(w, m - hw, y1, 0)], neg(ax), color, mat);
    this.quad([wp(w, m + hw, y0, 0), wp(w, m + hw, y0, -D), wp(w, m + hw, y1, -D), wp(w, m + hw, y1, 0)], ax, color, mat);
    // Konsole und Dach
    this.quad([wp(w, m - hw, y0, -D), wp(w, m + hw, y0, -D), wp(w, m + hw, y0 - 0.5, 0), wp(w, m - hw, y0 - 0.5, 0)], [o[0], -1, o[2]], darker(color, 0.85), mat);
    const roofMat = { mat: w.side === 'n' || w.side === 's' ? 'wtile' : 'wtile90' };
    this.quad([wp(w, m - hw - 0.12, y1, -D - 0.14), wp(w, m + hw + 0.12, y1, -D - 0.14), wp(w, m + hw, y1 + 0.55, 0), wp(w, m - hw, y1 + 0.55, 0)], [o[0], 1, o[2]], 0xd8c8b4, roofMat);
    for (const s of [-1, 1]) this.tri([wp(w, m + s * (hw + 0.12), y1, -D - 0.14), wp(w, m + s * hw, y1, 0), wp(w, m + s * hw, y1 + 0.55, 0)], [ax[0] * s, 0.6, ax[2] * s], 0xd8c8b4, roofMat);
  }

  // Bodenanschluss: Steinstreifen am Sockel, Pflanzen an geschützten Stellen, Eingänge bleiben frei
  plinthGround(lot, sides, yG, y, doors, rnd) {
    const stone = { mat: 'stone', layer: 'det', shade: false };
    for (const k of ['n', 's', 'e', 'w']) {
      const w = sides[k];
      for (const [r0, r1] of this.openRuns(w, yG, lot.id)) {
        if (r1 - r0 < 1) continue;
        this.quad([wp(w, r0, y + 0.03, -0.46), wp(w, r1, y + 0.03, -0.46), wp(w, r1, y + 0.03, 0), wp(w, r0, y + 0.03, 0)], [0, 1, 0], 0x7a7468, stone);
        // Rinnstein vor dem Steinstreifen
        this.quad([wp(w, r0, y + 0.018, -0.68), wp(w, r1, y + 0.018, -0.68), wp(w, r1, y + 0.018, -0.46), wp(w, r0, y + 0.018, -0.46)], [0, 1, 0], 0x68635a, stone);
        const near = (a) => doors.some((d) => d.side === k && Math.abs(d.a - a) < 1.3);
        const n = Math.floor((r1 - r0) / 3.5);
        for (let i = 0; i < n; i++) {
          if (rnd() > (lot.pilot ? 0.55 : 0.28)) continue;
          const a = r0 + 0.6 + rnd() * (r1 - r0 - 1.2);
          if (near(a)) continue;
          const p = wp(w, a, y + 0.02, -0.34);
          this.model(rnd() < 0.5 ? 'v_weed' : 'v_flowers', p, rnd() * 6, 0.45 + rnd() * 0.4, { layer: 'det', double: true });
        }
      }
    }
  }

  // Regenrinnen an beiden Traufen, Fallrohr an einer freien Hausecke ohne Tür
  eaves(lot, alongX, M, P0, P1, Q0, Q1, e, g, yE, rnd) {
    const zc = pick(rnd, PAL.zinc);
    const det = { mat: 'metal', layer: 'det' };
    const L = P1 - P0 + 2 * g;
    for (const [qw, sq] of [[Q0, -1], [Q1, 1]]) {
      const qg = qw + sq * (e + 0.02);
      const gc = M((P0 + P1) / 2, 0, qg);
      this.b.box(gc[0], yE - 0.27, gc[2], alongX ? L : 0.15, 0.13, alongX ? 0.15 : L, zc, det);
      // Fallrohre nur an Wohnhäusern (Sonderbauten wie der Dom haben kein lot.level)
      if (rnd() < 0.2 || lot.level == null) continue;
      const side = alongX ? (sq < 0 ? 'n' : 's') : (sq < 0 ? 'w' : 'e');
      const pA = rnd() < 0.5 ? P0 + 0.3 : P1 - 0.3;
      const probe = M(pA, 0, qw + sq * 0.7);
      if ((lot.doors || []).some((d) => d.side === side && Math.abs(d.a - pA) < 1.1)) continue;
      if (!this.openAt(probe[0], probe[2], lot.level + 0.15, lot.id)) continue;
      const qp = qw + sq * 0.2, y0 = lot.level + 0.12, y1 = yE - 0.42;
      const pc = M(pA, 0, qp);
      this.b.addPrim('cyl8', pc[0], (y0 + y1) / 2, pc[2], 0.055, y1 - y0, 0.055, zc, det);
      // Bogen zur Rinne und kurzer Stutzen
      const bc = M(pA, 0, (qp + qg) / 2), dq = Math.abs(qg - qp) + 0.1;
      this.b.box(bc[0], y1 - 0.05, bc[2], alongX ? 0.1 : dq, 0.1, alongX ? dq : 0.1, zc, det);
      const sc = M(pA, 0, qg);
      this.b.box(sc[0], y1 - 0.05, sc[2], 0.1, yE - 0.22 - y1, 0.1, zc, det);
      // Auslauf am Boden
      const oc = M(pA, 0, qp + sq * 0.12);
      this.b.box(oc[0], y0, oc[2], alongX ? 0.11 : 0.32, 0.09, alongX ? 0.32 : 0.11, zc, det);
    }
  }

  hipRoof(lot, alongX, ye, pitch, color) {
    const { x0, x1, z0, z1 } = lot;
    const M = (p, yy, q) => (alongX ? [p, yy, q] : [q, yy, p]);
    const N = (np, ny, nq) => (alongX ? [np, ny, nq] : [nq, ny, np]);
    const e = 1.0;
    const P0 = (alongX ? x0 : z0) - e, P1 = (alongX ? x1 : z1) + e, Q0 = (alongX ? z0 : x0) - e, Q1 = (alongX ? z1 : x1) + e;
    const hs = (Q1 - Q0) / 2, qc = (Q0 + Q1) / 2;
    const yE = ye - e * pitch;
    const yr = yE + hs * pitch;
    const inset = Math.min(hs, (P1 - P0) / 2 - 0.3);
    const R1 = M(P0 + inset, yr, qc), R2 = M(P1 - inset, yr, qc);
    const A0 = M(P0, yE, Q0), A1 = M(P1, yE, Q0), B0 = M(P0, yE, Q1), B1 = M(P1, yE, Q1);
    const nl = Math.hypot(1, pitch);
    const under = darker(PAL.wood[1], 0.85);
    const faces = [
      [[A0, A1, R2, R1], N(0, 1 / nl, -pitch / nl)],
      [[B1, B0, R1, R2], N(0, 1 / nl, pitch / nl)],
    ];
    for (const [pts, n] of faces) {
      this.quad(pts, n, color, { mat: 'wood' });
      this.quad(pts, [-n[0], -n[1], -n[2]], under);
    }
    const nlp = Math.hypot(1, (yr - yE) / inset);
    this.tri([B0, A0, R1], N(-((yr - yE) / inset) / nlp, 1 / nlp, 0), color, { mat: 'wood' });
    this.tri([B0, A0, R1], N(((yr - yE) / inset) / nlp, -1 / nlp, 0), under);
    this.tri([A1, B1, R2], N(((yr - yE) / inset) / nlp, 1 / nlp, 0), color, { mat: 'wood' });
    this.tri([A1, B1, R2], N(-((yr - yE) / inset) / nlp, -1 / nlp, 0), under);
    // Firstbalken
    const rp = M((P0 + P1) / 2, yr - 0.04, qc);
    const rl = P1 - P0 - 2 * inset + 0.3;
    if (rl > 0.3) this.b.box(rp[0], rp[1], rp[2], alongX ? rl : 0.28, 0.2, alongX ? 0.28 : rl, darker(color, 0.7), { mat: 'wood' });
    // Kollision
    const ys = ye + (hs - e) * pitch;
    const c1 = [M(P0 + e, ye, Q0 + e + 0.4), M(P1 - e, ye + (ys - ye) * 0.55, Q1 - e - 0.4)];
    const c2 = [M(P0 + e + 1, ye + (ys - ye) * 0.55, qc - (hs - e) * 0.4), M(P1 - e - 1, ye + (ys - ye) * 0.9, qc + (hs - e) * 0.4)];
    for (const [a, b] of [c1, c2]) this.col.addBox(Math.min(a[0], b[0]), a[1], Math.min(a[2], b[2]), Math.max(a[0], b[0]), b[1], Math.max(a[2], b[2]));
  }

  flatRoof(lot, ytop, rnd, hole) {
    const { x0, x1, z0, z1 } = lot;
    const pc = pick(rnd, PAL.stone);
    const ph = 0.9;
    const gaps = lot.gaps || [];
    // Brüstung, mit Lücken für Brücken
    const run = (side, a0, a1, mk) => {
      const gs = gaps.filter((g) => g.side === side).sort((p, q) => p.a0 - q.a0);
      let cur = a0;
      for (const g of gs) {
        if (g.a0 > cur) mk(cur, Math.min(g.a0, a1));
        cur = Math.max(cur, g.a1);
      }
      if (cur < a1) mk(cur, a1);
    };
    run('n', x0, x1, (a, c) => this.box((a + c) / 2, ytop, z0 + 0.15, c - a, ph, 0.3, pc));
    run('s', x0, x1, (a, c) => this.box((a + c) / 2, ytop, z1 - 0.15, c - a, ph, 0.3, pc));
    run('w', z0 + 0.3, z1 - 0.3, (a, c) => this.box(x0 + 0.15, ytop, (a + c) / 2, 0.3, ph, c - a, pc));
    run('e', z0 + 0.3, z1 - 0.3, (a, c) => this.box(x1 - 0.15, ytop, (a + c) / 2, 0.3, ph, c - a, pc));
    // Wassertank auf Stelzen (abseits der Dachluke)
    const W = x1 - x0, D = z1 - z0;
    if (lot.style === 'hochhaus' && Math.min(W, D) >= 7) this.roofClutter(lot, ytop, rnd, hole);
    if (!lot.noTank && Math.min(W, D) >= 5 && rnd() < 0.85) {
      const r = Math.min(1.7, Math.min(W, D) * 0.22);
      let tx = x0 + r + 0.7, tz = z0 + r + 0.7;
      if (hole) {
        const cand = [[x0 + r + 0.7, z0 + r + 0.7], [x1 - r - 0.7, z0 + r + 0.7], [x0 + r + 0.7, z1 - r - 0.7], [x1 - r - 0.7, z1 - r - 0.7]];
        const free = cand.find(([cx, cz]) => cx + r < hole.x0 - 0.3 || cx - r > hole.x1 + 0.3 || cz + r < hole.z0 - 0.3 || cz - r > hole.z1 + 0.3);
        if (free) [tx, tz] = free;
        else return;
      }
      const k = r / 1.4;
      if (this.model('kc_tank', [tx, ytop, tz], rnd() * 6, k)) {
        this.col.addCyl(tx, tz, r, ytop + 1.9 * k, ytop + 5.4 * k);
        this.col.addCyl(tx, tz, 0.5, ytop, ytop + 1.9 * k);
        return;
      }
      const legH = 1.6 + rnd() * 0.8, th = 2.4 + rnd() * 0.8;
      const rust = pick(rnd, [0x7a5236, 0x6b4a34, 0x5d6266, 0x86603e]);
      for (const [dx, dz] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) this.b.box(tx + dx * r * 0.6, ytop, tz + dz * r * 0.6, 0.16, legH, 0.16, 0x3a3532, { mat: 'metal' });
      this.b.addPrim('cyl16', tx, ytop + legH + th / 2, tz, r, th, r, rust, { mat: 'metal' });
      this.b.addPrim('cone8', tx, ytop + legH + th + 0.35, tz, r * 1.05, 0.7, r * 1.05, darker(rust, 0.8), { mat: 'metal' });
      this.col.addCyl(tx, tz, r, ytop + legH, ytop + legH + th + 0.7);
      for (const [dx, dz] of [[-1, -1], [1, 1]]) this.col.addCyl(tx + dx * r * 0.6, tz + dz * r * 0.6, 0.12, ytop, ytop + legH);
    }
  }

  // Ladenfront: Schildbrett, gestreifte Markisen über Tür und Schaufenstern, Auslage dahinter, Ausleger
  shopFront(lot, w, a, yG, rnd) {
    const out = OUTN[w.side];
    const col = pick(rnd, PAL.awning), yT = yG + 2.78;
    const dw = 0.8;
    const spans = [[a - dw, a + dw], ...(lot.shopWins || [])].sort((p, q) => p[0] - q[0]);
    // Überlappende Spannen zusammenfassen, damit Markisen nicht ineinander stecken
    const merged = [];
    for (const [p, q] of spans) {
      const last = merged[merged.length - 1];
      if (last && p < last[1] + 0.5) last[1] = Math.max(last[1], q);
      else merged.push([p, q]);
    }
    for (const [p, q] of merged) {
      const p0 = Math.max(w.a0 + 0.05, p - 0.2), q0 = Math.min(w.a1 - 0.05, q + 0.2);
      this.ledge(w, p0, q0, yT + 0.03, yT + 0.36, 0.05, darker(pick(rnd, PAL.wood), 1.15), 'wood', 'det');
      this.awning(w, p0, q0, yT, col, 0xe6dfcc);
    }
    for (const [p, q] of lot.shopWins || []) this.shopDisplay(w, p, q, yG, rnd);
    if (this.inst && this.inst.has('kc_sign')) this.model('kc_sign', wp(w, Math.min(a + 1.1, w.a1 - 0.3), yG + 3.3, 0), yawFront(out[0], out[2]), 1, { tint: col, layer: 'det' });
  }

  // Markise aus Stoffbahnen: leicht durchhängend, vorne ein Volant mit Zacken, seitlich Eisenarme
  awning(w, a0, a1, yT, colA, colB) {
    const opt = { mat: 'cloth', double: true, layer: 'det' };
    const n = Math.max(3, Math.round((a1 - a0) / 0.32)), s = (a1 - a0) / n;
    const dFront = -1.25, yF = yT - 0.6, dMid = -0.62, yM = yT - 0.33;
    for (let i = 0; i < n; i++) {
      const p = a0 + i * s, q = p + s, c = i % 2 ? colB : colA;
      this.b.addQuad(wp(w, p, yT, -0.02), wp(w, q, yT, -0.02), wp(w, q, yM, dMid), wp(w, p, yM, dMid), c, opt);
      this.b.addQuad(wp(w, p, yM, dMid), wp(w, q, yM, dMid), wp(w, q, yF, dFront), wp(w, p, yF, dFront), c, opt);
      this.b.addQuad(wp(w, p, yF, dFront), wp(w, q, yF, dFront), wp(w, q, yF - 0.13, dFront), wp(w, p, yF - 0.13, dFront), c, opt);
      this.b.addTri(wp(w, p, yF - 0.13, dFront), wp(w, q, yF - 0.13, dFront), wp(w, p + s / 2, yF - 0.22, dFront), c, opt);
    }
    for (const e of [a0, a1]) {
      this.b.addQuad(wp(w, e, yT - 0.42, -0.02), wp(w, e, yF + 0.0, dFront), wp(w, e, yF + 0.03, dFront), wp(w, e, yT - 0.38, -0.02), 0x2a2724, { mat: 'metal', double: true, layer: 'det' });
      this.b.addTri(wp(w, e, yT, -0.02), wp(w, e, yM, dMid), wp(w, e, yT - 0.12, -0.02), darker(colA, 0.8), opt);
    }
  }

  // Auslage hinter dem Schaufenster
  shopDisplay(w, a0, a1, yG, rnd) {
    const al = w.side === 'n' || w.side === 's', m = (a0 + a1) / 2;
    const c = wp(w, m, yG, T + 0.32);
    this.b.box(c[0], yG, c[2], al ? a1 - a0 : 0.55, 0.4, al ? 0.55 : a1 - a0, 0x6a5238, { mat: 'wood', layer: 'in' });
    const WARES = ['d_jug', 'd_ceramic', 'd_basket', 'd_bucket1', 'd_pot', 'd_lantern', 'd_crate1', 'd_watering', 'd_jerrycan'];
    const n = Math.max(2, Math.floor((a1 - a0) / 0.5));
    for (let i = 0; i < n; i++) {
      const t = (i + 0.5) / n, p = wp(w, a0 + (a1 - a0) * t, yG, T + 0.3 + (rnd() - 0.5) * 0.12);
      this.B.dress(pick(rnd, WARES), p[0], p[2], rnd() * 6, { y: yG + 0.4, layer: 'in', color: 0xd0d0d0, scale: 0.8 + rnd() * 0.2 });
    }
  }

  // Offener Türflügel im Raum: Füllungen, Bänder und Griff sind eigene Flächen
  doorLeaf(w, a0, dw, y, h, col) {
    const ang = 1.2, ca = Math.cos(ang), sa = Math.sin(ang);
    const P = (t, yy, off) => wp(w, a0 + ca * dw * t + off * sa, yy, T + 0.03 + sa * dw * t - off * ca);
    const face = (t0, t1, y0, y1, off, color, mat) => this.b.addQuad(P(t0, y0, off), P(t1, y0, off), P(t1, y1, off), P(t0, y1, off), color, { mat, double: true, layer: 'det' });
    face(0, 1, y, y + h, 0, col, 'wood');
    const dark = darker(col, 0.72);
    for (const off of [-0.012, 0.012]) {
      for (const [t0, t1] of [[0.1, 0.46], [0.54, 0.9]]) {
        face(t0, t1, y + 0.14, y + 0.86, off, dark, 'wood');
        face(t0, t1, y + 1.02, y + h - 0.14, off, dark, 'wood');
      }
      for (const yy of [y + 0.28, y + h - 0.32]) face(0, 0.24, yy, yy + 0.04, off * 1.4, 0x2a2724, 'metal');
      face(0.87, 0.93, y + 0.98, y + 1.13, off * 1.8, 0x2a2724, 'metal');
    }
  }

  // Fassade der Hochhäuser in drei verwandten Typen: Bänder, Pfeiler, Balkone
  towerFacade(lot, w, runs, ops, f, isTop, y, h, c) {
    const type = lot.id % 3, rnd = c.rnd, al = w.side === 'n' || w.side === 's', out = OUTN[w.side];
    for (const [r0, r1] of runs) {
      if (f > 0 && type !== 1) this.ledge(w, r0, r1, y - 0.04, y + 0.1, 0.06, c.fasche, 'wplaster', 'det');
      if (isTop) {
        this.ledge(w, r0, r1, y + h - 0.46, y + h - 0.3, 0.12, c.fasche, 'wplaster', 'det');
        this.ledge(w, r0, r1, y + h - 0.3, y + h, 0.26, c.fasche, 'wplaster', 'det');
      }
      if (type === 1 && f > 0) {
        const wins = ops.filter((o) => o.kind === 'win' && o.a0 >= r0 && o.a1 <= r1).sort((p, q) => p.a0 - q.a0);
        for (let i = 0; i + 1 < wins.length; i++) {
          const m = (wins[i].a1 + wins[i + 1].a0) / 2;
          this.ledge(w, m - 0.17, m + 0.17, y, y + h, 0.12, c.fasche, 'wplaster', 'det');
        }
        for (const e of [r0 + 0.18, r1 - 0.18]) this.ledge(w, e - 0.18, e + 0.18, y, y + h, 0.14, c.fasche, 'wplaster', 'det');
      }
    }
    for (const o of ops) {
      if (o.kind !== 'win' || f === 0) continue;
      const Y0 = y + o.y0, Y1 = y + o.y1, m = (o.a0 + o.a1) / 2, ow = o.a1 - o.a0;
      this.ledge(w, o.a0 - 0.06, o.a1 + 0.06, Y1 + 0.02, Y1 + 0.14, 0.03, c.fasche, 'wplaster', 'det');
      // Regenspuren unter der Fensterbank
      if (rnd() < 0.45) this.quad([wp(w, o.a0 + 0.08, Y0 - 0.2 - rnd() * 0.75, -0.003), wp(w, o.a1 - 0.08, Y0 - 0.2 - rnd() * 0.75, -0.003), wp(w, o.a1 - 0.03, Y0 - 0.1, -0.003), wp(w, o.a0 + 0.03, Y0 - 0.1, -0.003)], out, darker(c.wall, 0.72), { mat: 'wplaster', layer: 'det' });
      // Klimageräte, schräg an der Wand hängend
      if (rnd() < 0.07) {
        const p = wp(w, m, Y0 - 0.55, -0.24);
        this.b.box(p[0], p[1], p[2], al ? 0.72 : 0.44, 0.42, al ? 0.44 : 0.72, 0x8d908c, { mat: 'metal', layer: 'det' });
        const q = wp(w, m, Y0 - 0.5, -0.465);
        this.b.box(q[0], q[1], q[2], al ? 0.58 : 0.01, 0.3, al ? 0.01 : 0.58, 0x33342f, { mat: 'metal', layer: 'det' });
      }
      // Balkon mit Stahlgeländer
      if (type === 2 && rnd() < 0.38 && ow < 2) {
        const s = wp(w, m, y - 0.02, -0.55);
        this.b.box(s[0], s[1], s[2], al ? ow + 0.8 : 1.1, 0.12, al ? 1.1 : ow + 0.8, c.fasche, { mat: 'wplaster', layer: 'det' });
        const rail = wp(w, m, y + 0.98, -1.05);
        this.b.box(rail[0], rail[1], rail[2], al ? ow + 0.8 : 0.04, 0.04, al ? 0.04 : ow + 0.8, 0x34322e, { mat: 'metal', layer: 'det' });
        for (let t = -0.5; t <= 0.5 + 1e-6; t += 0.25) {
          const b = wp(w, m + t * (ow + 0.76), y + 0.1, -1.05);
          this.b.box(b[0], b[1], b[2], 0.025, 0.88, 0.025, 0x34322e, { mat: 'metal', layer: 'det' });
        }
      }
    }
  }

  // Feuertreppe auf einer Seite ohne Tür: nur Ansicht, die Etagen erreicht man durchs Haus
  fireEscape(lot, sides, doors, yG, floors, fh) {
    const iron = { mat: 'metal', layer: 'det' }, col = 0x8a6a52, thin = 0x6a584a;
    for (const k of ['n', 's', 'e', 'w']) {
      if (doors.some((d) => d.side === k)) continue;
      const w = sides[k], al = k === 'n' || k === 's';
      const run = this.openRuns(w, yG + fh, lot.id).find(([p, q]) => q - p >= 5.5);
      if (!run) continue;
      lot.escape = { side: k, a: (run[0] + run[1]) / 2 };
      const m = (run[0] + run[1]) / 2, B = (a, y, d, dx, h, dd, c = col) => {
        const p = wp(w, a, y, d);
        this.b.box(p[0], p[1], p[2], al ? dx : dd, h, al ? dd : dx, c, iron);
      };
      for (let f = 1; f < floors; f++) {
        const y = yG + f * fh;
        B(m, y - 0.05, -0.6, 4.4, 0.05, 1.05);
        B(m, y + 0.95, -1.1, 4.4, 0.04, 0.04, thin);
        for (const e of [-2.2, -1.1, 0, 1.1, 2.2]) B(m + e, y, -1.1, 0.04, 0.95, 0.04, thin);
        for (const e of [-1.9, 1.9]) B(m + e, y - 0.3, -0.25, 0.05, 0.25, 0.5, thin);
        // Treppenlauf zum nächsten Podest
        if (f < floors - 1) {
          const dir = f % 2 ? 1 : -1, a0 = m - dir * 1.8, a1 = m + dir * 1.8;
          // Zwei Wangen und einzelne Stufen, man sieht die Wand hindurch
          for (const d of [-0.15, -0.95]) this.b.addQuad(wp(w, a0, y, d), wp(w, a1, y + fh, d), wp(w, a1, y + fh + 0.14, d), wp(w, a0, y + 0.14, d), col, { ...iron, double: true });
          for (let i = 1; i < 11; i++) B(a0 + (a1 - a0) * i / 11, y + fh * i / 11 + 0.08, -0.55, 0.26, 0.03, 0.8);
          this.b.addQuad(wp(w, a0, y + 0.85, -0.95), wp(w, a1, y + fh + 0.85, -0.95), wp(w, a1, y + fh + 0.9, -0.95), wp(w, a0, y + 0.9, -0.95), thin, { ...iron, double: true });
        }
      }
      // Hochgezogene Leiter über dem Erdgeschoss
      const y1 = yG + fh;
      for (const e of [-0.2, 0.2]) B(m + 1.9 + e, y1 - 1.1, -0.9, 0.035, 1.1, 0.035, thin);
      for (let i = 0; i < 4; i++) B(m + 1.9, y1 - 1 + i * 0.26, -0.9, 0.42, 0.03, 0.03, thin);
      return;
    }
  }

  // Dachaufbauten: Lüfter, Rohre, Antennen, abseits von Luke und Geschütz in der Mitte
  roofClutter(lot, ytop, rnd, hole) {
    const { x0, x1, z0, z1 } = lot, c = pick(rnd, [0x7a7c78, 0x6c6e6a, 0x85827a]);
    const spots = [[x1 - 1.7, z1 - 1.7], [x0 + 1.7, z1 - 1.7], [x1 - 1.7, z0 + 1.7]];
    const free = (x, z) => !hole || x + 1 < hole.x0 || x - 1 > hole.x1 || z + 1 < hole.z0 || z - 1 > hole.z1;
    for (const [x, z] of spots) {
      if (!free(x, z)) continue;
      const r = rnd();
      if (r < 0.4) {
        this.box(x, ytop, z, 1.3, 0.8, 1, c, { mat: 'metal' });
        this.b.addPrim('cyl16', x, ytop + 0.86, z, 0.34, 0.12, 0.34, 0x3c3d3a, { mat: 'metal', layer: 'det' });
      } else if (r < 0.7) {
        this.box(x, ytop, z, 0.7, 1.1, 0.7, darker(c, 0.85), { mat: 'metal' });
        this.b.box(x, ytop + 1.1, z, 0.9, 0.08, 0.9, c, { mat: 'metal', layer: 'det' });
      } else {
        this.b.addPrim('cyl8', x, ytop + 2.4, z, 0.03, 4.8, 0.03, 0x34322e, { mat: 'metal' });
        for (const yy of [3.4, 3.9, 4.4]) this.b.box(x, ytop + yy, z, 1.2 - yy * 0.12, 0.02, 0.02, 0x34322e, { mat: 'metal', layer: 'det' });
        this.col.addCyl(x, z, 0.05, ytop, ytop + 4.8);
      }
    }
    // Rohrstrang über das Dach
    const pz = (z0 + z1) / 2 + 2.2;
    if (free((x0 + x1) / 2, pz)) this.b.addPrim('cyl8', (x0 + x1) / 2, ytop + 0.16, pz, 0.08, x1 - x0 - 1.4, 0.08, 0x59544c, { mat: 'metal', rz: Math.PI / 2, layer: 'det' });
  }

  // Ruine: Mauerreste ohne Dach, Schutthaufen
  ruin(lot, rnd) {
    const { x0, x1, z0, z1 } = lot;
    const y = lot.level;
    const col = lot.district === 'wald' ? pick(rnd, PAL.stone) : mix(pick(rnd, PAL.plaster), 0x8a8478, 0.45);
    const mat = lot.district === 'wald' ? 'stone' : 'wplaster';
    const inner = { x0: x0 + T - 0.02, x1: x1 - T + 0.02, z0: z0 + T - 0.02, z1: z1 - T + 0.02, k: 0.75 };
    const sides = [
      { side: 's', fixed: z1, a0: x0, a1: x1, ia0: x0 + T, ia1: x1 - T },
      { side: 'n', fixed: z0, a0: x0, a1: x1, ia0: x0 + T, ia1: x1 - T },
      { side: 'e', fixed: x1, a0: z0, a1: z1, ia0: z0 + T, ia1: z1 - T },
      { side: 'w', fixed: x0, a0: z0, a1: z1, ia0: z0 + T, ia1: z1 - T },
    ];
    for (const w of sides) {
      // in Stücke zerbrochen, unterschiedlich hoch, mit Lücken
      let a = w.a0;
      while (a < w.a1 - 0.2) {
        const len = Math.min(w.a1 - a, 1.2 + rnd() * 2.6);
        if (rnd() < 0.78) {
          const h = 0.8 + rnd() * (rnd() < 0.3 ? 5.5 : 3.0);
          const ops = h > 2 && len > 1.6 && rnd() < 0.5 ? [{ a0: a + len / 2 - 0.45, a1: a + len / 2 + 0.45, y0: 1.05, y1: Math.min(h - 0.2, 2.3), kind: 'win' }] : [];
          this.wall({ ...w, a0: a, a1: a + len }, y, h, ops, col, { mat, inner });
          // Abbruchkante oben
          this.quad([wp(w, a, y + h, 0), wp(w, a + len, y + h, 0), wp(w, a + len, y + h, T), wp(w, a, y + h, T)], [0, 1, 0], darker(col, 0.85));
        }
        a += len;
      }
    }
    this.slab({ x0, x1, z0, z1 }, y, y + 0.15, null, PAL.rubble[0], null, true, true);
    // Schutt
    const n = 3 + Math.floor(rnd() * 4);
    for (let i = 0; i < n; i++) {
      const px = x0 + 1 + rnd() * (x1 - x0 - 2), pz = z0 + 1 + rnd() * (z1 - z0 - 2), s = 0.6 + rnd() * 1.3;
      this.b.addPrim('ico', px, y + s * 0.3, pz, s * 1.3, s * 0.65, s, pick(rnd, PAL.rubble), { mat: 'stone', ry: rnd() * 6 });
      if (s > 1.1) this.col.addBox(px - s * 0.7, y, pz - s * 0.6, px + s * 0.7, y + s * 0.7, pz + s * 0.6);
    }
    // lose Steinplatten (eigener Zufall, damit der übrige Aufbau gleich bleibt)
    const pr = mulberry32(lot.id + 31);
    for (let i = 0; i < 5; i++) {
      this.B.prop('sr_stone', x0 + 1 + pr() * (x1 - x0 - 2), z0 + 1 + pr() * (z1 - z0 - 2), pr() * 6.28, {
        y: y + 0.17, scale: 2.4 + pr() * 1.6, rx: (pr() - 0.5) * 0.14, rz: (pr() - 0.5) * 0.14, collide: null, shadow: false,
      });
    }
    // Balken liegen herum
    for (let i = 0; i < 2; i++) {
      const px = x0 + 1.5 + rnd() * (x1 - x0 - 3), pz = z0 + 1.5 + rnd() * (z1 - z0 - 3);
      this.b.box(px, y + 0.15, pz, 3 + rnd() * 2, 0.22, 0.22, pick(rnd, PAL.timber), { ry: rnd() * 3, rz: (rnd() - 0.5) * 0.4, mat: 'wood' });
    }
    this.B.spots.push({ x: (x0 + x1) / 2 + (rnd() - 0.5) * 2, y: y + 0.15, z: (z0 + z1) / 2 + (rnd() - 0.5) * 2, tier: lot.tier, poi: lot.poi, indoor: false });
    this.stats.ruins++;
    lot.roofY = y + 3;
    return lot;
  }
}

export function darker(c, k) {
  const r = ((c >> 16) & 255) * k, g = ((c >> 8) & 255) * k, b = (c & 255) * k;
  return (Math.round(r) << 16) | (Math.round(g) << 8) | Math.round(b);
}

export function mix(a, b, t) {
  const ar = (a >> 16) & 255, ag = (a >> 8) & 255, ab = a & 255;
  const br = (b >> 16) & 255, bg = (b >> 8) & 255, bb = b & 255;
  return (Math.round(ar + (br - ar) * t) << 16) | (Math.round(ag + (bg - ag) * t) << 8) | Math.round(ab + (bb - ab) * t);
}
