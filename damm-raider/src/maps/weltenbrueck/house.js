// Begehbare Stadthäuser für Weltenbrück: Treppenhaus, Zimmer, Türen,
// echte Fensteröffnungen, Fachwerk, Läden, Möbel, Beuteplätze und Dächer.
// Wände bestehen aus Flächen (sparsamer als Boxen) plus Kollisionsboxen.

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
  rubble: [0x8f877a, 0x7d766a, 0x9c9384],
};

// Stilparameter: Geschosshöhe, Dach, Materialien
export const STYLE = {
  fachwerk: { fh: 3.0, roof: 'gable', pitch: 1.35, roofMat: 'tile', spacing: 2.3, winW: 0.95, winH: 1.3 },
  stein: { fh: 3.3, roof: 'gable', pitch: 1.0, roofMat: 'slate', spacing: 2.5, winW: 1.1, winH: 1.6 },
  backstein: { fh: 3.2, roof: 'gable', pitch: 1.2, roofMat: 'tile', spacing: 2.4, winW: 1.0, winH: 1.5, stepped: true },
  schwarzwald: { fh: 3.0, roof: 'hip', pitch: 1.15, roofMat: 'wood', spacing: 2.2, winW: 1.0, winH: 1.15 },
  turm: { fh: 3.0, roof: 'flat', pitch: 0, roofMat: 'world', spacing: 2.4, winW: 0.9, winH: 1.2 },
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
  wall(w, y, h, ops, color, { mat = 'world', inner = null, frameCol = PAL.frameLight, shutterCol = null, rnd } = {}) {
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
      this.quad([wp(w, a0, y + s0, 0), wp(w, a1, y + s0, 0), wp(w, a1, y + s1, 0), wp(w, a0, y + s1, 0)], out, color, { mat });
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
        && this.model('kc_window', wp(w, (o.a0 + o.a1) / 2, Y0, 0), yawFront(out2[0], out2[2]), [o.a1 - o.a0, Y1 - Y0, 1], { tint: frameCol, layer: 'det', double: true })) {
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
        const tilt = o.shutter === 'hang' ? 0.35 : 0;
        for (const [edge, dir] of [[o.a1, 1], [o.a0, -1]]) {
          if (dir > 0 ? edge + hw > w.a1 - 0.05 : edge - hw < w.a0 + 0.05) continue;
          const ry = yawX(ax[0] * dir, ax[2] * dir);
          const front = [Math.sin(ry), Math.cos(ry)];
          const inward = front[0] * out2[0] + front[1] * out2[2] < 0;
          this.model('kc_shutter', wp(w, edge, Y0, inward ? -0.12 : -0.01), ry, [hw / 0.5, Y1 - Y0, 1], { tint: shutterCol, layer: 'det', rz: dir > 0 ? -tilt : tilt, double: true });
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
    const d = -0.03;
    const dx = q[0] - p[0], dy = q[1] - p[1];
    const l = Math.hypot(dx, dy) || 1;
    const nx = (-dy / l) * width / 2, ny = (dx / l) * width / 2;
    this.quad([wp(w, p[0] - nx, p[1] - ny, d), wp(w, q[0] - nx, q[1] - ny, d), wp(w, q[0] + nx, q[1] + ny, d), wp(w, p[0] + nx, p[1] + ny, d)], OUTN[w.side], color, { shade: false });
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
    const wallCol = lot.style === 'backstein' ? pick(rnd, PAL.brick) : lot.style === 'fachwerk' ? pick(rnd, PAL.infill) : pick(rnd, PAL.plaster);
    const baseCol = lot.style === 'fachwerk' || lot.style === 'schwarzwald' || lot.style === 'turm' ? pick(rnd, PAL.stone) : wallCol;
    const upperCol = lot.style === 'schwarzwald' || lot.style === 'turm' ? pick(rnd, PAL.wood) : wallCol;
    const timber = pick(rnd, PAL.timber);
    const frameCol = lot.style === 'schwarzwald' || lot.style === 'turm' ? pick(rnd, PAL.wood) : PAL.frameLight;
    const shutterCol = rnd() < 0.6 ? pick(rnd, PAL.shutter) : null;
    const wallMat = lot.style === 'backstein' ? 'brick' : 'world';
    let baseMat = lot.style === 'fachwerk' || lot.style === 'schwarzwald' || lot.style === 'turm' ? 'stone' : wallMat;
    let upperMat = lot.style === 'schwarzwald' ? 'wood' : lot.style === 'turm' ? 'wood' : wallMat;
    let baseColor = baseCol, upperColor = upperCol, frameColor = frameCol, shutters = shutterCol;
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

    const shopSide = lot.shop ? doors[0]?.side : null;
    lot.doors = doors;
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
            ops.push({
              a0, a1, y0: off + sill, y1: off + sill + winH, kind: 'win',
              frame: lot.style !== 'fachwerk' || f === 0 ? r < 0.85 : r < 0.5,
              shutter: !isShop && shutters != null && r < 0.7 ? (r < 0.08 ? 'hang' : 'open') : null,
            });
          }
        }
        this.wall(w, yw0, hw, ops, col, { mat, inner: innerRect, frameCol: frameColor, shutterCol: shutters, rnd });
        const anyOpen = runs.length > 0;
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
          const w = sides[d.side], axv = AXIS[d.side];
          const shop = shopSide === d.side && d === doors[0];
          const dw = shop ? 1.6 : DOOR_W;
          this.model('kc_door', wp(w, d.a - dw / 2, fy, T + 0.05), yawX(-axv[0], -axv[2]), [dw, DOOR_H, 1], { tint: pick(rnd, PAL.door), layer: 'det' });
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
      if (hasPart) this.partition(rectUV, uPart, S, fy, fh, upperColor, innerRect);
      // Möbel und Beuteplätze im vorderen Zimmer
      this.furnish(lot, rnd, rectUV, toX, toZ, frontRoomEnd, S, fy, f, sw, doors, aOfU, aOfV, sideB, front, innerRect);
    }

    // Dach
    const yTop = yG + floors * fh;
    const roofCol = st.roofMat === 'slate' ? pick(rnd, PAL.slate) : st.roofMat === 'wood' ? pick(rnd, PAL.wood) : pick(rnd, PAL.tile);
    const mossy = rnd() < 0.45;
    const rc = mossy ? mix(roofCol, PAL.moss, 0.18 + rnd() * 0.2) : roofCol;
    if (st.roof === 'gable') this.gableRoof(lot, alongX, yTop, st.pitch, rc, st.roofMat, wallCol, timber, rnd, st.stepped && rnd() < 0.7);
    else if (st.roof === 'hip') this.hipRoof(lot, alongX, yTop, st.pitch, rc);
    else this.flatRoof(lot, yTop, rnd, roofAccess ? rectUV(uS0 - 0.05, L - T, (floors - 1) % 2 === 0 ? T : S - T - sw - 0.05, (floors - 1) % 2 === 0 ? T + sw + 0.05 : S - T) : null);

    // Efeu und Ladenschilder
    if (rnd() < 0.3) this.ivy(sides[front], y, Math.min(floors, 1 + Math.floor(rnd() * 3)) * fh, rnd);
    if (lot.shop && doors[0]) this.shopSign(sides[doors[0].side], doors[0].a, yG + 2.9, rnd);
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

  partition(rectUV, u, S, fy, fh, color, inner) {
    const t = 0.14, dw = 1.0, dh = 2.15;
    const vm = S / 2;
    const segs = [[T, vm - dw / 2, 0, fh], [vm + dw / 2, S - T, 0, fh], [vm - dw / 2, vm + dw / 2, dh, fh]];
    for (const [v0, v1, h0, h1] of segs) {
      const r = rectUV(u - t / 2, u + t / 2, v0, v1);
      this.b.box((r.x0 + r.x1) / 2, fy + h0, (r.z0 + r.z1) / 2, r.x1 - r.x0, h1 - h0, r.z1 - r.z0, color, { inner, layer: 'in' });
      this.col.addBox(r.x0, fy + h0, r.z0, r.x1, fy + h1, r.z1);
    }
  }

  furnish(lot, rnd, rectUV, toX, toZ, uEnd, S, fy, f, sw, doors, aOfU, aOfV, sideB, front, inner) {
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
    const du = [toX(1, 0) - toX(0, 0), toZ(1, 0) - toZ(0, 0)];
    const dv = [toX(0, 1) - toX(0, 0), toZ(0, 1) - toZ(0, 0)];
    const at = (u, v) => [toX(u, v), fy, toZ(u, v)];
    const mopts = { layer: 'in', shade: 0.62 };
    // Tisch mit Stühlen an Seite A
    if (S >= 4.6 && roomL >= 3.0) {
      const u = Math.min(T + roomL * 0.5, uEnd - 1.9), v = T + 0.75;
      if (u > T + 0.9 && !nearDoor(u, v)) {
        const r = rectUV(u - 0.7, u + 0.7, v - 0.4, v + 0.4);
        if (this.model('kc_table', at(u, v), yawX(du[0], du[1]), 1, mopts)) {
          for (const k of [-0.35, 0.35]) if (rnd() < 0.8) this.model('kc_chair', at(u + k, v + 0.62), yawFront(-dv[0], -dv[1]) + (rnd() - 0.5) * 0.4, 1, mopts);
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
    // Regal oder Kachelofen an der Vorderwand (Seite A)
    if (rnd() < 0.5 && S >= 5) {
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
    const e = 0.45, g = stepped ? -0.05 : 0.35;
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
        this.tri([M(p, ye, Q0), M(p, ye, Q1), M(p, yr, qc)], N(s, 0, 0), wallCol, { mat: lot.style === 'backstein' ? 'brick' : 'world' });
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
          this.quad([wp(ww, qc - 0.35, wy, -0.035), wp(ww, qc + 0.35, wy, -0.035), wp(ww, qc + 0.35, wy + 0.8, -0.035), wp(ww, qc - 0.35, wy + 0.8, -0.035)], OUTN[ww.side], 0x1d1a17, { shade: false });
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

  ivy(w, y, h, rnd) {
    const a0 = w.a0 + 0.3 + rnd() * (w.a1 - w.a0) * 0.5;
    const width = 1 + rnd() * 2.5;
    let yy = y;
    const col = pick(rnd, PAL.ivy);
    while (yy < y + h) {
      const ww = width * (0.6 + rnd() * 0.5), hh = 0.6 + rnd() * 1.2;
      const a = a0 + (rnd() - 0.5) * 0.8;
      this.quad([wp(w, a, yy, -0.06), wp(w, Math.min(w.a1, a + ww), yy, -0.06), wp(w, Math.min(w.a1, a + ww * 0.8), yy + hh, -0.06), wp(w, a + ww * 0.15, yy + hh, -0.06)], OUTN[w.side], col, { shade: false, layer: 'det' });
      yy += hh * 0.8;
    }
  }

  shopSign(w, a, y, rnd) {
    const out = OUTN[w.side];
    const col = pick(rnd, PAL.awning);
    if (this.inst && this.inst.has('kc_sign')) {
      const ry = yawFront(out[0], out[2]);
      this.model('kc_sign', wp(w, a + 0.9, y - 0.3, 0), ry, 1, { tint: col, layer: 'det' });
      if (rnd() < 0.55) this.model('kc_awning', wp(w, a, y + 0.1, 0), ry, 1, { tint: pick(rnd, PAL.awning), layer: 'det' });
      return;
    }
    // Ausleger mit Schild quer zur Wand
    const p0 = wp(w, a + 0.9, y, -0.05), p1 = wp(w, a + 0.9, y, -1.1);
    this.b.box((p0[0] + p1[0]) / 2, y + 0.55, (p0[2] + p1[2]) / 2, Math.abs(p0[0] - p1[0]) + 0.06, 0.06, Math.abs(p0[2] - p1[2]) + 0.06, 0x2a2622, { mat: 'metal', layer: 'det' });
    const q = [wp(w, a + 0.9, y - 0.25, -0.25), wp(w, a + 0.9, y - 0.25, -1.0), wp(w, a + 0.9, y + 0.5, -1.0), wp(w, a + 0.9, y + 0.5, -0.25)];
    this.quad(q, AXIS[w.side], col, { double: true, layer: 'det' });
    // Markise über dem Schaufenster
    if (rnd() < 0.55) {
      const b0 = a - 1.4, b1 = a + 1.4;
      this.quad([wp(w, b0, y - 0.5, -1.2), wp(w, b1, y - 0.5, -1.2), wp(w, b1, y + 0.1, -0.02), wp(w, b0, y + 0.1, -0.02)], [out[0], 0.7, out[2]], col, { double: true, layer: 'det' });
    }
  }

  // Ruine: Mauerreste ohne Dach, Schutthaufen
  ruin(lot, rnd) {
    const { x0, x1, z0, z1 } = lot;
    const y = lot.level;
    const col = pick(rnd, lot.district === 'wald' ? PAL.stone : PAL.plaster);
    const mat = lot.district === 'wald' ? 'stone' : 'world';
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
