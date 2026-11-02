// Baukasten für Gebäude und Props. Jede Form landet im Batcher (Optik) und
// optional in der Kollisionswelt. Wo ein Blender-Modell vorhanden ist, wird
// dieses eingebacken; sonst entstehen einfache Grundformen.
import * as THREE from 'three';
import { materialClass } from './assets.js';

const _Y = new THREE.Vector3(0, 1, 0);
const SIDE_RY = { s: 0, n: Math.PI, e: Math.PI / 2, w: -Math.PI / 2 };

export const COL = {
  concrete: 0xa7a296,
  concreteDark: 0x8a867b,
  concreteLight: 0xc2bdae,
  pale: 0xcfc8b4,
  paleDark: 0xb3ab96,
  trim: 0x3c3c37,
  rust: 0x8c5534,
  rustDark: 0x6a4028,
  teal: 0x3e6e68,
  tealDark: 0x2c4f4b,
  orange: 0xd2732e,
  yellow: 0xd8a92b,
  red: 0xa8392e,
  metal: 0x74777a,
  metalDark: 0x4b4e51,
  wood: 0x6e5438,
  woodDark: 0x4d3a27,
  floor: 0x8f887a,
  roof: 0x6d685e,
  glass: 0x9fc6d0,
  sand: 0x9a8a62,
  green: 0x4f6b3a,
  black: 0x252523,
};

export class Builder {
  constructor(batcher, col, terrain, assets = null) {
    this.b = batcher;
    this.col = col;
    this.t = terrain;
    this.assets = assets;
    this.spots = []; // mögliche Beute-Positionen {x,y,z,tier,poi}
    this.turretSpots = [];
    this.rnd = 1;
  }

  ground(x, z) { return this.t.height(x, z); }

  // deterministischer Zufall (gleiche Karte bei jedem Laden)
  rand() {
    this.rnd = (this.rnd * 16807) % 2147483647;
    return (this.rnd - 1) / 2147483646;
  }

  has(name) { return !!(this.assets && this.assets.has(name)); }

  // Blender-Modell einbacken. scale: Zahl oder [sx, sy, sz]; collide: 'box' = AABB aus den Modellgrenzen
  model(name, x, y, z, ry = 0, { scale = 1, tint = null, collide = null, shrink = 1 } = {}) {
    if (!this.has(name)) return false;
    const sc = typeof scale === 'number' ? [scale, scale, scale] : scale;
    const m = new THREE.Matrix4().compose(new THREE.Vector3(x, y, z), new THREE.Quaternion().setFromAxisAngle(_Y, ry), new THREE.Vector3(...sc));
    this.b.addModel(this.assets.meshes(name), m, tint != null ? new THREE.Color(tint) : null, materialClass);
    if (collide === 'box') {
      const bb = this.assets.bounds(name).clone().applyMatrix4(m);
      const cx = (bb.min.x + bb.max.x) / 2, cz = (bb.min.z + bb.max.z) / 2;
      const hx = ((bb.max.x - bb.min.x) / 2) * shrink, hz = ((bb.max.z - bb.min.z) / 2) * shrink;
      this.col.addBox(cx - hx, bb.min.y, cz - hz, cx + hx, bb.max.y, cz + hz);
    }
    return true;
  }

  // SimReady-Requisit als Instanz (behält seine Texturen, nicht eingebacken).
  // collide: 'box' = AABB aus den Modellgrenzen, 'cyl' = stehender Zylinder, null = begehbar
  prop(name, x, z, ry = 0, { y = null, scale = 1, collide = 'box', shrink = 0.9, rx = 0, rz = 0, layer = 'det', shadow = true } = {}) {
    if (!this.inst || !this.inst.has(name)) return false;
    const gy = y ?? this.ground(x, z);
    this.inst.add(name, x, gy, z, ry, scale, { layer, rx, rz, shadow });
    const bb = this.assets.bounds(name);
    const sc = typeof scale === 'number' ? [scale, scale, scale] : scale;
    const w = (bb.max.x - bb.min.x) * sc[0], d = (bb.max.z - bb.min.z) * sc[2], h = (bb.max.y - bb.min.y) * sc[1];
    if (collide === 'box') this.solidCol(x, gy, z, w, h, d, ry, shrink);
    else if (collide === 'cyl') this.col.addCyl(x, z, (Math.max(w, d) / 2) * shrink, gy, gy + h);
    return true;
  }

  // Requisit aus dem gemeinsamen Atlas: ein Draw-Call je Kachel, egal wie viele Objekte
  dress(name, x, z, ry = 0, { y = null, scale = 1, rx = 0, rz = 0, color = 0xffffff, bright = 1, collide = null, shrink = 0.9, layer = 'det' } = {}) {
    if (!this.has(name)) return false;
    const gy = y ?? this.ground(x, z);
    const sc = typeof scale === 'number' ? [scale, scale, scale] : scale;
    const m = new THREE.Matrix4().compose(new THREE.Vector3(x, gy, z), new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz, 'YXZ')), new THREE.Vector3(...sc));
    this.b.addModelUV(this.assets.meshes(name), m, color, 'dress', layer, bright);
    if (collide) {
      const bb = this.assets.bounds(name);
      const w = (bb.max.x - bb.min.x) * sc[0], d = (bb.max.z - bb.min.z) * sc[2], h = (bb.max.y - bb.min.y) * sc[1];
      if (collide === 'box') this.solidCol(x, gy, z, w, h, d, ry, shrink);
      else this.col.addCyl(x, z, (Math.max(w, d) / 2) * shrink, gy, gy + h);
    }
    return true;
  }

  // Nur Kollision (Box mit Unterkante y)
  solidCol(x, y, z, w, h, d, ry = 0, shrink = 1) {
    let hw = w / 2, hd = d / 2;
    if (ry) {
      const c = Math.abs(Math.cos(ry)), s = Math.abs(Math.sin(ry));
      const nw = hw * c + hd * s, nd = hw * s + hd * c;
      hw = nw * shrink; hd = nd * shrink;
    }
    this.col.addBox(x - hw, y, z - hd, x + hw, y + h, z + hd);
  }

  // Box mit Unterkante y, Mittelpunkt x/z. ry nur in 90°-Schritten kollisionsgenau.
  solid(x, y, z, w, h, d, color, opts = {}) {
    const ry = opts.ry || 0;
    this.b.box(x, y, z, w, h, d, color, { ry, mat: opts.mat || 'world' });
    if (opts.noCol) return;
    let hw = w / 2, hd = d / 2;
    if (ry) {
      const c = Math.abs(Math.cos(ry)), s = Math.abs(Math.sin(ry));
      const nw = hw * c + hd * s, nd = hw * s + hd * c;
      hw = nw; hd = nd;
      if (opts.shrink) { hw *= opts.shrink; hd *= opts.shrink; }
    }
    this.col.addBox(x - hw, y, z - hd, x + hw, y + h, z + hd, opts.tag || null);
  }

  deco(prim, x, y, z, sx, sy, sz, color, opts = {}) {
    this.b.addPrim(prim, x, y, z, sx, sy, sz, color, opts);
  }

  // Wand entlang X (bei z) oder Z (bei x) mit Öffnungen {c, w, y0, h}.
  // side ('s','n','e','w') bestimmt, wohin Rahmen und Läden zeigen.
  wall(axis, a1, a2, fixed, y, h, t, color, openings = [], side = null) {
    const ry = side ? SIDE_RY[side] : axis === 'x' ? 0 : Math.PI / 2;
    const ops = openings.slice().sort((p, q) => p.c - q.c);
    const seg = (s0, s1, y0, y1) => {
      if (s1 - s0 < 0.05 || y1 - y0 < 0.05) return;
      const mid = (s0 + s1) / 2, len = s1 - s0;
      if (axis === 'x') this.solid(mid, y + y0, fixed, len, y1 - y0, t, color);
      else this.solid(fixed, y + y0, mid, t, y1 - y0, len, color);
    };
    let cur = a1;
    for (const op of ops) {
      const a = Math.max(a1, op.c - op.w / 2), b2 = Math.min(a2, op.c + op.w / 2);
      if (a > cur) seg(cur, a, 0, h);
      if (op.y0 > 0) seg(a, b2, 0, op.y0);
      if (op.y0 + op.h < h) seg(a, b2, op.y0 + op.h, h);
      const mid = (a + b2) / 2;
      const px = axis === 'x' ? mid : fixed, pz = axis === 'x' ? fixed : mid;
      const ow = b2 - a;
      let framed = false;
      if (op.y0 > 0.3) {
        framed = this.model('k_window', px, y + op.y0 + op.h / 2, pz, ry, { scale: [ow / 1.4, op.h / 1.3, t / 0.35] });
        if (framed && this.rand() < 0.3) this.model('k_shutter', px, y + op.y0 + op.h / 2, pz, ry, { scale: [ow / 1.4, op.h / 1.3, 1] });
      } else if (op.door) {
        framed = ow >= 5
          ? this.model('k_rolldoor', px, y, pz, ry, { scale: [ow / 8, op.h / 7, 1] })
          : this.model('k_door', px, y, pz, ry, { scale: [ow / 1.8, op.h / 2.4, t / 0.35] });
      }
      // Fensterbank (ohne Modell)
      if (!framed && op.y0 > 0.3) {
        if (axis === 'x') this.deco('box', mid, y + op.y0 - 0.06, fixed, b2 - a + 0.2, 0.12, t + 0.24, COL.trim);
        else this.deco('box', fixed, y + op.y0 - 0.06, mid, t + 0.24, 0.12, b2 - a + 0.2, COL.trim);
      }
      cur = b2;
    }
    if (cur < a2) seg(cur, a2, 0, h);
  }

  // Platte mit optionalem rechteckigem Loch {x0,x1,z0,z1}
  slab(x0, x1, z0, z1, y, th, color, hole = null) {
    if (!hole) {
      this.solid((x0 + x1) / 2, y - th, (z0 + z1) / 2, x1 - x0, th, z1 - z0, color);
      return;
    }
    const parts = [
      [x0, x1, z0, hole.z0],
      [x0, x1, hole.z1, z1],
      [x0, hole.x0, hole.z0, hole.z1],
      [hole.x1, x1, hole.z0, hole.z1],
    ];
    for (const [a, b2, c, d] of parts) {
      if (b2 - a > 0.05 && d - c > 0.05) this.solid((a + b2) / 2, y - th, (c + d) / 2, b2 - a, th, d - c, color);
    }
  }

  // Treppe: steigt in Richtung dir ('n' = -z, 's' = +z, 'e' = +x, 'w' = -x)
  stairs(x, z, y0, y1, dir, width, color = COL.concreteDark) {
    const stepH = 0.3, stepD = 0.34;
    const n = Math.max(1, Math.round((y1 - y0) / stepH));
    const hh = (y1 - y0) / n;
    const len = n * stepD;
    for (let i = 0; i < n; i++) {
      const top = y0 + hh * (i + 1);
      const off = -len / 2 + stepD * (i + 0.5);
      let sx = x, sz = z, w = width, d = stepD;
      if (dir === 'n') sz = z - off;
      else if (dir === 's') sz = z + off;
      else if (dir === 'e') { sx = x + off; w = stepD; d = width; }
      else { sx = x - off; w = stepD; d = width; }
      this.solid(sx, y0, sz, w, top - y0, d, color);
    }
    return len;
  }

  // Mehrstöckiges Gebäude mit Türen, Fenstern, Treppen und Dach.
  building(o) {
    const {
      x, z, w, d,
      floors = 1, fh = 3.4, t = 0.35,
      wall = COL.concrete, trim = COL.trim, floorCol = COL.floor, roofCol = COL.roof,
      doors = { s: [0.5] }, windows = true, roofAccess = true, tier = 1, poi = null,
      baseY = null, winY0 = 1.0, winH = 1.3, winEvery = 3.6, parapet = 0.9, foundation = true,
    } = o;
    const y = baseY ?? this.ground(x, z);
    const x0 = x - w / 2, x1 = x + w / 2, z0 = z - d / 2, z1 = z + d / 2;
    const stairW = 1.4;
    const levels = floors + (roofAccess ? 0 : -1);

    // Fundament
    if (foundation) this.solid(x, y - 1.2, z, w + 0.4, 1.35, d + 0.4, COL.concreteDark);

    const stairFoot = (f) => {
      // gerade Stockwerke: Ostwand nach Norden, ungerade: Westwand nach Süden
      const len = Math.round(fh / 0.3) * 0.34;
      const sx = f % 2 === 0 ? x1 - t / 2 - stairW / 2 - 0.05 : x0 + t / 2 + stairW / 2 + 0.05;
      const cz = z;
      return { sx, cz, len, x0: sx - stairW / 2 - 0.1, x1: sx + stairW / 2 + 0.1, z0: cz - len / 2 - 0.1, z1: cz + len / 2 + 0.1 };
    };

    for (let f = 0; f < floors; f++) {
      const fy = y + 0.15 + f * fh;
      const wh = fh;
      const mkOps = (len, side, a0) => {
        const ops = [];
        // Türen: "s" = Erdgeschoss, "s@3" = Tür im 4. Stock
        const list = f === 0 ? doors[side] : doors[`${side}@${f}`];
        if (list) {
          for (const dc of list) {
            const dw = typeof dc === 'object' ? dc.w : 1.8;
            const dh = typeof dc === 'object' ? dc.h : 2.4;
            const c = typeof dc === 'object' ? dc.c : dc;
            ops.push({ c: a0 + c * len, w: dw, y0: 0, h: dh, door: true });
          }
        }
        if (windows) {
          const n = Math.floor((len - 1.5) / winEvery);
          for (let i = 0; i < n; i++) {
            const c = a0 + (len / n) * (i + 0.5);
            if (ops.some((op) => Math.abs(op.c - c) < op.w / 2 + 1.0)) continue;
            ops.push({ c, w: 1.4, y0: winY0, h: Math.min(winH, wh - winY0 - 0.4) });
          }
        }
        return ops;
      };
      this.wall('x', x0, x1, z1 - t / 2, fy, wh, t, wall, mkOps(w, 's', x0), 's');
      this.wall('x', x0, x1, z0 + t / 2, fy, wh, t, wall, mkOps(w, 'n', x0), 'n');
      this.wall('z', z0 + t, z1 - t, x1 - t / 2, fy, wh, t, wall, mkOps(d - 2 * t, 'e', z0 + t), 'e');
      this.wall('z', z0 + t, z1 - t, x0 + t / 2, fy, wh, t, wall, mkOps(d - 2 * t, 'w', z0 + t), 'w');
      // Gesims
      this.deco('box', x, fy + wh - 0.1, z, w + 0.3, 0.25, d + 0.3, trim);

      // Boden
      if (f === 0) this.slab(x0 + t, x1 - t, z0 + t, z1 - t, fy, 0.16, floorCol);
      else {
        const sf = stairFoot(f - 1);
        this.slab(x0, x1, z0, z1, fy, 0.3, floorCol, sf);
      }
      // Treppe nach oben
      if (f < levels) {
        const sf = stairFoot(f);
        this.stairs(sf.sx, sf.cz, fy, fy + fh, f % 2 === 0 ? 'n' : 's', stairW);
      }
      // Beute-Spots: Ecken und Wandmitten
      const inset = 0.9;
      const cand = [
        [x0 + inset + t, z0 + inset + t], [x1 - inset - t, z0 + inset + t],
        [x0 + inset + t, z1 - inset - t], [x1 - inset - t, z1 - inset - t],
        [x, z0 + inset + t], [x, z1 - inset - t],
      ];
      const sf = stairFoot(f);
      const sfPrev = f > 0 ? stairFoot(f - 1) : null;
      for (const [sx, sz] of cand) {
        const inFoot = (s) => s && sx > s.x0 - 0.6 && sx < s.x1 + 0.6 && sz > s.z0 - 0.6 && sz < s.z1 + 0.6;
        if (inFoot(sf) || inFoot(sfPrev)) continue;
        this.spots.push({ x: sx, y: fy, z: sz, tier, poi, indoor: true });
      }
    }
    // Dach
    const ry = y + 0.15 + floors * fh;
    this.slab(x0, x1, z0, z1, ry, 0.35, roofCol, roofAccess ? stairFoot(floors - 1) : null);
    if (parapet > 0) {
      this.solid(x, ry, z0 + 0.15, w, parapet, 0.3, wall);
      this.solid(x, ry, z1 - 0.15, w, parapet, 0.3, wall);
      this.solid(x0 + 0.15, ry, z, 0.3, parapet, d - 0.6, wall);
      this.solid(x1 - 0.15, ry, z, 0.3, parapet, d - 0.6, wall);
    }
    // Lüftung und Klimagerät auf dem Dach
    if (w > 8) {
      if (this.model('k_vent', x - w * 0.2, ry, z + d * 0.15, 0, { collide: 'box' })) {
        if (w > 12) this.model('k_ac', x + w * 0.22, ry, z + d * 0.2, Math.PI, { collide: 'box' });
      } else this.solid(x - w * 0.2, ry, z + d * 0.15, 2.2, 1.2, 1.6, COL.metal);
    }
    if (roofAccess) this.spots.push({ x: x + w * 0.25, y: ry, z: z - d * 0.25, tier, poi, indoor: false });
    return { x0, x1, z0, z1, y, roofY: ry, fh };
  }

  // Schiffscontainer
  container(x, z, ry, color, y = null) {
    const gy = y ?? this.ground(x, z);
    const w = 6.1, h = 2.6, d = 2.45;
    const rot = Math.round(ry / (Math.PI / 2)) * (Math.PI / 2);
    if (this.model('p_container', x, gy, z, rot, { tint: color })) {
      this.solidCol(x, gy, z, w, h, d, rot);
      return gy + h;
    }
    this.solid(x, gy, z, w, h, d, color, { ry: rot });
    // Rippen
    const ca = Math.cos(rot), sa = Math.sin(rot);
    for (let i = -2; i <= 2; i++) {
      const lx = i * 1.2;
      this.deco('box', x + ca * lx, gy + h / 2, z - sa * lx, 0.12, h - 0.1, d + 0.1, color, { ry: rot });
    }
    return gy + h;
  }

  car(x, z, ry, color) {
    const gy = this.ground(x, z);
    const kind = this.rand() < 0.65 ? 'p_car_a' : 'p_car_b';
    // Modelle zeigen nach -Z; die alten Autos lagen entlang X
    if (this.model(kind, x, gy - 0.05, z, ry + Math.PI / 2, { tint: color, collide: 'box', shrink: 0.85 })) return;
    const ca = Math.cos(ry), sa = Math.sin(ry);
    this.solid(x, gy + 0.35, z, 4.2, 0.85, 1.8, color, { ry, shrink: 0.85 });
    this.deco('box', x - ca * 0.3, gy + 1.2, z + sa * 0.3, 2.2, 0.7, 1.6, color, { ry });
    this.deco('box', x - ca * 0.3, gy + 1.25, z + sa * 0.3, 2.25, 0.5, 1.5, COL.black, { ry });
    for (const [lx, lz] of [[1.35, 0.85], [-1.35, 0.85], [1.35, -0.85], [-1.35, -0.85]]) {
      const wx = x + ca * lx + sa * lz, wz = z - sa * lx + ca * lz;
      this.deco('cyl8', wx, gy + 0.35, wz, 0.35, 0.25, 0.35, COL.black, { rx: Math.PI / 2, ry });
    }
  }

  barrier(x, z, ry, color = COL.concreteLight) {
    const gy = this.ground(x, z);
    if (this.model('p_barrier', x, gy, z, ry, { collide: 'box', shrink: 0.9 })) return;
    this.solid(x, gy, z, 3, 1.05, 0.7, color, { ry, shrink: 0.9 });
    this.deco('box', x, gy + 0.9, z, 3.05, 0.12, 0.72, COL.orange, { ry });
  }

  sandbags(x, z, len, ry) {
    const gy = this.ground(x, z);
    if (this.model('p_sandbags', x, gy - 0.05, z, ry, { scale: [len / 2.8, 1, 1] })) {
      this.solidCol(x, gy, z, len, 1.0, 1.0, ry, 0.9);
      return;
    }
    this.solid(x, gy, z, len, 1.1, 1.0, COL.sand, { ry, shrink: 0.9 });
    this.deco('box', x, gy + 1.1, z, len * 0.9, 0.25, 0.8, 0xa89a70, { ry });
  }

  crate(x, z, s = 1.2, y = null, color = COL.wood) {
    const gy = y ?? this.ground(x, z);
    const ry = this.rand() * Math.PI;
    if (this.prop('sr_crate', x, z, ry, { y: gy, scale: s, collide: null, layer: '' })) {
      this.solidCol(x, gy, z, s, s, s);
      return;
    }
    if (this.model('p_crate', x, gy, z, ry, { scale: s / 1.2 })) {
      this.solidCol(x, gy, z, s, s, s);
      return;
    }
    this.solid(x, gy, z, s, s, s, color);
    this.deco('box', x, gy + s / 2, z, s + 0.05, 0.12, s + 0.05, COL.woodDark);
  }

  // Strommast (Gittermast)
  pylon(x, z, h = 26, ry = 0) {
    const gy = this.ground(x, z);
    const base = 3.2, top = 1.0;
    const ca = Math.cos(ry), sa = Math.sin(ry);
    const legs = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
    if (this.model('p_pylon', x, gy - 0.1, z, ry, { scale: [1, h / 24, 1] })) {
      for (const [lx, lz] of legs) {
        const bx = x + (lx * ca + lz * sa) * base, bz = z + (-lx * sa + lz * ca) * base;
        this.col.addBox(bx - 0.4, gy, bz - 0.4, bx + 0.4, gy + 3, bz + 0.4);
      }
      return;
    }
    for (const [lx, lz] of legs) {
      const bx = x + (lx * ca + lz * sa) * base, bz = z + (-lx * sa + lz * ca) * base;
      const tx = x + (lx * ca + lz * sa) * top, tz = z + (-lx * sa + lz * ca) * top;
      const mx = (bx + tx) / 2, mz = (bz + tz) / 2;
      const len = Math.hypot(bx - tx, h, bz - tz);
      const tiltX = Math.atan2(tz - bz, h), tiltZ = -Math.atan2(tx - bx, h);
      this.deco('box', mx, gy + h / 2, mz, 0.25, len, 0.25, COL.metalDark, { rx: tiltX, rz: tiltZ });
      this.col.addBox(bx - 0.3, gy, bz - 0.3, bx + 0.3, gy + 3, bz + 0.3);
    }
    for (let k = 1; k < 6; k++) {
      const yy = gy + (h / 6) * k;
      const s = base + (top - base) * (k / 6);
      this.deco('box', x, yy, z, s * 2, 0.15, 0.15, COL.metalDark, { ry });
      this.deco('box', x, yy, z, 0.15, 0.15, s * 2, COL.metalDark, { ry });
    }
    this.deco('box', x, gy + h, z, 12, 0.35, 0.5, COL.metalDark, { ry });
    this.deco('box', x, gy + h - 4, z, 9, 0.3, 0.4, COL.metalDark, { ry });
    for (const o of [-5.5, 5.5, -4, 4]) {
      this.deco('cyl8', x + ca * o, gy + h - (Math.abs(o) < 5 ? 4.6 : 0.6), z - sa * o, 0.18, 0.9, 0.18, 0xd9d4c4);
    }
  }

  lamp(x, z, y = null, h = 6) {
    const gy = y ?? this.ground(x, z);
    if (this.model('p_lamp', x, gy, z, 0, { scale: h / 6 })) {
      this.col.addBox(x - 0.15, gy, z - 0.15, x + 0.15, gy + h, z + 0.15);
      return;
    }
    this.solid(x, gy, z, 0.25, h, 0.25, COL.metalDark);
    this.deco('box', x + 0.6, gy + h, z, 1.4, 0.18, 0.4, COL.metalDark);
    this.deco('box', x + 1.0, gy + h - 0.12, z, 0.5, 0.1, 0.3, 0xfff2c8, { mat: 'glow' });
  }

  fence(x0, z0, x1, z1, y = null) {
    const len = Math.hypot(x1 - x0, z1 - z0);
    const n = Math.max(1, Math.floor(len / 3));
    const ry = -Math.atan2(z1 - z0, x1 - x0);
    for (let i = 0; i <= n; i++) {
      const px = x0 + ((x1 - x0) * i) / n, pz = z0 + ((z1 - z0) * i) / n;
      const gy = y ?? this.ground(px, pz);
      this.deco('box', px, gy + 1.1, pz, 0.12, 2.2, 0.12, COL.metalDark);
    }
    const mx = (x0 + x1) / 2, mz = (z0 + z1) / 2;
    const gy = y ?? this.ground(mx, mz);
    this.deco('box', mx, gy + 2.05, mz, len, 0.08, 0.08, COL.metalDark, { ry });
    this.deco('box', mx, gy + 1.1, mz, len, 1.9, 0.03, 0x7a7d78, { ry, mat: 'fence' });
    // Kollision grob (nur achsenparallele Zäune exakt)
    if (Math.abs(x1 - x0) < 0.01 || Math.abs(z1 - z0) < 0.01) {
      this.col.addBox(Math.min(x0, x1) - 0.1, gy - 1, Math.min(z0, z1) - 0.1, Math.max(x0, x1) + 0.1, gy + 2.2, Math.max(z0, z1) + 0.1);
    }
  }

  tank(x, z, r, h, color, y = null, top = null) {
    const gy = y ?? this.ground(x, z);
    this.deco('cyl16', x, gy + h / 2, z, r, h, r, color);
    this.deco('cyl16', x, gy + h + 0.1, z, r + 0.15, 0.25, r + 0.15, COL.trim);
    if (top) this.deco('cyl16', x, gy + h + 0.02, z, r - 0.4, 0.1, r - 0.4, top, { mat: top === 'water' ? 'world' : 'world' });
    this.col.addCyl(x, z, r, gy - 1, gy + h);
    return gy + h;
  }

  // Rohr zwischen zwei Punkten (nur Optik), Kollision optional per Box-Kette
  pipe(ax, ay, az, bx, by, bz, r, color, collide = false) {
    const dx = bx - ax, dy = by - ay, dz = bz - az;
    const len = Math.hypot(dx, dy, dz);
    const mx = (ax + bx) / 2, my = (ay + by) / 2, mz = (az + bz) / 2;
    // Zylinder zeigt entlang Y: in Richtung (dx,dy,dz) drehen
    const yaw = Math.atan2(dx, dz);
    const pitch = Math.acos(Math.max(-1, Math.min(1, dy / len)));
    this.deco('cyl16', mx, my, mz, r, len, r, color, { rx: pitch, ry: yaw });
    if (collide) {
      const n = Math.ceil(len / 2);
      for (let i = 0; i < n; i++) {
        const t = (i + 0.5) / n;
        const px = ax + dx * t, py = ay + dy * t, pz = az + dz * t;
        this.col.addBox(px - r, py - r, pz - r, px + r, py + r, pz + r);
      }
    }
  }
}
