// Weltausstattung: Kleine Szenen statt einzelner Objekte, wie in großen Open-World-Spielen.
// Werkstatt, Holzlager, Wäscheleine, verlassenes Lager, Barrikade, Müll, Hafenkram ...
// Die Requisiten teilen sich Atlas und Material (assets/props.glb) und kosten pro Kachel einen Draw-Call.
import { mulberry32 } from './util.js';

const WOOD = 0x6a4a30, WOOD_DARK = 0x3a2a1e, LOG = 0x7a5a3a;
const CLOTH = [0xd8d0bc, 0xb9c4c8, 0xc8a890, 0x8a9a7a, 0xa8483a, 0x4a5a7a, 0xe0d8c8];
const BANNER = { markt: [0x8a2f2a, 0xb48a2c], wald: [0x3e5c2e, 0x6a4a30], weiss: [0xa8392e, 0xe8e2d4], turm: [0x2f4f6a, 0x5ce1e6], hafen: [0x2f5e6a, 0xc8a890] };

// Große Objekte bekommen eine Kollision, kleine sind begehbar
const COLLIDE = {
  d_wine_barrel: 'cyl', d_barrels: 'box', d_mil_crate: 'box', d_crate2: 'box', d_table: 'box', d_bench: 'box', d_cannon: 'box', d_barrier: 'box',
  d_generator: 'box', d_toolcart: 'box', d_rack: 'box', d_buoy: 'cyl', d_propane: 'cyl', d_trashcan: 'box', d_compost: 'box', d_firepit: null,
};

export function makeDresser(B, seed) {
  const rnd = mulberry32(seed);
  const R = (a, c) => a + rnd() * (c - a);
  const pick = (arr) => arr[Math.floor(rnd() * arr.length) % arr.length];
  const b = B.b, col = B.col;
  const has = B.assets?.has('d_crate1');

  // Rahmen: Ursprung an (x, z), a = entlang der Wand, o = von der Wand weg
  const frame = (x, z, y, ax, az, ox, oz) => ({
    pos: (a, o) => [x + ax * a + ox * o, z + az * a + oz * o],
    y,
    yaw: Math.atan2(ox, oz),
    put(name, a, o, { ry = 0, h = 0, scale = 1, rx = 0, rz = 0, color = 0xffffff, collide } = {}) {
      const [px, pz] = this.pos(a, o);
      const c = collide === undefined ? COLLIDE[name] : collide;
      return B.dress(name, px, pz, this.yaw + ry, { y: y + h, scale, rx, rz, color, collide: c ?? null });
    },
  });

  // Nächster freier Platz um einen Wunschpunkt, auf der Spirale nach außen
  const freeAt = (x, z, y, r) => !col.query(x - r, z - r, x + r, z + r).some((s) => s.minY < y + 2.2 && s.maxY > y + 0.25);
  const spot = (x, z, r, clear = () => true) => {
    for (let d = 0; d < 24; d += 2) {
      for (let k = 0; k < (d ? 10 : 1); k++) {
        const px = x + Math.cos(k * 0.628) * d, pz = z + Math.sin(k * 0.628) * d, y = B.ground(px, pz);
        if (clear(px, pz, r) && freeAt(px, pz, y, r)) return [px, pz, y];
      }
    }
    return null;
  };

  // Holzstapel aus einzelnen Scheiten, längs der Wand
  const logs = (f, a, o, len = 2.4, rows = 3) => {
    const [cx, cz] = f.pos(a, o);
    const ry = f.yaw + Math.PI / 2;
    for (let r = 0; r < rows; r++) {
      const n = Math.max(1, Math.round(len / 0.22) - r);
      for (let i = 0; i < n; i++) {
        const d = (i - (n - 1) / 2) * 0.22 + (r % 2) * 0.05;
        const px = cx + Math.cos(f.yaw) * d, pz = cz - Math.sin(f.yaw) * d;
        const yy = f.y + 0.11 + r * 0.19;
        const color = LOG + (Math.floor(rnd() * 3) * 0x080808);
        b.addPrim('cyl8', px, yy, pz, 0.105, 0.95, 0.105, color, { rx: Math.PI / 2, ry: ry + Math.PI / 2 + R(-0.04, 0.04), layer: 'det' });
      }
    }
    const [hx, hz] = f.pos(a, o);
    const hw = Math.abs(Math.cos(f.yaw)) * len / 2 + Math.abs(Math.sin(f.yaw)) * 0.5, hd = Math.abs(Math.sin(f.yaw)) * len / 2 + Math.abs(Math.cos(f.yaw)) * 0.5;
    col.addBox(hx - hw, f.y, hz - hd, hx + hw, f.y + rows * 0.2, hz + hd);
  };

  // Plane über einem Stapel
  const tarp = (f, a, o, w, d, h, color = 0x5a6048) => {
    const pts = [[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]].map(([u, v]) => {
      const [px, pz] = f.pos(a + u, o + v);
      return [px, f.y + h + (v < 0 ? 0.12 : -0.1) + (u > 0 ? 0.04 : -0.03), pz];
    });
    b.addQuad(pts[0], pts[1], pts[2], pts[3], color, { double: true, layer: 'det' });
  };

  const SCENES = {
    faesser(f) {
      f.put('d_wine_barrel', 0, 0.45, { ry: R(0, 6) });
      if (rnd() < 0.6) f.put('d_wine_barrel', 0.85, 0.42, { ry: R(0, 6) });
      if (rnd() < 0.5) f.put('d_bucket1', -0.62, 0.3, { ry: R(0, 6), scale: 0.9 });
    },
    kisten(f) {
      f.put('d_crate2', 0, 0.42, { ry: Math.PI / 2 + R(-0.15, 0.15), scale: 1.2 });
      f.put('d_crate1', 0.1, 0.42, { h: 0.55, ry: R(-0.3, 0.3), scale: 1.1, collide: null });
      if (rnd() < 0.5) f.put('d_basket', -1, 0.3, { ry: R(0, 6), scale: 1.5 });
    },
    werkstatt(f) {
      logs(f, 0, 0.55, 2.2, 3);
      f.put('d_ladder', 1.7, 0.1, { rx: -0.2, scale: 1.8, collide: null });
      f.put('d_axe', -1.45, 0.25, { ry: 0.5, rx: 0.3 });
      f.put('d_broom', -1.9, 0.08, { rx: -0.12, ry: R(-0.3, 0.3) });
      f.put('d_bucket2', -1.2, 0.7, { ry: R(0, 6), scale: 0.8 });
    },
    holz(f) {
      logs(f, 0, 0.55, 3, 4);
      if (rnd() < 0.5) tarp(f, 0, 0.55, 3.2, 1.3, 0.92);
    },
    garten(f) {
      f.put('d_pot', 0, 0.3, { scale: 1.4 });
      f.put('d_plant', 0, 0.3, { h: 0.16, scale: 2.4 });
      f.put('d_pot', 0.55, 0.28, { scale: 1.1 });
      f.put('d_watering', -0.6, 0.35, { ry: R(0, 6) });
      if (rnd() < 0.6) f.put('d_stool', 1.2, 0.5, { ry: R(0, 6) });
    },
    tisch(f) {
      f.put('d_table', 0, 2, { ry: R(-0.1, 0.1) });
      f.put('d_jug', 0.2, 2, { h: 0.75, ry: R(0, 6) });
      f.put('d_ceramic', -0.3, 2.2, { h: 0.75, scale: 0.5 });
      f.put('d_chair', 2, 2.2, { ry: 1.2 + R(-0.3, 0.3) });
    },
    sitz(f) {
      f.put('d_bench', 0, 0.3);
      f.put('d_pot', 1, 0.3);
      if (rnd() < 0.5) f.put('d_lantern', -0.85, 0.25);
    },
    werkzeug(f) {
      f.put('d_spade', 0, 0.08, { rx: -0.18, ry: R(-0.2, 0.2) });
      f.put('d_bucket1', 0.6, 0.32, { scale: 0.9 });
      f.put('d_cement', -0.8, 0.5, { ry: R(0, 3) });
      f.put('d_cement', -0.8, 0.5, { h: 0.17, ry: R(0, 3) });
    },
    muell(f) {
      f.put('d_trashbag', 0, 0.4, { ry: R(0, 6) });
      if (rnd() < 0.7) f.put('d_trashbag', 0.55, 0.45, { ry: R(0, 6), scale: 0.8 });
      f.put('d_can', -0.5, 0.7, { rx: 1.5, ry: R(0, 6), h: 0.07 });
      if (rnd() < 0.5) f.put('d_tyre', 1.1, 0.2, { rx: -0.25, ry: 1.57 });
    },
    technik(f) {
      f.put('d_generator', 0, 0.55, { ry: R(-0.2, 0.2) });
      f.put('d_jerrycan', 0.75, 0.35, { ry: R(-0.4, 0.4) });
      f.put('d_jerrycan', 1.1, 0.4, { ry: R(-0.4, 0.4) });
      f.put('d_can', -0.7, 0.4);
    },
    regal(f) {
      f.put('d_rack', 0, 0.35);
      f.put('d_propane', 1, 0.3);
      f.put('d_jerrycan', -0.9, 0.3, { ry: R(-0.4, 0.4) });
    },
    wagen(f) {
      f.put('d_toolcart', 0, 0.5, { ry: R(-0.3, 0.3) });
      f.put('d_rim', 1.1, 0.12, { rx: -0.2, ry: 1.57 });
      f.put('d_tyre', 1.1, 0.5, { h: 0.0, rx: 1.57, ry: 0, scale: 1 });
    },
    hafen(f) {
      f.put('d_wine_barrel', 0, 0.45, { ry: R(0, 6) });
      f.put('d_crate2', 1, 0.42, { ry: Math.PI / 2, scale: 1.2 });
      f.put('d_lifebuoy', -0.7, 0.05, { h: 0.9, ry: 0, scale: 0.8, rx: -0.1 });
      f.put('d_basket', -0.8, 0.6, { scale: 1.6, ry: R(0, 6) });
    },
    stapel(f) {
      f.put('d_mil_crate', 0, 0.6, { ry: R(-0.08, 0.08) });
      f.put('d_mil_crate', 0.1, 0.6, { h: 0.3, ry: R(-0.15, 0.15), collide: null });
      f.put('d_crate2', -0.2, 0.6, { h: 0.6, ry: Math.PI / 2 + R(-0.3, 0.3), scale: 1.1, collide: null });
      f.put('d_crate2', 1.5, 0.5, { ry: Math.PI / 2 + R(-0.2, 0.2), scale: 1.3 });
      f.put('d_wine_barrel', -1.5, 0.45, { ry: R(0, 6) });
      if (rnd() < 0.5) f.put('d_basket', 1.5, 0.5, { h: 0.6, ry: R(0, 6), scale: 1.5 });
    },
    tonne(f) {
      f.put('d_trashcan', 0, 0.5, { ry: R(-0.1, 0.1) });
      f.put('d_trashbag', 1.3, 0.45, { ry: R(0, 6) });
      f.put('d_trashbag', -1.25, 0.4, { ry: R(0, 6), scale: 0.85 });
      f.put('d_can', 1, 1, { rx: 1.5, ry: R(0, 6), h: 0.07 });
    },
    kompost(f) {
      f.put('d_compost', 0, 0.55, { ry: R(-0.1, 0.1) });
      f.put('d_spade', 1.2, 0.08, { rx: -0.18, ry: R(-0.2, 0.2) });
      f.put('d_bucket2', -1.25, 0.4, { ry: R(0, 6), scale: 0.8 });
    },
    netze(f) {
      // zum Trocknen aufgehängtes Netz zwischen zwei Pfählen
      const [ax, az] = f.pos(-1.6, 0.25), [cx, cz] = f.pos(1.6, 0.25);
      for (const [px, pz] of [[ax, az], [cx, cz]]) {
        b.box(px, f.y, pz, 0.1, 2.3, 0.1, WOOD_DARK, { layer: 'det' });
        col.addCyl(px, pz, 0.08, f.y, f.y + 2.3);
      }
      // Maschen aus dünnen Schnüren, leicht durchhängend
      const along = Math.abs(cx - ax) > Math.abs(cz - az);
      const w = along ? Math.abs(cx - ax) : Math.abs(cz - az), mx = (ax + cx) / 2, mz = (az + cz) / 2;
      for (let i = 0; i <= 7; i++) {
        const yy = f.y + 0.75 + i * 0.2;
        b.box(mx, yy - 0.08, mz, along ? w : 0.015, 0.015, along ? 0.015 : w, 0x8a8468, { layer: 'det' });
      }
      for (let i = 1; i < 16; i++) {
        const t = i / 16 - 0.5;
        const sag = (0.25 - t * t) * 0.5;
        b.box(mx + (along ? t * w : 0), f.y + 0.67 - sag, mz + (along ? 0 : t * w), 0.015, 1.4 + sag, 0.015, 0x8a8468, { layer: 'det' });
      }
      f.put('d_buoy', 2.3, 0.7, { scale: 0.42, collide: null });
    },
  };

  // Lager: Feuerstelle, Sitzplätze, Vorräte. So ein Platz erzählt, dass hier jemand war.
  const camp = (x, z, y) => {
    const f = frame(x, z, y, 1, 0, 0, 1);
    f.put('d_firepit', 0, 0, { ry: R(0, 6) });
    const n = 2 + Math.floor(rnd() * 3);
    for (let i = 0; i < n; i++) {
      const a = (i / n) * 6.28 + R(-0.3, 0.3);
      f.put(rnd() < 0.5 ? 'd_stool' : 'd_crate1', Math.cos(a) * 1.8, Math.sin(a) * 1.8, { ry: R(0, 6), scale: 1 });
    }
    f.put('d_mil_crate', 2.8, -1.6, { ry: R(-0.4, 0.4), collide: 'box' });
    f.put('d_lantern', 2.4, -1.3, { h: 0.3 });
    f.put('d_jerrycan', -2.6, -1.3, { ry: R(0, 6) });
    // Schlafplatz: Plane auf dem Boden, Sonnensegel darüber
    const tw = 2.2, th = 1.6;
    b.addQuad([x - 3.4, y + 0.04, z + 1], [x - 3.4 + tw, y + 0.04, z + 1], [x - 3.4 + tw, y + 0.04, z + 1 + th], [x - 3.4, y + 0.04, z + 1 + th], pick(CLOTH), { double: true, layer: 'det' });
    b.addQuad([x + 1.2, y + 1.8, z + 1], [x + 4, y + 1.8, z + 1], [x + 4, y + 1.1, z + 3.2], [x + 1.2, y + 1.1, z + 3.2], 0x5a6048, { double: true, layer: 'det' });
    for (const [px, pz, h] of [[x + 1.2, z + 1, 1.8], [x + 4, z + 1, 1.8], [x + 1.2, z + 3.2, 1.1], [x + 4, z + 3.2, 1.1]]) b.box(px, y, pz, 0.07, h, 0.07, WOOD_DARK, { layer: 'det' });
  };

  // Barrikade: Hindernis mit Lücke, gibt Deckung im Kampf
  const barricade = (x, z, y, alongX) => {
    const f = alongX ? frame(x, z, y, 1, 0, 0, 1) : frame(x, z, y, 0, 1, 1, 0);
    f.put('d_barrier', -1.1, 0, { ry: Math.PI / 2 * 0 + R(-0.08, 0.08), collide: 'box' });
    f.put('d_barrier', 0.65, 0.1, { ry: R(-0.1, 0.1), collide: 'box' });
    f.put('d_crate2', 2.2, 0.15, { ry: Math.PI / 2 + R(-0.2, 0.2), scale: 1.3, collide: 'box' });
    f.put('d_wine_barrel', -2.6, 0.2, { ry: R(0, 6) });
    f.put('d_tyre', 3.3, 0.45, { rx: 1.57 });
    for (let i = 0; i < 3; i++) {
      const [px, pz] = f.pos(R(-2.5, 2.5), R(0.5, 0.9));
      b.box(px, y + 0.03, pz, 2.2, 0.05, 0.22, WOOD, { ry: f.yaw + Math.PI / 2 + R(-0.4, 0.4), layer: 'det' });
    }
  };

  return { rnd, R, pick, has, frame, SCENES, camp, barricade, logs, freeAt, spot };
}

// ---------------------------------------------------------------- Stadt
const STREET_SCENES = {
  markt: ['faesser', 'kisten', 'stapel', 'stapel', 'werkzeug', 'sitz', 'tonne', 'garten', 'holz'],
  wald: ['holz', 'holz', 'werkstatt', 'garten', 'faesser', 'kompost', 'stapel'],
  weiss: ['garten', 'garten', 'sitz', 'kisten', 'stapel', 'kompost', 'faesser'],
  turm: ['muell', 'tonne', 'tonne', 'technik', 'regal', 'wagen', 'stapel'],
  hafen: ['hafen', 'hafen', 'netze', 'stapel', 'stapel', 'faesser', 'tonne'],
};
const YARD_SCENES = {
  markt: ['werkstatt', 'holz', 'tisch', 'kisten', 'garten'],
  wald: ['holz', 'werkstatt', 'tisch', 'garten'],
  weiss: ['tisch', 'garten', 'werkstatt', 'sitz'],
  turm: ['technik', 'regal', 'wagen', 'muell'],
  hafen: ['netze', 'hafen', 'kisten', 'holz'],
};
const OUT = { s: [0, 1], n: [0, -1], e: [1, 0], w: [-1, 0] };

export function dressCity({ B, lots, layout, kit, occ, camps, barricades, cannons }) {
  if (!B.assets?.has('d_crate1')) return { scenes: 0 };
  const D = makeDresser(B, 909);
  const { rnd, R, pick, frame, SCENES } = D;
  const { freeAt } = D;
  const distOf = (x, z) => (Math.abs(x) > 205 && z < 205 ? (x < 0 ? 'weiss' : 'turm') : z > 205 ? 'hafen' : z < -2 ? 'wald' : 'markt');
  let scenes = 0, lines = 0, banners = 0, scaffolds = 0;
  const samples = {}; // je Szene ein Beispiel, für die Prüfbilder
  const note = (kind, x, z, y, ox, oz) => { (samples[kind] ||= []).length < 3 && samples[kind].push([x, z, y, ox, oz]); };
  const clearOf = (x, z, r) => [[0, 0], [r, 0], [-r, 0], [0, r], [0, -r]].every(([u, v]) => occ.at(x + u, z + v) === -1);
  const spot = (x, z, r) => D.spot(x, z, r, clearOf);

  // 1) An den Hauswänden, nie vor Türen und nur wenn die Gasse breit genug bleibt
  for (const l of lots) {
    if (l.kind !== 'house' || !l.doors) continue;
    for (const side of ['n', 's', 'e', 'w']) {
      const fixed = side === 's' ? l.z1 : side === 'n' ? l.z0 : side === 'e' ? l.x1 : l.x0;
      const a0 = side === 'n' || side === 's' ? l.x0 : l.z0, a1 = side === 'n' || side === 's' ? l.x1 : l.z1;
      if (a1 - a0 < 5) continue;
      const [ox, oz] = OUT[side];
      const along = side === 'n' || side === 's';
      for (let tries = Math.ceil((a1 - a0) / 8); tries > 0; tries--) {
        if (rnd() > 0.62) continue;
        const a = R(a0 + 2.3, a1 - 2.3);
        if (l.doors.some((d) => d.side === side && Math.abs(d.a - a) < 3.2)) continue;
        const wx = along ? a : fixed, wz = along ? fixed : a;
        const y = B.ground(wx + ox * 0.8, wz + oz * 0.8);
        if (Math.abs(y - l.level) > 1.2) continue;
        // Standort und seine Enden frei, 3,2 m weiter muss man noch vorbeikommen
        const ax = along ? 2 : 0, az = along ? 0 : 2;
        const ok = [[0, 0.8], [1, 0.8], [-1, 0.8], [0, 3.2]].every(([t, o]) => {
          const px = wx + ax * t + ox * o, pz = wz + az * t + oz * o;
          return occ.at(px, pz) === -1 && freeAt(px, pz, y, o > 1 ? 0.5 : 0.55);
        });
        if (!ok) continue;
        const f = frame(wx, wz, y, Math.abs(oz), Math.abs(ox), ox, oz);
        const kind = pick(STREET_SCENES[distOf(wx, wz)]);
        SCENES[kind](f);
        note(kind, wx, wz, y, ox, oz);
        scenes++;
      }
    }
  }

  // 2) In den Höfen: größere Szene in einer Ecke
  for (const yard of layout.yards) {
    const w = yard.x1 - yard.x0, d = yard.z1 - yard.z0;
    if (w < 6 || d < 6 || yard.passage || rnd() > 0.85) continue;
    const cx = (yard.x0 + yard.x1) / 2, cz = (yard.z0 + yard.z1) / 2;
    const south = rnd() < 0.5;
    const x = cx + R(-w * 0.2, w * 0.2), z = south ? yard.z1 - 0.1 : yard.z0 + 0.1;
    const y = B.ground(x, z);
    if (!freeAt(x, z + (south ? -1 : 1), y, 1.2)) continue;
    const f = frame(x, z, y, 1, 0, 0, south ? -1 : 1);
    const kind = pick(YARD_SCENES[distOf(cx, cz)]);
    SCENES[kind](f);
    note(kind, x, z, y, 0, south ? -1 : 1);
    scenes++;
  }

  // 3) Wäscheleinen quer über schmale Gassen
  const b = B.b;
  for (const a of layout.alleys) {
    const w = a.x1 - a.x0, d = a.z1 - a.z0;
    if (Math.min(w, d) > 5.5 || Math.max(w, d) < 6 || rnd() > 0.75) continue;
    const alongX = w > d; // Gasse verläuft entlang X, Leine quer dazu
    const m = (alongX ? a.x0 : a.z0) + R(1.5, Math.max(w, d) - 1.5);
    const p0 = alongX ? [m, a.z0] : [a.x0, m], p1 = alongX ? [m, a.z1] : [a.x1, m];
    const id0 = occ.at(p0[0] - (alongX ? 0 : 0.6), p0[1] - (alongX ? 0.6 : 0)), id1 = occ.at(p1[0] + (alongX ? 0 : 0.6), p1[1] + (alongX ? 0.6 : 0));
    const l0 = lots[id0], l1 = lots[id1];
    if (!l0 || !l1 || l0.kind !== 'house' || l1.kind !== 'house' || l0.level !== l1.level || l0.floors < 2 || l1.floors < 2) continue;
    const y = l0.level + 5.2 + R(0, 0.8);
    const len = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
    const cx = (p0[0] + p1[0]) / 2, cz = (p0[1] + p1[1]) / 2;
    b.box(cx, y, cz, alongX ? 0.025 : len, 0.025, alongX ? len : 0.025, 0x8a7a5a, { layer: 'det' });
    const n = Math.floor(len / 0.8);
    for (let i = 0; i < n; i++) {
      if (rnd() < 0.35) continue;
      const t = (i + 0.5) / n, cw = R(0.35, 0.7), ch = R(0.5, 1.05);
      const px = p0[0] + (p1[0] - p0[0]) * t, pz = p0[1] + (p1[1] - p0[1]) * t;
      const sag = Math.sin(t * Math.PI) * 0.1, sway = R(-0.05, 0.05), col2 = pick(CLOTH);
      const A = alongX ? [px + sway, y - sag - ch, pz - cw / 2] : [px - cw / 2, y - sag - ch, pz + sway];
      const C = alongX ? [px, y - sag, pz + cw / 2] : [px + cw / 2, y - sag, pz];
      const B2 = alongX ? [px + sway, y - sag - ch, pz + cw / 2] : [px + cw / 2, y - sag - ch, pz + sway];
      const D2 = alongX ? [px, y - sag, pz - cw / 2] : [px - cw / 2, y - sag, pz];
      b.addQuad(A, B2, C, D2, col2, { double: true, layer: 'det' });
    }
    note('leine', cx, cz, l0.level, alongX ? 1 : 0, alongX ? 0 : 1);
    lines++;
  }

  // 4) Banner an den Häusern der Hauptstraßen, Stangen und Fahnen je Viertel
  for (const l of lots) {
    if (l.kind !== 'house' || !l.doors || l.floors < 2 || rnd() > 0.2) continue;
    const side = pick(['n', 's', 'e', 'w']);
    const [ox, oz] = OUT[side];
    const along = side === 'n' || side === 's';
    const fixed = side === 's' ? l.z1 : side === 'n' ? l.z0 : side === 'e' ? l.x1 : l.x0;
    const a = (along ? (l.x0 + l.x1) / 2 : (l.z0 + l.z1) / 2) + R(-1, 1);
    const wx = along ? a : fixed, wz = along ? fixed : a;
    if (occ.at(wx + ox * 1.5, wz + oz * 1.5) !== -1) continue;
    const pal = BANNER[distOf(wx, wz)], c1 = pal[0], c2 = pal[1];
    const yt = l.level + l.floors * 3 - 0.3, h = 2.4, bw = 0.42;
    const off = 0.5;
    const px = wx + ox * off, pz = wz + oz * off;
    const ax = along ? 1 : 0, az = along ? 0 : 1;
    b.box(wx + ox * 0.3, yt, wz + oz * 0.3, along ? 0.05 : 0.6, 0.05, along ? 0.6 : 0.05, 0x2b2724, { mat: 'metal', layer: 'det' });
    b.addQuad([px - ax * bw, yt - h, pz - az * bw], [px + ax * bw, yt - h, pz + az * bw], [px + ax * bw, yt, pz + az * bw], [px - ax * bw, yt, pz - az * bw], c1, { double: true, layer: 'det' });
    b.addQuad([px - ax * bw * 0.6, yt - 1.1, pz - az * bw * 0.6 + oz * 0.01 + 0], [px + ax * bw * 0.6, yt - 1.1, pz + az * bw * 0.6], [px + ax * bw * 0.6, yt - 0.6, pz + az * bw * 0.6], [px - ax * bw * 0.6, yt - 0.6, pz - az * bw * 0.6], c2, { double: true, layer: 'det', shade: false });
    note('banner', wx, wz, l.level, ox, oz);
    banners++;
  }

  // 5) Baugerüste an einzelnen Fassaden
  for (const l of lots) {
    if (l.kind !== 'house' || !l.doors || l.floors < 2 || rnd() > 0.07) continue;
    // Erste Seite ohne Tür mit genug Platz davor
    const side = ['n', 's', 'e', 'w'].find((k) => {
      if (l.doors.some((d) => d.side === k)) return false;
      const [kx, kz] = OUT[k], al = k === 'n' || k === 's';
      const fx = k === 's' ? l.z1 : k === 'n' ? l.z0 : k === 'e' ? l.x1 : l.x0, m = al ? (l.x0 + l.x1) / 2 : (l.z0 + l.z1) / 2;
      const qx = al ? m : fx, qz = al ? fx : m;
      return occ.at(qx + kx * 2.8, qz + kz * 2.8) === -1 && freeAt(qx + kx * 1.4, qz + kz * 1.4, l.level, 0.5);
    });
    if (!side) continue;
    const [ox, oz] = OUT[side];
    const along = side === 'n' || side === 's';
    const fixed = side === 's' ? l.z1 : side === 'n' ? l.z0 : side === 'e' ? l.x1 : l.x0;
    const a0 = along ? l.x0 : l.z0, a1 = along ? l.x1 : l.z1, mid = (a0 + a1) / 2;
    const wx = along ? mid : fixed, wz = along ? fixed : mid;
    const y = l.level, H = Math.min(l.floors, 3) * 3, len = Math.min(a1 - a0 - 0.6, 7);
    const P = (t, o) => (along ? [mid + t, fixed + oz * o] : [fixed + ox * o, mid + t]);
    for (let t = -len / 2; t <= len / 2 + 0.01; t += len / Math.max(2, Math.round(len / 2.4))) {
      for (const o of [0.18, 1.18]) {
        const [px, pz] = P(t, o);
        b.box(px, y, pz, 0.08, H, 0.08, WOOD, { layer: 'det' });
      }
    }
    for (let yy = y + 2.4; yy < y + H - 0.2; yy += 2.4) {
      const [cx, cz] = P(0, 0.68);
      b.box(cx, yy, cz, along ? len : 1, 0.05, along ? 1 : len, WOOD, { layer: 'det' });
      const [rx, rz] = P(0, 1.18);
      b.box(rx, yy + 0.95, rz, along ? len : 0.05, 0.05, along ? 0.05 : len, WOOD_DARK, { layer: 'det' });
    }
    const [lx, lz] = P(-len / 2 + 0.5, 1.5);
    B.dress('d_ladder', lx, lz, Math.atan2(ox, oz), { y, rx: -0.14, scale: 2, collide: null });
    note('geruest', wx, wz, y, ox, oz);
    scaffolds++;
  }

  // 6) Lager, Barrikaden, Kanonen an festen Orten
  let placed = 0;
  for (const [x, z] of camps) {
    const p = spot(x, z, 4.2);
    if (!p) continue;
    D.camp(...p);
    note('lager', p[0], p[1], p[2], 0, 1);
    placed++;
  }
  for (const [x, z, alongX] of barricades) {
    const p = spot(x, z, 1.8);
    // Nur wo links und rechts ein Weg bleibt
    if (!p || !clearOf(p[0] + (alongX ? 5.6 : 0), p[1] + (alongX ? 0 : 5.6), 0.6) || !clearOf(p[0] - (alongX ? 5.6 : 0), p[1] - (alongX ? 0 : 5.6), 0.6)) continue;
    D.barricade(...p, alongX);
    note('barrikade', p[0], p[1], p[2], alongX ? 0 : 1, alongX ? 1 : 0);
    placed++;
  }
  for (const [x, z, ry] of cannons) {
    const p = spot(x, z, 1.5);
    if (!p) continue;
    B.dress('d_cannon', p[0], p[1], ry, { y: p[2], collide: 'box' });
    note('kanone', p[0], p[1], p[2], 0, -1);
    placed++;
  }
  B.dressSamples = samples;
  return { scenes, lines, banners, scaffolds, placed };
}

// ---------------------------------------------------------------- Damm
export function dressDamm(B) {
  if (!B.assets?.has('d_crate1')) return;
  const D = makeDresser(B, 1212);
  const { frame, SCENES, camp, barricade, spot } = D;
  const samples = (B.dressSamples = {});
  const note = (kind, x, z, y, ox, oz) => { (samples[kind] ||= []).push([x, z, y, ox, oz]); };
  for (const [x, z] of [[-60, 60], [120, 110], [-210, 40], [30, 130], [-120, -60], [205, 60]]) {
    const p = spot(x, z, 4.2);
    if (p) { camp(...p); note('lager', p[0], p[1], p[2], 0, 1); }
  }
  for (const [x, z, alongX] of [[60, 8, true], [-60, 30, false], [150, 62, true], [-150, 60, true], [118, 130, false]]) {
    const p = spot(x, z, 1.8);
    if (p) { barricade(...p, alongX); note('barrikade', p[0], p[1], p[2], alongX ? 0 : 1, alongX ? 1 : 0); }
  }
  // Vor Hallen und Häusern: [Szene, x, z, Blick entlang z?]
  const list = [
    ['technik', 30, -33, 0], ['regal', 86, -33, 0], ['wagen', 95, -46, 1], ['muell', 62, -33, 0], ['technik', 141, 34, 0], ['werkzeug', 135, 12, 1],
    ['muell', -170, 15, 1], ['technik', -135, 38, 0], ['regal', -60, 122, 0], ['muell', 44, 152, 0], ['wagen', 96, 146, 1], ['muell', 62, 172, 0], ['technik', -222, -148, 0],
  ];
  for (const [name, x, z, dir] of list) {
    const p = spot(x, z, 1.6);
    if (!p) continue;
    SCENES[name](dir ? frame(p[0], p[1], p[2], 0, 1, 1, 0) : frame(p[0], p[1], p[2], 1, 0, 0, 1));
    note(name, p[0], p[1], p[2], dir ? 1 : 0, dir ? 0 : 1);
  }
}
