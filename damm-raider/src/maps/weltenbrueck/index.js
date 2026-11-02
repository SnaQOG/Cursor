// Karte "Weltenbrück": eine Fantasy-Stadt aus deutscher Architektur.
import * as THREE from 'three';
import {
  CITY, TERRACES, STAIRS, STAIR_RUN, PLAZA, DOM, RATHAUS, STREETS, OUTER_STREETS, WHITE_TOWER, ENERGY_TOWER, GIANT_TREES, FACES, STEIG,
  EXTRACTS, POIS, PLAYER_SPAWNS, levelAt, stairRamp, generateLayout, Occupancy,
} from './layout.js';
import { buildCity } from './build.js';
import {
  makeCobbleSet, makeBrickSet, makePlankSet, makeTileSet, makeSlateSet, makeAshlarSet, makeShingleSet, makePlasterSet, makeTimberSet, makeGrassSet, makeClothSet,
} from '../../textures.js';
import { Noise2D, mulberry32, clamp, smoothstep, lerp } from '../../util.js';

let LAYOUT = null;
const layout = () => (LAYOUT ||= generateLayout());

// Bodenart im 2-m-Raster (für Farben): 0 Pflaster, 1 Platz, 2 Hof/Wiese, 3 Garten, 4 Hauptstraße
let GROUND = null;
function groundGrid() {
  if (GROUND) return GROUND;
  const L = layout();
  const occ = new Occupancy(CITY.OUTER + 2, 2);
  occ.grid.fill(0);
  for (const s of [...STREETS, ...OUTER_STREETS]) if (s.main) occ.mark(s, 4);
  for (const y of L.yards) occ.mark(y, y.garden ? 3 : y.passage ? 0 : 2);
  for (const q of L.squares) occ.mark(q, 1);
  occ.mark(PLAZA, 1);
  occ.mark(DOM.area, 1);
  for (const t of GIANT_TREES) occ.mark({ x0: t.x - t.r, x1: t.x + t.r, z0: t.z - t.r, z1: t.z + t.r }, 3);
  GROUND = occ;
  return occ;
}

const C = {
  cobble: new THREE.Color(0xc4baa9), cobbleDark: new THREE.Color(0x9c9383), plaza: new THREE.Color(0xb3a993),
  grass: new THREE.Color(0x5d7337), grassDry: new THREE.Color(0x8a8a4e), garden: new THREE.Color(0x4f6a2f),
  moss: new THREE.Color(0x5a6a3a), rock: new THREE.Color(0xb9b2a5), rockDark: new THREE.Color(0x938c80),
  forest: new THREE.Color(0x3e4f2c), pale: new THREE.Color(0xc9c1ad), sand: new THREE.Color(0xa8986f), seabed: new THREE.Color(0x4f5a52),
  main: new THREE.Color(0xaea594),
};

const terrain = {
  half: CITY.HALF,
  res: CITY.RES,
  seed: 1871,
  uvScale: 3,

  rawHeight(T, x, z) {
    const n = T.noise;
    const ax = Math.abs(x);
    if (ax <= 200 && z <= 200 && z >= CITY.CLIFF_Z) {
      const r = stairRamp(x, z);
      if (r != null) return r;
      return levelAt(z);
    }
    // Hügel hinter den Außenmauern
    const hill = CITY.GROUND + smoothstep(CITY.OUTER + 3, CITY.HALF, ax) * 70 + smoothstep(CITY.OUTER + 3, CITY.OUTER + 30, ax) * n.fbm(x * 0.015, z * 0.015, 3) * 8;
    if (z < CITY.CLIFF_Z) {
      // Gründerfelsen und Plateau mit Wald; vor den Vierteln reicht die Wand bis zum tieferen Boden
      const base = lerp(18, CITY.GROUND, smoothstep(200, 207, ax));
      const k = smoothstep(CITY.CLIFF_Z, CITY.CLIFF_Z - 20, z);
      const rough = n.fbm(x * 0.06, z * 0.06, 3) * 2.5 * Math.min(1, k * 3) * (1 - k * 0.6);
      return Math.max(hill, base + k * (CITY.CLIFF_TOP - base) + rough + n.fbm(x * 0.01, z * 0.01, 2) * 4 * k);
    }
    if (z <= CITY.QUAY - 8) return hill;
    // Kaikante zum Meer, im Westen ein flacher Strand (dort kommt man schwimmend wieder an Land)
    const quay = z <= CITY.QUAY ? hill : lerp(hill, -3, smoothstep(CITY.QUAY, CITY.QUAY + 5, z));
    const sand = CITY.GROUND - (z - (CITY.QUAY - 8)) * 0.125;
    return lerp(quay, sand, smoothstep(-135, -125, x) * (1 - smoothstep(-55, -45, x)));
  },

  waterAt(T, x, z) {
    if (z > CITY.QUAY) return { level: 0.6, toxic: false };
    // Marktbrunnen
    if (Math.hypot(x, z - 96) < 4.6) return { level: 3.5, toxic: false };
    return null;
  },

  groundColor(T, x, z, h, sl, c) {
    const n = T.noise;
    const nv = n.fbm(x * 0.05 + 3, z * 0.05 - 9, 3);
    const inside = Math.abs(x) <= CITY.OUTER && z <= CITY.QUAY && z >= CITY.CLIFF_Z;
    if (z > CITY.QUAY - 8 && x > -135 && x < -45) {
      c.copy(C.sand).lerp(C.seabed, smoothstep(CITY.QUAY + 4, CITY.QUAY + 24, z));
    } else if (inside) {
      const g = groundGrid().at(x, z);
      if (g === 1) c.copy(C.plaza);
      else if (g === 2) c.copy(C.grass).lerp(C.grassDry, clamp(0.4 + nv, 0, 1));
      else if (g === 3) c.copy(C.garden).lerp(C.grass, clamp(0.5 + nv, 0, 1));
      else if (g === 4) c.copy(C.main);
      else c.copy(C.cobble).lerp(C.cobbleDark, clamp(0.5 + nv, 0, 1));
      // Moos und Unkraut zwischen den Steinen (verlassene Stadt)
      if (g !== 2 && g !== 3) {
        const m = smoothstep(0.05, 0.45, n.fbm(x * 0.09 - 7, z * 0.09 + 3, 3));
        if (m > 0) c.lerp(C.moss, m * 0.55);
      }
      // Weiße Stadt hell, Turmstadt dunkel
      if (x < -205 && z < 205) c.lerp(C.pale, 0.4);
      else if (x > 205 && z < 205) c.lerp(C.cobbleDark, 0.5);
    } else if (z < CITY.CLIFF_Z) {
      const k = smoothstep(CITY.CLIFF_Z, CITY.CLIFF_Z - 4, z) * (1 - smoothstep(CITY.CLIFF_Z - 18, CITY.CLIFF_Z - 24, z));
      c.copy(C.forest).lerp(C.grass, clamp(0.4 + nv, 0, 1) * 0.5);
      if (k > 0 || sl > 0.8) c.lerp(nv > 0 ? C.rock : C.rockDark, Math.max(k, smoothstep(0.8, 1.4, sl)));
    } else if (z > CITY.QUAY) {
      c.copy(C.seabed);
    } else c.copy(C.forest).lerp(C.grass, clamp(0.5 + nv, 0, 1));
    c.multiplyScalar(1.1 + n.noise(x * 0.4, z * 0.4) * 0.12);
  },

  mapColor() { return [40, 44, 46]; },

  // Anteil Fels-Textur: Gründerfelsen und steile Hänge
  rockMix(T, x, z, h, sl) {
    if (z > CITY.CLIFF_Z + 1) return 0;
    return clamp(Math.max(smoothstep(CITY.CLIFF_Z + 1, CITY.CLIFF_Z - 1, z) * (1 - smoothstep(CITY.CLIFF_Z - 18, CITY.CLIFF_Z - 26, z)), smoothstep(0.7, 1.3, sl)), 0, 1);
  },

  // Anteil Gras-Textur am Boden (0 = Pflaster)
  grassMix(T, x, z) {
    if (z < CITY.CLIFF_Z - 16) return 1;
    if (z < CITY.CLIFF_Z) return 0;
    if (z > CITY.QUAY - 8) return 0;
    if (Math.abs(x) > CITY.OUTER) return 1;
    const g = groundGrid().at(x, z);
    return g === 2 || g === 3 ? 1 : 0;
  },
};

function materials(world) {
  const mk = (set, roughness = 0.88, metalness = 0.03, ns = 0.9) => new THREE.MeshStandardMaterial({
    vertexColors: true, map: set.map, normalMap: set.normal, normalScale: new THREE.Vector2(ns, ns), roughness, metalness,
  });
  // Fotografische CC0-Texturen; ohne sie gelten die prozeduralen
  const tex = world.assets?.tex || {};
  const photo = (t, roughness, ns, rot = 0) => {
    const map = rot ? t.map.clone() : t.map, normal = rot ? t.normal.clone() : t.normal;
    if (rot) for (const x of [map, normal]) { x.rotation = rot; x.needsUpdate = true; }
    return new THREE.MeshStandardMaterial({ vertexColors: true, map, normalMap: normal, normalScale: new THREE.Vector2(ns, ns), roughness, metalness: 0 });
  };
  // Ziegel in echter Größe: eine Kachel sind rund 6,5 m Dach
  if (tex.rooftile) for (const x of [tex.rooftile.map, tex.rooftile.normal]) x.repeat.set(0.6, 0.6);
  const base = mk(makePlasterSet(), 0.93, 0, 0.7), tile = mk(makeTileSet(), 0.8, 0.02, 1.0);
  // Welt-UV: eine Kachel sind 4 m, die Fotos zeigen weniger
  const scaled = (t, rep, roughness, ns) => {
    if (!t) return null;
    for (const x of [t.map, t.normal]) x.repeat.set(rep, rep);
    return photo(t, roughness, ns);
  };
  const cobble2 = () => {
    if (!tex.cobble) return base;
    const c = { map: tex.cobble.map.clone(), normal: tex.cobble.normal.clone() };
    for (const x of [c.map, c.normal]) { x.repeat.set(4 / 3, 4 / 3); x.needsUpdate = true; }
    return photo(c, 0.96, 0.9);
  };
  return {
    wplaster: tex.plaster ? photo(tex.plaster, 0.95, 0.9) : base,
    wtile: tex.rooftile ? photo(tex.rooftile, 0.82, 1.0) : tile,
    wtile90: tex.rooftile ? photo(tex.rooftile, 0.82, 1.0, Math.PI / 2) : tile,
    world: base,
    brick: scaled(tex.brick, 2, 0.9, 1.0) || mk(makeBrickSet(), 0.9),
    stone: scaled(tex.ashlar, 1.7, 0.92, 1.0) || mk(makeAshlarSet(), 0.92),
    wood: scaled(tex.planks, 1.5, 0.84, 0.9) || mk(makePlankSet(), 0.82),
    timber: scaled(tex.beam, 2.5, 0.82, 0.9) || mk(makeTimberSet(), 0.8, 0, 0.8),
    cobble2: cobble2(),
    puddle: new THREE.MeshStandardMaterial({ vertexColors: true, transparent: true, opacity: 0.34, roughness: 0.1, metalness: 0.3, depthWrite: false }),
    shingle: mk(makeShingleSet(), 0.85),
    tile,
    slate: mk(makeSlateSet(), 0.62, 0.06),
    cloth: mk(makeClothSet(), 0.97, 0, 0.6),
  };
}

// Boden: Pflaster, in Höfen, Gärten und im Wald Gras
function groundTexture(world) {
  const c = world.assets?.tex?.cobble || makeCobbleSet(), g = makeGrassSet();
  return { map: c.map, normal: c.normal, grass: g.map, grassNormal: g.normal, rock: world.assets?.tex?.cliff?.map };
}

// Meer, Brunnenwasser
function decorate(world) {
  const normalTex = world.waterNormal;
  const mk = (color, opacity, rep) => {
    const t = normalTex.clone();
    t.needsUpdate = true;
    t.repeat.set(rep, rep);
    return new THREE.MeshStandardMaterial({ color, roughness: 0.1, metalness: 0.2, transparent: true, opacity, normalMap: t, normalScale: new THREE.Vector2(0.5, 0.5), depthWrite: false });
  };
  const sea = mk(0x2f5a66, 0.92, 40);
  const m = new THREE.Mesh(new THREE.PlaneGeometry(1800, 500), sea);
  m.rotation.x = -Math.PI / 2;
  m.position.set(0, 0.6, CITY.QUAY + 250);
  world.scene.add(m);
  const fountain = mk(0x3d6c72, 0.85, 3);
  const f = new THREE.Mesh(new THREE.CircleGeometry(4.6, 32), fountain);
  f.rotation.x = -Math.PI / 2;
  f.position.set(0, 3.5, 96);
  world.scene.add(f);
  world.waterMats.push(sea, fountain);
  if (world.mapData.glowMats) world.glowPulse = world.mapData.glowMats;
}

function updateDecor(world) {
  // Energiekern des Erfinderturms pulsiert
  if (world.glowPulse) {
    const k = 0.75 + Math.sin(world.time * 2.2) * 0.25;
    for (const m of world.glowPulse) m.color.setScalar(k);
  }
}

// Bäume in Höfen und Gärten, Wald auf dem Plateau, Gras in Ritzen
function vegetation(world) {
  const rnd = mulberry32(77);
  const T = world.terrain;
  const L = layout();
  const n = new Noise2D(31);
  const col = world.col;
  const trees = [], bushes = [], rocks = [], grass = [];
  const linden = [], birches = [], flowers = [], weeds = [], meadow = [];
  const free = (x, z, r, h = 1) => !col.query(x - r, z - r, x + r, z + r).some((s) => s.maxY > T.height(x, z) + 0.3 && s.minY < T.height(x, z) + h);
  // Höfe, Gärten, Plätze
  const areas = [...L.yards.filter((y) => !y.passage), ...L.squares];
  for (const a of areas) {
    const w = a.x1 - a.x0, d = a.z1 - a.z0;
    const area = w * d;
    const nt = Math.floor(area / (a.court ? 70 : 110) * (0.6 + rnd()));
    for (let i = 0; i < nt; i++) {
      const x = a.x0 + 2 + rnd() * (w - 4), z = a.z0 + 2 + rnd() * (d - 4);
      // Baumkronen sind Bildflächen und dürfen nicht durch Hauswände ragen
      if (w < 6 || d < 6 || !free(x, z, 3.2, 8)) continue;
      const it = { x, z, y: T.height(x, z), s: 0.42 + rnd() * 0.3, pine: false, r: rnd() * 6 };
      const k = rnd();
      if (z < -2 && k < 0.45) birches.push(it);
      else if (z >= -2 && k < 0.55) linden.push(it);
      else { it.pine = k > 0.9; trees.push(it); }
    }
    // Blumen in Gärten und Innenhöfen
    if (a.garden || a.court) {
      const nf = Math.floor(area / 14 * (0.6 + rnd()));
      for (let i = 0; i < nf; i++) {
        const x = a.x0 + 0.6 + rnd() * (w - 1.2), z = a.z0 + 0.6 + rnd() * (d - 1.2);
        if (w < 2 || d < 2 || !free(x, z, 0.4, 0.6)) continue;
        flowers.push({ x, z, y: T.height(x, z), s: 0.7 + rnd() * 0.6, r: rnd() * 6 });
      }
    }
    const nb = Math.floor(area / 30 * (0.5 + rnd()));
    for (let i = 0; i < nb; i++) {
      const x = a.x0 + 0.8 + rnd() * (w - 1.6), z = a.z0 + 0.8 + rnd() * (d - 1.6);
      if (!free(x, z, 0.5, 1.2)) continue;
      bushes.push({ x, z, y: T.height(x, z), s: 0.5 + rnd() * 0.9, r: rnd() * 6 });
    }
  }
  // Wald auf dem Plateau und am Felsfuß
  for (let i = 0; i < 9000 && trees.length < 1900; i++) {
    const x = (rnd() * 2 - 1) * 396, z = -396 + rnd() * (396 + CITY.CLIFF_Z - 20);
    if (n.fbm(x * 0.02, z * 0.02, 2) < -0.25) continue;
    trees.push({ x, z, y: T.height(x, z), s: 0.9 + rnd() * 0.8, pine: rnd() < 0.8, r: rnd() * 6 });
  }
  // Hügel hinter den Außenmauern
  for (let i = 0; i < 900; i++) {
    const x = (rnd() < 0.5 ? -1 : 1) * (CITY.OUTER + 8 + rnd() * 95), z = CITY.CLIFF_Z + rnd() * (CITY.QUAY - CITY.CLIFF_Z);
    trees.push({ x, z, y: T.height(x, z), s: 0.9 + rnd() * 0.8, pine: rnd() < 0.7, r: rnd() * 6 });
  }
  // Felsbrocken am Gründerfelsen
  for (let i = 0; i < 260; i++) {
    const x = (rnd() * 2 - 1) * 396, z = CITY.CLIFF_Z - 2 - rnd() * 20;
    rocks.push({ x, z, y: T.height(x, z), s: 1.5 + rnd() * 3.5, r: rnd() * 6, t: rnd() * 6 });
  }
  // Gras zwischen Pflaster und in Höfen
  const maxGrass = { niedrig: 0, mittel: 14000, hoch: 28000 }[world.quality] ?? 14000;
  const gg = groundGrid();
  for (let i = 0; i < maxGrass * 8 && grass.length < maxGrass; i++) {
    const x = (rnd() * 2 - 1) * (CITY.OUTER - 2), z = CITY.CLIFF_Z + 2 + rnd() * (CITY.QUAY - 4 - CITY.CLIFF_Z - 2);
    const g = gg.at(x, z);
    const green = g === 2 || g === 3;
    if (!green && n.fbm(x * 0.09 - 7, z * 0.09 + 3, 3) < 0.15) continue;
    // auf Pflaster wächst Unkraut nur an geschützten Stellen: nah an Mauern, nie mitten auf der Straße
    if (!green && (g === 4 || !col.query(x - 1, z - 1, x + 1, z + 1).some((s) => s.maxY > T.height(x, z) + 1))) continue;
    if (!free(x, z, 0.3, 0.6)) continue;
    const tuft = { x, z, y: T.height(x, z), s: green ? 0.7 + rnd() * 0.6 : 0.3 + rnd() * 0.3, r: rnd() * 6 };
    if (!green && rnd() < 0.5) weeds.push(tuft);
    else if (green && rnd() < 0.24) meadow.push(tuft);
    else grass.push(tuft);
  }
  // Stämme von Linden und Birken
  for (const t of linden) col.addCyl(t.x, t.z, 0.35 * t.s, t.y - 1, t.y + 4 * t.s, { tree: true });
  for (const t of birches) col.addCyl(t.x, t.z, 0.18 * t.s, t.y - 1, t.y + 5 * t.s, { tree: true });
  const leaf = (it, c) => {
    const v = 0.82 + (it.r % 1) * 0.3;
    return c.setRGB(v * (0.94 + (it.r % 0.37) * 0.3), v, v * (0.85 + (it.r % 0.53) * 0.25));
  };
  const upright = (it, p, q, sc, e) => { p.set(it.x, it.y - 0.08, it.z); q.setFromEuler(e.set(0, it.r, 0)); sc.setScalar(it.s); };
  const extra = [
    { name: world.assets?.has('n_leaf0') ? 'n_leaf0' : 'v_linde', list: linden, place: upright, opts: { tint: leaf, wind: 0.0025 } },
    { name: world.assets?.has('n_leaf0') ? 'n_leaf0' : 'v_birke', list: birches, place: upright, opts: { tint: leaf, wind: 0.004 } },
    { name: 'v_flowers', list: flowers, place: upright, opts: { shadow: false, doubleSide: true, cull: true, wind: 0.25 } },
    { name: 'v_meadow', list: meadow, place: upright, opts: { shadow: false, doubleSide: true, cull: true, wind: 0.2 } },
    { name: 'v_weed', list: weeds, place: upright, opts: { shadow: false, doubleSide: true, tint: leaf, cull: true, wind: 0.12 } },
  ];
  return { trees, bushes, rocks, grass, extra };
}

// Übersichtskarte direkt aus dem Stadtplan
function drawMap(g, size, toPx, k) {
  const L = layout();
  const R = (r, fill) => { g.fillStyle = fill; g.fillRect(toPx(r.x0), toPx(r.z0), (r.x1 - r.x0) * k, (r.z1 - r.z0) * k); };
  R({ x0: -320, x1: 320, z0: -320, z1: 320 }, '#47553a');
  R({ x0: -320, x1: 320, z0: -320, z1: CITY.CLIFF_Z - 18 }, '#4c5a3c');
  R({ x0: -320, x1: 320, z0: CITY.CLIFF_Z - 18, z1: CITY.CLIFF_Z }, '#6c665c');
  R({ x0: -320, x1: 320, z0: CITY.QUAY, z1: 320 }, '#3d6670');
  R({ x0: -130, x1: -50, z0: CITY.QUAY - 8, z1: CITY.QUAY + 14 }, '#b7a67d');
  R({ x0: -CITY.OUTER, x1: -200, z0: CITY.CLIFF_Z, z1: 205 }, '#b9b3a2');
  R({ x0: 200, x1: CITY.OUTER, z0: CITY.CLIFF_Z, z1: 205 }, '#7e7f7b');
  R({ x0: -CITY.OUTER, x1: CITY.OUTER, z0: 205, z1: CITY.QUAY }, '#a49c8a');
  const tcol = { t0: '#a49c8a', t1: '#9f9884', t2: '#9a937e', t3: '#958f79' };
  for (const t of TERRACES) R({ x0: -197, x1: 197, z0: t.z0, z1: t.z1 }, tcol[t.id]);
  R({ x0: -197, x1: 197, z0: -2, z1: 197 }, '#a49c8a');
  for (const a of L.alleys) R(a, '#b5ac98');
  for (const s of [...STREETS, ...OUTER_STREETS]) R(s, '#c2b9a4');
  for (const y of L.yards) R(y, y.passage ? '#b5ac98' : '#7d8c5a');
  for (const q of L.squares) R(q, '#d3c9b2');
  R(PLAZA, '#ddd2ba');
  const STY = { fachwerk: '#9a5a42', stein: '#6f6a69', backstein: '#7f3f30', schwarzwald: '#5a4532', turm: '#7d8186', weiss: '#ece7da', hochhaus: '#4d545b', speicher: '#7a4634' };
  for (const l of L.lots) R({ x0: l.x0 + 0.3, x1: l.x1 - 0.3, z0: l.z0 + 0.3, z1: l.z1 - 0.3 }, l.kind === 'ruin' ? '#8d867b' : STY[l.style]);
  g.fillStyle = '#4a433a';
  for (const s of STAIRS) {
    g.fillRect(toPx(-197), toPx(s.z - 0.2), 394 * k, 2.4 * k);
    g.fillStyle = '#c9bfa6';
    for (const [cx, w] of s.list) g.fillRect(toPx(cx - w / 2), toPx(s.z - STAIR_RUN), w * k, (STAIR_RUN + 2.2) * k);
    g.fillStyle = '#4a433a';
  }
  R(DOM.nave, '#3f4348'); R(DOM.choir, '#3f4348'); R(DOM.tower, '#2f3338'); R(RATHAUS, '#7f3f30');
  g.strokeStyle = '#5ce1e6'; g.lineWidth = 2; g.beginPath(); g.arc(toPx(0), toPx(65), 5 * k, 0, Math.PI * 2); g.stroke();
  for (const t of GIANT_TREES) {
    g.fillStyle = 'rgba(48,74,36,0.95)'; g.beginPath(); g.arc(toPx(t.x), toPx(t.z), (t.r + 4) * k, 0, Math.PI * 2); g.fill();
  }
  for (const f of FACES) {
    g.fillStyle = '#a8a091'; g.beginPath(); g.ellipse(toPx(f.x), toPx(f.z), 8 * k, 5 * k, 0, 0, Math.PI * 2); g.fill();
  }
  g.fillStyle = '#2f2b26';
  g.fillRect(toPx(-200), toPx(CITY.CLIFF_Z), 3 * k, (200 - CITY.CLIFF_Z) * k);
  g.fillRect(toPx(197), toPx(CITY.CLIFF_Z), 3 * k, (200 - CITY.CLIFF_Z) * k);
  g.fillRect(toPx(-200), toPx(197), 400 * k, 3 * k);
  R(STEIG.tower, '#6a4a2e');
  R(WHITE_TOWER, '#fbf8ef');
  g.fillStyle = '#5ce1e6'; g.beginPath(); g.arc(toPx(ENERGY_TOWER.x), toPx(ENERGY_TOWER.z), ENERGY_TOWER.r * k, 0, Math.PI * 2); g.fill();
  R({ x0: 47, x1: 93, z0: 268, z1: 280 }, '#4a3424');
  R({ x0: -157, x1: -143, z0: 252, z1: 266 }, '#8a8274');
}

export const WELTENBRUECK = {
  id: 'weltenbrueck',
  name: 'Weltenbrück',
  tag: 'Stadt · eng & verwinkelt',
  kicker: 'Oberfläche · Gesamte Stadt',
  lead: 'Die alte Gildenstadt am Gründerfelsen. Fachwerkgassen um den Kreuzmarkt, darüber die Waldsiedlung mit Riesenbäumen. Durch die Tore: die Weiße Stadt, die Turmstadt mit Himmelsbrücken und der Hafen mit gestrandeter Galeone. Jedes Haus ist begehbar – und in jedem kann etwas lauern.',
  loading: 'Weltenbrück wird errichtet …',
  half: CITY.HALF,
  bound: CITY.BOUND,
  mapHalf: 300,
  // Fotografische Modelle (blender/nature_build.py)
  models: {
    pine: ['n_fir0', 'n_fir1', 'n_fir2'],
    tree: ['n_leaf0'],
    bush: ['n_bush0', 'n_bush1'],
    rock: ['n_rock0', 'n_rock1', 'n_rock2', 'n_rock3', 'n_rock4', 'n_rock5', 'n_rock6', 'n_rock7'],
  },
  enemyDraw: 170, // enge Gassen, Wände verdecken das meiste
  pois: POIS,
  extracts: EXTRACTS,
  spawns: PLAYER_SPAWNS,
  menuCamera: { x: 0, z: 10, r: 150, h: 70, lookY: 14 },
  spotChance: 0.24,
  terrain,
  groundTexture,
  materials,
  decorate,
  updateDecor,
  vegetation,
  drawMap,
  build: (ctx) => buildCity({ ...ctx, layout: layout() }),
};
