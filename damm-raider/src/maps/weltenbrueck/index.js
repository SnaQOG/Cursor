// Karte "Weltenbrück": eine Fantasy-Stadt aus deutscher Architektur.
import * as THREE from 'three';
import {
  CITY, TERRACES, STAIRS, STAIR_RUN, PLAZA, DOM, RATHAUS, STREETS, GIANT_TREES, FACES, STEIG,
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
  const occ = new Occupancy(CITY.WALL + 2, 2);
  occ.grid.fill(0);
  for (const s of STREETS) if (s.main) occ.mark(s, 4);
  for (const y of L.yards) occ.mark(y, y.garden ? 3 : y.passage ? 0 : 2);
  for (const q of L.squares) occ.mark(q, 1);
  occ.mark(PLAZA, 1);
  occ.mark(DOM.area, 1);
  for (const t of GIANT_TREES) occ.mark({ x0: t.x - t.r, x1: t.x + t.r, z0: t.z - t.r, z1: t.z + t.r }, 3);
  GROUND = occ;
  return occ;
}

const C = {
  cobble: new THREE.Color(0x8f877a), cobbleDark: new THREE.Color(0x6f695f), plaza: new THREE.Color(0xb3a993),
  grass: new THREE.Color(0x5d7337), grassDry: new THREE.Color(0x8a8a4e), garden: new THREE.Color(0x4f6a2f),
  moss: new THREE.Color(0x5a6a3a), rock: new THREE.Color(0x7a7366), rockDark: new THREE.Color(0x56514a),
  forest: new THREE.Color(0x3e4f2c), pale: new THREE.Color(0xc9c1ad), sand: new THREE.Color(0xa8986f), seabed: new THREE.Color(0x4f5a52),
  main: new THREE.Color(0x7d766a),
};

const terrain = {
  half: CITY.HALF,
  res: CITY.RES,
  seed: 1871,
  uvScale: 4,

  rawHeight(T, x, z) {
    const n = T.noise;
    const inside = Math.abs(x) <= 200 && z <= 200;
    if (inside && z >= CITY.CLIFF_Z) {
      const r = stairRamp(x, z);
      if (r != null) return r;
      return levelAt(z);
    }
    if (z < CITY.CLIFF_Z) {
      // Gründerfelsen und Plateau mit Wald
      const k = smoothstep(CITY.CLIFF_Z, CITY.CLIFF_Z - 20, z);
      const rough = n.fbm(x * 0.06, z * 0.06, 3) * 2.5 * Math.min(1, k * 3) * (1 - k * 0.6);
      return 18 + k * (CITY.CLIFF_TOP - 18) + rough + n.fbm(x * 0.01, z * 0.01, 2) * 4 * k;
    }
    // außerhalb der Mauer: sanfte Übergänge zwischen den Terrassen
    let h = 3;
    if (z < -2) {
      const t = (-2 - z) / (CITY.CLIFF_Z + 2);
      h = 3 + smoothstep(0, 1, clamp(-t, 0, 1)) * 15;
    }
    h += n.fbm(x * 0.02, z * 0.02, 2) * 0.6;
    // Meer im Süden
    if (z > CITY.SEA_Z - 30) h = lerp(h, -3, smoothstep(CITY.SEA_Z - 30, CITY.SEA_Z, z));
    return h;
  },

  waterAt(T, x, z) {
    if (z > CITY.SEA_Z - 12) return { level: 0.6, toxic: false };
    // Marktbrunnen
    if (Math.hypot(x, z - 96) < 4.6) return { level: 3.5, toxic: false };
    return null;
  },

  groundColor(T, x, z, h, sl, c) {
    const n = T.noise;
    const nv = n.fbm(x * 0.05 + 3, z * 0.05 - 9, 3);
    const inside = Math.abs(x) <= 200 && z <= 200 && z >= CITY.CLIFF_Z;
    if (inside) {
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
    } else if (z < CITY.CLIFF_Z) {
      const k = smoothstep(CITY.CLIFF_Z, CITY.CLIFF_Z - 4, z) * (1 - smoothstep(CITY.CLIFF_Z - 18, CITY.CLIFF_Z - 24, z));
      c.copy(C.forest).lerp(C.grass, clamp(0.4 + nv, 0, 1) * 0.5);
      if (k > 0 || sl > 0.8) c.lerp(nv > 0 ? C.rock : C.rockDark, Math.max(k, smoothstep(0.8, 1.4, sl)));
    } else if (z > CITY.SEA_Z - 20) {
      c.copy(C.sand).lerp(C.seabed, smoothstep(CITY.SEA_Z - 14, CITY.SEA_Z, z));
    } else if (x < -200) c.copy(C.pale).lerp(C.grass, 0.15 + nv * 0.2);
    else if (x > 200) c.copy(C.cobbleDark).lerp(C.cobble, 0.5 + nv * 0.5);
    else c.copy(C.cobble).lerp(C.sand, 0.3);
    c.multiplyScalar(0.92 + n.noise(x * 0.4, z * 0.4) * 0.1);
  },

  mapColor() { return [40, 44, 46]; },

  // Anteil Gras-Textur am Boden (0 = Pflaster)
  grassMix(T, x, z) {
    if (z < CITY.CLIFF_Z - 16) return 1;
    if (z < CITY.CLIFF_Z) return 0;
    if (Math.abs(x) > 200 || z > 200) return z > CITY.SEA_Z - 20 ? 0 : 0.6;
    const g = groundGrid().at(x, z);
    return g === 2 || g === 3 ? 1 : 0;
  },
};

function materials() {
  const mk = (set, roughness = 0.88, metalness = 0.03, ns = 0.9) => new THREE.MeshStandardMaterial({
    vertexColors: true, map: set.map, normalMap: set.normal, normalScale: new THREE.Vector2(ns, ns), roughness, metalness,
  });
  return {
    world: mk(makePlasterSet(), 0.93, 0, 0.7),
    brick: mk(makeBrickSet(), 0.9),
    stone: mk(makeAshlarSet(), 0.92),
    wood: mk(makePlankSet(), 0.82),
    timber: mk(makeTimberSet(), 0.8, 0, 0.8),
    shingle: mk(makeShingleSet(), 0.85),
    tile: mk(makeTileSet(), 0.8, 0.02, 1.0),
    slate: mk(makeSlateSet(), 0.62, 0.06),
    cloth: mk(makeClothSet(), 0.97, 0, 0.6),
  };
}

// Boden: Pflaster, in Höfen, Gärten und im Wald Gras
function groundTexture() {
  const c = makeCobbleSet(), g = makeGrassSet();
  return { map: c.map, normal: c.normal, grass: g.map, grassNormal: g.normal };
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
  const m = new THREE.Mesh(new THREE.PlaneGeometry(1400, 500), sea);
  m.rotation.x = -Math.PI / 2;
  m.position.set(0, 0.6, CITY.SEA_Z + 240);
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
      if (w < 4 || d < 4 || !free(x, z, 1.4, 6)) continue;
      const it = { x, z, y: T.height(x, z), s: 0.7 + rnd() * 0.6, pine: false, r: rnd() * 6 };
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
  for (let i = 0; i < 6000 && trees.length < 1300; i++) {
    const x = (rnd() * 2 - 1) * 296, z = -296 + rnd() * (296 + CITY.CLIFF_Z - 20);
    if (n.fbm(x * 0.02, z * 0.02, 2) < -0.25) continue;
    trees.push({ x, z, y: T.height(x, z), s: 0.9 + rnd() * 0.8, pine: rnd() < 0.8, r: rnd() * 6 });
  }
  // Felsbrocken am Gründerfelsen
  for (let i = 0; i < 260; i++) {
    const x = (rnd() * 2 - 1) * 296, z = CITY.CLIFF_Z - 2 - rnd() * 20;
    rocks.push({ x, z, y: T.height(x, z), s: 1.5 + rnd() * 3.5, r: rnd() * 6, t: rnd() * 6 });
  }
  // Gras zwischen Pflaster und in Höfen
  const maxGrass = { niedrig: 0, mittel: 9000, hoch: 18000 }[world.quality] ?? 9000;
  const gg = groundGrid();
  for (let i = 0; i < maxGrass * 8 && grass.length < maxGrass; i++) {
    const x = (rnd() * 2 - 1) * 196, z = CITY.CLIFF_Z + 2 + rnd() * (196 - CITY.CLIFF_Z - 2);
    const g = gg.at(x, z);
    const green = g === 2 || g === 3;
    if (!green && n.fbm(x * 0.09 - 7, z * 0.09 + 3, 3) < 0.15) continue;
    if (!free(x, z, 0.3, 0.6)) continue;
    const tuft = { x, z, y: T.height(x, z), s: (green ? 0.7 : 0.45) + rnd() * 0.6, r: rnd() * 6 };
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
    { name: 'v_linde', list: linden, place: upright, opts: { tint: leaf } },
    { name: 'v_birke', list: birches, place: upright, opts: { tint: leaf } },
    { name: 'v_flowers', list: flowers, place: upright, opts: { shadow: false, doubleSide: true, cull: true } },
    { name: 'v_meadow', list: meadow, place: upright, opts: { shadow: false, doubleSide: true, cull: true } },
    { name: 'v_weed', list: weeds, place: upright, opts: { shadow: false, doubleSide: true, tint: leaf, cull: true } },
  ];
  return { trees, bushes, rocks, grass, extra };
}

// Übersichtskarte direkt aus dem Stadtplan
function drawMap(g, size, toPx, k) {
  const L = layout();
  const R = (r, fill) => { g.fillStyle = fill; g.fillRect(toPx(r.x0), toPx(r.z0), (r.x1 - r.x0) * k, (r.z1 - r.z0) * k); };
  R({ x0: -230, x1: 230, z0: -230, z1: 230 }, '#3a3d3a');
  R({ x0: -230, x1: 230, z0: -230, z1: CITY.CLIFF_Z - 18 }, '#4c5a3c');
  R({ x0: -230, x1: 230, z0: CITY.CLIFF_Z - 18, z1: CITY.CLIFF_Z }, '#6c665c');
  const tcol = { t0: '#a49c8a', t1: '#9f9884', t2: '#9a937e', t3: '#958f79' };
  for (const t of TERRACES) R({ x0: -197, x1: 197, z0: t.z0, z1: t.z1 }, tcol[t.id]);
  R({ x0: -197, x1: 197, z0: -2, z1: 197 }, '#a49c8a');
  for (const a of L.alleys) R(a, '#b5ac98');
  for (const s of STREETS) R(s, '#c2b9a4');
  for (const y of L.yards) R(y, y.passage ? '#b5ac98' : '#7d8c5a');
  for (const q of L.squares) R(q, '#d3c9b2');
  R(PLAZA, '#ddd2ba');
  const STY = { fachwerk: '#9a5a42', stein: '#6f6a69', backstein: '#7f3f30', schwarzwald: '#5a4532', turm: '#7d8186' };
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
}

export const WELTENBRUECK = {
  id: 'weltenbrueck',
  name: 'Weltenbrück',
  tag: 'Stadt · eng & verwinkelt',
  kicker: 'Oberfläche · Etappe 1 · Kreuzmarkt & Waldsiedlung',
  lead: 'Die alte Gildenstadt am Gründerfelsen. Enge Fachwerkgassen um den Kreuzmarkt, darüber die Waldsiedlung auf vier Terrassen mit Riesenbäumen und Hängebrücken. Jedes Haus ist begehbar – und in jedem kann etwas lauern.',
  loading: 'Weltenbrück wird errichtet …',
  half: CITY.HALF,
  bound: CITY.BOUND,
  mapHalf: 212,
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
