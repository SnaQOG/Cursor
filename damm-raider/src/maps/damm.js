// Karte "Der Damm": Terrain, Wasser, Vegetation und Aufbau.
import * as THREE from 'three';
import { MAP, PADS, ROADS, POIS, EXTRACTS, PLAYER_SPAWNS } from '../config.js';
import { COVER_BUDGET, placeMeadowPlant } from '../groundcover.js';
import { buildMap } from '../map.js';
import { makeSignTexture, makeSoftDot } from '../textures.js';
import { Noise2D, clamp, lerp, smoothstep, distToPolyline, mulberry32 } from '../util.js';

function sdRoundBox(px, pz, hx, hz, r) {
  const qx = Math.abs(px) - hx + r, qz = Math.abs(pz) - hz + r;
  const ox = Math.max(qx, 0), oz = Math.max(qz, 0);
  return Math.sqrt(ox * ox + oz * oz) + Math.min(Math.max(qx, qz), 0) - r;
}

function lakeMask(T, x, z) {
  const L = MAP.LAKE;
  const d = sdRoundBox(x - L.cx, z - L.cz, L.hx, L.hz, L.r) + T.noise.noise(x * 0.02, z * 0.02) * 10;
  return 1 - smoothstep(-10, 6, d);
}

function swampMask(T, x, z) {
  const S = MAP.SWAMP;
  const d = Math.hypot(x - S.x, z - S.z) + T.noise.noise(x * 0.03 + 5, z * 0.03) * 14;
  return 1 - smoothstep(S.r, S.r + S.blend, d);
}

const grass = new THREE.Color(0x64865a), dry = new THREE.Color(0x868e6a), lush = new THREE.Color(0x547450);
const rock = new THREE.Color(0x77705f), rockDark = new THREE.Color(0x4f4a42), mud = new THREE.Color(0x5b4c38);
const sand = new THREE.Color(0x9b8d6a), swamp = new THREE.Color(0x3c4526), road = new THREE.Color(0x6f685a);
const gravel = new THREE.Color(0x8b867a);

// Höhenfeld: Tal unterhalb des Damms, Stausee-Plateau im Norden, Fluss,
// Giftsumpf und Berge am Rand.
const terrain = {
  half: MAP.HALF,
  res: MAP.RES,
  seed: 4242,
  uvScale: 3,

  init(T) {
    T.pads = PADS.map((p) => ({ ...p, h: p.h === 'auto' ? this.rawHeight(T, p.x, p.z, false) : p.h }));
  },

  rawHeight(T, x, z, withPads) {
    const n = T.noise;
    const ax = Math.abs(x);
    // Plateau (Norden) gegen Tal (Süden). Im Dammbereich harter Übergang.
    const pDam = smoothstep(-70, -90, z);
    const pOpen = smoothstep(-15, -108, z);
    const side = smoothstep(128, 168, ax);
    const p = lerp(pDam, pOpen, side);
    let h = lerp(MAP.VALLEY, MAP.PLATEAU, p);

    const n1 = n.fbm(x * 0.0065 + 11.3, z * 0.0065 - 7.1, 4);
    const n2 = n.fbm(x * 0.03, z * 0.03, 3);
    h += Math.max(0, n1 + 0.08) * 16 + n2 * 1.1;

    // Stausee
    if (z < -60) {
      const l = lakeMask(T, x, z);
      if (l > 0) h = lerp(h, MAP.LAKE_FLOOR + n2 * 2, l);
    }

    // Berge am Kartenrand
    const e = Math.max(ax, Math.abs(z));
    const m = smoothstep(262, 318, e);
    if (m > 0) h += m * (42 + n1 * 22 + n2 * 4);

    // Fluss (nur im Tal)
    if (z > -76) {
      const d = distToPolyline(x, z, MAP.RIVER) + n.noise(x * 0.05, z * 0.05) * 3;
      const r = 1 - smoothstep(5, 15, d);
      if (r > 0) h = lerp(h, -0.7, r * smoothstep(-76, -70, z));
    }

    // Giftsumpf: flache Tümpel
    const s = swampMask(T, x, z);
    if (s > 0) h = lerp(h, 0.5 + n.fbm(x * 0.05, z * 0.05, 3) * 1.8, s);

    if (withPads) {
      for (const pad of T.pads) {
        const d = Math.hypot(x - pad.x, z - pad.z);
        if (d > pad.r + pad.blend) continue;
        const w = 1 - smoothstep(pad.r, pad.r + pad.blend, d);
        h = lerp(h, pad.h, w);
      }
    }
    return h;
  },

  // Wasser an dieser Stelle? Liefert Pegel und ob giftig.
  waterAt(T, x, z) {
    const S = MAP.SWAMP;
    if (Math.hypot(x - S.x, z - S.z) < S.r + S.blend + 10) return { level: MAP.SWAMP_WATER, toxic: true };
    if (z < MAP.DAM_Z && Math.abs(x) < MAP.LAKE.hx + 2) return { level: MAP.LAKE_LEVEL, toxic: false };
    if (z > MAP.DAM_Z + 4) return { level: MAP.VALLEY_WATER, toxic: false };
    return null;
  },

  // Fotos statt Einheitsfarbe: Wiese, Erde auf Wegen und Bauplätzen, Fels an steilen Hängen
  grassMix(T, x, z, h) {
    let rd = Infinity;
    for (const r of ROADS) rd = Math.min(rd, distToPolyline(x, z, r));
    let m = smoothstep(2.5, 6, rd);
    for (const pad of T.pads) {
      if (pad.r < 20) continue;
      m = Math.min(m, smoothstep(pad.r * 0.55, pad.r * 0.95, Math.hypot(x - pad.x, z - pad.z)));
    }
    const w = T.waterAt(x, z);
    if (w) m = Math.min(m, smoothstep(0.1, 1.2, h - w.level));
    return m;
  },

  rockMix(T, x, z, h, sl) {
    return clamp(smoothstep(0.55, 1.0, sl) + smoothstep(290, 310, Math.max(Math.abs(x), Math.abs(z))) * 0.6, 0, 1);
  },

  wearMix(T, x, z, out) {
    let rd = Infinity;
    for (const r of ROADS) rd = Math.min(rd, distToPolyline(x, z, r));
    out[0] = Math.max(smoothstep(0.1, 0.5, T.noise.fbm(x * 0.06 + 11, z * 0.06 - 5, 3)) * 0.6, 1 - smoothstep(4, 10, rd));
    out[1] = 0;
  },

  groundColor(T, x, z, h, sl, c) {
    const n = T.noise;
    const nv = n.fbm(x * 0.04 + 3, z * 0.04 - 9, 3);
    c.copy(grass).lerp(dry, clamp(0.45 + nv * 1.4, 0, 1));
    if (z < -90) c.lerp(lush, 0.35);
    // Ufer / Schlamm
    const w = T.waterAt(x, z);
    if (w) {
      const above = h - w.level;
      if (above < 1.2) c.lerp(w.toxic ? swamp : (w.level > 10 ? sand : mud), clamp(1 - above / 1.2, 0, 1));
    }
    const sm = swampMask(T, x, z);
    if (sm > 0) c.lerp(swamp, sm * 0.8);
    // Straßen
    let rd = Infinity;
    for (const r of ROADS) rd = Math.min(rd, distToPolyline(x, z, r));
    if (rd < 5) c.lerp(road, (1 - smoothstep(2.5, 5, rd)) * 0.85);
    // Bauplätze
    for (const pad of T.pads) {
      if (pad.r < 20) continue;
      const d = Math.hypot(x - pad.x, z - pad.z);
      if (d < pad.r) c.lerp(gravel, (1 - smoothstep(pad.r * 0.6, pad.r, d)) * 0.55);
    }
    // Fels an steilen Hängen und in den Bergen
    const rk = smoothstep(0.55, 1.0, sl) + smoothstep(290, 310, Math.max(Math.abs(x), Math.abs(z))) * 0.6;
    if (rk > 0) c.lerp(nv > 0 ? rock : rockDark, clamp(rk, 0, 1));
    const v = 0.92 + n.noise(x * 0.3, z * 0.3) * 0.12;
    c.multiplyScalar(v);
  },

  // Farbe für die Übersichtskarte
  mapColor(T, x, z, h) {
    const w = T.waterAt(x, z);
    if (w && h < w.level) {
      const depth = Math.min(1, (w.level - h) / 6);
      return w.toxic ? [96, 120, 40] : [Math.round(60 - depth * 30), Math.round(96 - depth * 30), Math.round(110 - depth * 20)];
    }
    const e = 0.8;
    const shade = clamp(0.75 + (T.height(x - e, z - e) - T.height(x + e, z + e)) * 0.25, 0.45, 1.25);
    const k = clamp((h - 0) / 60, 0, 1);
    const base = [128 - k * 30, 132 - k * 20, 100 - k * 10];
    return base.map((v) => Math.round(clamp(v * shade, 0, 255)));
  },
};

// Stausee, Tal-Wasser, Giftsumpf, Wasserfall und Schriftzug
function decorate(world) {
  const scene = world.scene;
  const normalTex = world.waterNormal;
  const mk = (color, opacity, rep) => {
    const t = normalTex.clone();
    t.needsUpdate = true;
    t.repeat.set(rep, rep);
    return new THREE.MeshStandardMaterial({ color, roughness: 0.12, metalness: 0.15, transparent: true, opacity, normalMap: t, normalScale: new THREE.Vector2(0.6, 0.6), depthWrite: false });
  };
  // Stausee
  const L = MAP.LAKE;
  const lakeMat = mk(0x2c5a66, 0.9, 30);
  const lake = new THREE.Mesh(new THREE.PlaneGeometry(L.hx * 2 + 6, L.hz * 2 + 4), lakeMat);
  lake.rotation.x = -Math.PI / 2;
  lake.position.set(0, MAP.LAKE_LEVEL, (L.cz - L.hz + MAP.DAM_Z) / 2);
  lake.scale.set(1, (MAP.DAM_Z - (L.cz - L.hz)) / (L.hz * 2 + 4), 1);
  scene.add(lake);
  // Tal-Wasser (Fluss, Tümpel)
  const valMat = mk(0x3c5f55, 0.82, 60);
  const val = new THREE.Mesh(new THREE.PlaneGeometry(640, 400), valMat);
  val.rotation.x = -Math.PI / 2;
  val.position.set(0, MAP.VALLEY_WATER, MAP.DAM_Z + 6 + 200);
  scene.add(val);
  // Giftsumpf
  const S = MAP.SWAMP;
  const swMat = mk(0x7c9a2a, 0.85, 18);
  swMat.emissive = new THREE.Color(0x3a4a08);
  swMat.emissiveIntensity = 0.6;
  const sw = new THREE.Mesh(new THREE.CircleGeometry(S.r + S.blend + 10, 48), swMat);
  sw.rotation.x = -Math.PI / 2;
  sw.position.set(S.x, MAP.SWAMP_WATER, S.z);
  scene.add(sw);
  world.waterMats.push(lakeMat, valMat, swMat);

  // Wasserfall am Überlauf
  const c = document.createElement('canvas');
  c.width = 64; c.height = 256;
  const g = c.getContext('2d');
  for (let i = 0; i < 90; i++) {
    g.fillStyle = `rgba(255,255,255,${0.15 + Math.random() * 0.5})`;
    g.fillRect(Math.random() * 64, Math.random() * 256, 1 + Math.random() * 3, 20 + Math.random() * 60);
  }
  const ft = new THREE.CanvasTexture(c);
  ft.wrapS = ft.wrapT = THREE.RepeatWrapping;
  ft.repeat.set(3, 2);
  world.fallTex = ft;
  const fallMat = new THREE.MeshBasicMaterial({ map: ft, color: 0xcfe8ec, transparent: true, opacity: 0.75, depthWrite: false, side: THREE.DoubleSide });
  const fall = new THREE.Mesh(new THREE.PlaneGeometry(22, 30), fallMat);
  fall.position.set(0, 15.5, MAP.DAM_Z + MAP.DAM_THICK / 2 + 0.6);
  scene.add(fall);
  // Gischt
  const dot = makeSoftDot();
  const mistMat = new THREE.SpriteMaterial({ map: dot, color: 0xe8f2f2, transparent: true, opacity: 0.35, depthWrite: false });
  world.mist = [];
  for (let i = 0; i < 14; i++) {
    const s = new THREE.Sprite(mistMat);
    s.position.set(-10 + Math.random() * 20, 1.5 + Math.random() * 3, MAP.DAM_Z + 10 + Math.random() * 6);
    s.scale.setScalar(6 + Math.random() * 6);
    s.userData.phase = Math.random() * 6;
    scene.add(s);
    world.mist.push(s);
  }

  // Großer Schriftzug auf der Staumauer
  const tex = makeSignTexture('TALSPERRE 7', '#b9b3a3', '#3b3a35');
  const m = new THREE.Mesh(new THREE.PlaneGeometry(40, 10), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.9, transparent: true, opacity: 0.85 }));
  m.position.set(-60, 24, MAP.DAM_Z + MAP.DAM_THICK / 2 + 0.05);
  scene.add(m);
}

function updateDecor(world, dt) {
  world.fallTex.offset.y += dt * 1.6;
  for (const s of world.mist) {
    s.userData.phase += dt;
    s.material.opacity = 0.3;
    s.position.y = 1.5 + Math.sin(s.userData.phase) * 0.8;
  }
}

// Bäume, Büsche, Felsen und Gras
// Die Karte ist zu groß für eine geschlossene Wiese: dichte Flecken um Orte und Straßen, dazwischen nur wenig
function placeDamMeadow(world, n, rnd) {
  const budget = COVER_BUDGET[world.quality] ?? COVER_BUDGET.mittel;
  const T = world.terrain, col = world.col, b = world.batcher;
  let placed = 0;
  for (let i = 0; i < budget * 14 && placed < budget; i++) {
    const x = (rnd() * 2 - 1) * 290, z = (rnd() * 2 - 1) * 290;
    const patch = n.fbm(x * 0.045 - 4, z * 0.045 + 2, 2);
    if (rnd() > clamp(0.12 + (patch + 0.05) * 2.4, 0.04, 1)) continue;
    let near = Infinity;
    for (const p of POIS) near = Math.min(near, Math.hypot(x - p.x, z - p.z));
    if (near > 90 && rnd() > 0.2) continue;
    const y = T.height(x, z);
    if (T.slope(x, z) > 0.6 || terrain.grassMix(T, x, z, y) < 0.8) continue;
    if (Math.abs(z - MAP.DAM_Z) < 12 && Math.abs(x) < MAP.DAM_HALF_X + 4) continue;
    if (col.query(x - 0.3, z - 0.3, x + 0.3, z + 0.3).some((s) => s.maxY > y + 0.3 && s.minY < y + 0.8)) continue;
    placeMeadowPlant(b, n, rnd, x, y, z, 1.35);
    placed++;
  }
  world.coverStats = { wiese: placed };
}

function vegetation(world) {
  const rnd = mulberry32(2024);
  const T = world.terrain;
  const n = new Noise2D(77);
  const trees = [], bushes = [], rocks = [];
  const blocked = (x, z, r) => {
    if (Math.abs(x) > 300 || Math.abs(z) > 300) return true;
    const w = T.waterAt(x, z);
    const h = T.height(x, z);
    if (w && h < w.level + 0.4) return true;
    if (Math.abs(z - MAP.DAM_Z) < 14 && Math.abs(x) < MAP.DAM_HALF_X + 6) return true;
    for (const p of PADS) if (Math.hypot(x - p.x, z - p.z) < p.r * 0.9 + r) return true;
    for (const rd of ROADS) if (distToPolyline(x, z, rd) < 4 + r) return true;
    if (world.col.query(x - r, z - r, x + r, z + r).length) return true;
    return false;
  };
  for (let i = 0; i < 9000 && trees.length < 1100; i++) {
    const x = (rnd() * 2 - 1) * 300, z = (rnd() * 2 - 1) * 300;
    const dens = n.fbm(x * 0.012, z * 0.012, 3) + (z < -90 ? 0.15 : 0) + smoothstep(220, 290, Math.max(Math.abs(x), Math.abs(z))) * 0.4;
    if (dens < 0.08 + rnd() * 0.25) continue;
    if (T.slope(x, z) > 0.8 || blocked(x, z, 1.5)) continue;
    trees.push({ x, z, y: T.height(x, z), s: 0.8 + rnd() * 0.7, pine: rnd() < (z < -90 || Math.abs(x) > 240 ? 0.75 : 0.35), r: rnd() * 6 });
  }
  for (let i = 0; i < 12000 && bushes.length < 1800; i++) {
    const x = (rnd() * 2 - 1) * 300, z = (rnd() * 2 - 1) * 300;
    if (n.fbm(x * 0.02 + 9, z * 0.02, 2) < -0.1 + rnd() * 0.2) continue;
    if (blocked(x, z, 0.5)) continue;
    bushes.push({ x, z, y: T.height(x, z), s: 0.35 + rnd() * 0.55, r: rnd() * 6 });
  }
  for (let i = 0; i < 4000 && rocks.length < 420; i++) {
    const x = (rnd() * 2 - 1) * 300, z = (rnd() * 2 - 1) * 300;
    if (blocked(x, z, 2)) continue;
    const sl = T.slope(x, z);
    if (rnd() > 0.25 + sl) continue;
    rocks.push({ x, z, y: T.height(x, z), s: 0.8 + rnd() * rnd() * 3.5, r: rnd() * 6, t: rnd() * 6 });
  }
  // Grasbüschel auf den Wiesen
  const grassTufts = [];
  const cover = !!world.assets?.tex?.cover;
  const maxGrass = cover ? 0 : ({ niedrig: 0, mittel: 7000, hoch: 14000 }[world.quality] ?? 7000);
  if (cover) placeDamMeadow(world, n, rnd);
  for (let i = 0; i < maxGrass * 7 && grassTufts.length < maxGrass; i++) {
    const x = (rnd() * 2 - 1) * 290, z = (rnd() * 2 - 1) * 290;
    if (n.fbm(x * 0.03 - 4, z * 0.03 + 2, 2) < -0.05) continue;
    if (T.slope(x, z) > 0.6) continue;
    const w = T.waterAt(x, z);
    const h = T.height(x, z);
    if (w && h < w.level + 0.2) continue;
    if (Math.abs(z - MAP.DAM_Z) < 12 && Math.abs(x) < MAP.DAM_HALF_X + 4) continue;
    let bad = false;
    for (const pd of PADS) if (Math.hypot(x - pd.x, z - pd.z) < pd.r * 0.8) { bad = true; break; }
    if (bad) continue;
    for (const rd of ROADS) if (distToPolyline(x, z, rd) < 3.5) { bad = true; break; }
    if (bad) continue;
    grassTufts.push({ x, z, y: h, s: 0.7 + rnd() * 0.8, r: rnd() * 6 });
  }
  return { trees, bushes, rocks, grass: grassTufts };
}

export const DAMM = {
  id: 'damm',
  name: 'Der Damm',
  tag: 'Talsperre · offen',
  kicker: 'Oberfläche · Einsatzgebiet',
  lead: 'Talsperre 7 staut den Fluss noch immer. Unter der Mauer liegen Kraftwerk, Forschungsanlage und die Wohnblöcke – bewacht von ARC-Maschinen. Hol dir, was du tragen kannst, und schaff es zu einem Aufzug, bevor die Zeit abläuft.',
  loading: 'Staumauer wird gegossen …',
  half: MAP.HALF,
  bound: MAP.BOUND,
  mapHalf: MAP.HALF,
  pois: POIS,
  extracts: EXTRACTS,
  spawns: PLAYER_SPAWNS,
  menuCamera: { x: 0, z: -30, r: 200, h: 85, lookY: 18 },
  terrain,
  // Fotografische Modelle wie in der Stadt (blender/nature_build.py)
  models: {
    pine: ['n_fir0', 'n_fir1', 'n_fir2'],
    tree: ['n_lind0', 'n_lind1', 'n_birch0'],
    bush: ['n_bush0', 'n_bush1'],
    rock: ['n_rock0', 'n_rock1', 'n_rock2', 'n_rock3', 'n_rock4', 'n_rock5', 'n_rock6', 'n_rock7'],
  },
  groundTexture: (world) => {
    const t = world.assets?.tex;
    return t?.meadow && t?.path ? { map: t.path.map, normal: t.path.normal, grass: t.meadow.map, grassNormal: t.meadow.normal, rock: t.cliff?.map, path: t.path.map } : world.grunge;
  },
  decorate,
  updateDecor,
  vegetation,
  build: ({ batcher, col, terrain: T, assets, instancer }) => buildMap(batcher, col, T, assets, instancer),
};
