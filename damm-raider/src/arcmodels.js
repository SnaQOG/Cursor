// Modelle der ARC-Maschinen aus Grundformen. Jede Funktion liefert
// { root, parts: [{name, obj, r, mult, armor, weak, hp?}], anim: {...} }
import * as THREE from 'three';
import { assets, Assets } from './assets.js';

const M = {
  body: new THREE.MeshStandardMaterial({ color: 0x4d5155, roughness: 0.5, metalness: 0.45 }),
  plate: new THREE.MeshStandardMaterial({ color: 0xd9d4c6, roughness: 0.5, metalness: 0.15 }),
  plateDark: new THREE.MeshStandardMaterial({ color: 0x86837a, roughness: 0.55, metalness: 0.3 }),
  eye: new THREE.MeshStandardMaterial({ color: 0xff4a1c, emissive: 0xff3a10, emissiveIntensity: 2.2, roughness: 0.3 }),
  weak: new THREE.MeshStandardMaterial({ color: 0xffb020, emissive: 0xff9a10, emissiveIntensity: 1.6, roughness: 0.3 }),
  thrust: new THREE.MeshStandardMaterial({ color: 0x9fe8ff, emissive: 0x40c8ff, emissiveIntensity: 2.0, roughness: 0.2 }),
  orange: new THREE.MeshStandardMaterial({ color: 0xd8642a, roughness: 0.5, metalness: 0.3 }),
  wreck: new THREE.MeshStandardMaterial({ color: 0x2b2a28, roughness: 0.9, metalness: 0.1 }),
};
export const ARC_MATS = M;

function mesh(geo, mat, parent, x = 0, y = 0, z = 0) {
  const m = new THREE.Mesh(geo, mat);
  m.position.set(x, y, z);
  m.castShadow = true;
  parent.add(m);
  return m;
}

const G = {
  sph: (r, w = 12, h = 10) => new THREE.SphereGeometry(r, w, h),
  box: (x, y, z) => new THREE.BoxGeometry(x, y, z),
  cyl: (r1, r2, h, s = 10) => new THREE.CylinderGeometry(r1, r2, h, s),
};

export function buildTick() {
  const root = new THREE.Group();
  const body = mesh(G.sph(0.32, 10, 8), M.body, root, 0, 0.35, 0);
  body.scale.set(1, 0.6, 1.2);
  mesh(G.box(0.5, 0.08, 0.5), M.plateDark, root, 0, 0.5, 0);
  const eye = mesh(G.sph(0.08), M.eye, root, 0, 0.42, -0.34);
  const legs = [];
  for (let i = 0; i < 6; i++) {
    const side = i < 3 ? -1 : 1;
    const k = (i % 3) - 1;
    const pivot = new THREE.Group();
    pivot.position.set(side * 0.22, 0.35, k * 0.22);
    root.add(pivot);
    mesh(G.box(0.45, 0.05, 0.05), M.body, pivot, side * 0.22, -0.12, 0).rotation.z = side * 0.6;
    legs.push(pivot);
  }
  return { root, legs, parts: [
    { name: 'Körper', obj: body, r: 0.4, mult: 1, armor: 0 },
    { name: 'Auge', obj: eye, r: 0.12, mult: 2, armor: 0, weak: true },
  ] };
}

export function buildPop() {
  const root = new THREE.Group();
  const ball = new THREE.Group();
  ball.position.y = 0.5;
  root.add(ball);
  const core = mesh(G.sph(0.5, 14, 12), M.plate, ball);
  for (let i = 0; i < 3; i++) {
    const band = mesh(new THREE.TorusGeometry(0.5, 0.05, 6, 20), M.orange, ball);
    band.rotation.set(i * 1.1, i * 0.7, 0);
  }
  const light = mesh(G.sph(0.12), M.eye, ball, 0, 0.45, 0);
  return { root, ball, light, parts: [
    { name: 'Hülle', obj: core, r: 0.55, mult: 1, armor: 0 },
  ] };
}

export function buildWasp(scale = 1) {
  const root = new THREE.Group();
  const body = mesh(G.sph(0.42, 12, 10), M.body, root);
  body.scale.set(1, 0.7, 1.3);
  const lid = mesh(new THREE.SphereGeometry(0.46, 12, 8, 0, Math.PI * 2, 0, Math.PI / 2), M.plate, root, 0, 0.05, 0);
  lid.scale.set(1, 0.55, 1.3);
  const eye = mesh(G.sph(0.1), M.eye, root, 0, 0, -0.52);
  mesh(G.cyl(0.05, 0.05, 0.5, 6), M.body, root, 0, -0.18, -0.35).rotation.x = Math.PI / 2;
  const thrusters = [];
  const parts = [
    { name: 'Rumpf', obj: body, r: 0.5, mult: 1, armor: 0.2 },
    { name: 'Auge', obj: eye, r: 0.14, mult: 1.6, armor: 0, weak: true },
  ];
  for (const [x, z] of [[-0.7, -0.45], [0.7, -0.45], [-0.7, 0.45], [0.7, 0.45]]) {
    mesh(G.box(Math.abs(x) * 1.1, 0.07, 0.1), M.body, root, x / 2, 0.05, z * 0.8);
    const t = mesh(G.cyl(0.22, 0.18, 0.2, 10), M.plateDark, root, x, 0.05, z);
    const glow = mesh(G.cyl(0.16, 0.16, 0.05, 10), M.thrust, t, 0, -0.1, 0);
    thrusters.push(t);
    parts.push({ name: 'Rotor', obj: t, r: 0.26, mult: 2, armor: 0, weak: true, hp: 18, thruster: true, glow });
  }
  root.scale.setScalar(scale);
  return { root, thrusters, parts };
}

export function buildHornet() {
  const root = new THREE.Group();
  const body = mesh(G.sph(0.6, 14, 10), M.body, root);
  body.scale.set(1.1, 0.8, 1.5);
  // Frontpanzer
  const shield = mesh(new THREE.SphereGeometry(0.75, 14, 10, 0, Math.PI * 2, 0, Math.PI / 2), M.plate, root, 0, 0, -0.35);
  shield.rotation.x = -Math.PI / 2;
  shield.scale.set(1, 1, 0.5);
  mesh(G.sph(0.12), M.eye, root, 0, 0.05, -0.78);
  mesh(G.box(0.12, 0.12, 0.7), M.body, root, 0.35, -0.3, -0.5);
  mesh(G.box(0.12, 0.12, 0.7), M.body, root, -0.35, -0.3, -0.5);
  const parts = [
    { name: 'Frontpanzer', obj: shield, r: 0.62, mult: 1, armor: 0.8, offset: new THREE.Vector3(0, 0, -0.55) },
    { name: 'Rumpf', obj: body, r: 0.65, mult: 1, armor: 0.35 },
  ];
  const thrusters = [];
  for (const [x, z] of [[-0.85, 0.55], [0.85, 0.55]]) {
    mesh(G.box(Math.abs(x) * 1.1, 0.1, 0.14), M.body, root, x / 2, 0.1, z);
    const t = mesh(G.cyl(0.32, 0.26, 0.3, 12), M.plateDark, root, x, 0.1, z);
    const glow = mesh(G.cyl(0.24, 0.24, 0.06, 12), M.thrust, t, 0, -0.15, 0);
    thrusters.push(t);
    parts.push({ name: 'Antrieb', obj: t, r: 0.36, mult: 2.2, armor: 0, weak: true, hp: 45, thruster: true, glow });
  }
  return { root, thrusters, parts };
}

export function buildSnitch() {
  const root = new THREE.Group();
  const body = mesh(G.cyl(0.35, 0.5, 0.4, 12), M.plate, root);
  mesh(G.sph(0.3, 10, 8), M.body, root, 0, -0.25, 0);
  const eye = mesh(G.sph(0.14), M.eye, root, 0, -0.38, 0);
  const ring = mesh(new THREE.TorusGeometry(0.75, 0.06, 6, 24), M.body, root, 0, 0.1, 0);
  ring.rotation.x = Math.PI / 2;
  const antenna = mesh(G.cyl(0.03, 0.03, 0.8, 6), M.body, root, 0, 0.6, 0);
  const beacon = mesh(G.sph(0.09), M.eye, antenna, 0, 0.42, 0);
  return { root, ring, beacon, parts: [
    { name: 'Rumpf', obj: body, r: 0.55, mult: 1, armor: 0 },
    { name: 'Scanner', obj: eye, r: 0.18, mult: 2, armor: 0, weak: true },
  ] };
}

export function buildRocketeer() {
  const root = new THREE.Group();
  const body = mesh(G.box(1.8, 1.1, 2.4), M.body, root);
  mesh(G.box(1.9, 0.35, 2.3), M.plate, root, 0, 0.65, 0);
  mesh(G.box(1.2, 0.25, 0.6), M.orange, root, 0, 0.85, 0.7);
  const eye = mesh(G.sph(0.2), M.eye, root, 0, 0.1, -1.25);
  const pods = [];
  for (const x of [-1.25, 1.25]) {
    const pod = mesh(G.box(0.6, 0.6, 1.4), M.plateDark, root, x, -0.1, -0.3);
    for (let i = 0; i < 4; i++) mesh(G.cyl(0.08, 0.08, 0.1, 8), M.orange, pod, (i % 2 - 0.5) * 0.28, (Math.floor(i / 2) - 0.5) * 0.28, -0.72).rotation.x = Math.PI / 2;
    pods.push(pod);
  }
  const parts = [
    { name: 'Panzer', obj: body, r: 1.35, mult: 1, armor: 0.55 },
    { name: 'Sensor', obj: eye, r: 0.28, mult: 2, armor: 0, weak: true },
  ];
  const thrusters = [];
  for (const [x, z] of [[-1.6, 0.9], [1.6, 0.9]]) {
    mesh(G.box(1, 0.15, 0.2), M.body, root, x / 2, 0.2, z);
    const t = mesh(G.cyl(0.55, 0.45, 0.5, 14), M.plateDark, root, x, 0.2, z);
    const glow = mesh(G.cyl(0.42, 0.42, 0.08, 14), M.thrust, t, 0, -0.25, 0);
    thrusters.push(t);
    parts.push({ name: 'Düse', obj: t, r: 0.6, mult: 2, armor: 0, weak: true, hp: 130, thruster: true, glow });
  }
  return { root, thrusters, pods, parts };
}

export function buildLeaper() {
  const root = new THREE.Group();
  const hull = new THREE.Group();
  hull.position.y = 2.2;
  root.add(hull);
  const body = mesh(G.sph(1.3, 14, 10), M.body, hull);
  body.scale.set(1.2, 0.7, 1.4);
  const shell = mesh(new THREE.SphereGeometry(1.45, 14, 10, 0, Math.PI * 2, 0, Math.PI / 2), M.plate, hull, 0, 0.1, 0.1);
  shell.scale.set(1.1, 0.6, 1.3);
  const eye = mesh(G.sph(0.3), M.weak, hull, 0, 0, -1.75);
  const legs = [];
  for (const [x, z] of [[-1.3, -1.1], [1.3, -1.1], [-1.3, 1.1], [1.3, 1.1]]) {
    const hip = new THREE.Group();
    hip.position.set(x, 0, z);
    hull.add(hip);
    const upper = mesh(G.box(0.35, 1.8, 0.35), M.plateDark, hip, Math.sign(x) * 0.6, 0.4, 0);
    upper.rotation.z = Math.sign(x) * -0.9;
    const lower = mesh(G.box(0.25, 2.6, 0.25), M.body, hip, Math.sign(x) * 1.35, -0.9, 0);
    lower.rotation.z = Math.sign(x) * 0.25;
    legs.push(hip);
  }
  return { root, hull, legs, parts: [
    { name: 'Panzer', obj: body, r: 1.5, mult: 1, armor: 0.6 },
    { name: 'Kernauge', obj: eye, r: 0.4, mult: 2.5, armor: 0, weak: true },
  ] };
}

export function buildBastion() {
  const root = new THREE.Group();
  const hull = new THREE.Group();
  hull.position.y = 4.2;
  root.add(hull);
  const body = mesh(G.box(3.4, 2.4, 3.8), M.plate, hull);
  mesh(G.box(3.6, 0.5, 4), M.plateDark, hull, 0, 1.4, 0);
  mesh(G.box(3.2, 1.6, 0.4), M.plate, hull, 0, 0.1, -2.0);
  mesh(G.box(3.5, 0.3, 3.9), M.body, hull, 0, -1.25, 0);
  mesh(G.box(0.3, 1.2, 3.9), M.orange, hull, 1.72, 0.2, 0);
  mesh(G.box(0.3, 1.2, 3.9), M.orange, hull, -1.72, 0.2, 0);
  mesh(G.box(1.2, 0.3, 0.2), M.eye, hull, 0, 0.7, -2.2);
  const core = mesh(G.sph(0.6, 12, 10), M.weak, hull, 0, 0.3, 2.0);
  mesh(G.box(1.8, 1.8, 0.3), M.plateDark, hull, 0, 0.3, 1.75);
  const guns = [];
  for (const x of [-2.1, 2.1]) {
    const g = new THREE.Group();
    g.position.set(x, 0, -0.8);
    hull.add(g);
    mesh(G.box(0.8, 0.8, 1.6), M.plateDark, g);
    for (let i = 0; i < 3; i++) mesh(G.cyl(0.09, 0.09, 1.6, 6), M.body, g, (i - 1) * 0.18, 0, -1.4).rotation.x = Math.PI / 2;
    guns.push(g);
  }
  const legs = [];
  for (const [x, z] of [[-1.7, -1.4], [1.7, -1.4], [-1.7, 1.4], [1.7, 1.4]]) {
    const hip = new THREE.Group();
    hip.position.set(x, -1, z);
    hull.add(hip);
    mesh(G.box(0.8, 3.4, 0.8), M.plateDark, hip, 0, -1.6, 0);
    mesh(G.box(1.2, 0.3, 1.4), M.body, hip, 0, -3.25, 0);
    legs.push(hip);
  }
  return { root, hull, guns, legs, parts: [
    { name: 'Rückenkern', obj: core, r: 0.75, mult: 3, armor: 0, weak: true },
    { name: 'Panzer', obj: body, r: 2.4, mult: 1, armor: 0.85 },
  ] };
}

export function buildTurret() {
  const root = new THREE.Group();
  mesh(G.cyl(0.7, 0.9, 0.5, 10), M.plateDark, root, 0, 0.25, 0);
  const head = new THREE.Group();
  head.position.y = 1.0;
  root.add(head);
  const body = mesh(G.box(0.9, 0.7, 1.1), M.body, head);
  mesh(G.box(1.0, 0.18, 1.2), M.plate, head, 0, 0.42, 0);
  mesh(G.cyl(0.08, 0.08, 1.4, 8), M.body, head, 0, 0, -1.1).rotation.x = Math.PI / 2;
  const lens = mesh(G.sph(0.18), M.eye, head, 0, 0.12, -0.58);
  return { root, head, parts: [
    { name: 'Gehäuse', obj: body, r: 0.75, mult: 1, armor: 0.7 },
    { name: 'Linse', obj: lens, r: 0.24, mult: 2.5, armor: 0, weak: true },
  ] };
}

// ---------------------------------------------------------------- Blender-Modelle

// Baut ein ARC-Modell aus assets/models.glb. Trefferzonen kommen aus den
// Custom Properties der Blender-Objekte (hit, r, mult, armor, hp, thruster).
export function buildArcModel(type) {
  const name = 'arc_' + type;
  if (!assets.has(name)) return null;
  const obj = assets.clone(name);
  const root = new THREE.Group();
  root.add(obj);
  const n = Assets.nodes(obj, name);
  const parts = [];
  obj.traverse((o) => {
    const u = o.userData;
    if (!u || !u.hit) return;
    const nm = o.name.slice(name.length + 2);
    parts.push({
      name: nm, obj: o, r: u.r, mult: u.mult ?? 1, armor: u.armor ?? 0, weak: u.hit === 'weak',
      hp: u.hp, thruster: !!u.thruster, glow: n[nm + 'Glow'],
    });
  });
  const list = (prefix) => {
    const out = [];
    for (let i = 0; n[prefix + i]; i++) out.push(n[prefix + i]);
    return out;
  };
  const fans = Object.keys(n).filter((k) => k.endsWith('Fan')).map((k) => n[k]);
  const model = {
    root, parts, glb: true, fans,
    legs: list('Leg'), thrusters: parts.filter((p) => p.thruster).map((p) => p.obj),
    ball: n.Ball, light: n.Light, ring: n.Ring, beacon: n.Beacon, pods: list('Pod'), hull: n.Hull, guns: list('Gun'), head: n.Head,
  };
  for (const k of ['hull', 'ball']) if (model[k]) model[k].userData.restY = model[k].position.y;
  obj.traverse((o) => { if (o.isMesh) o.castShadow = true; });
  return model;
}
