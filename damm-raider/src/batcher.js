// Sammelt viele statische Formen in wenige große Meshes (pro Material und
// 80-m-Kachel), damit die Karte mit wenigen Draw-Calls auskommt.
import * as THREE from 'three';

const CHUNK = 80;
const _m = new THREE.Matrix4();
const _n = new THREE.Matrix3();
const _v = new THREE.Vector3();
const _nv = new THREE.Vector3();
const _c = new THREE.Color();
const _q = new THREE.Quaternion();
const _e = new THREE.Euler();
const _s = new THREE.Vector3();
const _p = new THREE.Vector3();

// Wiederverwendbare Grundformen (nicht indiziert)
const ni = (g) => (g.index ? g.toNonIndexed() : g);
const PRIMS = {
  box: ni(new THREE.BoxGeometry(1, 1, 1)),
  cyl8: ni(new THREE.CylinderGeometry(1, 1, 1, 8, 1)),
  cyl16: ni(new THREE.CylinderGeometry(1, 1, 1, 16, 1)),
  cyl24open: ni(new THREE.CylinderGeometry(1, 1, 1, 24, 1, true)),
  cone8: ni(new THREE.ConeGeometry(1, 1, 8, 1)),
  sphere: ni(new THREE.SphereGeometry(1, 12, 8)),
  dome: ni(new THREE.SphereGeometry(1, 20, 10, 0, Math.PI * 2, 0, Math.PI / 2)),
  ico: ni(new THREE.IcosahedronGeometry(1, 0)),
};

class Bucket {
  constructor() {
    this.pos = [];
    this.nor = [];
    this.col = [];
    this.uv = [];
  }
}

export class Batcher {
  constructor(materials) {
    this.materials = materials; // key -> Material
    this.buckets = new Map(); // `${mat}|${cx}|${cz}` -> Bucket
  }

  bucket(mat, x, z) {
    const cx = Math.floor(x / CHUNK), cz = Math.floor(z / CHUNK);
    const key = `${mat}|${cx}|${cz}`;
    let b = this.buckets.get(key);
    if (!b) {
      b = new Bucket();
      b.mat = mat;
      this.buckets.set(key, b);
    }
    return b;
  }

  // Fügt eine Grundform mit Transformation hinzu.
  addPrim(prim, x, y, z, sx, sy, sz, color, { rx = 0, ry = 0, rz = 0, mat = 'world', shade = true } = {}) {
    const g = PRIMS[prim];
    _e.set(rx, ry, rz, 'YXZ');
    _q.setFromEuler(_e);
    _s.set(sx, sy, sz);
    _p.set(x, y, z);
    _m.compose(_p, _q, _s);
    _n.getNormalMatrix(_m);
    const b = this.bucket(mat, x, z);
    const P = g.attributes.position, N = g.attributes.normal;
    _c.set(color);
    for (let i = 0; i < P.count; i++) {
      _v.fromBufferAttribute(P, i).applyMatrix4(_m);
      _nv.fromBufferAttribute(N, i).applyMatrix3(_n).normalize();
      b.pos.push(_v.x, _v.y, _v.z);
      b.nor.push(_nv.x, _nv.y, _nv.z);
      // leichte Schattierung nach unten für mehr Tiefe
      const k = shade ? 0.86 + 0.14 * (_nv.y * 0.5 + 0.5) : 1;
      b.col.push(_c.r * k, _c.g * k, _c.b * k);
      // Welt-UV (triplanar grob nach Normalen gewählt)
      const ax = Math.abs(_nv.x), ay = Math.abs(_nv.y), az = Math.abs(_nv.z);
      if (ay >= ax && ay >= az) b.uv.push(_v.x / 4, _v.z / 4);
      else if (ax >= az) b.uv.push(_v.z / 4, _v.y / 4);
      else b.uv.push(_v.x / 4, _v.y / 4);
    }
  }

  // Blender-Modell einbacken. meshes: Liste aus Assets.meshes(), matrix: Platzierung,
  // tint: Farbe für die "key"-Lackflächen (lineare THREE.Color)
  addModel(meshes, matrix, tint, classify) {
    const px = matrix.elements[12], pz = matrix.elements[14];
    for (const m of meshes) {
      const g = m.geometry;
      const mat = classify(m.material);
      const b = this.bucket(mat, px, pz);
      _m.multiplyMatrices(matrix, m.matrix);
      _n.getNormalMatrix(_m);
      const P = g.attributes.position, N = g.attributes.normal, C = g.attributes.color;
      const idx = g.index;
      const count = idx ? idx.count : P.count;
      const base = mat === 'glow' ? m.material.emissive : m.material.color;
      for (let k = 0; k < count; k++) {
        const i = idx ? idx.getX(k) : k;
        _v.fromBufferAttribute(P, i).applyMatrix4(_m);
        _nv.fromBufferAttribute(N, i).applyMatrix3(_n).normalize();
        b.pos.push(_v.x, _v.y, _v.z);
        b.nor.push(_nv.x, _nv.y, _nv.z);
        let r = base.r, gg = base.g, bl = base.b;
        if (C && mat !== 'glow') {
          r *= C.getX(i); gg *= C.getY(i); bl *= C.getZ(i);
          // Lackflächen umfärben: Magenta = Lack, dunkles Magenta = dunkler Lack
          if (tint && gg < 0.05 && Math.abs(r - bl) < 0.02) {
            if (r > 0.9) { r = tint.r; gg = tint.g; bl = tint.b; }
            else if (r > 0.15 && r < 0.3) { r = tint.r * 0.7; gg = tint.g * 0.7; bl = tint.b * 0.7; }
          }
        }
        b.col.push(r, gg, bl);
        const ax = Math.abs(_nv.x), ay = Math.abs(_nv.y), az = Math.abs(_nv.z);
        if (ay >= ax && ay >= az) b.uv.push(_v.x / 4, _v.z / 4);
        else if (ax >= az) b.uv.push(_v.z / 4, _v.y / 4);
        else b.uv.push(_v.x / 4, _v.y / 4);
      }
    }
  }

  // Box mit Unterkante bei y
  box(x, y, z, w, h, d, color, opts) {
    this.addPrim('box', x, y + h / 2, z, w, h, d, color, opts);
  }

  build(scene) {
    const meshes = [];
    for (const b of this.buckets.values()) {
      if (!b.pos.length) continue;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(b.pos, 3));
      geo.setAttribute('normal', new THREE.Float32BufferAttribute(b.nor, 3));
      geo.setAttribute('color', new THREE.Float32BufferAttribute(b.col, 3));
      geo.setAttribute('uv', new THREE.Float32BufferAttribute(b.uv, 2));
      geo.computeBoundingSphere();
      const mat = this.materials[b.mat];
      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = !mat.transparent && b.mat !== 'glow';
      mesh.receiveShadow = !mat.transparent;
      mesh.matrixAutoUpdate = false;
      mesh.updateMatrix();
      scene.add(mesh);
      meshes.push(mesh);
    }
    this.buckets.clear();
    return meshes;
  }
}
