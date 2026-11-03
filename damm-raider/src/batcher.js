// Sammelt viele statische Formen in wenige große Meshes (pro Material und
// 80-m-Kachel), damit die Karte mit wenigen Draw-Calls auskommt.
// Innenräume (layer 'in') landen in kleineren Kacheln, die die Welt nur in
// der Nähe des Spielers zeichnet.
import * as THREE from 'three';

const CHUNK = 80;
const CHUNK_IN = 40;
const _m = new THREE.Matrix4();
const _n = new THREE.Matrix3();
const _v = new THREE.Vector3();
const _nv = new THREE.Vector3();
const _c = new THREE.Color();
const _q = new THREE.Quaternion();
const _e = new THREE.Euler();
const _s = new THREE.Vector3();
const _p = new THREE.Vector3();
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();

// Wiederverwendbare Grundformen (nicht indiziert)
const ni = (g) => (g.index ? g.toNonIndexed() : g);
const PRIMS = {
  box: ni(new THREE.BoxGeometry(1, 1, 1)),
  cyl8: ni(new THREE.CylinderGeometry(1, 1, 1, 8, 1)),
  cyl16: ni(new THREE.CylinderGeometry(1, 1, 1, 16, 1)),
  cyl24open: ni(new THREE.CylinderGeometry(1, 1, 1, 24, 1, true)),
  cone8: ni(new THREE.ConeGeometry(1, 1, 8, 1)),
  cone4: ni(new THREE.ConeGeometry(1, 1, 4, 1)),
  sphere: ni(new THREE.SphereGeometry(1, 12, 8)),
  dome: ni(new THREE.SphereGeometry(1, 20, 10, 0, Math.PI * 2, 0, Math.PI / 2)),
  ico: ni(new THREE.IcosahedronGeometry(1, 0)),
  ico1: ni(new THREE.IcosahedronGeometry(1, 1)),
  torus: ni(new THREE.TorusGeometry(1, 0.08, 6, 24)),
};

// Wachsender Float32-Puffer (spart gegenüber JS-Arrays die Hälfte Speicher)
class FBuf {
  constructor(n = 3072) { this.a = new Float32Array(n); this.n = 0; }
  push3(x, y, z) {
    if (this.n + 3 > this.a.length) this.grow();
    const a = this.a, n = this.n;
    a[n] = x; a[n + 1] = y; a[n + 2] = z;
    this.n = n + 3;
  }
  push2(x, y) {
    if (this.n + 2 > this.a.length) this.grow();
    this.a[this.n] = x; this.a[this.n + 1] = y;
    this.n += 2;
  }
  grow() {
    const b = new Float32Array(this.a.length * 2);
    b.set(this.a);
    this.a = b;
  }
  array() { return this.a.subarray(0, this.n); }
}

class Bucket {
  constructor(mat, layer, cx, cz, size) {
    this.mat = mat;
    this.layer = layer;
    this.center = new THREE.Vector3((cx + 0.5) * size, 0, (cz + 0.5) * size);
    this.ySum = 0;
    this.pos = new FBuf();
    this.nor = new FBuf();
    this.col = new FBuf();
    this.uv = new FBuf();
  }
}

export class Batcher {
  constructor(materials) {
    this.materials = materials; // key -> Material
    this.buckets = new Map(); // `${mat}|${layer}|${cx}|${cz}` -> Bucket
    this.vertexCount = 0;
  }

  bucket(mat, x, z, layer = '') {
    const size = layer ? CHUNK_IN : CHUNK;
    const cx = Math.floor(x / size), cz = Math.floor(z / size);
    const key = `${mat}|${layer}|${cx}|${cz}`;
    let b = this.buckets.get(key);
    if (!b) {
      b = new Bucket(mat, layer, cx, cz, size);
      this.buckets.set(key, b);
    }
    return b;
  }

  // Ecke abdunkeln, wenn sie in einem Innenraum liegt (falsche Umgebungsverdeckung)
  static innerK(inner, x, y, z) {
    if (!inner) return 1;
    const list = Array.isArray(inner) ? inner : [inner];
    for (const r of list) {
      if (x > r.x0 && x < r.x1 && z > r.z0 && z < r.z1 && (r.y0 == null || (y > r.y0 && y < r.y1))) return r.k ?? 0.55;
    }
    return 1;
  }

  pushVertex(b, v, nv, color, k) {
    b.pos.push3(v.x, v.y, v.z);
    b.nor.push3(nv.x, nv.y, nv.z);
    b.col.push3(color.r * k, color.g * k, color.b * k);
    b.ySum += v.y;
    // Welt-UV (triplanar grob nach Normalen gewählt)
    const ax = Math.abs(nv.x), ay = Math.abs(nv.y), az = Math.abs(nv.z);
    if (ay >= ax && ay >= az) b.uv.push2(v.x / 4, v.z / 4);
    else if (ax >= az) b.uv.push2(v.z / 4, v.y / 4);
    else b.uv.push2(v.x / 4, v.y / 4);
    this.vertexCount++;
  }

  // Fügt eine Grundform mit Transformation hinzu.
  addPrim(prim, x, y, z, sx, sy, sz, color, { rx = 0, ry = 0, rz = 0, mat = 'world', shade = true, layer = '', inner = null } = {}) {
    const g = PRIMS[prim];
    _e.set(rx, ry, rz, 'YXZ');
    _q.setFromEuler(_e);
    _s.set(sx, sy, sz);
    _p.set(x, y, z);
    _m.compose(_p, _q, _s);
    _n.getNormalMatrix(_m);
    const b = this.bucket(mat, x, z, layer);
    const P = g.attributes.position, N = g.attributes.normal;
    _c.set(color);
    for (let i = 0; i < P.count; i++) {
      _v.fromBufferAttribute(P, i).applyMatrix4(_m);
      _nv.fromBufferAttribute(N, i).applyMatrix3(_n).normalize();
      // leichte Schattierung nach unten für mehr Tiefe
      let k = shade ? 0.86 + 0.14 * (_nv.y * 0.5 + 0.5) : 1;
      if (inner) k *= Batcher.innerK(inner, _v.x - _nv.x * 0.01, _v.y, _v.z - _nv.z * 0.01);
      this.pushVertex(b, _v, _nv, _c, k);
    }
  }

  // Freies Viereck (a,b,c,d gegen den Uhrzeigersinn von vorn gesehen); double = beidseitig
  addQuad(a, b, c, d, color, { mat = 'world', double = false, layer = '', inner = null, shade = true, kf = null } = {}) {
    this.addTri(a, b, c, color, { mat, double, layer, inner, shade, kf });
    this.addTri(a, c, d, color, { mat, double, layer, inner, shade, kf });
  }

  // kf: Helligkeit abhängig von der Höhe (für Schmutz am Sockel und Schatten unter der Traufe)
  addTri(a, b, c, color, { mat = 'world', double = false, layer = '', inner = null, shade = true, kf = null } = {}) {
    const cx = (a[0] + b[0] + c[0]) / 3, cz = (a[2] + b[2] + c[2]) / 3;
    const bk = this.bucket(mat, cx, cz, layer);
    _a.set(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    _b.set(c[0] - a[0], c[1] - a[1], c[2] - a[2]);
    _nv.crossVectors(_a, _b).normalize();
    _c.set(color);
    const side = (pts, n) => {
      let k = shade ? 0.86 + 0.14 * (n.y * 0.5 + 0.5) : 1;
      for (const p of pts) {
        _v.set(p[0], p[1], p[2]);
        let kk = inner ? k * Batcher.innerK(inner, p[0] - n.x * 0.01, p[1], p[2] - n.z * 0.01) : k;
        if (kf) kk *= kf(p[1]);
        this.pushVertex(bk, _v, n, _c, kk);
      }
    };
    side([a, b, c], _nv);
    if (double) {
      const back = _nv.clone().negate();
      side([a, c, b], back);
    }
  }

  // Blender-Modell einbacken. meshes: Liste aus Assets.meshes(), matrix: Platzierung,
  // tint: Farbe für die "key"-Lackflächen (lineare THREE.Color)
  addModel(meshes, matrix, tint, classify, layer = '') {
    const px = matrix.elements[12], pz = matrix.elements[14];
    for (const m of meshes) {
      const g = m.geometry;
      const mat = classify(m.material);
      const b = this.bucket(mat, px, pz, layer);
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
        let r = base.r, gg = base.g, bl = base.b;
        if (C && mat !== 'glow') {
          r *= C.getX(i); gg *= C.getY(i); bl *= C.getZ(i);
          // Lackflächen umfärben: Magenta = Lack, dunkles Magenta = dunkler Lack
          if (tint && gg < 0.05 && Math.abs(r - bl) < 0.02) {
            if (r > 0.9) { r = tint.r; gg = tint.g; bl = tint.b; }
            else if (r > 0.15 && r < 0.3) { r = tint.r * 0.7; gg = tint.g * 0.7; bl = tint.b * 0.7; }
          }
        }
        _c.setRGB(r, gg, bl);
        this.pushVertex(b, _v, _nv, _c, 1);
      }
    }
  }

  // Modell mit eigener Textur einbacken (Requisiten im gemeinsamen Atlas): UVs bleiben erhalten
  addModelUV(meshes, matrix, color, mat = 'dress', layer = 'det', bright = 1) {
    const b = this.bucket(mat, matrix.elements[12], matrix.elements[14], layer);
    _c.set(color).multiplyScalar(bright);
    for (const m of meshes) {
      const g = m.geometry;
      _m.multiplyMatrices(matrix, m.matrix);
      _n.getNormalMatrix(_m);
      const P = g.attributes.position, N = g.attributes.normal, U = g.attributes.uv, idx = g.index;
      const count = idx ? idx.count : P.count;
      for (let k = 0; k < count; k++) {
        const i = idx ? idx.getX(k) : k;
        _v.fromBufferAttribute(P, i).applyMatrix4(_m);
        _nv.fromBufferAttribute(N, i).applyMatrix3(_n).normalize();
        b.pos.push3(_v.x, _v.y, _v.z);
        b.nor.push3(_nv.x, _nv.y, _nv.z);
        b.col.push3(_c.r, _c.g, _c.b);
        b.uv.push2(U.getX(i), U.getY(i));
        b.ySum += _v.y;
        this.vertexCount++;
      }
    }
  }

  // Bodenpflanze aus dem Atlas: zwei gekreuzte Bildflächen, Normalen nach oben, damit sie wie der Boden beleuchtet wird.
  // cell: [Spalte, Zeile (von oben)], color: Farbe oben, unten wird sie dunkler und läuft in den Boden aus
  addCover(x, y, z, size, ry, cell, cols, rows, color, k = 1) {
    const b = this.bucket('cover', x, z, 'det');
    const u0 = (cell[0] + 0.004) / cols, u1 = (cell[0] + 0.996) / cols;
    const v0 = (rows - 1 - cell[1] + 0.004) / rows, v1 = (rows - cell[1] - 0.004) / rows;
    _c.set(color).multiplyScalar(k);
    const h = size * 0.5, base = 0.62;
    for (let i = 0; i < 2; i++) {
      const a = ry + i * Math.PI / 2, dx = Math.cos(a) * h, dz = Math.sin(a) * h;
      const corners = [[-1, 0, u0, v0, base], [1, 0, u1, v0, base], [1, 1, u1, v1, 1], [-1, 0, u0, v0, base], [1, 1, u1, v1, 1], [-1, 1, u0, v1, 1]];
      for (const [s, t, u, v, kk] of corners) {
        b.pos.push3(x + dx * s, y + t * size, z + dz * s);
        b.nor.push3(0, 1, 0);
        b.col.push3(_c.r * kk, _c.g * kk, _c.b * kk);
        b.uv.push2(u, v);
        b.ySum += y;
        this.vertexCount++;
      }
    }
  }

  // Viereck mit festem Bildausschnitt (Schilder): r = [u0, v0, u1, v1], a unten links, gegen den Uhrzeigersinn
  addUVQuad(a, b, c, d, n, r, mat, layer = 'det') {
    const bk = this.bucket(mat, (a[0] + c[0]) / 2, (a[2] + c[2]) / 2, layer);
    const uv = [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]], pts = [a, b, c, d];
    for (const i of [0, 1, 2, 0, 2, 3]) {
      const p = pts[i];
      bk.pos.push3(p[0], p[1], p[2]);
      bk.nor.push3(n[0], n[1], n[2]);
      bk.col.push3(1, 1, 1);
      bk.uv.push2(uv[i][0], uv[i][1]);
      bk.ySum += p[1];
      this.vertexCount++;
    }
  }

  // Weicher Kontaktschatten am Boden: verbindet ein Objekt mit dem Untergrund, auch außerhalb der Schattenkarte
  addContact(x, y, z, sx, sz, ry = 0, layer = 'det') {
    const c = Math.cos(ry), s = Math.sin(ry), P = (u, v) => [x + u * sx * c + v * sz * s, y + 0.025, z - u * sx * s + v * sz * c];
    const a = P(-1, 1), b = P(1, 1), cc = P(1, -1), d = P(-1, -1);
    const bk = this.bucket('contact', x, z, layer), uv = [[0, 0], [1, 0], [1, 1], [0, 1]], pts = [a, b, cc, d];
    for (const i of [0, 1, 2, 0, 2, 3]) {
      bk.pos.push3(pts[i][0], pts[i][1], pts[i][2]);
      bk.nor.push3(0, 1, 0);
      bk.col.push3(0, 0, 0);
      bk.uv.push2(uv[i][0], uv[i][1]);
      bk.ySum += y;
      this.vertexCount++;
    }
  }

  // Box mit Unterkante bei y
  box(x, y, z, w, h, d, color, opts) {
    this.addPrim('box', x, y + h / 2, z, w, h, d, color, opts);
  }

  build(scene) {
    const meshes = [];
    for (const b of this.buckets.values()) {
      if (!b.pos.n) continue;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(b.pos.array().slice(), 3));
      geo.setAttribute('normal', new THREE.BufferAttribute(b.nor.array().slice(), 3));
      geo.setAttribute('color', new THREE.BufferAttribute(b.col.array().slice(), 3));
      geo.setAttribute('uv', new THREE.BufferAttribute(b.uv.array().slice(), 2));
      geo.computeBoundingSphere();
      const mat = this.materials[b.mat];
      const mesh = new THREE.Mesh(geo, mat);
      const inside = b.layer === 'in';
      mesh.castShadow = !mat.transparent && b.mat !== 'glow' && (!b.layer || b.mat === 'dress');
      mesh.receiveShadow = !mat.transparent;
      mesh.matrixAutoUpdate = false;
      mesh.updateMatrix();
      mesh.userData.layer = b.layer;
      mesh.userData.cover = b.mat === 'cover';
      mesh.userData.center = b.center.clone().setY(b.ySum / (b.pos.n / 3));
      if (inside) mesh.visible = false;
      scene.add(mesh);
      meshes.push(mesh);
    }
    this.buckets.clear();
    return meshes;
  }
}
