// Viele gleiche Blender-Modelle als InstancedMesh: pro Modell, Ebene und
// Kachel ein Mesh (damit Frustum- und Entfernungs-Culling greifen).
// Teile namens "...__Paint" werden pro Instanz eingefärbt.
import * as THREE from 'three';

const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _p = new THREE.Vector3();
const _s = new THREE.Vector3();
const _e = new THREE.Euler();
const _c = new THREE.Color();

export class Instancer {
  constructor(assets) {
    this.assets = assets;
    this.groups = new Map();
    this.count = 0;
  }

  has(name) {
    return !!(this.assets && this.assets.has(name));
  }

  // layer: '' (immer), 'det' (mittlere Entfernung), 'in' (Innenräume, nah)
  add(name, x, y, z, ry = 0, scale = 1, { tint = null, layer = '', rx = 0, rz = 0, shade = 1, double = false } = {}) {
    if (!this.has(name)) return false;
    const size = layer ? 40 : 80;
    const ix = Math.floor(x / size), iz = Math.floor(z / size);
    const key = `${name}|${layer}|${ix}|${iz}`;
    let g = this.groups.get(key);
    if (!g) {
      g = { name, layer, double, items: [], cx: (ix + 0.5) * size, cz: (iz + 0.5) * size, ySum: 0 };
      this.groups.set(key, g);
    }
    const sc = typeof scale === 'number' ? [scale, scale, scale] : scale;
    g.items.push({ x, y, z, rx, ry, rz, sc, tint, shade });
    g.ySum += y;
    this.count++;
    return true;
  }

  // doppelseitige Kopie eines Materials (für Modelle aus einzelnen Flächen)
  doubleSided(mat) {
    this.dsCache ||= new Map();
    let m = this.dsCache.get(mat);
    if (!m) {
      m = mat.clone();
      m.side = THREE.DoubleSide;
      this.dsCache.set(mat, m);
    }
    return m;
  }

  build(scene) {
    const out = [];
    for (const g of this.groups.values()) {
      const n = g.items.length;
      for (const part of this.assets.bakedMeshes(g.name)) {
        const paint = part.name.includes('__Paint');
        const lamp = part.name.includes('__Lamp');
        const im = new THREE.InstancedMesh(part.geometry, g.double ? this.doubleSided(part.material) : part.material, n);
        g.items.forEach((it, i) => {
          _e.set(it.rx, it.ry, it.rz, 'YXZ');
          _q.setFromEuler(_e);
          _p.set(it.x, it.y, it.z);
          _s.set(it.sc[0], it.sc[1], it.sc[2]);
          im.setMatrixAt(i, _m.compose(_p, _q, _s));
          if (paint && it.tint != null) _c.set(it.tint).multiplyScalar(it.shade);
          else _c.setScalar(lamp ? 1 : it.shade);
          im.setColorAt(i, _c);
        });
        im.instanceMatrix.needsUpdate = true;
        im.instanceColor.needsUpdate = true;
        im.computeBoundingSphere();
        im.castShadow = !g.layer && !lamp;
        im.receiveShadow = true;
        im.userData.layer = g.layer;
        im.userData.center = new THREE.Vector3(g.cx, g.ySum / n, g.cz);
        if (g.layer === 'in') im.visible = false;
        scene.add(im);
        out.push(im);
      }
    }
    this.groups.clear();
    return out;
  }
}
