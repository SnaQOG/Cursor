// Exportiert einen Ausschnitt der laufenden Welt für den Unreal-Vergleich: je Material und 40-m-Kachel eine .glb, dazu Texturen und manifest.json
//   node tools/export_unreal.mjs <zielordner> [x z radius]
import fs from 'node:fs';
import path from 'node:path';
import { open } from './tests/common.mjs';

const out = process.argv[2];
const [CX, CZ, R] = process.argv.slice(3).map(Number);
const center = [CX ?? -60, CZ ?? 96, R ?? 85];
fs.mkdirSync(path.join(out, 'tex'), { recursive: true });

const { browser, page, errors } = await open();
await page.evaluate(() => window.__game.selectMap('weltenbrueck'));
await page.waitForFunction(() => window.__game.mapId === 'weltenbrueck' && !window.__game.loadingMap, null, { timeout: 300000 });

// Geometrie im Browser einsammeln, die Welt läuft dort ohnehin
await page.evaluate(([CX, CZ, R]) => {
  const g = window.__game, w = g.world;
  g.startRaid();
  const THREE_M = w.staticMeshes[0].matrixWorld.constructor;
  const V3 = w.sunDir.constructor;
  const mats = new Map(), images = new Map(), buckets = new Map();
  const imageId = (tex, kind) => {
    if (!tex?.image) return null;
    if (!images.has(tex.image)) images.set(tex.image, { id: `t${images.size}`, kind, tex });
    return images.get(tex.image).id;
  };
  const keyOf = new Map(Object.entries(w.batcher.materials).map(([k, m]) => [m, k]));
  const matId = (m, terrain) => {
    if (mats.has(m)) return mats.get(m).id;
    const key = terrain ? 'terrain' : keyOf.get(m) || m.name || 'mat';
    const kind = terrain ? 'terrain' : m.transparent && key === 'glass' ? 'glass' : m.isMeshBasicMaterial ? 'glow' : m.alphaTest > 0 ? 'masked' : m.transparent ? 'skip' : 'opaque';
    const c = m.color ? [m.color.r, m.color.g, m.color.b] : [1, 1, 1];
    mats.set(m, { id: `m${mats.size}`, key, kind, color: c, rough: m.roughness ?? 0.9, metal: m.metalness ?? 0, map: imageId(m.map, 'color'), normal: imageId(m.normalMap, 'normal'), two: m.side === 2 });
    return mats.get(m).id;
  };
  const add = (mesh, mat, terrain) => {
    const id = matId(mat, terrain);
    if (mats.get(mat).kind === 'skip') return;
    const G = mesh.geometry, P = G.attributes.position, N = G.attributes.normal, U = G.attributes.uv, C = G.attributes.color, I = G.index;
    const T0 = G.attributes.gmix, T1 = G.attributes.rmix, T2 = G.attributes.wear;
    const map = mat.map; if (map) map.updateMatrix();
    const e = map ? map.matrix.elements : [1, 0, 0, 0, 1, 0, 0, 0, 1], flip = map ? map.flipY : true;
    const count = mesh.isInstancedMesh ? mesh.count : 1, M = new THREE_M(), W = new THREE_M(), ic = [1, 1, 1], tmp = { r: 1, g: 1, b: 1 };
    const v = new V3(), n = new V3(), pts = [new V3(), new V3(), new V3()], nors = [new V3(), new V3(), new V3()];
    const nTri = (I ? I.count : P.count) / 3;
    for (let k = 0; k < count; k++) {
      W.copy(mesh.matrixWorld);
      if (mesh.isInstancedMesh) {
        mesh.getMatrixAt(k, M);
        W.multiply(M);
        if (mesh.instanceColor) { mesh.getColorAt(k, tmp); ic[0] = tmp.r; ic[1] = tmp.g; ic[2] = tmp.b; }
      }
      for (let t = 0; t < nTri; t++) {
        const ids = [0, 1, 2].map((q) => (I ? I.getX(t * 3 + q) : t * 3 + q));
        for (let q = 0; q < 3; q++) {
          pts[q].fromBufferAttribute(P, ids[q]).applyMatrix4(W);
          nors[q].fromBufferAttribute(N, ids[q]).transformDirection(W);
        }
        const mx = (pts[0].x + pts[1].x + pts[2].x) / 3, mz = (pts[0].z + pts[1].z + pts[2].z) / 3;
        if (Math.hypot(mx - CX, mz - CZ) > R) continue;
        const key = `${id}|${Math.floor(mx / 40)}|${Math.floor(mz / 40)}`;
        let b = buckets.get(key);
        if (!b) buckets.set(key, (b = { mat: id, pos: [], nor: [], uv: [], col: [] }));
        for (let q = 0; q < 3; q++) {
          const i = ids[q];
          b.pos.push(pts[q].x, pts[q].y, pts[q].z);
          b.nor.push(nors[q].x, nors[q].y, nors[q].z);
          let u = U ? U.getX(i) : pts[q].x / 4, vv = U ? U.getY(i) : pts[q].z / 4;
          const tu = e[0] * u + e[3] * vv + e[6], tv = e[1] * u + e[4] * vv + e[7];
          b.uv.push(tu, flip ? 1 - tv : tv);
          if (terrain) b.col.push(T0 ? T0.getX(i) : 0, T1 ? T1.getX(i) : 0, T2 ? T2.getX(i) : 0);
          else if (C) b.col.push(C.getX(i) * ic[0], C.getY(i) * ic[1], C.getZ(i) * ic[2]);
          else b.col.push(ic[0], ic[1], ic[2]);
        }
      }
    }
    void v; void n;
  };
  const all = new Set([...w.staticMeshes, ...(w.vegDetail || []), ...(w.farMeshes || [])]);
  for (const m of all) {
    if (!m.isMesh) continue;
    for (const mat of Array.isArray(m.material) ? m.material : [m.material]) add(m, mat, false);
  }
  g.scene.traverse((o) => { if (o.isMesh && o.geometry.attributes.gmix) add(o, o.material, true); });
  // Bodentexturen für den Mischshader in Unreal
  const extra = {};
  for (const [name, s] of Object.entries(w.assets.tex || {})) {
    extra[name] = { map: imageId(s.map, 'color'), normal: s.normal ? imageId(s.normal, 'normal') : null };
  }
  window.__exp = { mats: [...mats.values()], images: [...images.values()], buckets: [...buckets.entries()], extra };
}, center);

const summary = await page.evaluate(() => {
  const e = window.__exp;
  return { mats: e.mats, extra: e.extra, images: e.images.map((i) => ({ id: i.id, kind: i.kind })), buckets: e.buckets.map(([k, b]) => [k, b.mat, b.pos.length / 3]) };
});

// Texturen als PNG
for (let i = 0; i < summary.images.length; i++) {
  const url = await page.evaluate((i) => {
    const { tex } = window.__exp.images[i], img = tex.image, k = Math.min(1, 2048 / Math.max(img.width, img.height));
    const c = document.createElement('canvas');
    c.width = Math.round(img.width * k); c.height = Math.round(img.height * k);
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    return c.toDataURL('image/png');
  }, i);
  fs.writeFileSync(path.join(out, 'tex', `${summary.images[i].id}.png`), Buffer.from(url.split(',')[1], 'base64'));
}

// Ein glb je Kachel
function glb(name, pos, nor, uv, col) {
  const n = pos.length / 3, lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], pos[i * 3 + a]); hi[a] = Math.max(hi[a], pos[i * 3 + a]); }
  const bin = Buffer.concat([pos, nor, uv, col].map((f) => Buffer.from(f.buffer, f.byteOffset, f.byteLength)));
  const views = [pos, nor, uv, col]; let off = 0;
  const json = {
    asset: { version: '2.0', generator: 'damm-raider' }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ name, mesh: 0 }],
    meshes: [{ name, primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2, COLOR_0: 3 } }] }],
    buffers: [{ byteLength: bin.length }],
    bufferViews: views.map((f) => { const r = { buffer: 0, byteOffset: off, byteLength: f.byteLength }; off += f.byteLength; return r; }),
    accessors: [
      { bufferView: 0, componentType: 5126, count: n, type: 'VEC3', min: lo, max: hi }, { bufferView: 1, componentType: 5126, count: n, type: 'VEC3' },
      { bufferView: 2, componentType: 5126, count: n, type: 'VEC2' }, { bufferView: 3, componentType: 5126, count: n, type: 'VEC3' },
    ],
  };
  let js = Buffer.from(JSON.stringify(json));
  js = Buffer.concat([js, Buffer.alloc((4 - (js.length % 4)) % 4, 0x20)]);
  const head = Buffer.alloc(20);
  head.writeUInt32LE(0x46546c67, 0); head.writeUInt32LE(2, 4); head.writeUInt32LE(12 + 8 + js.length + 8 + bin.length, 8);
  head.writeUInt32LE(js.length, 12); head.writeUInt32LE(0x4e4f534a, 16);
  const bh = Buffer.alloc(8); bh.writeUInt32LE(bin.length, 0); bh.writeUInt32LE(0x004e4942, 4);
  return Buffer.concat([head, js, bh, bin]);
}

const files = [];
let verts = 0;
for (let i = 0; i < summary.buckets.length; i++) {
  const [key, mat, n] = summary.buckets[i];
  if (!n) continue;
  const arrs = await page.evaluate((i) => {
    const b = window.__exp.buckets[i][1];
    const enc = (a) => { const u = new Uint8Array(new Float32Array(a).buffer); let s = ''; for (let k = 0; k < u.length; k += 32768) s += String.fromCharCode.apply(null, u.subarray(k, k + 32768)); return btoa(s); };
    return [enc(b.pos), enc(b.nor), enc(b.uv), enc(b.col)];
  }, i);
  const [pos, nor, uv, col] = arrs.map((s) => { const b = Buffer.from(s, 'base64'); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); });
  const name = `S_${key.replace(/\|/g, '_').replace(/-/g, 'n')}`;
  fs.writeFileSync(path.join(out, `${name}.glb`), glb(name, pos, nor, uv, col));
  files.push({ name, mat });
  verts += n;
}
fs.writeFileSync(path.join(out, 'manifest.json'), JSON.stringify({ center, mats: summary.mats, images: summary.images, extra: summary.extra, files }, null, 1));
console.log('Kacheln', files.length, 'Ecken', verts, 'Materialien', summary.mats.length, 'Texturen', summary.images.length);
console.log(summary.mats.map((m) => `${m.key}:${m.kind}`).join(' '));
console.log('Fehler:', errors.filter((e) => !/fonts|ERR_|GPU stall|GL Driver/.test(e)).slice(0, 8));
await browser.close();
