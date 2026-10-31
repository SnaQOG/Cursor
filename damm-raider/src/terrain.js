// Höhenfeld der Karte: Tal unterhalb des Damms, Stausee-Plateau im Norden,
// Fluss, Giftsumpf und Berge am Rand.
import * as THREE from 'three';
import { MAP, PADS, ROADS } from './config.js';
import { Noise2D, clamp, lerp, smoothstep, distToPolyline } from './util.js';

function sdRoundBox(px, pz, hx, hz, r) {
  const qx = Math.abs(px) - hx + r, qz = Math.abs(pz) - hz + r;
  const ox = Math.max(qx, 0), oz = Math.max(qz, 0);
  return Math.sqrt(ox * ox + oz * oz) + Math.min(Math.max(qx, qz), 0) - r;
}

export class Terrain {
  constructor() {
    this.noise = new Noise2D(4242);
    this.N = MAP.RES;
    this.half = MAP.HALF;
    this.step = (MAP.HALF * 2) / (this.N - 1);
    this.heights = new Float32Array(this.N * this.N);
    this.pads = PADS.map((p) => ({ ...p, h: p.h === 'auto' ? this.rawHeight(p.x, p.z, false) : p.h }));
    for (let iz = 0; iz < this.N; iz++) {
      for (let ix = 0; ix < this.N; ix++) {
        const x = -this.half + ix * this.step, z = -this.half + iz * this.step;
        this.heights[iz * this.N + ix] = this.rawHeight(x, z, true);
      }
    }
  }

  lakeMask(x, z) {
    const L = MAP.LAKE;
    const d = sdRoundBox(x - L.cx, z - L.cz, L.hx, L.hz, L.r) + this.noise.noise(x * 0.02, z * 0.02) * 10;
    return 1 - smoothstep(-10, 6, d);
  }

  swampMask(x, z) {
    const S = MAP.SWAMP;
    const d = Math.hypot(x - S.x, z - S.z) + this.noise.noise(x * 0.03 + 5, z * 0.03) * 14;
    return 1 - smoothstep(S.r, S.r + S.blend, d);
  }

  rawHeight(x, z, withPads) {
    const n = this.noise;
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
      const l = this.lakeMask(x, z);
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
    const s = this.swampMask(x, z);
    if (s > 0) h = lerp(h, 0.5 + n.fbm(x * 0.05, z * 0.05, 3) * 1.8, s);

    if (withPads) {
      for (const pad of this.pads) {
        const d = Math.hypot(x - pad.x, z - pad.z);
        if (d > pad.r + pad.blend) continue;
        const w = 1 - smoothstep(pad.r, pad.r + pad.blend, d);
        h = lerp(h, pad.h, w);
      }
    }
    return h;
  }

  // Exakt dieselbe Dreiecksaufteilung wie das Mesh
  height(x, z) {
    const N = this.N;
    let fx = (x + this.half) / this.step, fz = (z + this.half) / this.step;
    fx = clamp(fx, 0, N - 1.001);
    fz = clamp(fz, 0, N - 1.001);
    const ix = Math.floor(fx), iz = Math.floor(fz);
    const tx = fx - ix, tz = fz - iz;
    const H = this.heights;
    const h00 = H[iz * N + ix], h10 = H[iz * N + ix + 1];
    const h01 = H[(iz + 1) * N + ix], h11 = H[(iz + 1) * N + ix + 1];
    if (tx + tz <= 1) return h00 + (h10 - h00) * tx + (h01 - h00) * tz;
    return h11 + (h01 - h11) * (1 - tx) + (h10 - h11) * (1 - tz);
  }

  slope(x, z) {
    const e = 1.0;
    const dx = this.height(x + e, z) - this.height(x - e, z);
    const dz = this.height(x, z + e) - this.height(x, z - e);
    return Math.sqrt(dx * dx + dz * dz) / (2 * e);
  }

  // Wasser an dieser Stelle? Liefert Pegel und ob giftig.
  waterAt(x, z) {
    const S = MAP.SWAMP;
    if (Math.hypot(x - S.x, z - S.z) < S.r + S.blend + 10) return { level: MAP.SWAMP_WATER, toxic: true };
    if (z < MAP.DAM_Z && Math.abs(x) < MAP.LAKE.hx + 2) return { level: MAP.LAKE_LEVEL, toxic: false };
    if (z > MAP.DAM_Z + 4) return { level: MAP.VALLEY_WATER, toxic: false };
    return null;
  }

  buildMesh(detailTex) {
    const N = this.N;
    const pos = new Float32Array(N * N * 3);
    const col = new Float32Array(N * N * 3);
    const uv = new Float32Array(N * N * 2);
    const c = new THREE.Color();
    const grass = new THREE.Color(0x5f7a36), dry = new THREE.Color(0xa09a55), lush = new THREE.Color(0x46652c);
    const rock = new THREE.Color(0x77705f), rockDark = new THREE.Color(0x4f4a42), mud = new THREE.Color(0x5b4c38);
    const sand = new THREE.Color(0x9b8d6a), swamp = new THREE.Color(0x3c4526), road = new THREE.Color(0x6f685a);
    const gravel = new THREE.Color(0x8b867a);
    const n = this.noise;

    for (let iz = 0; iz < N; iz++) {
      for (let ix = 0; ix < N; ix++) {
        const i = iz * N + ix;
        const x = -this.half + ix * this.step, z = -this.half + iz * this.step;
        const h = this.heights[i];
        pos[i * 3] = x; pos[i * 3 + 1] = h; pos[i * 3 + 2] = z;
        uv[i * 2] = x / 6; uv[i * 2 + 1] = z / 6;

        const sl = this.slope(x, z);
        const nv = n.fbm(x * 0.04 + 3, z * 0.04 - 9, 3);
        c.copy(grass).lerp(dry, clamp(0.45 + nv * 1.4, 0, 1));
        if (z < -90) c.lerp(lush, 0.35);
        // Ufer / Schlamm
        const w = this.waterAt(x, z);
        if (w) {
          const above = h - w.level;
          if (above < 1.2) c.lerp(w.toxic ? swamp : (w.level > 10 ? sand : mud), clamp(1 - above / 1.2, 0, 1));
        }
        const sm = this.swampMask(x, z);
        if (sm > 0) c.lerp(swamp, sm * 0.8);
        // Straßen
        let rd = Infinity;
        for (const r of ROADS) rd = Math.min(rd, distToPolyline(x, z, r));
        if (rd < 5) c.lerp(road, (1 - smoothstep(2.5, 5, rd)) * 0.85);
        // Bauplätze
        for (const pad of this.pads) {
          if (pad.r < 20) continue;
          const d = Math.hypot(x - pad.x, z - pad.z);
          if (d < pad.r) c.lerp(gravel, (1 - smoothstep(pad.r * 0.6, pad.r, d)) * 0.55);
        }
        // Fels an steilen Hängen und in den Bergen
        const rk = smoothstep(0.55, 1.0, sl) + smoothstep(290, 310, Math.max(Math.abs(x), Math.abs(z))) * 0.6;
        if (rk > 0) c.lerp(nv > 0 ? rock : rockDark, clamp(rk, 0, 1));
        const v = 0.92 + n.noise(x * 0.3, z * 0.3) * 0.12;
        col[i * 3] = c.r * v; col[i * 3 + 1] = c.g * v; col[i * 3 + 2] = c.b * v;
      }
    }

    const idx = new Uint32Array((N - 1) * (N - 1) * 6);
    let k = 0;
    for (let iz = 0; iz < N - 1; iz++) {
      for (let ix = 0; ix < N - 1; ix++) {
        const a = iz * N + ix, b = a + 1, c2 = a + N, d = a + N + 1;
        idx[k++] = a; idx[k++] = c2; idx[k++] = b;
        idx[k++] = b; idx[k++] = c2; idx[k++] = d;
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeVertexNormals();
    geo.computeBoundingSphere();

    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      map: detailTex,
      roughness: 0.96,
      metalness: 0,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.receiveShadow = true;
    mesh.name = 'terrain';
    return mesh;
  }

  // Farbiges Kartenbild (für die Übersichtskarte)
  renderMapImage(size, colorFn) {
    const c = document.createElement('canvas');
    c.width = c.height = size;
    const g = c.getContext('2d');
    const img = g.createImageData(size, size);
    for (let py = 0; py < size; py++) {
      for (let px = 0; px < size; px++) {
        const x = -this.half + (px / size) * this.half * 2;
        const z = -this.half + (py / size) * this.half * 2;
        const rgb = colorFn(x, z, this.height(x, z));
        const i = (py * size + px) * 4;
        img.data[i] = rgb[0]; img.data[i + 1] = rgb[1]; img.data[i + 2] = rgb[2]; img.data[i + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
    return c;
  }
}
