// Höhenfeld einer Karte. Form, Wasser und Bodenfarben liefert die
// Kartendefinition (maps/*.js), hier liegen Raster, Abfragen und Mesh.
import * as THREE from 'three';
import { Noise2D, clamp } from './util.js';

export class Terrain {
  constructor(def) {
    this.def = def;
    this.noise = new Noise2D(def.seed ?? 4242);
    this.N = def.res;
    this.half = def.half;
    this.step = (def.half * 2) / (this.N - 1);
    this.heights = new Float32Array(this.N * this.N);
    if (def.init) def.init(this);
    for (let iz = 0; iz < this.N; iz++) {
      for (let ix = 0; ix < this.N; ix++) {
        const x = -this.half + ix * this.step, z = -this.half + iz * this.step;
        this.heights[iz * this.N + ix] = def.rawHeight(this, x, z, true);
      }
    }
  }

  rawHeight(x, z, withPads) {
    return this.def.rawHeight(this, x, z, withPads);
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
    return this.def.waterAt ? this.def.waterAt(this, x, z) : null;
  }

  buildMesh(detailTex) {
    const N = this.N;
    const pos = new Float32Array(N * N * 3);
    const col = new Float32Array(N * N * 3);
    const uv = new Float32Array(N * N * 2);
    const c = new THREE.Color();
    const uvScale = this.def.uvScale ?? 6;
    const set = detailTex && !detailTex.isTexture ? detailTex : null;
    const gmix = set && this.def.grassMix ? new Float32Array(N * N) : null;
    const rmix = set && set.rock && this.def.rockMix ? new Float32Array(N * N) : null;
    // Abnutzung: x = Erde statt Gras, y = Moos in den Pflasterfugen
    const wear = set && set.path && gmix && this.def.wearMix ? new Float32Array(N * N * 2) : null;
    const wearOut = [0, 0];

    for (let iz = 0; iz < N; iz++) {
      for (let ix = 0; ix < N; ix++) {
        const i = iz * N + ix;
        const x = -this.half + ix * this.step, z = -this.half + iz * this.step;
        const h = this.heights[i];
        pos[i * 3] = x; pos[i * 3 + 1] = h; pos[i * 3 + 2] = z;
        uv[i * 2] = x / uvScale; uv[i * 2 + 1] = z / uvScale;
        this.def.groundColor(this, x, z, h, this.slope(x, z), c);
        col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
        if (gmix) gmix[i] = this.def.grassMix(this, x, z, h);
        if (rmix) rmix[i] = this.def.rockMix(this, x, z, h, this.slope(x, z));
        if (wear) {
          this.def.wearMix(this, x, z, wearOut);
          wear[i * 2] = wearOut[0]; wear[i * 2 + 1] = wearOut[1];
        }
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
    if (gmix) geo.setAttribute('gmix', new THREE.BufferAttribute(gmix, 1));
    if (rmix) geo.setAttribute('rmix', new THREE.BufferAttribute(rmix, 1));
    if (wear) geo.setAttribute('wear', new THREE.BufferAttribute(wear, 2));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeVertexNormals();
    geo.computeBoundingSphere();

    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      map: set ? set.map : detailTex,
      roughness: 0.96,
      metalness: 0,
    });
    if (set) {
      mat.normalMap = set.normal;
      mat.normalScale = new THREE.Vector2(0.9, 0.9);
      if (gmix) {
        // Pflaster und Gras pro Ecke überblenden
        mat.onBeforeCompile = (sh) => {
          sh.uniforms.grassMap = { value: set.grass };
          sh.uniforms.grassNormal = { value: set.grassNormal };
          if (rmix) sh.uniforms.rockMap = { value: set.rock };
          if (wear) sh.uniforms.pathMap = { value: set.path };
          sh.vertexShader = sh.vertexShader
            .replace('#include <common>', `#include <common>\nattribute float gmix;\nvarying float vGmix;${rmix ? '\nattribute float rmix;\nvarying float vRmix;\nvarying vec3 vWPos;\nvarying vec3 vWNrm;' : ''}${wear ? '\nattribute vec2 wear;\nvarying vec2 vWear;' : ''}`)
            .replace('#include <uv_vertex>', `#include <uv_vertex>\nvGmix = gmix;${rmix ? '\nvRmix = rmix; vWPos = position; vWNrm = normal;' : ''}${wear ? '\nvWear = wear;' : ''}`);
          sh.fragmentShader = sh.fragmentShader
            .replace('#include <common>', `#include <common>\nuniform sampler2D grassMap;\nuniform sampler2D grassNormal;\nvarying float vGmix;${rmix ? '\nuniform sampler2D rockMap;\nvarying float vRmix;\nvarying vec3 vWPos;\nvarying vec3 vWNrm;' : ''}${wear ? '\nuniform sampler2D pathMap;\nvarying vec2 vWear;' : ''}`)
            .replace('#include <map_fragment>', `#ifdef USE_MAP
  ${wear ? `// Große Flecken gegen sichtbare Wiederholung, unregelmäßige Ränder zwischen Pflaster, Gras und Erde
  float macroA = texture2D(grassMap, vMapUv * 0.043).g, macroB = texture2D(grassMap, vMapUv * 0.0127).g;
  vec4 cob = texture2D(map, vMapUv);
  float jointL = dot(cob.rgb, vec3(0.333));
  float joint = smoothstep(0.26, 0.1, jointL);
  cob.rgb = mix(cob.rgb, mix(vec3(0.42, 0.38, 0.3), vec3(0.17, 0.24, 0.1), vWear.y), joint * (0.42 + 0.4 * vWear.y));
  cob.rgb = max(cob.rgb - 0.3, vec3(-0.3)) * 1.22 + 0.3; // Steine plastischer, auch im Schatten
  vec4 gra = texture2D(grassMap, vMapUv * 0.8);
  gra.rgb = mix(gra.rgb, texture2D(pathMap, vMapUv * 0.6).rgb, smoothstep(0.32, 0.68, vWear.x + (macroA - 0.45) * 0.9));
  float gEdge = smoothstep(0.32, 0.68, vGmix + (macroA - 0.45) * 1.4 * vGmix * (1.0 - vGmix) * 4.0);
  vec4 sampledDiffuseColor = mix(cob, gra, gEdge);
  sampledDiffuseColor.rgb *= 0.74 + macroB * 0.72;` : 'vec4 sampledDiffuseColor = mix(texture2D(map, vMapUv), texture2D(grassMap, vMapUv * 0.8), vGmix);'}
  ${rmix ? `// Fels: von der Seite projiziert, sonst wäre die Textur auf steilen Hängen verzerrt
  vec3 rw = abs(normalize(vWNrm)); rw = pow(rw, vec3(4.0)); rw /= rw.x + rw.y + rw.z;
  vec4 rockCol = texture2D(rockMap, vWPos.zy / 14.0) * rw.x + texture2D(rockMap, vWPos.xz / 14.0) * rw.y + texture2D(rockMap, vWPos.xy / 14.0) * rw.z;
  sampledDiffuseColor = mix(sampledDiffuseColor, rockCol, vRmix);` : ''}
  diffuseColor *= sampledDiffuseColor;
#endif`)
            .replace('#include <normal_fragment_maps>', THREE.ShaderChunk.normal_fragment_maps.replace(
              'vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;',
              'vec3 mapN = mix(texture2D( normalMap, vNormalMapUv ).xyz, texture2D( grassNormal, vNormalMapUv * 0.8 ).xyz, vGmix) * 2.0 - 1.0;'));
        };
        mat.customProgramCacheKey = () => `terrain-grass${rmix ? '-rock' : ''}${wear ? '-wear' : ''}`;
      }
    }
    const mesh = new THREE.Mesh(geo, mat);
    mesh.receiveShadow = true;
    mesh.name = 'terrain';
    return mesh;
  }

  // Farbiges Kartenbild (für die Übersichtskarte), Ausschnitt ±view
  renderMapImage(size, colorFn, view = this.half) {
    const c = document.createElement('canvas');
    c.width = c.height = size;
    const g = c.getContext('2d');
    const img = g.createImageData(size, size);
    for (let py = 0; py < size; py++) {
      for (let px = 0; px < size; px++) {
        const x = -view + (px / size) * view * 2;
        const z = -view + (py / size) * view * 2;
        const rgb = colorFn(x, z, this.height(x, z));
        const i = (py * size + px) * 4;
        img.data[i] = rgb[0]; img.data[i + 1] = rgb[1]; img.data[i + 2] = rgb[2]; img.data[i + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
    return c;
  }
}
