// Szene, Licht, Wetter, Wasser, Vegetation und statische Karte.
import * as THREE from 'three';
import { Terrain } from './terrain.js';
import { CollisionWorld } from './collision.js';
import { Batcher } from './batcher.js';
import { Instancer } from './instancer.js';
import { coverMaterial } from './groundcover.js';
import { makeGrungeTexture, makeSignTexture } from './textures.js';
import { Noise2D, clamp } from './util.js';

const CONDITION_LOOK = {
  tag: {
    sunElev: 0.72, sunAz: 2.3, sun: 0xffe8c8, sunI: 3.1, hemiSky: 0xe2e6e8, hemiGround: 0x7d6e58, hemiI: 0.72, envI: 0.38,
    top: 0x4f86c2, horizon: 0xd6dccf, fog: 0xc6d0cc, fogNear: 90, fogFar: 620, exposure: 1.0,
  },
  daemmerung: {
    sunElev: 0.2, sunAz: -1.9, sun: 0xffa45c, sunI: 2.4, hemiSky: 0x9fa4c8, hemiGround: 0x5a4636, hemiI: 0.62, envI: 0.6,
    top: 0x34466f, horizon: 0xf2a06a, fog: 0xc79377, fogNear: 50, fogFar: 380, exposure: 1.05,
  },
  sturm: {
    sunElev: 0.8, sunAz: 1.2, sun: 0xc8d2dc, sunI: 0.7, hemiSky: 0x8c979e, hemiGround: 0x3d3f3a, hemiI: 0.9, envI: 0.7,
    top: 0x363e43, horizon: 0x687277, fog: 0x5f686c, fogNear: 12, fogFar: 170, exposure: 0.95,
  },
};

// Sichtweiten: Innenräume sieht man nur durch Fenster, Details nur in der Nähe
const INTERIOR_RANGE = 60, DETAIL_RANGE = 90, PROP_RANGE = 200;

export class World {
  constructor(renderer, quality, assets = null, def) {
    this.def = def;
    this.renderer = renderer;
    this.quality = quality;
    this.assets = assets && assets.ready ? assets : null;
    this.scene = new THREE.Scene();
    this.time = 0;
    this.lightning = 0;
    this.nextLightning = 8;

    this.grunge = makeGrungeTexture();
    this.terrain = new Terrain(def.terrain);
    this.col = new CollisionWorld(this.terrain);

    // Wind für Vegetation: Zeit und Stärke, gemeinsam für alle Pflanzen-Shader
    this.wind = { time: { value: 0 }, strength: { value: 1 } };
    this.setupLights();
    this.setupSky();

    const terrainMesh = this.terrain.buildMesh(def.groundTexture ? def.groundTexture(this) : this.grunge);
    this.scene.add(terrainMesh);

    const mats = {
      world: new THREE.MeshStandardMaterial({ vertexColors: true, map: this.grunge, roughness: 0.88, metalness: 0.04 }),
      metal: new THREE.MeshStandardMaterial({ vertexColors: true, map: this.grunge, roughness: 0.5, metalness: 0.5 }),
      glass: new THREE.MeshStandardMaterial({ vertexColors: true, transparent: true, opacity: 0.28, roughness: 0.05, metalness: 0.3, depthWrite: false }),
      frame: new THREE.MeshStandardMaterial({ vertexColors: true, wireframe: true, roughness: 0.6, metalness: 0.4 }),
      fence: new THREE.MeshStandardMaterial({ vertexColors: true, transparent: true, opacity: 0.35, roughness: 0.8, depthWrite: false, side: THREE.DoubleSide }),
      glow: new THREE.MeshBasicMaterial({ vertexColors: true }),
    };
    if (def.materials) Object.assign(mats, def.materials(this));
    const dress = this.assets?.dressMaterial();
    if (dress) mats.dress = dress;
    const cover = coverMaterial(this);
    if (cover) mats.cover = cover;
    this.batcher = new Batcher(mats);
    this.farMeshes = [];
    this.instancer = new Instancer(this.assets);
    this.mapData = def.build({ batcher: this.batcher, col: this.col, terrain: this.terrain, assets: this.assets, world: this, instancer: this.instancer });
    this.plantVegetation();
    this.staticMeshes = [...this.batcher.build(this.scene), ...this.instancer.build(this.scene)];
    // Kleine Modelle ohne Layer (Laternen, Karren, Tanks) brauchen nicht bis zum Horizont gezeichnet werden
    for (const m of this.staticMeshes) if (m.isInstancedMesh && !m.userData.layer) { m.userData.far = PROP_RANGE; this.farMeshes.push(m); }
    // Innenräume (Treppen, Möbel …) nur in der Nähe zeichnen
    this.interiorMeshes = this.staticMeshes.filter((m) => m.userData.layer === 'in');
    // Kleinkram (Rahmen, Läden, Efeu …) nur bis mittlere Entfernung
    this.detailMeshes = [...this.staticMeshes.filter((m) => m.userData.layer === 'det'), ...(this.vegDetail || [])];
    // Statische Meshes bewegen sich nie: die Matrixprüfung je Bild entfällt
    for (const m of [...this.staticMeshes, ...this.farMeshes, ...(this.vegDetail || [])]) { m.matrixAutoUpdate = false; m.matrixWorldAutoUpdate = false; }
    this.waterMats = [];
    this.waterNormal = makeWaterNormal();
    this.waterNormal.wrapS = this.waterNormal.wrapT = THREE.RepeatWrapping;
    if (def.decorate) def.decorate(this);
    this.buildSigns();
    this.buildRain();
  }

  // Alles freigeben (Kartenwechsel)
  dispose() {
    const seen = new Set();
    this.scene.traverse((o) => {
      if (o.geometry && !seen.has(o.geometry)) { seen.add(o.geometry); o.geometry.dispose(); }
      const mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
      for (const m of mats) {
        if (seen.has(m)) continue;
        seen.add(m);
        for (const k of ['map', 'normalMap', 'emissiveMap']) if (m[k] && !seen.has(m[k])) { seen.add(m[k]); m[k].dispose(); }
        m.dispose();
      }
    });
    if (this.envRT) this.envRT.dispose();
    if (this.pmrem) this.pmrem.dispose();
    this.scene.clear();
  }

  setupLights() {
    this.hemi = new THREE.HemisphereLight(0xcfe2ff, 0x6a5c44, 1.1);
    this.scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(0xfff1d6, 2.8);
    this.sun.castShadow = this.quality !== 'niedrig';
    const s = this.sun.shadow;
    // feinere, weichere Schatten über einen größeren Bereich (PCF mit Vogel-Scheibe)
    const size = this.quality === 'hoch' ? 4096 : 2048;
    s.mapSize.set(size, size);
    s.radius = this.quality === 'hoch' ? 3 : 2;
    const ext = this.quality === 'hoch' ? 90 : 70;
    s.camera.left = -ext; s.camera.right = ext; s.camera.top = ext; s.camera.bottom = -ext;
    s.camera.near = 1; s.camera.far = 400;
    s.bias = -0.0004;
    s.normalBias = 0.6;
    this.shadowExt = ext;
    this.scene.add(this.sun);
    this.scene.add(this.sun.target);
    this.scene.fog = new THREE.Fog(0xc9d2cc, 60, 460);
  }

  setupSky() {
    const geo = new THREE.SphereGeometry(900, 32, 16);
    this.skyUniforms = {
      top: { value: new THREE.Color(0x5f93c7) },
      horizon: { value: new THREE.Color(0xd9ddcf) },
      sunDir: { value: new THREE.Vector3(0, 1, 0) },
      sunColor: { value: new THREE.Color(0xfff1d6) },
      cloud: { value: 0.3 },
      time: { value: 0 },
    };
    const mat = new THREE.ShaderMaterial({
      uniforms: this.skyUniforms,
      side: THREE.BackSide,
      depthWrite: false,
      fog: false,
      vertexShader: `varying vec3 vDir; void main(){ vDir = normalize(position); vec4 p = modelViewMatrix * vec4(position,1.0); gl_Position = projectionMatrix * p; gl_Position.z = gl_Position.w * 0.9999; }`,
      fragmentShader: `
        uniform vec3 top; uniform vec3 horizon; uniform vec3 sunDir; uniform vec3 sunColor; uniform float cloud; uniform float time;
        varying vec3 vDir;
        float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
        float noise(vec2 p){ vec2 i=floor(p); vec2 f=fract(p); f=f*f*(3.0-2.0*f);
          return mix(mix(hash(i),hash(i+vec2(1,0)),f.x), mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x), f.y); }
        float fbm(vec2 p){ float s=0.0; float a=0.5; for(int i=0;i<5;i++){ s+=a*noise(p); p*=2.03; a*=0.5; } return s; }
        void main(){
          vec3 d = normalize(vDir);
          float h = clamp(d.y, -0.2, 1.0);
          vec3 s3 = normalize(sunDir);
          float sd = max(dot(d, s3), 0.0);
          // Verlauf Zenit -> Horizont mit Dunstband
          vec3 col = mix(horizon, top, pow(max(h, 0.0), 0.5));
          col = mix(col, horizon * 1.06, exp(-max(h, 0.0) * 9.0) * 0.55);
          if (h < 0.0) col = mix(horizon, horizon * 0.72, -h * 5.0);
          // Sonne: Scheibe, Korona und weiter Schein (in Horizontnähe stärker)
          col += sunColor * (pow(sd, 1400.0) * 4.0 + pow(sd, 70.0) * 0.32 + pow(sd, 6.0) * 0.12 * (1.0 - 0.6 * max(h, 0.0)));
          // Wolken: verwirbeltes Rauschen, zur Sonne hin aufgehellt, Silberrand
          if (d.y > 0.0) {
            vec2 q = d.xz / (d.y + 0.12) * 0.8 + vec2(time * 0.006, time * 0.003);
            vec2 warp = vec2(fbm(q * 0.7 + 3.1), fbm(q * 0.7 - 1.7));
            float n = fbm(q * 1.3 + warp * 0.8);
            float cov = mix(0.6, 0.36, cloud);
            float dens = smoothstep(cov, cov + 0.26, n);
            vec2 toSun = normalize(s3.xz + 1e-4) * 0.09;
            float n2 = fbm((q + toSun) * 1.3 + warp * 0.8);
            float lit = clamp(0.55 + (n - n2) * 3.0, 0.0, 1.0);
            vec3 cLit = mix(vec3(1.0), sunColor, 0.35) * (1.0 + 0.5 * pow(sd, 8.0));
            vec3 cShade = mix(horizon, top, 0.35) * 0.74;
            vec3 cc = mix(cShade, cLit, lit);
            cc += sunColor * pow(sd, 10.0) * (1.0 - dens) * 0.7;
            col = mix(col, cc, dens * smoothstep(0.0, 0.18, d.y) * (0.6 + 0.4 * cloud));
          }
          gl_FragColor = vec4(col, 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    });
    this.sky = new THREE.Mesh(geo, mat);
    this.sky.frustumCulled = false;
    this.sky.renderOrder = -1;
    this.scene.add(this.sky);
  }

  applyCondition(id) {
    const L = CONDITION_LOOK[id] || CONDITION_LOOK.tag;
    this.condition = id;
    this.look = L;
    this.sunDir = new THREE.Vector3(Math.cos(L.sunAz) * Math.cos(L.sunElev), Math.sin(L.sunElev), Math.sin(L.sunAz) * Math.cos(L.sunElev)).normalize();
    this.sun.color.set(L.sun);
    this.sun.intensity = L.sunI;
    this.hemi.color.set(L.hemiSky);
    this.hemi.groundColor.set(L.hemiGround);
    this.hemi.intensity = L.hemiI;
    this.baseHemi = L.hemiI;
    this.scene.fog.color.set(L.fog);
    this.scene.fog.near = L.fogNear;
    this.scene.fog.far = L.fogFar;
    this.skyUniforms.top.value.set(L.top);
    this.skyUniforms.horizon.value.set(L.horizon);
    this.skyUniforms.sunDir.value.copy(this.sunDir);
    this.skyUniforms.sunColor.value.set(L.sun);
    this.skyUniforms.cloud.value = id === 'sturm' ? 1.0 : id === 'daemmerung' ? 0.45 : 0.35;
    this.renderer.toneMappingExposure = L.exposure;
    this.rain.visible = id === 'sturm';
    this.wind.strength.value = id === 'sturm' ? 2.6 : id === 'daemmerung' ? 0.7 : 1;
    this.updateEnvironment(L.envI);
  }

  // Umgebungslicht/-reflexion aus dem aktuellen Himmel
  updateEnvironment(intensity) {
    if (!this.pmrem) this.pmrem = new THREE.PMREMGenerator(this.renderer);
    const envScene = new THREE.Scene();
    envScene.add(new THREE.Mesh(this.sky.geometry, this.sky.material));
    const t = this.skyUniforms.time.value;
    this.skyUniforms.time.value = 0;
    if (this.envRT) this.envRT.dispose();
    this.envRT = this.pmrem.fromScene(envScene, 0.02, 0.1, 100);
    this.skyUniforms.time.value = t;
    this.scene.environment = this.envRT.texture;
    this.scene.environmentIntensity = intensity;
  }

  buildSigns() {
    for (const l of this.mapData.labels || []) {
      const tex = makeSignTexture(l.text, l.bg, l.fg);
      const w = l.w || 6, h = l.h || 1.5;
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.8 }));
      m.position.set(l.x, l.y, l.z);
      m.rotation.y = l.ry;
      this.scene.add(m);
    }
  }

  plantVegetation() {
    const v = this.def.vegetation ? this.def.vegetation(this) : { trees: [], bushes: [], rocks: [], grass: [] };
    const { trees, bushes, rocks } = v;
    if (this.assets) {
      this.plantModels(v);
      return;
    }
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), p = new THREE.Vector3(), sc = new THREE.Vector3(), e = new THREE.Euler();
    const c = new THREE.Color();
    const mkInst = (geo, color, count) => {
      const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.9, flatShading: true });
      const im = new THREE.InstancedMesh(geo, mat, count);
      im.castShadow = true;
      im.receiveShadow = true;
      this.scene.add(im);
      return im;
    };
    const pines = trees.filter((t) => t.pine), leafy = trees.filter((t) => !t.pine);
    const trunk = mkInst(new THREE.CylinderGeometry(0.18, 0.3, 1, 6), 0x5a4330, trees.length);
    const pineCrown = mkInst(new THREE.ConeGeometry(1, 1, 7), 0xffffff, pines.length * 2);
    const leafCrown = mkInst(new THREE.IcosahedronGeometry(1, 1), 0xffffff, leafy.length);
    trees.forEach((t, i) => {
      const h = (t.pine ? 7 : 4.5) * t.s;
      p.set(t.x, t.y + h * 0.35, t.z); sc.set(t.s, h * 0.7, t.s); q.identity();
      trunk.setMatrixAt(i, m.compose(p, q, sc));
      this.col.addCyl(t.x, t.z, 0.35 * t.s, t.y - 1, t.y + h, { tree: true });
    });
    pines.forEach((t, i) => {
      const h = 7 * t.s;
      for (let k = 0; k < 2; k++) {
        p.set(t.x, t.y + h * (0.45 + k * 0.3), t.z);
        sc.set(t.s * (2.4 - k * 0.8), h * (0.6 - k * 0.15), t.s * (2.4 - k * 0.8));
        q.setFromEuler(e.set(0, t.r, 0));
        pineCrown.setMatrixAt(i * 2 + k, m.compose(p, q, sc));
        c.setHSL(0.27 + (t.r % 1) * 0.04, 0.35, 0.2 + k * 0.03);
        pineCrown.setColorAt(i * 2 + k, c);
      }
    });
    leafy.forEach((t, i) => {
      const h = 4.5 * t.s;
      p.set(t.x, t.y + h * 0.95, t.z);
      sc.set(t.s * 2.5, t.s * 2.2, t.s * 2.5);
      q.setFromEuler(e.set(t.r, t.r * 2, 0));
      leafCrown.setMatrixAt(i, m.compose(p, q, sc));
      c.setHSL(0.17 + (t.r % 1) * 0.1, 0.4, 0.28 + (t.r % 0.5) * 0.12);
      leafCrown.setColorAt(i, c);
    });
    const bush = mkInst(new THREE.IcosahedronGeometry(1, 1), 0xffffff, bushes.length);
    bush.castShadow = false;
    bushes.forEach((b, i) => {
      p.set(b.x, b.y + 0.15 * b.s, b.z); sc.set(b.s * 1.1, b.s * 0.65, b.s * 1.1);
      q.setFromEuler(e.set(0, b.r * 3, 0));
      bush.setMatrixAt(i, m.compose(p, q, sc));
      c.setHSL(0.2 + (b.r % 1) * 0.09, 0.45, 0.17 + (b.r % 0.3) * 0.25);
      bush.setColorAt(i, c);
    });
    const rock = mkInst(new THREE.DodecahedronGeometry(1, 0), 0xffffff, rocks.length);
    rocks.forEach((r, i) => {
      p.set(r.x, r.y + r.s * 0.2, r.z); sc.set(r.s * 1.3, r.s * 0.8, r.s);
      q.setFromEuler(e.set(r.t, r.r, r.t * 0.5));
      rock.setMatrixAt(i, m.compose(p, q, sc));
      const v = 0.3 + (r.t % 1) * 0.1;
      c.setRGB(v, v * 0.96, v * 0.9);
      rock.setColorAt(i, c);
      if (r.s > 1.1) this.col.addCyl(r.x, r.z, r.s * 0.95, r.y - 2, r.y + r.s * 0.9, { rock: true });
    });
    for (const im of [trunk, pineCrown, leafCrown, bush, rock]) {
      im.instanceMatrix.needsUpdate = true;
      if (im.instanceColor) im.instanceColor.needsUpdate = true;
      im.computeBoundingSphere();
    }
  }

  // Vegetation aus den Blender-Modellen als InstancedMesh
  plantModels({ trees, bushes, rocks, grass = [], extra = [] }) {
    const T = this.terrain;
    const q = new THREE.Quaternion(), p = new THREE.Vector3(), sc = new THREE.Vector3(), e = new THREE.Euler();
    const m = new THREE.Matrix4();
    const c = new THREE.Color();
    // In Kacheln aufteilen, damit Frustum- und Schatten-Culling greifen
    // Bodendecker (Gras, Blumen) in kleineren Kacheln, die ab mittlerer Entfernung ausgeblendet werden
    this.vegDetail = [];
    const inst = (name, all, place, { shadow = true, tint = null, doubleSide = false, cull = false, wind = 0, far = 0 } = {}) => {
      if (!all.length) return;
      // Große Kacheln sparen Draw-Calls, die Bildflächen und Felsen kosten kaum Dreiecke
      const CH = cull ? 40 : 220;
      const chunks = new Map();
      for (const it of all) {
        const key = `${Math.floor((it.x + 1000) / CH)}|${Math.floor((it.z + 1000) / CH)}`;
        if (!chunks.has(key)) chunks.set(key, []);
        chunks.get(key).push(it);
      }
      const mats = this.assets.bakedMeshes(name).map((bm) => {
        const mat = bm.material.clone();
        if (doubleSide) mat.side = THREE.DoubleSide;
        if (wind) this.addWind(mat, wind);
        return { geo: bm.geometry, mat };
      });
      for (const list of chunks.values()) {
        for (const { geo, mat } of mats) {
          const im = new THREE.InstancedMesh(geo, mat, list.length);
          im.name = name;
          im.castShadow = shadow;
          im.receiveShadow = !name.startsWith('n_'); // Bildflächen würden sich selbst verschatten
          list.forEach((it, i) => {
            place(it);
            im.setMatrixAt(i, m.compose(p, q, sc));
            if (tint) im.setColorAt(i, tint(it, c));
          });
          im.instanceMatrix.needsUpdate = true;
          if (im.instanceColor) im.instanceColor.needsUpdate = true;
          im.computeBoundingSphere();
          if (cull || far) {
            let cx = 0, cy = 0, cz = 0;
            for (const it of list) { cx += it.x; cy += it.y; cz += it.z; }
            im.userData.center = new THREE.Vector3(cx / list.length, cy / list.length, cz / list.length);
            if (cull) {
              im.userData.layer = 'det';
              this.vegDetail.push(im);
            } else {
              im.userData.far = far;
              this.farMeshes.push(im);
            }
          }
          this.scene.add(im);
        }
      }
    };
    const leafTint = (it, col) => {
      const v = 0.82 + (it.r % 1) * 0.3;
      return col.setRGB(v * (0.94 + (it.r % 0.37) * 0.3), v, v * (0.85 + (it.r % 0.53) * 0.25));
    };
    const pines = trees.filter((t) => t.pine), leafy = trees.filter((t) => !t.pine);
    for (const t of trees) this.col.addCyl(t.x, t.z, 0.3 * t.s, t.y - 1, t.y + 7 * t.s, { tree: true });
    // Die Karte kann eigene Modelle festlegen; mehrere Namen werden gleichmäßig verteilt
    const M = this.def.models || {};
    const each = (key, fallback, list, fn) => {
      const names = (M[key] || []).filter((n) => this.assets.has(n));
      if (!names.length) return fn(fallback, list);
      names.forEach((n, k) => fn(n, list.filter((_, i) => i % names.length === k)));
    };
    each('pine', 'v_pine', pines, (nm, ls) => inst(nm, ls, (t) => { p.set(t.x, t.y - 0.1, t.z); q.setFromEuler(e.set(0, t.r, 0)); sc.set(t.s, t.s * (0.9 + (t.r % 0.4)), t.s); }, { tint: leafTint, wind: 0.0018 }));
    each('tree', 'v_tree', leafy, (nm, ls) => inst(nm, ls, (t) => { p.set(t.x, t.y - 0.1, t.z); q.setFromEuler(e.set(0, t.r, 0)); sc.setScalar(t.s); }, { tint: leafTint, wind: 0.003 }));
    each('bush', 'v_bush', bushes, (nm, ls) => inst(nm, ls, (b) => { p.set(b.x, b.y - 0.05, b.z); q.setFromEuler(e.set(0, b.r * 3, 0)); sc.set(b.s, b.s * 0.9, b.s); }, { shadow: false, tint: leafTint, wind: 0.03, far: 280 }));
    const rockNames = (M.rock || []).filter((n) => this.assets.has(n));
    const rockSet = rockNames.length ? rockNames : ['v_rock0', 'v_rock1', 'v_rock2'];
    for (let k = 0; k < rockSet.length; k++) {
      const list = rocks.filter((_, i) => i % rockSet.length === k);
      inst(rockSet[k], list, (r) => { p.set(r.x, r.y - r.s * (0.15 + Math.min(0.6, T.slope(r.x, r.z) * 0.45)), r.z); q.setFromEuler(e.set(0, r.r, 0)); sc.setScalar(r.s * 0.8); }, {
        tint: (r, col) => { const v = 0.85 + (r.t % 1) * 0.3; return col.setRGB(v, v, v); },
        far: 340,
      });
    }
    for (const r of rocks) if (r.s > 1.1) this.col.addCyl(r.x, r.z, r.s * 0.95, r.y - 2, r.y + r.s * 0.8, { rock: true });
    inst('v_grass', grass, (g) => { p.set(g.x, g.y - 0.05, g.z); q.setFromEuler(e.set(0, g.r, 0)); sc.setScalar(g.s); }, { shadow: false, doubleSide: true, tint: leafTint, cull: true, wind: 0.12 });
    // weitere Modelle der Karte (z. B. Riesenbäume): { name, list, place(it, p, q, sc, e), opts }
    for (const ex of extra) inst(ex.name, ex.list, (it) => ex.place(it, p, q, sc, e), ex.opts || {});
  }

  // Pflanzen im Wind wiegen: Auslenkung wächst mit der Höhe über dem Modellursprung,
  // Phase je Instanz aus ihrer Position
  addWind(mat, amp) {
    const w = this.wind;
    mat.onBeforeCompile = (sh) => {
      sh.uniforms.uWindTime = w.time;
      sh.uniforms.uWindK = w.strength;
      sh.vertexShader = sh.vertexShader
        .replace('#include <common>', '#include <common>\nuniform float uWindTime;\nuniform float uWindK;')
        .replace('#include <begin_vertex>', `#include <begin_vertex>
          {
            float hgt = max(position.y, 0.0);
            #ifdef USE_INSTANCING
              vec3 ip = instanceMatrix[3].xyz;
            #else
              vec3 ip = vec3(0.0);
            #endif
            float ph = ip.x * 0.31 + ip.z * 0.23;
            float bend = hgt * hgt * ${amp.toFixed(5)} * uWindK;
            float t = uWindTime * (0.8 + 0.4 * uWindK);
            transformed.x += (sin(t * 1.6 + ph) + 0.35 * sin(t * 3.9 + ph * 2.3)) * bend;
            transformed.z += (cos(t * 1.25 + ph * 1.3) + 0.3 * sin(t * 4.3 + ph)) * bend * 0.7;
          }`);
    };
    mat.customProgramCacheKey = () => 'wind' + amp;
  }

  buildRain() {
    const count = 5000;
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(count * 6);
    for (let i = 0; i < count; i++) {
      const x = (Math.random() - 0.5) * 80, y = Math.random() * 40, z = (Math.random() - 0.5) * 80;
      pos.set([x, y, z, x + 0.05, y - 0.9, z + 0.05], i * 6);
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.rain = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0xaab8c0, transparent: true, opacity: 0.35 }));
    this.rain.frustumCulled = false;
    this.rain.visible = false;
    this.scene.add(this.rain);
    this.rainOffset = 0;
  }

  update(dt, focus, sfx, near = true) {
    this.time += dt;
    this.wind.time.value = this.time;
    this.skyUniforms.time.value = this.time;
    this.sky.position.copy(focus);
    // Schatten folgen dem Spieler (auf Texel gerastert, damit nichts flimmert)
    const ext = this.shadowExt;
    const texel = (ext * 2) / this.sun.shadow.mapSize.x;
    const fx = Math.round(focus.x / texel) * texel, fz = Math.round(focus.z / texel) * texel;
    this.sun.target.position.set(fx, focus.y, fz);
    this.sun.position.set(fx + this.sunDir.x * 200, focus.y + this.sunDir.y * 200, fz + this.sunDir.z * 200);
    for (const mat of this.waterMats) {
      mat.normalMap.offset.x += dt * 0.01;
      mat.normalMap.offset.y += dt * 0.006;
    }
    if (this.def.updateDecor) this.def.updateDecor(this, dt, focus);
    for (const m of this.interiorMeshes) {
      const c = m.userData.center;
      const dx = c.x - focus.x, dz = c.z - focus.z;
      m.visible = near && dx * dx + dz * dz < INTERIOR_RANGE * INTERIOR_RANGE && Math.abs(c.y - focus.y) < 80;
    }
    for (const m of this.detailMeshes) {
      const c = m.userData.center;
      const dx = c.x - focus.x, dz = c.z - focus.z;
      const d2 = dx * dx + dz * dz;
      m.visible = near && d2 < DETAIL_RANGE * DETAIL_RANGE;
      // Wiese: mit der Entfernung weniger Pflanzen zeichnen. Sie liegen zufällig sortiert im Mesh
      if (m.userData.cover && m.visible) {
        const n = m.geometry.attributes.position.count, frac = d2 < 32 * 32 ? 1 : d2 < 58 * 58 ? 0.5 : 0.22;
        m.geometry.setDrawRange(0, Math.floor((n * frac) / 12) * 12);
      }
    }
    for (const m of this.farMeshes) {
      const c = m.userData.center, r = m.userData.far;
      const dx = c.x - focus.x, dz = c.z - focus.z;
      m.visible = !near || dx * dx + dz * dz < r * r;
    }
    if (this.rain.visible) {
      this.rainOffset = (this.rainOffset + dt * 30) % 40;
      this.rain.position.set(focus.x, focus.y - 10 - this.rainOffset + 20, focus.z);
      // Blitze
      this.nextLightning -= dt;
      if (this.nextLightning <= 0) {
        this.nextLightning = 7 + Math.random() * 14;
        this.lightning = 1;
        if (sfx) setTimeout(() => sfx.play('thunder', null, { vol: 0.8 }), 400 + Math.random() * 1600);
      }
    }
    if (this.lightning > 0) {
      this.lightning = Math.max(0, this.lightning - dt * 3);
      const flick = this.lightning > 0.5 ? (Math.random() < 0.5 ? 1 : 0.3) : this.lightning;
      this.hemi.intensity = this.baseHemi + flick * 3;
    } else if (this.hemi.intensity !== this.baseHemi) this.hemi.intensity = this.baseHemi;
  }
}

function makeWaterNormal() {
  const size = 128;
  const n = new Noise2D(5);
  const hts = new Float32Array(size * size);
  const per = 4;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = (x / size) * per, v = (y / size) * per;
      // kachelbar
      const fx = x / size, fy = y / size;
      const s = (a, b) => n.fbm(a, b, 3);
      hts[y * size + x] = (s(u, v) * (1 - fx) + s(u - per, v) * fx) * (1 - fy) + (s(u, v - per) * (1 - fx) + s(u - per, v - per) * fx) * fy;
    }
  }
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  const img = g.createImageData(size, size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const h = (xx, yy) => hts[((yy + size) % size) * size + ((xx + size) % size)];
      const dx = (h(x + 1, y) - h(x - 1, y)) * 6, dy = (h(x, y + 1) - h(x, y - 1)) * 6;
      const l = Math.hypot(dx, dy, 1);
      const i = (y * size + x) * 4;
      img.data[i] = clamp((-dx / l * 0.5 + 0.5) * 255, 0, 255);
      img.data[i + 1] = clamp((-dy / l * 0.5 + 0.5) * 255, 0, 255);
      img.data[i + 2] = clamp((1 / l * 0.5 + 0.5) * 255, 0, 255);
      img.data[i + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  return new THREE.CanvasTexture(c);
}
