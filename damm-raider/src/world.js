// Szene, Licht, Wetter, Wasser, Vegetation und statische Karte.
import * as THREE from 'three';
import { MAP } from './config.js';
import { Terrain } from './terrain.js';
import { CollisionWorld } from './collision.js';
import { Batcher } from './batcher.js';
import { buildMap } from './map.js';
import { makeGrungeTexture, makeSignTexture, makeSoftDot } from './textures.js';
import { mulberry32, Noise2D, clamp, smoothstep, distToPolyline } from './util.js';
import { ROADS, PADS } from './config.js';

const CONDITION_LOOK = {
  tag: {
    sunElev: 0.95, sunAz: 2.3, sun: 0xfff1d6, sunI: 2.8, hemiSky: 0xcfe2ff, hemiGround: 0x6a5c44, hemiI: 0.75, envI: 0.55,
    top: 0x5f93c7, horizon: 0xd9ddcf, fog: 0xc9d2cc, fogNear: 60, fogFar: 460, exposure: 1.0,
  },
  daemmerung: {
    sunElev: 0.2, sunAz: -1.9, sun: 0xffa45c, sunI: 2.4, hemiSky: 0x9fa4c8, hemiGround: 0x5a4636, hemiI: 0.5, envI: 0.5,
    top: 0x34466f, horizon: 0xf2a06a, fog: 0xc79377, fogNear: 50, fogFar: 380, exposure: 1.05,
  },
  sturm: {
    sunElev: 0.8, sunAz: 1.2, sun: 0xc8d2dc, sunI: 0.7, hemiSky: 0x8c979e, hemiGround: 0x3d3f3a, hemiI: 0.9, envI: 0.7,
    top: 0x363e43, horizon: 0x687277, fog: 0x5f686c, fogNear: 12, fogFar: 170, exposure: 0.95,
  },
};

export class World {
  constructor(renderer, quality, assets = null) {
    this.renderer = renderer;
    this.quality = quality;
    this.assets = assets && assets.ready ? assets : null;
    this.scene = new THREE.Scene();
    this.time = 0;
    this.lightning = 0;
    this.nextLightning = 8;

    this.grunge = makeGrungeTexture();
    this.terrain = new Terrain();
    this.col = new CollisionWorld(this.terrain);

    this.setupLights();
    this.setupSky();

    const terrainMesh = this.terrain.buildMesh(this.grunge);
    this.scene.add(terrainMesh);

    const mats = {
      world: new THREE.MeshStandardMaterial({ vertexColors: true, map: this.grunge, roughness: 0.88, metalness: 0.04 }),
      metal: new THREE.MeshStandardMaterial({ vertexColors: true, map: this.grunge, roughness: 0.5, metalness: 0.5 }),
      glass: new THREE.MeshStandardMaterial({ vertexColors: true, transparent: true, opacity: 0.28, roughness: 0.05, metalness: 0.3, depthWrite: false }),
      frame: new THREE.MeshStandardMaterial({ vertexColors: true, wireframe: true, roughness: 0.6, metalness: 0.4 }),
      fence: new THREE.MeshStandardMaterial({ vertexColors: true, transparent: true, opacity: 0.35, roughness: 0.8, depthWrite: false, side: THREE.DoubleSide }),
      glow: new THREE.MeshBasicMaterial({ vertexColors: true }),
    };
    this.batcher = new Batcher(mats);
    this.mapData = buildMap(this.batcher, this.col, this.terrain, this.assets);
    this.plantVegetation();
    this.batcher.build(this.scene);
    this.buildWater();
    this.buildSigns();
    this.buildRain();
  }

  setupLights() {
    this.hemi = new THREE.HemisphereLight(0xcfe2ff, 0x6a5c44, 1.1);
    this.scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(0xfff1d6, 2.8);
    this.sun.castShadow = this.quality !== 'niedrig';
    const s = this.sun.shadow;
    const size = this.quality === 'hoch' ? 2048 : 1024;
    s.mapSize.set(size, size);
    const ext = this.quality === 'hoch' ? 75 : 60;
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
          vec3 col = mix(horizon, top, pow(max(h,0.0), 0.55));
          if (h < 0.0) col = mix(horizon, horizon*0.7, -h*5.0);
          float sd = max(dot(d, normalize(sunDir)), 0.0);
          col += sunColor * (pow(sd, 900.0) * 3.0 + pow(sd, 12.0) * 0.25);
          vec2 uv = d.xz / max(d.y, 0.08) * 0.9 + vec2(time*0.004, time*0.002);
          float c = smoothstep(0.45, 0.85, fbm(uv*1.4)) * smoothstep(0.02, 0.25, d.y);
          vec3 cc = mix(vec3(1.0), horizon, 0.35) * (0.8 + 0.3*sd);
          col = mix(col, cc, c * cloud);
          gl_FragColor = vec4(col, 1.0);
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

  buildWater() {
    const normalTex = makeWaterNormal();
    normalTex.wrapS = normalTex.wrapT = THREE.RepeatWrapping;
    this.waterNormal = normalTex;
    const mk = (color, opacity, rep) => {
      const t = normalTex.clone();
      t.needsUpdate = true;
      t.repeat.set(rep, rep);
      return new THREE.MeshStandardMaterial({ color, roughness: 0.12, metalness: 0.15, transparent: true, opacity, normalMap: t, normalScale: new THREE.Vector2(0.6, 0.6), depthWrite: false });
    };
    this.waterMats = [];
    // Stausee
    const L = MAP.LAKE;
    const lakeMat = mk(0x2c5a66, 0.9, 30);
    const lake = new THREE.Mesh(new THREE.PlaneGeometry(L.hx * 2 + 6, L.hz * 2 + 4), lakeMat);
    lake.rotation.x = -Math.PI / 2;
    lake.position.set(0, MAP.LAKE_LEVEL, (L.cz - L.hz + MAP.DAM_Z) / 2);
    lake.scale.set(1, (MAP.DAM_Z - (L.cz - L.hz)) / (L.hz * 2 + 4), 1);
    this.scene.add(lake);
    // Tal-Wasser (Fluss, Tümpel)
    const valMat = mk(0x3c5f55, 0.82, 60);
    const val = new THREE.Mesh(new THREE.PlaneGeometry(640, 400), valMat);
    val.rotation.x = -Math.PI / 2;
    val.position.set(0, MAP.VALLEY_WATER, MAP.DAM_Z + 6 + 200);
    this.scene.add(val);
    // Giftsumpf
    const S = MAP.SWAMP;
    const swMat = mk(0x7c9a2a, 0.85, 18);
    swMat.emissive = new THREE.Color(0x3a4a08);
    swMat.emissiveIntensity = 0.6;
    const sw = new THREE.Mesh(new THREE.CircleGeometry(S.r + S.blend + 10, 48), swMat);
    sw.rotation.x = -Math.PI / 2;
    sw.position.set(S.x, MAP.SWAMP_WATER, S.z);
    this.scene.add(sw);
    this.waterMats.push(lakeMat, valMat, swMat);

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
    this.fallTex = ft;
    const fallMat = new THREE.MeshBasicMaterial({ map: ft, color: 0xcfe8ec, transparent: true, opacity: 0.75, depthWrite: false, side: THREE.DoubleSide });
    const fall = new THREE.Mesh(new THREE.PlaneGeometry(22, 30), fallMat);
    fall.position.set(0, 15.5, MAP.DAM_Z + MAP.DAM_THICK / 2 + 0.6);
    this.scene.add(fall);
    // Gischt
    const dot = makeSoftDot();
    const mistMat = new THREE.SpriteMaterial({ map: dot, color: 0xe8f2f2, transparent: true, opacity: 0.35, depthWrite: false });
    this.mist = [];
    for (let i = 0; i < 14; i++) {
      const s = new THREE.Sprite(mistMat);
      s.position.set(-10 + Math.random() * 20, 1.5 + Math.random() * 3, MAP.DAM_Z + 10 + Math.random() * 6);
      s.scale.setScalar(6 + Math.random() * 6);
      s.userData.phase = Math.random() * 6;
      this.scene.add(s);
      this.mist.push(s);
    }
  }

  buildSigns() {
    for (const l of this.mapData.labels) {
      const tex = makeSignTexture(l.text);
      const m = new THREE.Mesh(new THREE.PlaneGeometry(6, 1.5), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.8 }));
      m.position.set(l.x, l.y, l.z);
      m.rotation.y = l.ry;
      this.scene.add(m);
    }
    // Großer Schriftzug auf der Staumauer
    const tex = makeSignTexture('TALSPERRE 7', '#b9b3a3', '#3b3a35');
    const m = new THREE.Mesh(new THREE.PlaneGeometry(40, 10), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.9, transparent: true, opacity: 0.85 }));
    m.position.set(-60, 24, MAP.DAM_Z + MAP.DAM_THICK / 2 + 0.05);
    this.scene.add(m);
  }

  plantVegetation() {
    const rnd = mulberry32(2024);
    const T = this.terrain;
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
      if (this.col.query(x - r, z - r, x + r, z + r).length) return true;
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
      bushes.push({ x, z, y: T.height(x, z), s: 0.6 + rnd() * 1.0, r: rnd() * 6 });
    }
    for (let i = 0; i < 4000 && rocks.length < 420; i++) {
      const x = (rnd() * 2 - 1) * 300, z = (rnd() * 2 - 1) * 300;
      if (blocked(x, z, 2)) continue;
      const sl = T.slope(x, z);
      if (rnd() > 0.25 + sl) continue;
      rocks.push({ x, z, y: T.height(x, z), s: 0.8 + rnd() * rnd() * 3.5, r: rnd() * 6, t: rnd() * 6 });
    }

    if (this.assets) {
      this.plantModels(trees, bushes, rocks, rnd, blocked, n);
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
  plantModels(trees, bushes, rocks, rnd, blocked, n) {
    const T = this.terrain;
    const q = new THREE.Quaternion(), p = new THREE.Vector3(), sc = new THREE.Vector3(), e = new THREE.Euler();
    const m = new THREE.Matrix4();
    const c = new THREE.Color();
    // In Kacheln aufteilen, damit Frustum- und Schatten-Culling greifen
    const CH = 110;
    const inst = (name, all, place, { shadow = true, tint = null, doubleSide = false } = {}) => {
      if (!all.length) return;
      const chunks = new Map();
      for (const it of all) {
        const key = `${Math.floor((it.x + 330) / CH)}|${Math.floor((it.z + 330) / CH)}`;
        if (!chunks.has(key)) chunks.set(key, []);
        chunks.get(key).push(it);
      }
      const mats = this.assets.bakedMeshes(name).map((bm) => {
        const mat = bm.material.clone();
        if (doubleSide) mat.side = THREE.DoubleSide;
        return { geo: bm.geometry, mat };
      });
      for (const list of chunks.values()) {
        for (const { geo, mat } of mats) {
          const im = new THREE.InstancedMesh(geo, mat, list.length);
          im.castShadow = shadow;
          im.receiveShadow = true;
          list.forEach((it, i) => {
            place(it);
            im.setMatrixAt(i, m.compose(p, q, sc));
            if (tint) im.setColorAt(i, tint(it, c));
          });
          im.instanceMatrix.needsUpdate = true;
          if (im.instanceColor) im.instanceColor.needsUpdate = true;
          im.computeBoundingSphere();
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
    inst('v_pine', pines, (t) => { p.set(t.x, t.y - 0.1, t.z); q.setFromEuler(e.set(0, t.r, 0)); sc.set(t.s, t.s * (0.9 + (t.r % 0.4)), t.s); }, { tint: leafTint });
    inst('v_tree', leafy, (t) => { p.set(t.x, t.y - 0.1, t.z); q.setFromEuler(e.set(0, t.r, 0)); sc.setScalar(t.s); }, { tint: leafTint });
    inst('v_bush', bushes, (b) => { p.set(b.x, b.y - 0.05, b.z); q.setFromEuler(e.set(0, b.r * 3, 0)); sc.set(b.s, b.s * 0.9, b.s); }, { shadow: false, tint: leafTint });
    for (let k = 0; k < 3; k++) {
      const list = rocks.filter((_, i) => i % 3 === k);
      inst(`v_rock${k}`, list, (r) => { p.set(r.x, r.y - r.s * (0.15 + Math.min(0.6, T.slope(r.x, r.z) * 0.45)), r.z); q.setFromEuler(e.set(0, r.r, 0)); sc.setScalar(r.s * 0.8); }, {
        tint: (r, col) => { const v = 0.85 + (r.t % 1) * 0.3; return col.setRGB(v, v, v); },
      });
    }
    for (const r of rocks) if (r.s > 1.1) this.col.addCyl(r.x, r.z, r.s * 0.95, r.y - 2, r.y + r.s * 0.8, { rock: true });
    // Grasbüschel auf den Wiesen
    const grass = [];
    const maxGrass = { niedrig: 0, mittel: 7000, hoch: 14000 }[this.quality] ?? 7000;
    for (let i = 0; i < maxGrass * 7 && grass.length < maxGrass; i++) {
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
      grass.push({ x, z, y: h, s: 0.7 + rnd() * 0.8, r: rnd() * 6 });
    }
    inst('v_grass', grass, (g) => { p.set(g.x, g.y - 0.05, g.z); q.setFromEuler(e.set(0, g.r, 0)); sc.setScalar(g.s); }, { shadow: false, doubleSide: true, tint: leafTint });
    void blocked;
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

  update(dt, focus, sfx) {
    this.time += dt;
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
    this.fallTex.offset.y += dt * 1.6;
    for (const s of this.mist) {
      s.userData.phase += dt;
      s.material.opacity = 0.3;
      s.position.y = 1.5 + Math.sin(s.userData.phase) * 0.8;
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
