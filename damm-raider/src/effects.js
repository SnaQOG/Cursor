// Partikel, Leuchtspuren, Mündungsfeuer, Lichtblitze und Schadenszahlen.
import * as THREE from 'three';
import { rand } from './util.js';

const VERT = `
  attribute float alpha; attribute float size; attribute vec3 color;
  varying float vAlpha; varying vec3 vColor;
  uniform float scale;
  void main(){
    vAlpha = alpha; vColor = color;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = size * scale / max(-mv.z, 0.1);
    gl_Position = projectionMatrix * mv;
  }`;
const FRAG = `
  varying float vAlpha; varying vec3 vColor;
  void main(){
    vec2 c = gl_PointCoord - 0.5;
    float d = length(c);
    if (d > 0.5) discard;
    float a = smoothstep(0.5, 0.1, d) * vAlpha;
    gl_FragColor = vec4(vColor, a);
  }`;

class ParticleSystem {
  constructor(scene, capacity, additive) {
    this.cap = capacity;
    this.n = 0;
    const g = new THREE.BufferGeometry();
    this.pos = new Float32Array(capacity * 3);
    this.col = new Float32Array(capacity * 3);
    this.alpha = new Float32Array(capacity);
    this.size = new Float32Array(capacity);
    this.vel = new Float32Array(capacity * 3);
    this.life = new Float32Array(capacity);
    this.max = new Float32Array(capacity);
    this.grav = new Float32Array(capacity);
    this.drag = new Float32Array(capacity);
    this.grow = new Float32Array(capacity);
    this.a0 = new Float32Array(capacity);
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('color', new THREE.BufferAttribute(this.col, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('alpha', new THREE.BufferAttribute(this.alpha, 1).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('size', new THREE.BufferAttribute(this.size, 1).setUsage(THREE.DynamicDrawUsage));
    g.setDrawRange(0, 0);
    this.geo = g;
    this.mat = new THREE.ShaderMaterial({
      uniforms: { scale: { value: 400 } },
      vertexShader: VERT,
      fragmentShader: FRAG,
      transparent: true,
      depthWrite: false,
      blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
    });
    this.points = new THREE.Points(g, this.mat);
    this.points.frustumCulled = false;
    this.points.renderOrder = additive ? 3 : 2;
    scene.add(this.points);
  }

  emit(x, y, z, vx, vy, vz, r, g, b, alpha, size, life, grav = 0, drag = 0, grow = 0) {
    if (this.n >= this.cap) return;
    const i = this.n++;
    this.pos[i * 3] = x; this.pos[i * 3 + 1] = y; this.pos[i * 3 + 2] = z;
    this.vel[i * 3] = vx; this.vel[i * 3 + 1] = vy; this.vel[i * 3 + 2] = vz;
    this.col[i * 3] = r; this.col[i * 3 + 1] = g; this.col[i * 3 + 2] = b;
    this.alpha[i] = alpha; this.a0[i] = alpha;
    this.size[i] = size;
    this.life[i] = life; this.max[i] = life;
    this.grav[i] = grav; this.drag[i] = drag; this.grow[i] = grow;
  }

  update(dt) {
    let i = 0;
    while (i < this.n) {
      this.life[i] -= dt;
      if (this.life[i] <= 0) {
        // mit dem letzten tauschen
        const j = --this.n;
        if (i !== j) {
          for (let k = 0; k < 3; k++) {
            this.pos[i * 3 + k] = this.pos[j * 3 + k];
            this.vel[i * 3 + k] = this.vel[j * 3 + k];
            this.col[i * 3 + k] = this.col[j * 3 + k];
          }
          this.alpha[i] = this.alpha[j]; this.a0[i] = this.a0[j]; this.size[i] = this.size[j];
          this.life[i] = this.life[j]; this.max[i] = this.max[j];
          this.grav[i] = this.grav[j]; this.drag[i] = this.drag[j]; this.grow[i] = this.grow[j];
        }
        continue;
      }
      const d = Math.max(0, 1 - this.drag[i] * dt);
      this.vel[i * 3] *= d; this.vel[i * 3 + 1] = this.vel[i * 3 + 1] * d - this.grav[i] * dt; this.vel[i * 3 + 2] *= d;
      this.pos[i * 3] += this.vel[i * 3] * dt;
      this.pos[i * 3 + 1] += this.vel[i * 3 + 1] * dt;
      this.pos[i * 3 + 2] += this.vel[i * 3 + 2] * dt;
      const t = this.life[i] / this.max[i];
      this.alpha[i] = this.a0[i] * Math.min(1, t * 2.5);
      this.size[i] += this.grow[i] * dt;
      i++;
    }
    this.geo.setDrawRange(0, this.n);
    for (const k of ['position', 'color', 'alpha', 'size']) this.geo.attributes[k].needsUpdate = true;
  }

  clear() { this.n = 0; this.geo.setDrawRange(0, 0); }
}

export class Effects {
  constructor(scene, camera, hudLayer) {
    this.scene = scene;
    this.camera = camera;
    this.glow = new ParticleSystem(scene, 3000, true);
    this.smoke = new ParticleSystem(scene, 1500, false);
    this.shake = 0;
    this.flash = 0;

    // Leuchtspuren
    const tg = new THREE.BoxGeometry(1, 1, 1);
    tg.translate(0, 0, 0);
    this.tracers = [];
    for (let i = 0; i < 40; i++) {
      const m = new THREE.Mesh(tg, new THREE.MeshBasicMaterial({ color: 0xffe1a0, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
      m.visible = false;
      m.frustumCulled = false;
      scene.add(m);
      this.tracers.push({ mesh: m, t: 0, max: 0.06 });
    }
    this.tracerIdx = 0;

    // Feste Anzahl Punktlichter (sonst Shader-Neukompilierung)
    this.lights = [];
    for (let i = 0; i < 3; i++) {
      const l = new THREE.PointLight(0xffb060, 0, 12, 2);
      scene.add(l);
      this.lights.push({ light: l, t: 0, max: 0.05, i0: 0 });
    }
    this.lightIdx = 0;

    // Mündungsblitz
    const c = document.createElement('canvas');
    c.width = c.height = 64;
    const g = c.getContext('2d');
    const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grd.addColorStop(0, 'rgba(255,250,220,1)');
    grd.addColorStop(0.3, 'rgba(255,190,80,0.8)');
    grd.addColorStop(1, 'rgba(255,120,20,0)');
    g.fillStyle = grd;
    g.fillRect(0, 0, 64, 64);
    const flashTex = new THREE.CanvasTexture(c);
    this.muzzleSprites = [];
    for (let i = 0; i < 6; i++) {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: flashTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
      s.visible = false;
      scene.add(s);
      this.muzzleSprites.push({ s, t: 0 });
    }
    this.muzzleIdx = 0;

    // Schadenszahlen
    this.layer = hudLayer;
    this.numbers = [];
    for (let i = 0; i < 28; i++) {
      const el = document.createElement('div');
      el.className = 'dmgnum';
      el.hidden = true;
      hudLayer.appendChild(el);
      this.numbers.push({ el, t: 0, pos: new THREE.Vector3(), vy: 0 });
    }
    this.numIdx = 0;
    this._v = new THREE.Vector3();
  }

  setViewport(h) {
    const s = h / 2 / Math.tan((this.camera.fov * Math.PI) / 360);
    this.glow.mat.uniforms.scale.value = s;
    this.smoke.mat.uniforms.scale.value = s;
  }

  light(pos, color, intensity, dist, dur) {
    const L = this.lights[this.lightIdx];
    this.lightIdx = (this.lightIdx + 1) % this.lights.length;
    L.light.position.copy(pos);
    L.light.color.set(color);
    L.light.distance = dist;
    L.i0 = intensity;
    L.light.intensity = intensity;
    L.t = dur; L.max = dur;
  }

  tracer(from, to, color = 0xffe1a0, width = 0.035, life = 0.06) {
    const T = this.tracers[this.tracerIdx];
    this.tracerIdx = (this.tracerIdx + 1) % this.tracers.length;
    const len = from.distanceTo(to);
    if (len < 0.1) return;
    T.mesh.position.copy(from).lerp(to, 0.5);
    T.mesh.scale.set(width, width, len);
    T.mesh.lookAt(to);
    T.mesh.material.color.set(color);
    T.mesh.material.opacity = 0.9;
    T.mesh.visible = true;
    T.t = life; T.max = life;
  }

  muzzle(pos, big = 1) {
    const M = this.muzzleSprites[this.muzzleIdx];
    this.muzzleIdx = (this.muzzleIdx + 1) % this.muzzleSprites.length;
    M.s.position.copy(pos);
    M.s.scale.setScalar(0.5 * big + Math.random() * 0.25);
    M.s.material.rotation = Math.random() * 6;
    M.s.visible = true;
    M.t = 0.045;
    this.light(pos, 0xffb060, 6 * big, 10, 0.05);
  }

  sparks(pos, normal, count = 8, color = [1, 0.75, 0.35], speed = 6) {
    for (let i = 0; i < count; i++) {
      const vx = normal.x * speed * 0.6 + rand(-1, 1) * speed * 0.6;
      const vy = normal.y * speed * 0.6 + rand(-0.3, 1) * speed * 0.6;
      const vz = normal.z * speed * 0.6 + rand(-1, 1) * speed * 0.6;
      this.glow.emit(pos.x, pos.y, pos.z, vx, vy, vz, color[0], color[1], color[2], 1, rand(0.06, 0.12), rand(0.15, 0.4), 12, 1.5);
    }
  }

  dust(pos, normal, color = [0.55, 0.5, 0.42], count = 5) {
    for (let i = 0; i < count; i++) {
      this.smoke.emit(pos.x, pos.y, pos.z, normal.x * 1.5 + rand(-0.6, 0.6), normal.y * 1.5 + rand(0, 1), normal.z * 1.5 + rand(-0.6, 0.6),
        color[0], color[1], color[2], 0.55, rand(0.25, 0.45), rand(0.4, 0.8), -0.3, 2, 0.9);
    }
  }

  splash(pos, toxic = false) {
    const c = toxic ? [0.6, 0.8, 0.25] : [0.8, 0.9, 0.95];
    for (let i = 0; i < 8; i++) this.smoke.emit(pos.x, pos.y, pos.z, rand(-1, 1), rand(2, 5), rand(-1, 1), c[0], c[1], c[2], 0.7, rand(0.12, 0.25), rand(0.4, 0.7), 9, 0.5, 0.2);
  }

  blood(pos) {
    // ARC "bluten" Öl und Funken
    for (let i = 0; i < 4; i++) this.smoke.emit(pos.x, pos.y, pos.z, rand(-1.5, 1.5), rand(0, 2.5), rand(-1.5, 1.5), 0.12, 0.12, 0.13, 0.8, rand(0.1, 0.2), rand(0.3, 0.6), 9, 0.5, 0);
  }

  explosion(pos, radius = 5) {
    const s = radius / 5;
    for (let i = 0; i < 50 * s; i++) {
      const a = Math.random() * Math.PI * 2, e = Math.random() * 1.4 - 0.2, sp = rand(4, 16) * s;
      this.glow.emit(pos.x, pos.y, pos.z, Math.cos(a) * Math.cos(e) * sp, Math.sin(e) * sp + 2, Math.sin(a) * Math.cos(e) * sp,
        1, rand(0.5, 0.85), 0.25, 1, rand(0.1, 0.25), rand(0.3, 0.9), 14, 1.2);
    }
    for (let i = 0; i < 18 * s; i++) {
      this.glow.emit(pos.x + rand(-1, 1) * s, pos.y + rand(0, 1) * s, pos.z + rand(-1, 1) * s, rand(-3, 3), rand(1, 5), rand(-3, 3),
        1, rand(0.45, 0.7), 0.15, 0.9, rand(1.4, 2.6) * s, rand(0.25, 0.5), -1, 3, 4 * s);
    }
    for (let i = 0; i < 22 * s; i++) {
      const g = rand(0.18, 0.3);
      this.smoke.emit(pos.x + rand(-1, 1) * s, pos.y + rand(0, 1.5) * s, pos.z + rand(-1, 1) * s, rand(-2.5, 2.5), rand(1, 4.5), rand(-2.5, 2.5),
        g, g, g * 0.95, 0.75, rand(1.5, 3) * s, rand(1.5, 3.2), -0.6, 1.2, 2.2 * s);
    }
    this.light(pos, 0xff9a40, 60 * s, 30 * s, 0.35);
  }

  smokePuff(pos, dark = true) {
    const g = dark ? rand(0.12, 0.2) : rand(0.5, 0.6);
    this.smoke.emit(pos.x + rand(-0.3, 0.3), pos.y, pos.z + rand(-0.3, 0.3), rand(-0.3, 0.3), rand(1, 2), rand(-0.3, 0.3), g, g, g, 0.5, rand(0.6, 1.1), rand(1.5, 2.5), -0.2, 0.5, 0.8);
  }

  fire(pos) {
    this.glow.emit(pos.x + rand(-0.3, 0.3), pos.y, pos.z + rand(-0.3, 0.3), rand(-0.3, 0.3), rand(1.5, 3), rand(-0.3, 0.3), 1, rand(0.4, 0.7), 0.1, 0.8, rand(0.3, 0.6), rand(0.3, 0.6), -1, 1, -0.3);
  }

  thrusterTrail(pos, color = [0.4, 0.8, 1]) {
    this.glow.emit(pos.x, pos.y, pos.z, rand(-0.3, 0.3), rand(-1.5, -0.5), rand(-0.3, 0.3), color[0], color[1], color[2], 0.6, rand(0.15, 0.3), rand(0.1, 0.25), 0, 2, -0.3);
  }

  number(pos, amount, kind) {
    const N = this.numbers[this.numIdx];
    this.numIdx = (this.numIdx + 1) % this.numbers.length;
    N.pos.copy(pos);
    N.pos.x += rand(-0.3, 0.3);
    N.pos.y += rand(0, 0.4);
    N.t = 0.8;
    N.vy = 1.4;
    N.el.textContent = Math.max(1, Math.round(amount));
    N.el.className = 'dmgnum ' + (kind || '');
    N.el.hidden = false;
  }

  addShake(a) { this.shake = Math.min(1.5, this.shake + a); }

  update(dt, w, h) {
    this.glow.update(dt);
    this.smoke.update(dt);
    for (const T of this.tracers) {
      if (!T.mesh.visible) continue;
      T.t -= dt;
      if (T.t <= 0) T.mesh.visible = false;
      else T.mesh.material.opacity = 0.9 * (T.t / T.max);
    }
    for (const L of this.lights) {
      if (L.t <= 0) { L.light.intensity = 0; continue; }
      L.t -= dt;
      L.light.intensity = L.i0 * Math.max(0, L.t / L.max);
    }
    for (const M of this.muzzleSprites) {
      if (!M.s.visible) continue;
      M.t -= dt;
      if (M.t <= 0) M.s.visible = false;
    }
    for (const N of this.numbers) {
      if (N.el.hidden) continue;
      N.t -= dt;
      if (N.t <= 0) { N.el.hidden = true; continue; }
      N.pos.y += N.vy * dt;
      N.vy *= 0.94;
      const v = this._v.copy(N.pos).project(this.camera);
      if (v.z > 1) { N.el.style.opacity = 0; continue; }
      N.el.style.transform = `translate(${((v.x + 1) / 2) * w}px, ${((1 - v.y) / 2) * h}px) translate(-50%,-50%)`;
      N.el.style.opacity = Math.min(1, N.t * 3);
    }
    this.shake = Math.max(0, this.shake - dt * 2.5);
    this.flash = Math.max(0, this.flash - dt * 3);
  }

  clear() {
    this.glow.clear();
    this.smoke.clear();
    for (const N of this.numbers) N.el.hidden = true;
    for (const T of this.tracers) T.mesh.visible = false;
  }
}
