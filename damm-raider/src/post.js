// Nachbearbeitung: Umgebungsverdeckung (GTAO), Bloom, Tonemapping, Farbkorrektur mit
// Split-Toning, Vignette und feinem Korn. Stufen nach Grafikqualität:
//   niedrig – direktes Rendern ohne Nachbearbeitung
//   mittel  – Kantenglättung (MSAA), Bloom, Farbkorrektur
//   hoch    – zusätzlich GTAO (in halber Auflösung)
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

// Farblook je Wetterlage (im Anzeigeraum nach dem Tonemapping)
const LOOKS = {
  tag: { contrast: 1.07, saturation: 1.1, shadow: [0.95, 0.99, 1.06], high: [1.04, 1.0, 0.95], vignette: 0.26, bloom: 0.22 },
  daemmerung: { contrast: 1.1, saturation: 1.06, shadow: [0.92, 0.96, 1.1], high: [1.08, 0.99, 0.9], vignette: 0.32, bloom: 0.32 },
  sturm: { contrast: 1.05, saturation: 0.82, shadow: [0.95, 1.0, 1.06], high: [0.98, 1.0, 1.03], vignette: 0.36, bloom: 0.12 },
};

const GradeShader = {
  uniforms: {
    tDiffuse: { value: null },
    contrast: { value: 1.07 },
    saturation: { value: 1.1 },
    shadowTint: { value: new THREE.Vector3(0.95, 0.99, 1.06) },
    highTint: { value: new THREE.Vector3(1.04, 1.0, 0.95) },
    vignette: { value: 0.26 },
    grain: { value: 0.018 },
    time: { value: 0 },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform float contrast, saturation, vignette, grain, time;
    uniform vec3 shadowTint, highTint;
    varying vec2 vUv;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main() {
      vec4 c = texture2D(tDiffuse, vUv);
      vec3 col = c.rgb;
      float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
      col = mix(vec3(l), col, saturation);
      col = (col - 0.5) * contrast + 0.5;
      // kühle Schatten, warme Lichter
      col *= mix(shadowTint, highTint, smoothstep(0.05, 0.85, l));
      // Vignette
      vec2 d = vUv - 0.5;
      col *= 1.0 - vignette * smoothstep(0.2, 0.75, dot(d, d) * 2.0);
      // feines Filmkorn (vor allem in dunklen Bereichen sichtbar)
      col += (hash(vUv * vec2(1931.0, 1093.0) + fract(time)) - 0.5) * grain * (1.2 - l);
      gl_FragColor = vec4(clamp(col, 0.0, 1.0), c.a);
    }`,
};

export class Post {
  constructor(renderer) {
    this.renderer = renderer;
    this.composer = null;
    this.key = '';
    this.size = [1, 1, 1];
  }

  get active() {
    return !!this.composer;
  }

  // Bei Kartenwechsel oder geänderter Qualität neu aufbauen
  setup(scene, camera, quality) {
    const key = `${scene.uuid}|${quality}`;
    if (key === this.key) return;
    this.key = key;
    this.dispose();
    if (quality === 'niedrig') return;
    const r = this.renderer;
    const [w, h, pr] = this.size;
    const rt = new THREE.WebGLRenderTarget(w * pr, h * pr, { type: THREE.HalfFloatType, samples: 4 });
    const composer = new EffectComposer(r, rt);
    composer.addPass(new RenderPass(scene, camera));
    if (quality === 'hoch') {
      const ao = new GTAOPass(scene, camera, w, h);
      ao.output = GTAOPass.OUTPUT.Default;
      ao.blendIntensity = 0.85;
      ao.updateGtaoMaterial({ radius: 0.9, distanceExponent: 1.6, thickness: 2.0, scale: 1.1, samples: 12, distanceFallOff: 1.0 });
      ao.updatePdMaterial({ lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 6, radiusExponent: 1, rings: 2, samples: 12 });
      // halbe Auflösung genügt für weiche Verdeckung
      const setSize = ao.setSize.bind(ao);
      ao.setSize = (sw, sh) => setSize(Math.max(1, Math.floor(sw / 2)), Math.max(1, Math.floor(sh / 2)));
      composer.addPass(ao);
      this.ao = ao;
    }
    this.bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 0.22, 0.45, 0.92);
    composer.addPass(this.bloom);
    composer.addPass(new OutputPass());
    this.grade = new ShaderPass(GradeShader);
    composer.addPass(this.grade);
    this.composer = composer;
    composer.setPixelRatio(pr);
    composer.setSize(w, h);
    this.look = null;
  }

  setSize(w, h, pr) {
    this.size = [w, h, pr];
    if (!this.composer) return;
    this.composer.setPixelRatio(pr);
    this.composer.setSize(w, h);
  }

  setLook(id) {
    if (!this.composer || id === this.look) return;
    this.look = id;
    const L = LOOKS[id] || LOOKS.tag;
    const u = this.grade.uniforms;
    u.contrast.value = L.contrast;
    u.saturation.value = L.saturation;
    u.shadowTint.value.set(...L.shadow);
    u.highTint.value.set(...L.high);
    u.vignette.value = L.vignette;
    this.bloom.strength = L.bloom;
  }

  render(dt) {
    this.grade.uniforms.time.value += dt;
    this.composer.render(dt);
  }

  dispose() {
    if (!this.composer) return;
    for (const p of this.composer.passes) if (p.dispose) p.dispose();
    this.composer.renderTarget1.dispose();
    this.composer.renderTarget2.dispose();
    this.composer = null;
    this.ao = null;
    this.bloom = null;
    this.grade = null;
  }
}
