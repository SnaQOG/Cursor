// Lädt die in Blender gebauten Modelle (assets/models.glb) und den Icon-Atlas.
// Liefert Klone (für Gegner, Waffen, Figur) und vorbereitete Geometrie
// (zum Backen in statische Batches oder für InstancedMesh).
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { EMBED } from './assetdata.js';

function b64ToBuffer(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out.buffer;
}

// Materialklasse eines glTF-Materials für die statischen Batches
export function materialClass(m) {
  if (m.emissive && m.emissiveIntensity > 0 && (m.emissive.r + m.emissive.g + m.emissive.b) > 0.05) return 'glow';
  if (m.transparent && m.opacity < 0.9) return 'glass';
  if (m.metalness >= 0.3) return 'metal';
  return 'world';
}

// CC0-Texturen unter assets/tex/ (Ablauf in tools/fetch_textures.py)
const TEXTURES = ['plaster', 'rooftile', 'cobble', 'cliff'];

export class Assets {
  constructor() {
    this.tex = {};
    this.roots = new Map();
    this.meshCache = new Map();
    this.boundsCache = new Map();
    this.ready = false;
    this.icons = null;
  }

  async load() {
    try {
      let buf;
      if (EMBED) buf = b64ToBuffer(EMBED.models);
      else {
        const r = await fetch('assets/models.glb');
        if (!r.ok) throw new Error(`models.glb: ${r.status}`);
        buf = await r.arrayBuffer();
      }
      await this.parse(buf);
      this.ready = true;
    } catch (err) {
      console.warn('3D-Modelle nicht geladen, nutze einfache Formen:', err.message || err);
    }
    await this.loadTextures();
    // Zusätzliche Modelle mit Texturen: SimReady-Requisiten, Felsen und Pflanzen.
    // Fehlen sie, bleiben die Blender-Modelle.
    if (this.ready) {
      for (const name of ['simready', 'nature', 'props']) {
        try {
          let buf;
          if (EMBED) buf = EMBED[name] ? b64ToBuffer(EMBED[name]) : null;
          else {
            const r = await fetch(`assets/${name}.glb`);
            buf = r.ok ? await r.arrayBuffer() : null;
          }
          if (buf) await this.parse(buf);
        } catch (err) {
          console.warn(`${name}.glb nicht geladen:`, err.message || err);
        }
      }
    }
    try {
      if (EMBED) this.icons = { url: EMBED.icons, ...EMBED.iconMeta };
      else {
        const r = await fetch('assets/icons.json');
        // absolute URL, weil die CSS-Variable sonst relativ zur Stylesheet-Datei aufgelöst wird
        if (r.ok) this.icons = { url: new URL('assets/icons.webp', document.baseURI).href, ...(await r.json()) };
      }
    } catch {
      this.icons = null;
    }
    return this.ready;
  }

  async loadTextures() {
    const loader = new THREE.TextureLoader();
    await Promise.all(TEXTURES.map(async (name) => {
      try {
        const load = async (suffix, srgb) => {
          const key = `${name}_${suffix}`;
          const t = await loader.loadAsync(EMBED ? EMBED.tex[key] : `assets/tex/${key}.webp`);
          t.wrapS = t.wrapT = THREE.RepeatWrapping;
          t.anisotropy = 8;
          if (srgb) t.colorSpace = THREE.SRGBColorSpace;
          return t;
        };
        this.tex[name] = { map: await load('c', true), normal: await load('n', false) };
      } catch (err) {
        console.warn(`Textur ${name} nicht geladen:`, err.message || err);
      }
    }));
  }

  async parse(buf) {
    const gltf = await new GLTFLoader().parseAsync(buf, '');
    for (const child of gltf.scene.children) {
      child.updateMatrixWorld(true);
      this.roots.set(child.name, child);
    }
    gltf.scene.traverse((o) => {
      if (!o.isMesh) return;
      o.castShadow = true;
      o.receiveShadow = true;
      const m = o.material;
      // Pflanzen sind Bildflächen und werden ausgeschnitten
      if (/^n_(fir|leaf|bush)/.test(o.name)) {
        m.alphaTest = 0.45;
        // Die Bilder sind neutral beleuchtet und liegen im Spiel im eigenen Schatten: etwas anheben
        m.color.setRGB(1.75, 1.8, 1.55);
        m.transparent = false;
        m.side = THREE.DoubleSide;
      } else if (m.transparent) m.depthWrite = false;
    });
  }

  has(name) {
    return this.ready && this.roots.has(name);
  }

  // Gemeinsames Material der Requisiten (Atlas) für die zusammengefassten Meshes
  dressMaterial() {
    if (this._dress === undefined) {
      this._dress = null;
      const src = this.has('d_crate1') ? this.meshes('d_crate1')[0]?.material : null;
      if (src) {
        this._dress = src.clone();
        this._dress.vertexColors = true;
        this._dress.metalness = 0;
      }
    }
    return this._dress;
  }

  // Tiefe Kopie der Hierarchie (Geometrie und Materialien werden geteilt)
  clone(name) {
    const src = this.roots.get(name);
    const c = src.clone(true);
    c.position.set(0, 0, 0);
    c.rotation.set(0, 0, 0);
    c.scale.set(1, 1, 1);
    return c;
  }

  // Alle Meshes eines Assets mit ihrer Transformation relativ zur Wurzel
  meshes(name) {
    if (this.meshCache.has(name)) return this.meshCache.get(name);
    const root = this.roots.get(name);
    const inv = new THREE.Matrix4().copy(root.matrixWorld).invert();
    const out = [];
    root.traverse((o) => {
      if (!o.isMesh) return;
      const matrix = new THREE.Matrix4().multiplyMatrices(inv, o.matrixWorld);
      out.push({ name: o.name, geometry: o.geometry, material: o.material, matrix });
    });
    this.meshCache.set(name, out);
    return out;
  }

  // Geometrie mit eingebackener Knotentransformation (für InstancedMesh)
  bakedMeshes(name) {
    const key = name + '#baked';
    if (this.meshCache.has(key)) return this.meshCache.get(key);
    const out = this.meshes(name).map((m) => {
      const g = m.geometry.clone();
      g.applyMatrix4(m.matrix);
      g.computeBoundingSphere();
      return { name: m.name, geometry: g, material: m.material };
    });
    this.meshCache.set(key, out);
    return out;
  }

  bounds(name) {
    if (this.boundsCache.has(name)) return this.boundsCache.get(name);
    const box = new THREE.Box3();
    for (const m of this.meshes(name)) {
      if (!m.geometry.boundingBox) m.geometry.computeBoundingBox();
      box.union(m.geometry.boundingBox.clone().applyMatrix4(m.matrix));
    }
    this.boundsCache.set(name, box);
    return box;
  }

  // Benannte Knoten eines Klons: "arc_wasp__Rotor0" -> nodes.Rotor0
  static nodes(obj, prefix) {
    const map = {};
    obj.traverse((o) => {
      if (o.name.startsWith(prefix + '__')) map[o.name.slice(prefix.length + 2)] = o;
    });
    return map;
  }

  iconStyle(id) {
    const I = this.icons;
    if (!I || I.items[id] == null) return null;
    const i = I.items[id];
    const col = i % I.cols, row = Math.floor(i / I.cols);
    const px = I.cols > 1 ? (col / (I.cols - 1)) * 100 : 0;
    const py = I.rows > 1 ? (row / (I.rows - 1)) * 100 : 0;
    return `background-size:${I.cols * 100}% ${I.rows * 100}%;background-position:${px}% ${py}%`;
  }
}

export const assets = new Assets();
