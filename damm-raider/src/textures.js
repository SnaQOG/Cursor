// Prozedurale Canvas-Texturen (keine externen Assets).
import * as THREE from 'three';
import { Noise2D, clamp } from './util.js';

function canvas(size) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  return c;
}

// Kachelbare Schmutz-/Betonstruktur, wird mit Vertexfarben multipliziert.
export function makeGrungeTexture(size = 256, seed = 7) {
  const c = canvas(size);
  const g = c.getContext('2d');
  const img = g.createImageData(size, size);
  const n = new Noise2D(seed);
  const period = 8;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = (x / size) * period, v = (y / size) * period;
      // kachelbar über Überblendung von vier Samples
      const fx = x / size, fy = y / size;
      const s = (a, b) => n.fbm(a, b, 4);
      const v00 = s(u, v), v10 = s(u - period, v), v01 = s(u, v - period), v11 = s(u - period, v - period);
      let val = (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy;
      val = 0.82 + val * 0.35 + (Math.random() - 0.5) * 0.08;
      const c8 = clamp(Math.round(val * 235), 0, 255);
      const i = (y * size + x) * 4;
      img.data[i] = c8; img.data[i + 1] = c8; img.data[i + 2] = c8; img.data[i + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  // ein paar Flecken und Kratzer
  for (let k = 0; k < 60; k++) {
    g.fillStyle = `rgba(0,0,0,${Math.random() * 0.06})`;
    const r = 2 + Math.random() * 14;
    g.beginPath();
    g.arc(Math.random() * size, Math.random() * size, r, 0, Math.PI * 2);
    g.fill();
  }
  g.strokeStyle = 'rgba(255,255,255,0.05)';
  for (let k = 0; k < 40; k++) {
    g.beginPath();
    const x = Math.random() * size, y = Math.random() * size;
    g.moveTo(x, y);
    g.lineTo(x + (Math.random() - 0.5) * 30, y + (Math.random() - 0.5) * 30);
    g.stroke();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

// Weicher Kreis für Partikel
export function makeSoftDot(size = 64) {
  const c = canvas(size);
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grd.addColorStop(0, 'rgba(255,255,255,1)');
  grd.addColorStop(0.4, 'rgba(255,255,255,0.6)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd;
  g.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(c);
  return tex;
}

// Warnstreifen (gelb/schwarz) für Aufzüge usw.
export function makeHazardTexture() {
  const c = canvas(128);
  const g = c.getContext('2d');
  g.fillStyle = '#e8b21e';
  g.fillRect(0, 0, 128, 128);
  g.fillStyle = '#1d1d1b';
  for (let i = -128; i < 256; i += 32) {
    g.beginPath();
    g.moveTo(i, 0); g.lineTo(i + 16, 0); g.lineTo(i + 16 + 128, 128); g.lineTo(i + 128, 128);
    g.fill();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// Schild mit Text (z.B. Ortsnamen an Gebäuden)
export function makeSignTexture(text, bg = '#c9c1a8', fg = '#20231f') {
  const c = document.createElement('canvas');
  c.width = 512; c.height = 128;
  const g = c.getContext('2d');
  g.fillStyle = bg;
  g.fillRect(0, 0, 512, 128);
  g.strokeStyle = fg;
  g.lineWidth = 6;
  g.strokeRect(8, 8, 496, 112);
  g.fillStyle = fg;
  g.font = 'bold 64px "Big Shoulders Display", Impact, sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(text.toUpperCase(), 256, 68);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// ---------------------------------------------------------------- Stadt-Texturen
// Alle kachelbar und grau: die Farbe kommt aus den Vertexfarben. Eine Kachel
// deckt 4 m ab (Welt-UV im Batcher = Position / 4).
function finishTex(c, repeat = 1) {
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  tex.repeat.set(repeat, repeat);
  return tex;
}

function speckle(g, size, n, a = 0.05) {
  for (let k = 0; k < n; k++) {
    g.fillStyle = Math.random() < 0.5 ? `rgba(0,0,0,${Math.random() * a})` : `rgba(255,255,255,${Math.random() * a})`;
    g.fillRect(Math.random() * size, Math.random() * size, 1 + Math.random() * 3, 1 + Math.random() * 3);
  }
}

// Gemauerte Steine in versetzten Reihen (Backstein, Quader, Kopfsteinpflaster)
function bondTexture(size, cols, rows, { base, mortar, jitter, round = 0, seed = 1, stagger = 0.5, gap = 2 }) {
  const c = canvas(size);
  const g = c.getContext('2d');
  g.fillStyle = mortar;
  g.fillRect(0, 0, size, size);
  const n = new Noise2D(seed);
  const cw = size / cols, rh = size / rows;
  for (let r = 0; r < rows; r++) {
    const off = (r % 2) * stagger * cw;
    for (let k = -1; k <= cols; k++) {
      const x = k * cw + off, y = r * rh;
      const v = base + (Math.random() - 0.5) * jitter + n.noise(k * 0.7, r * 0.7) * jitter * 0.6;
      const c8 = clamp(Math.round(v), 0, 255);
      g.fillStyle = `rgb(${c8},${c8},${c8})`;
      if (round > 0) {
        g.beginPath();
        g.roundRect(x + gap / 2, y + gap / 2, cw - gap, rh - gap, round);
        g.fill();
      } else g.fillRect(x + gap / 2, y + gap / 2, cw - gap, rh - gap);
    }
  }
  speckle(g, size, size * 6, 0.07);
  return c;
}

export function makeBrickTexture() {
  return finishTex(bondTexture(512, 16, 56, { base: 200, mortar: '#8a8a8a', jitter: 60, seed: 3, gap: 2 }));
}

export function makeAshlarTexture() {
  return finishTex(bondTexture(512, 7, 13, { base: 214, mortar: '#9a9a9a', jitter: 34, seed: 5, gap: 3 }));
}

export function makeCobbleTexture() {
  const c = bondTexture(512, 18, 22, { base: 196, mortar: '#7c7c7c', jitter: 70, round: 7, seed: 9, stagger: 0.5, gap: 4 });
  return finishTex(c);
}

// Senkrechte Bretter mit Fugen und Maserung
export function makePlankTexture() {
  const size = 512, c = canvas(size), g = c.getContext('2d');
  const n = 20, w = size / n;
  for (let i = 0; i < n; i++) {
    const v = 170 + (Math.random() - 0.5) * 60;
    g.fillStyle = `rgb(${v},${v},${v})`;
    g.fillRect(i * w, 0, w, size);
    g.strokeStyle = 'rgba(0,0,0,0.12)';
    for (let k = 0; k < 6; k++) {
      g.beginPath();
      const x = i * w + Math.random() * w;
      g.moveTo(x, 0);
      g.bezierCurveTo(x + 3, size * 0.3, x - 3, size * 0.6, x + 1, size);
      g.stroke();
    }
    g.fillStyle = 'rgba(0,0,0,0.45)';
    g.fillRect(i * w, 0, 2, size);
  }
  speckle(g, size, 3000, 0.06);
  return finishTex(c);
}

// Biberschwanz-Dachziegel (Reihen quer, runde Unterkanten)
export function makeTileTexture() {
  const size = 512, c = canvas(size), g = c.getContext('2d');
  g.fillStyle = '#6a6a6a';
  g.fillRect(0, 0, size, size);
  const cols = 24, rows = 14, cw = size / cols, rh = size / rows;
  for (let r = 0; r < rows; r++) {
    const off = (r % 2) * cw * 0.5;
    for (let k = -1; k <= cols; k++) {
      const v = 190 + (Math.random() - 0.5) * 55;
      const x = k * cw + off, y = r * rh;
      const grd = g.createLinearGradient(0, y, 0, y + rh);
      grd.addColorStop(0, `rgb(${v * 0.75},${v * 0.75},${v * 0.75})`);
      grd.addColorStop(1, `rgb(${v},${v},${v})`);
      g.fillStyle = grd;
      g.beginPath();
      g.moveTo(x + 1, y);
      g.lineTo(x + cw - 1, y);
      g.lineTo(x + cw - 1, y + rh * 0.7);
      g.quadraticCurveTo(x + cw / 2, y + rh * 1.15, x + 1, y + rh * 0.7);
      g.closePath();
      g.fill();
    }
  }
  speckle(g, size, 4000, 0.08);
  return finishTex(c);
}

// Schiefer in Schuppen
export function makeSlateTexture() {
  const size = 512, c = canvas(size), g = c.getContext('2d');
  g.fillStyle = '#5a5a5a';
  g.fillRect(0, 0, size, size);
  const cols = 16, rows = 18, cw = size / cols, rh = size / rows;
  for (let r = 0; r < rows; r++) {
    const off = (r % 2) * cw * 0.5;
    for (let k = -1; k <= cols; k++) {
      const v = 165 + (Math.random() - 0.5) * 50;
      g.fillStyle = `rgb(${v},${v},${v})`;
      const x = k * cw + off, y = r * rh;
      g.beginPath();
      g.moveTo(x + 1, y);
      g.lineTo(x + cw - 1, y);
      g.lineTo(x + cw - 1, y + rh * 0.8);
      g.lineTo(x + cw / 2, y + rh * 1.2);
      g.lineTo(x + 1, y + rh * 0.8);
      g.closePath();
      g.fill();
    }
  }
  speckle(g, size, 3000, 0.07);
  return finishTex(c);
}
