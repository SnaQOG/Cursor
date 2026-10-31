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
