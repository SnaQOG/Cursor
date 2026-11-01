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
// Kachelbare Sets aus Farbbild (grau, die Vertexfarbe färbt) und Normal-Map, die aus
// einem gleichzeitig gezeichneten Höhenbild entsteht. Eine Kachel deckt 4 m ab.
function finishTex(c, repeat = 1, linear = false) {
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = linear ? THREE.NoColorSpace : THREE.SRGBColorSpace;
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

// Höhenbild weichzeichnen (kachelbar) und in eine Normal-Map umrechnen
function normalFromHeight(hc, size, strength, blur) {
  const src = hc.getContext('2d').getImageData(0, 0, size, size).data;
  let H = new Float32Array(size * size);
  for (let i = 0; i < size * size; i++) H[i] = src[i * 4] / 255;
  const tmp = new Float32Array(size * size);
  for (let pass = 0; pass < blur; pass++) {
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      tmp[y * size + x] = (H[y * size + ((x + size - 1) % size)] + H[y * size + x] + H[y * size + ((x + 1) % size)]) / 3;
    }
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      H[y * size + x] = (tmp[((y + size - 1) % size) * size + x] + tmp[y * size + x] + tmp[((y + 1) % size) * size + x]) / 3;
    }
  }
  const c = canvas(size);
  const g = c.getContext('2d');
  const img = g.createImageData(size, size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (H[y * size + ((x + 1) % size)] - H[y * size + ((x + size - 1) % size)]) * strength;
      const dy = (H[((y + 1) % size) * size + x] - H[((y + size - 1) % size) * size + x]) * strength;
      const l = Math.hypot(dx, dy, 1);
      const i = (y * size + x) * 4;
      img.data[i] = (-dx / l * 0.5 + 0.5) * 255;
      img.data[i + 1] = (dy / l * 0.5 + 0.5) * 255;
      img.data[i + 2] = (1 / l * 0.5 + 0.5) * 255;
      img.data[i + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  H = null;
  return c;
}

function texSet(size, draw, { strength = 3, blur = 1, repeat = 1 } = {}) {
  const col = canvas(size), hgt = canvas(size);
  draw(col.getContext('2d'), hgt.getContext('2d'), size);
  return { map: finishTex(col, repeat), normal: finishTex(normalFromHeight(hgt, size, strength, blur), repeat, true) };
}

const gray = (v) => { const c = clamp(Math.round(v), 0, 255); return `rgb(${c},${c},${c})`; };

// Steine im Verband: Farbe und Höhe zugleich (Backstein, Quader, Pflaster)
function drawBond(gc, gh, size, cols, rows, { base, mortar, jitter, round = 0, seed = 1, stagger = 0.5, gap = 2, dome = false, chisel = 0 }) {
  const n = new Noise2D(seed);
  gc.fillStyle = mortar; gc.fillRect(0, 0, size, size);
  gh.fillStyle = gray(40); gh.fillRect(0, 0, size, size);
  const cw = size / cols, rh = size / rows;
  for (let r = 0; r < rows; r++) {
    const off = (r % 2) * stagger * cw;
    for (let k = -1; k <= cols; k++) {
      const x = k * cw + off + (dome ? (Math.random() - 0.5) * cw * 0.15 : 0), y = r * rh;
      const v = base + (Math.random() - 0.5) * jitter + n.noise(k * 0.7, r * 0.7) * jitter * 0.6;
      const w = cw - gap, h = rh - gap;
      const shape = (g) => {
        if (round > 0) { g.beginPath(); g.roundRect(x + gap / 2, y + gap / 2, w, h, round); g.fill(); }
        else g.fillRect(x + gap / 2, y + gap / 2, w, h);
      };
      gc.fillStyle = gray(v);
      shape(gc);
      if (dome) {
        const grd = gh.createRadialGradient(x + gap / 2 + w / 2, y + gap / 2 + h / 2, 1, x + gap / 2 + w / 2, y + gap / 2 + h / 2, Math.max(w, h) * 0.6);
        grd.addColorStop(0, gray(250)); grd.addColorStop(1, gray(130));
        gh.fillStyle = grd;
      } else gh.fillStyle = gray(200 + (Math.random() - 0.5) * 30);
      shape(gh);
      // Meißelspuren / Abplatzungen
      for (let c = 0; c < chisel; c++) {
        const px = x + gap + Math.random() * (w - gap), py = y + gap + Math.random() * (h - gap), r2 = 1 + Math.random() * 3;
        gh.fillStyle = gray(150 + Math.random() * 60);
        gh.fillRect(px, py, r2, r2);
        gc.fillStyle = `rgba(0,0,0,${Math.random() * 0.12})`;
        gc.fillRect(px, py, r2, r2);
      }
    }
  }
  speckle(gc, size, size * 6, 0.07);
}

export function makeBrickSet() {
  return texSet(512, (gc, gh, s) => drawBond(gc, gh, s, 16, 56, { base: 200, mortar: '#8a8a8a', jitter: 60, seed: 3, gap: 2, chisel: 1 }), { strength: 4, blur: 1 });
}

export function makeAshlarSet() {
  return texSet(512, (gc, gh, s) => drawBond(gc, gh, s, 7, 13, { base: 214, mortar: '#9a9a9a', jitter: 34, seed: 5, gap: 3, chisel: 30 }), { strength: 3, blur: 2 });
}

export function makeCobbleSet() {
  return texSet(512, (gc, gh, s) => drawBond(gc, gh, s, 18, 22, { base: 196, mortar: '#6c6c6c', jitter: 70, round: 7, seed: 9, gap: 4, dome: true }), { strength: 5, blur: 2 });
}

// Senkrechte Bretter mit Fugen und Maserung
export function makePlankSet() {
  return texSet(512, (gc, gh, size) => {
    const n = 20, w = size / n;
    gh.fillStyle = gray(200); gh.fillRect(0, 0, size, size);
    for (let i = 0; i < n; i++) {
      const v = 170 + (Math.random() - 0.5) * 60;
      gc.fillStyle = gray(v);
      gc.fillRect(i * w, 0, w, size);
      for (let k = 0; k < 9; k++) {
        const x = i * w + Math.random() * w;
        gc.strokeStyle = 'rgba(0,0,0,0.13)'; gh.strokeStyle = 'rgba(0,0,0,0.25)';
        for (const g of [gc, gh]) { g.beginPath(); g.moveTo(x, 0); g.bezierCurveTo(x + 3, size * 0.3, x - 3, size * 0.6, x + 1, size); g.stroke(); }
      }
      // Astlöcher
      if (Math.random() < 0.4) {
        const ky = Math.random() * size;
        gc.fillStyle = 'rgba(40,25,10,0.5)'; gc.beginPath(); gc.ellipse(i * w + w / 2, ky, 3, 5, 0, 0, 7); gc.fill();
        gh.fillStyle = gray(150); gh.beginPath(); gh.ellipse(i * w + w / 2, ky, 3, 5, 0, 0, 7); gh.fill();
      }
      gc.fillStyle = 'rgba(0,0,0,0.5)'; gc.fillRect(i * w, 0, 2, size);
      gh.fillStyle = gray(60); gh.fillRect(i * w, 0, 2, size);
    }
    speckle(gc, size, 3000, 0.06);
  }, { strength: 4, blur: 1 });
}

// Balkenholz: feine Maserung ohne Fugen
export function makeTimberSet() {
  return texSet(256, (gc, gh, size) => {
    gc.fillStyle = gray(175); gc.fillRect(0, 0, size, size);
    gh.fillStyle = gray(180); gh.fillRect(0, 0, size, size);
    for (let k = 0; k < 160; k++) {
      const x = Math.random() * size, a = 0.05 + Math.random() * 0.15;
      gc.strokeStyle = `rgba(0,0,0,${a})`; gh.strokeStyle = `rgba(0,0,0,${a * 1.5})`;
      for (const g of [gc, gh]) { g.lineWidth = 0.5 + Math.random(); g.beginPath(); g.moveTo(x, 0); g.bezierCurveTo(x + 2, size * 0.4, x - 2, size * 0.7, x, size); g.stroke(); }
    }
    // Risse
    for (let k = 0; k < 6; k++) {
      const x = Math.random() * size, y = Math.random() * size, l = 20 + Math.random() * 60;
      gc.fillStyle = 'rgba(0,0,0,0.4)'; gc.fillRect(x, y, 1.5, l);
      gh.fillStyle = gray(80); gh.fillRect(x, y, 1.5, l);
    }
    speckle(gc, size, 1500, 0.08);
  }, { strength: 3, blur: 1 });
}

// Biberschwanz-Dachziegel (Reihen quer, runde Unterkanten)
export function makeTileSet() {
  return texSet(512, (gc, gh, size) => {
    gc.fillStyle = '#6a6a6a'; gc.fillRect(0, 0, size, size);
    gh.fillStyle = gray(30); gh.fillRect(0, 0, size, size);
    const cols = 24, rows = 14, cw = size / cols, rh = size / rows;
    for (let r = 0; r < rows; r++) {
      const off = (r % 2) * cw * 0.5;
      for (let k = -1; k <= cols; k++) {
        const v = 190 + (Math.random() - 0.5) * 55;
        const x = k * cw + off, y = r * rh;
        const path = (g) => {
          g.beginPath(); g.moveTo(x + 1, y); g.lineTo(x + cw - 1, y); g.lineTo(x + cw - 1, y + rh * 0.7);
          g.quadraticCurveTo(x + cw / 2, y + rh * 1.15, x + 1, y + rh * 0.7); g.closePath(); g.fill();
        };
        const grd = gc.createLinearGradient(0, y, 0, y + rh);
        grd.addColorStop(0, gray(v * 0.72)); grd.addColorStop(1, gray(v));
        gc.fillStyle = grd; path(gc);
        const hg = gh.createLinearGradient(0, y, 0, y + rh);
        hg.addColorStop(0, gray(90)); hg.addColorStop(1, gray(240));
        gh.fillStyle = hg; path(gh);
      }
    }
    speckle(gc, size, 4000, 0.08);
  }, { strength: 5, blur: 1 });
}

// Schiefer in Schuppen
export function makeSlateSet() {
  return texSet(512, (gc, gh, size) => {
    gc.fillStyle = '#5a5a5a'; gc.fillRect(0, 0, size, size);
    gh.fillStyle = gray(40); gh.fillRect(0, 0, size, size);
    const cols = 16, rows = 18, cw = size / cols, rh = size / rows;
    for (let r = 0; r < rows; r++) {
      const off = (r % 2) * cw * 0.5;
      for (let k = -1; k <= cols; k++) {
        const v = 165 + (Math.random() - 0.5) * 50;
        const x = k * cw + off, y = r * rh;
        const path = (g) => {
          g.beginPath(); g.moveTo(x + 1, y); g.lineTo(x + cw - 1, y); g.lineTo(x + cw - 1, y + rh * 0.8);
          g.lineTo(x + cw / 2, y + rh * 1.2); g.lineTo(x + 1, y + rh * 0.8); g.closePath(); g.fill();
        };
        gc.fillStyle = gray(v); path(gc);
        const hg = gh.createLinearGradient(0, y, 0, y + rh * 1.2);
        hg.addColorStop(0, gray(110)); hg.addColorStop(1, gray(230));
        gh.fillStyle = hg; path(gh);
      }
    }
    speckle(gc, size, 3000, 0.07);
  }, { strength: 4, blur: 1 });
}

// Holzschindeln (Schwarzwald): kleine, sich überlappende Schindeln mit runder Unterkante
export function makeShingleSet() {
  return texSet(512, (gc, gh, size) => {
    gc.fillStyle = '#4a4a4a'; gc.fillRect(0, 0, size, size);
    gh.fillStyle = gray(30); gh.fillRect(0, 0, size, size);
    const cols = 30, rows = 26, cw = size / cols, rh = size / rows;
    for (let r = 0; r < rows; r++) {
      const off = (r % 2) * cw * 0.5;
      for (let k = -1; k <= cols; k++) {
        const v = 150 + (Math.random() - 0.5) * 70;
        const x = k * cw + off, y = r * rh;
        const path = (g) => {
          g.beginPath(); g.moveTo(x + 0.5, y); g.lineTo(x + cw - 0.5, y); g.lineTo(x + cw - 0.5, y + rh);
          g.quadraticCurveTo(x + cw / 2, y + rh * 1.35, x + 0.5, y + rh); g.closePath(); g.fill();
        };
        gc.fillStyle = gray(v); path(gc);
        gc.strokeStyle = 'rgba(0,0,0,0.12)';
        for (let i = 0; i < 3; i++) { const gx = x + Math.random() * cw; gc.beginPath(); gc.moveTo(gx, y); gc.lineTo(gx, y + rh); gc.stroke(); }
        const hg = gh.createLinearGradient(0, y, 0, y + rh * 1.3);
        hg.addColorStop(0, gray(80)); hg.addColorStop(1, gray(235));
        gh.fillStyle = hg; path(gh);
      }
    }
    speckle(gc, size, 3000, 0.07);
  }, { strength: 4, blur: 1 });
}

// Kalkputz mit Rissen, Wasserflecken und abgeplatzten Stellen
export function makePlasterSet() {
  return texSet(512, (gc, gh, size) => {
    const n = new Noise2D(13);
    const ic = gc.createImageData(size, size), ih = gh.createImageData(size, size);
    const per = 6;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const u = (x / size) * per, v = (y / size) * per, fx = x / size, fy = y / size;
        const f = (a, b) => n.fbm(a, b, 4);
        const val = (f(u, v) * (1 - fx) + f(u - per, v) * fx) * (1 - fy) + (f(u, v - per) * (1 - fx) + f(u - per, v - per) * fx) * fy;
        const fine = (Math.random() - 0.5) * 0.06;
        const c = clamp(0.86 + val * 0.25 + fine, 0, 1) * 235;
        const h = clamp(0.6 + val * 0.5 + fine * 2, 0, 1) * 255;
        const i = (y * size + x) * 4;
        ic.data[i] = ic.data[i + 1] = ic.data[i + 2] = c; ic.data[i + 3] = 255;
        ih.data[i] = ih.data[i + 1] = ih.data[i + 2] = h; ih.data[i + 3] = 255;
      }
    }
    gc.putImageData(ic, 0, 0);
    gh.putImageData(ih, 0, 0);
    // Wasserflecken (senkrechte Schlieren)
    for (let k = 0; k < 14; k++) {
      const x = Math.random() * size, w = 6 + Math.random() * 30, y0 = Math.random() * size, l = 60 + Math.random() * 200;
      const grd = gc.createLinearGradient(0, y0, 0, y0 + l);
      grd.addColorStop(0, 'rgba(40,35,25,0.16)'); grd.addColorStop(1, 'rgba(40,35,25,0)');
      gc.fillStyle = grd;
      gc.fillRect(x, y0, w, l);
    }
    // abgeplatzte Stellen
    for (let k = 0; k < 5; k++) {
      const x = Math.random() * size, y = Math.random() * size, rx = 10 + Math.random() * 25, ry = 6 + Math.random() * 18;
      gc.fillStyle = 'rgba(70,60,50,0.28)'; gh.fillStyle = gray(70);
      for (const g of [gc, gh]) { g.beginPath(); g.ellipse(x, y, rx, ry, Math.random() * 3, 0, 7); g.fill(); }
    }
    // Risse
    for (let k = 0; k < 10; k++) {
      let x = Math.random() * size, y = Math.random() * size;
      gc.strokeStyle = 'rgba(30,25,20,0.45)'; gh.strokeStyle = gray(20);
      gc.lineWidth = gh.lineWidth = 1;
      gc.beginPath(); gh.beginPath(); gc.moveTo(x, y); gh.moveTo(x, y);
      for (let s = 0; s < 12; s++) {
        x += (Math.random() - 0.5) * 14; y += 4 + Math.random() * 8;
        gc.lineTo(x, y); gh.lineTo(x, y);
      }
      gc.stroke(); gh.stroke();
    }
  }, { strength: 2.5, blur: 1 });
}

// Wiese am Boden: kurze Halme in Grautönen (die Vertexfarbe färbt grün)
export function makeGrassSet() {
  return texSet(512, (gc, gh, size) => {
    gc.fillStyle = gray(150); gc.fillRect(0, 0, size, size);
    gh.fillStyle = gray(100); gh.fillRect(0, 0, size, size);
    for (let k = 0; k < 9000; k++) {
      const x = Math.random() * size, y = Math.random() * size, l = 4 + Math.random() * 10, a = (Math.random() - 0.5) * 1.2;
      const v = 110 + Math.random() * 140;
      gc.strokeStyle = gray(v); gh.strokeStyle = gray(120 + Math.random() * 135);
      for (const g of [gc, gh]) {
        g.lineWidth = 1 + Math.random() * 1.2;
        g.beginPath(); g.moveTo(x, y); g.lineTo(x + Math.sin(a) * l, y - Math.cos(a) * l); g.stroke();
        // am Rand wiederholen (kachelbar)
        if (x < l || y < l || x > size - l || y > size - l) {
          for (const [ox, oy] of [[size, 0], [-size, 0], [0, size], [0, -size]]) {
            g.beginPath(); g.moveTo(x + ox, y + oy); g.lineTo(x + ox + Math.sin(a) * l, y + oy - Math.cos(a) * l); g.stroke();
          }
        }
      }
    }
  }, { strength: 2, blur: 1 });
}

// Webstoff für Teppiche und Läufer: Kette und Schuss, Bordüren, Rautenmuster (Kachel 4 m, Muster ~1 m)
export function makeClothSet() {
  return texSet(512, (gc, gh, size) => {
    gc.fillStyle = gray(170); gc.fillRect(0, 0, size, size);
    gh.fillStyle = gray(140); gh.fillRect(0, 0, size, size);
    const cell = size / 4;
    for (let cy = 0; cy < 4; cy++) {
      for (let cx = 0; cx < 4; cx++) {
        const x0 = cx * cell, y0 = cy * cell, m = cell / 2;
        // Bänder
        gc.fillStyle = gray(120); gc.fillRect(x0, y0 + cell * 0.08, cell, cell * 0.06); gc.fillRect(x0, y0 + cell * 0.86, cell, cell * 0.06);
        // Raute mit innerer Raute
        for (const [r, v] of [[0.36, 215], [0.25, 110], [0.12, 235]]) {
          gc.fillStyle = gray(v);
          gc.beginPath(); gc.moveTo(x0 + m, y0 + m - cell * r); gc.lineTo(x0 + m + cell * r, y0 + m); gc.lineTo(x0 + m, y0 + m + cell * r); gc.lineTo(x0 + m - cell * r, y0 + m); gc.closePath(); gc.fill();
        }
        // Haken an den Rautenspitzen
        gc.fillStyle = gray(95);
        for (const [dx, dy] of [[0, -0.42], [0.42, 0], [0, 0.42], [-0.42, 0]]) gc.fillRect(x0 + m + dx * cell - 3, y0 + m + dy * cell - 3, 6, 6);
      }
    }
    // Webstruktur: feine Kett- und Schussfäden
    for (let i = 0; i < size; i += 3) {
      gh.fillStyle = gray(i % 6 ? 120 : 165); gh.fillRect(i, 0, 1, size);
      gh.fillStyle = gray(i % 6 ? 150 : 110); gh.fillRect(0, i, size, 1);
      gc.fillStyle = `rgba(0,0,0,${i % 6 ? 0.04 : 0.08})`; gc.fillRect(0, i, size, 1);
    }
    speckle(gc, size, 4000, 0.08);
  }, { strength: 1.5, blur: 0 });
}
