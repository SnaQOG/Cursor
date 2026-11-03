// Bodenpflanzen aus einem gemeinsamen Atlas (blender/ground_build.py): je Pflanze zwei gekreuzte Bildflächen,
// zusammengefasst in großen Meshes. So kostet eine ganze Wiese nur einen Draw-Call je 40-m-Kachel.
import * as THREE from 'three';

export const COVER_ROWS = 3, COVER_COLS = 4;
// id: Spalte, Zeile, Kantenlänge des Bildes in Metern
export const COVER = {
  gross1: [0, 0, 0.47], gross2: [1, 0, 0.39], mittel: [2, 0, 0.32], klein: [3, 0, 0.25],
  halme: [0, 1, 0.33], trocken: [1, 1, 0.28], loewenzahn: [2, 1, 0.35], gelb: [3, 1, 0.39],
  rosa: [0, 2, 0.08], nessel: [1, 2, 0.08], kraut: [2, 2, 0.38], farn: [3, 2, 1.37],
};
// Obergrenze je Grafikstufe: 12 Ecken je Pflanze, rund 0,5 kB
export const COVER_BUDGET = { niedrig: 0, mittel: 90000, hoch: 160000 };

// Wind nach der Höhe innerhalb des Atlasfelds, Phase aus der Lage in der Welt
export function coverMaterial(world) {
  const t = world.assets?.tex?.cover;
  if (!t) return null;
  const m = new THREE.MeshStandardMaterial({ vertexColors: true, map: t.map, alphaTest: 0.38, side: THREE.DoubleSide, roughness: 0.92, metalness: 0 });
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uWindTime = world.wind.time;
    sh.uniforms.uWindK = world.wind.strength;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nuniform float uWindTime;\nuniform float uWindK;')
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        {
          float hgt = fract(uv.y * ${COVER_ROWS.toFixed(1)});
          float ph = position.x * 0.53 + position.z * 0.37;
          float t = uWindTime * (0.9 + 0.5 * uWindK);
          float bend = hgt * hgt * 0.1 * uWindK;
          transformed.x += (sin(t * 1.7 + ph) + 0.4 * sin(t * 4.1 + ph * 2.1)) * bend;
          transformed.z += cos(t * 1.3 + ph * 1.3) * bend * 0.7;
        }`);
    // Beidseitige Flächen drehen sonst die Normale auf der Rückseite um: eine der gekreuzten Flächen würde fast schwarz
    sh.fragmentShader = sh.fragmentShader.replace('#include <normal_fragment_begin>', '#include <normal_fragment_begin>\nnormal = normalize(vNormal);');
  };
  m.customProgramCacheKey = () => 'cover-wind';
  return m;
}

// Eine Pflanze der Wiese. Hohe Gräser, Blumen und trockene Stellen wachsen in Flecken wie auf echten Wiesen.
// scale: 1 in Höfen, größer auf offenen Flächen, wo die Pflanzen weiter auseinander stehen
export function placeMeadowPlant(b, n, rnd, x, y, z, scale = 1) {
  const tall = n.fbm(x * 0.07 + 3, z * 0.07 + 9, 2), bloom = n.fbm(x * 0.15 + 40, z * 0.15 - 12, 2), dry = n.fbm(x * 0.05 - 21, z * 0.05 + 6, 2);
  const k = 0.86 + rnd() * 0.3;
  const dk = Math.max(0, dry);
  const r = (0.92 + dk * 0.9) * k, gr = k, bl = (0.84 - dk * 0.5) * k;
  const color = (Math.min(255, r * 214) << 16) | (Math.min(255, gr * 224) << 8) | Math.min(255, bl * 196);
  const u = rnd();
  let id, s;
  if (bloom > 0.14 && u < 0.34) { id = u < 0.14 ? 'loewenzahn' : u < 0.24 ? 'gelb' : 'rosa'; s = id === 'rosa' ? 4 + rnd() * 2 : 1 + rnd() * 0.5; }
  else if (dry > 0.12 && u < 0.45) { id = 'trocken'; s = 2.2 + rnd() * 1.4; }
  else if (u < 0.05) { id = 'halme'; s = 1.9 + rnd() * 1; }
  else if (u < 0.09) { id = rnd() < 0.5 ? 'kraut' : 'farn'; s = id === 'farn' ? 0.5 + rnd() * 0.3 : 1 + rnd() * 0.6; }
  else if (tall > -0.12) { id = rnd() < 0.5 ? 'gross1' : 'gross2'; s = (2.6 + rnd() * 1.6) * scale; }
  else { id = rnd() < 0.5 ? 'klein' : 'mittel'; s = (2.2 + rnd() * 1.4) * scale; }
  const c = COVER[id];
  b.addCover(x, y - 0.02, z, c[2] * s, rnd() * 6.28, c, COVER_COLS, COVER_ROWS, color);
}
