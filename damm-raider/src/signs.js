// Beschriftete Schilder: ein Atlas, der beim Start auf eine Leinwand gezeichnet wird und deshalb keine Datei braucht.
import * as THREE from 'three';
import { mulberry32 } from './util.js';

const W = 2048, H = 1200, CW = 512, CH = 96, NW = 64, NH = 48, NY = 1152;
export const SIGN_ASPECT = CW / CH;

// Hintergrund, Schrift, Rand; Schriftart: s = Serifen, g = Grotesk
const PAINT = {
  gruen: ['#23402f', '#d9bd6e', '#b49a55', 's'],
  rot: ['#5f2420', '#ecdfc0', '#c9b58a', 's'],
  schwarz: ['#1d1b19', '#d8b55c', '#a88c46', 's'],
  creme: ['#d8cfb4', '#2a241c', '#6b5b43', 's'],
  blau: ['#1f3f66', '#eee9dc', '#c9c5b8', 'g'],
  holz: ['#6f5233', '#f0e2c0', '#4a3522', 's'],
  warn: ['#d9a925', '#17140f', '#17140f', 'g'],
  amt: ['#dcdcd2', '#22262a', '#22262a', 'g'],
};

// Ladennamen nach Stadtteil, jeder Laden bekommt über seine Nummer immer denselben
export const SHOPS = {
  markt: ['Bäckerei', 'Apotheke', 'Krämerei', 'Metzgerei', 'Schneiderei', 'Uhrmacher', 'Gasthaus Zum Damm', 'Café Brücke', 'Buchbinder', 'Barbier', 'Kolonialwaren', 'Schuster'],
  wald: ['Sattlerei', 'Holzhandel', 'Schmiede', 'Kräuter & Tee', 'Gasthof Waldrast'],
  weiss: ['Salzhandel', 'Keramik', 'Stoffe & Tuch', 'Café Weiß', 'Silberschmied'],
  turm: ['Funkstube', 'Ersatzteile', 'Lampen & Öl', 'Kantine', 'Tauschladen', 'Altwaren'],
  hafen: ['Fischhalle', 'Seilerei', 'Schiffsbedarf', 'Hafenkneipe', 'Eisenwaren'],
};
const ROADS = ['Kreuzmarkt', 'Altstadt', 'Hafen', 'Weiße Stadt', 'Turmstadt', 'Waldterrasse'];
const PLANTS = [['KRAFTWERK', 'amt'], ['AUFBEREITUNG', 'amt'], ['UMSPANNWERK', 'amt'], ['HOCHSPANNUNG', 'warn'], ['BETRETEN VERBOTEN', 'warn'], ['EINSTURZGEFAHR', 'warn']];
const SHOP_PAINTS = ['gruen', 'rot', 'schwarz', 'creme', 'holz', 'blau'];

const cells = new Map();
let atlas = null;

function board(g, x, y, w, h, text, paintName, rnd) {
  const [bg, fg, edge, font] = PAINT[paintName];
  g.fillStyle = bg;
  g.fillRect(x, y, w, h);
  // Maserung und Verwitterung im Grund
  for (let i = 0; i < 26; i++) {
    g.fillStyle = `rgba(${rnd() < 0.5 ? '0,0,0' : '255,245,220'},${0.03 + rnd() * 0.05})`;
    g.fillRect(x, y + rnd() * h, w, 1 + rnd() * 2);
  }
  const pad = Math.max(3, h * 0.07);
  g.strokeStyle = edge;
  g.lineWidth = Math.max(1.5, h * 0.035);
  g.strokeRect(x + pad, y + pad, w - pad * 2, h - pad * 2);
  // Schrift passend zur Breite
  let size = h * 0.62;
  const face = font === 's' ? 'Georgia, "Times New Roman", serif' : 'Arial, Helvetica, sans-serif';
  g.font = `bold ${size}px ${face}`;
  const max = w - pad * 6, tw = g.measureText(text).width;
  if (tw > max) { size *= max / tw; g.font = `bold ${size}px ${face}`; }
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillStyle = 'rgba(0,0,0,0.35)';
  g.fillText(text, x + w / 2 + 1.5, y + h / 2 + 3);
  g.fillStyle = fg;
  g.fillText(text, x + w / 2, y + h / 2 + 1.5);
  // Abgeplatzte Farbe über Schrift und Rand
  for (let i = 0; i < w * h / 700; i++) {
    g.fillStyle = bg;
    g.globalAlpha = 0.35 + rnd() * 0.5;
    g.fillRect(x + rnd() * w, y + rnd() * h, 1 + rnd() * 5, 1 + rnd() * 2.5);
  }
  g.globalAlpha = 1;
  // Rostspuren vom Rand nach unten
  for (let i = 0; i < 5; i++) {
    g.fillStyle = 'rgba(70,40,20,0.16)';
    g.fillRect(x + rnd() * w, y, 2 + rnd() * 4, h * (0.3 + rnd() * 0.7));
  }
}

function buildAtlas() {
  const c = document.createElement('canvas');
  c.width = W;
  c.height = H;
  const g = c.getContext('2d');
  const rnd = mulberry32(4711);
  let n = 0;
  const add = (key, text, paint) => {
    const x = (n % 4) * CW, y = Math.floor(n / 4) * CH;
    board(g, x, y, CW, CH, text, paint, rnd);
    cells.set(key, [(x + 2) / W, 1 - (y + CH - 2) / H, (x + CW - 2) / W, 1 - (y + 2) / H]);
    n++;
  };
  for (const list of Object.values(SHOPS)) {
    for (const name of list) add(name, name, SHOP_PAINTS[n % SHOP_PAINTS.length]);
  }
  for (const name of ROADS) add(name, name, 'holz');
  for (const [name, paint] of PLANTS) add(name, name, paint);
  // Hausnummern als Emailschilder in der letzten Zeile
  for (let i = 0; i < W / NW; i++) {
    const x = i * NW;
    board(g, x, NY, NW, NH, String(i + 1), 'blau', rnd);
    cells.set(`#${i + 1}`, [(x + 1) / W, 1 - (NY + NH - 1) / H, (x + NW - 1) / W, 1 - (NY + 1) / H]);
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

export function signMaterial() {
  if (!atlas) atlas = buildAtlas();
  return new THREE.MeshStandardMaterial({ map: atlas, vertexColors: true, roughness: 0.72, metalness: 0.05 });
}

// Laden zur Lage in der Stadt, immer derselbe für dieselbe Nummer
export function shopName(id, district) {
  const list = SHOPS[district] || SHOPS.markt;
  return list[(id * 7 + (id >> 3)) % list.length];
}

export const houseNumber = (id) => `#${id % (W / NW) + 1}`;
export const roadName = (i) => ROADS[i % ROADS.length];

// Schild als Fläche: Mitte (x, y, z), Blickrichtung (nx, nz) von der Schrift weg zum Betrachter, Unterkante y
export function addSign(batcher, key, x, y, z, nx, nz, w, h, layer = 'det') {
  const r = cells.get(key);
  if (!r || !batcher.materials.sign) return false;
  const rx = nz * w / 2, rz = -nx * w / 2;
  batcher.addUVQuad([x - rx, y, z - rz], [x + rx, y, z + rz], [x + rx, y + h, z + rz], [x - rx, y + h, z - rz], [nx, 0, nz], r, 'sign', layer);
  return true;
}
