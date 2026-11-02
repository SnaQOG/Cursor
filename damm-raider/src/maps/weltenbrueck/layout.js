// Stadtplan von Weltenbrück: reine Daten ohne three.js. Wird für Terrain,
// 3D-Aufbau, Übersichtskarte und den Kartenentwurf benutzt.
// Innerhalb der Stadtmauer: Kreuzmarkt mit Dom (Süden) und die Waldsiedlung am
// Gründerfelsen (Norden). Durch die Tore: Weiße Stadt (Westen), Turmstadt (Osten)
// und Hafenviertel (Süden).
// Koordinaten in Metern, +X = Osten, -Z = Norden.
import { mulberry32 } from '../../util.js';

export const CITY = {
  HALF: 400, // Terrain von -400 bis +400 (hinter den Vierteln Hügel und Meer)
  RES: 401, // 2-m-Raster
  BOUND: 292, // spielbarer Bereich
  WALL: 197, // Innenkante der Stadtmauer
  WALL_T: 3,
  GROUND: 3, // Höhe von Markt und unterster Terrasse
  CLIFF_Z: -166, // Fuß des Gründerfelsens
  CLIFF_TOP: 52,
  QUAY: 240, // Kaikante, dahinter das Meer
  OUTER: 293, // Außenmauer der Viertel
};

// Terrassen der Waldsiedlung: z0 = Nordrand (Stützmauer zur nächsten Terrasse), z1 = Südrand
export const TERRACES = [
  { id: 't0', name: 'Untere Terrasse', z0: -40, z1: -2, h: 3 },
  { id: 't1', name: 'Mittlere Terrasse', z0: -80, z1: -40, h: 8 },
  { id: 't2', name: 'Obere Terrasse', z0: -120, z1: -80, h: 13 },
  { id: 't3', name: 'Felsterrasse', z0: -166, z1: -120, h: 18 },
];
export const RISE = 5;
// gleiche Stufen wie Builder.stairs (0,3 m hoch, 0,34 m tief)
export const STAIR_N = Math.round(RISE / 0.3);
export const STAIR_RUN = STAIR_N * 0.34;
// Treppengassen durch die Stützmauern: Mauerlinie z, Mitte x, Breite w (Kanten auf geraden Metern)
export const STAIRS = [
  { z: -40, list: [[-140, 4], [-70, 4], [0, 8], [76, 4], [150, 4]] },
  { z: -80, list: [[-120, 4], [-40, 4], [0, 8], [46, 4], [120, 4]] },
  { z: -120, list: [[-150, 4], [-86, 4], [0, 8], [60, 4], [140, 4]] },
];

// Ebene (ohne Treppen-Rampen) an Position z
export function levelAt(z, x = 0) {
  if (Math.abs(x) > 200 || z > -40) return 3; // die Viertel vor der Mauer liegen eben
  if (z > -80) return 8;
  if (z > -120) return 13;
  return 18;
}

// Rampe einer Treppengasse? Liefert Höhe oder null
export function stairRamp(x, z) {
  for (const s of STAIRS) {
    if (z > s.z || z < s.z - STAIR_RUN) continue;
    for (const [cx, w] of s.list) {
      if (Math.abs(x - cx) <= w / 2) return levelAt(s.z + 1) + ((s.z - z) / STAIR_RUN) * RISE;
    }
  }
  return null;
}

// ------------------------------------------------------------------ Hilfen
const R = (x0, x1, z0, z1, extra = {}) => ({ x0, x1, z0, z1, ...extra });
const overlap = (a, b, m = 0) => a.x0 < b.x1 + m && a.x1 > b.x0 - m && a.z0 < b.z1 + m && a.z1 > b.z0 - m;
const inside = (x, z, r, m = 0) => x > r.x0 - m && x < r.x1 + m && z > r.z0 - m && z < r.z1 + m;
const snap = (v) => Math.round(v * 2) / 2;

// ------------------------------------------------------------------ Feste Elemente
export const PLAZA = R(-34, 34, 72, 120, { name: 'Kreuzmarkt' });
export const DOM = {
  area: R(-16, 16, 6, 72),
  nave: R(-12, 12, 24, 58),
  tower: R(-8, 8, 58, 72),
  choir: R(-9, 9, 13, 24),
};
export const RATHAUS = R(-33, -6, 121, 145);

// Hauptstraßen (Kreuzmarkt-Viertel)
export const STREETS = [
  R(-197, 197, -2, 6, { name: 'Ringgasse', main: true }),
  R(-197, -34, 91, 101, { name: 'Weststraße', main: true }),
  R(34, 197, 91, 101, { name: 'Oststraße', main: true }),
  R(-5, 5, 120, 197, { name: 'Hafenstraße', main: true }),
  R(-24, -16, 6, 72, { name: 'Westliche Domgasse', main: true }),
  R(16, 24, 6, 72, { name: 'Östliche Domgasse', main: true }),
  R(-92, -86, 6, 197, { name: 'Gerbergasse' }),
  R(86, 92, 6, 197, { name: 'Zunftgasse' }),
  R(-152, -147, 6, 197, { name: 'Mauergasse West' }),
  R(147, 152, 6, 197, { name: 'Mauergasse Ost' }),
  R(-197, -24, 42, 47, { name: 'Domstraße' }),
  R(24, 197, 42, 47, { name: 'Domstraße' }),
  R(-197, -5, 150, 156, { name: 'Brunnengasse' }),
  R(5, 197, 150, 156, { name: 'Brunnengasse' }),
];

// Baublöcke im Kreuzmarkt-Viertel (zwischen den Straßen)
const MARKET_BLOCKS = [
  // Reihe A
  R(-197, -152, 6, 42), R(-147, -92, 6, 42), R(-86, -24, 6, 42), R(24, 86, 6, 42), R(92, 147, 6, 42), R(152, 197, 6, 42),
  // Reihe B
  R(-197, -152, 47, 91), R(-147, -92, 47, 91), R(-86, -24, 47, 72), R(-86, -34, 72, 91),
  R(24, 86, 47, 72), R(34, 86, 72, 91), R(92, 147, 47, 91), R(152, 197, 47, 91),
  // Reihe C
  R(-197, -152, 101, 150), R(-147, -92, 101, 150), R(-86, -34, 101, 120), R(-86, -34, 120, 150), R(-33, -5, 145, 150),
  R(34, 86, 101, 120), R(5, 86, 120, 150), R(92, 147, 101, 150), R(152, 197, 101, 150),
  // Reihe D
  R(-197, -152, 156, 197), R(-147, -92, 156, 197), R(-86, -5, 156, 197), R(5, 86, 156, 197), R(92, 147, 156, 197), R(152, 197, 156, 197),
];

// Riesenbäume der Waldsiedlung (mit Plattformen und Hängebrücken)
export const GIANT_TREES = [
  { id: 'riese', name: 'Alter Riese', x: -100, z: -60, r: 10, h: 44 },
  { id: 'zwilling', name: 'Zwillingsbaum', x: 96, z: -20, r: 9, h: 38 },
  { id: 'weise', name: 'Weise Eiche', x: 92, z: -100, r: 10, h: 42 },
  { id: 'wurzel', name: 'Wurzelthron', x: -48, z: -142, r: 10, h: 46 },
];

// Gesichter der vier Gildengründer im Felsen
export const FACES = [
  { id: 'seefahrer', name: 'Der Seefahrer', x: -66 },
  { id: 'schatten', name: 'Die Schattenläuferin', x: -22 },
  { id: 'erfinder', name: 'Der Erfinder', x: 22 },
  { id: 'waechter', name: 'Die Wächterin', x: 66 },
].map((f) => ({ ...f, z: -174, y: 30, h: 21 }));

// Felsensteig: Holzturm an der Felsterrasse und Felsband vor den Gesichtern
export const STEIG = { tower: R(-172, -164, -164, -156), ledge: { x0: -172, x1: -40, z0: -172, z1: -167, y: 33.15 } };

// Aufzüge (Extraktion): freie Plätze
export const EXTRACTS = [
  { id: 'felsen', name: 'Aufzug Felsterrasse', x: -128, z: -146, yaw: Math.PI },
  { id: 'wasser', name: 'Aufzug Wassertürme', x: 168, z: -100, yaw: -Math.PI / 2 },
  { id: 'gerber', name: 'Aufzug Gerberhof', x: -172, z: 176, yaw: 0 },
  { id: 'hafentor', name: 'Aufzug Hafentor', x: 120, z: 180, yaw: 0 },
  { id: 'weiss', name: 'Aufzug Weiße Stadt', x: -272, z: 176, yaw: 0 },
  { id: 'turm', name: 'Aufzug Turmstadt', x: 272, z: -70, yaw: -Math.PI / 2 },
  { id: 'kai', name: 'Aufzug Kai', x: -240, z: 222, yaw: 0 },
];

export const POIS = [
  { id: 'markt', name: 'Kreuzmarkt', x: 0, z: 96, tier: 3 },
  { id: 'dom', name: 'Dom & Erfinderturm', x: 0, z: 42, tier: 3 },
  { id: 'rathaus', name: 'Altes Rathaus', x: -19, z: 133, tier: 2 },
  { id: 'gerber', name: 'Gerberviertel', x: -120, z: 70, tier: 1 },
  { id: 'zunft', name: 'Zunftviertel', x: 120, z: 70, tier: 2 },
  { id: 'brunnen', name: 'Brunnengassen', x: -60, z: 176, tier: 1 },
  { id: 'hafentor', name: 'Hafentor', x: 0, z: 186, tier: 1 },
  { id: 'unten', name: 'Untere Terrasse', x: 30, z: -22, tier: 1 },
  { id: 'riese', name: 'Alter Riese', x: -100, z: -60, tier: 2 },
  { id: 'eiche', name: 'Weise Eiche', x: 92, z: -100, tier: 2 },
  { id: 'wurzel', name: 'Wurzelthron', x: -48, z: -142, tier: 2 },
  { id: 'felsen', name: 'Gründerfelsen', x: 0, z: -170, tier: 3 },
  { id: 'weiss', name: 'Weiße Stadt', x: -252, z: 96, tier: 2 },
  { id: 'weissturm', name: 'Weißer Turm', x: -252.5, z: 37, tier: 3 },
  { id: 'turm', name: 'Turmstadt', x: 252, z: 96, tier: 2 },
  { id: 'energie', name: 'Energieturm', x: 252.5, z: 37, tier: 3 },
  { id: 'hafen', name: 'Hafenviertel', x: 120, z: 222, tier: 1 },
  { id: 'galeone', name: 'Gestrandete Galeone', x: 70, z: 274, tier: 3 },
  { id: 'leuchtturm', name: 'Leuchtturm', x: -150, z: 259, tier: 2 },
];

// auf Straßen und Promenaden (nicht auf Dächern)
export const PLAYER_SPAWNS = [
  [-180, 96], [180, 96], [0, 190], [-149.5, 175], [149.5, 175], [-185, 2], [185, 2],
  [-170, -43.5], [170, -83.5], [-160, -123.5], [160, -123.5],
  [-285, 96], [285, 96], [-285, -97], [285, -97], [-200, 236], [200, 236],
];

// Viertel vor der Mauer. Sonderbauten stehen auf den Kreuzungen der Hauptstraßen.
export const WHITE_TOWER = R(-258.5, -246.5, 31, 43);
export const ENERGY_TOWER = { x: 252.5, z: 37, r: 9 };
const ROWS = [[-160, -100], [-94, -34], [-28, 34], [40, 91], [101, 150], [156, 205]];
const crossStreets = (x0, x1, side) => [
  R(x0, x1, 91, 101, { name: side === 'w' ? 'Weststraße' : 'Oststraße', main: true }),
  R(x0, x1, -100, -94, {}), R(x0, x1, -34, -28, {}), R(x0, x1, 34, 40, {}), R(x0, x1, 150, 156, {}),
];
const QUAY_LANES = [[-210, -205], [-150, -145], [-90, -85], [85, 90], [145, 150], [205, 210]];
export const OUTER_STREETS = [
  R(-211, -205, -160, 211, { name: 'Ringstraße West', main: true }),
  R(205, 211, -160, 211, { name: 'Ringstraße Ost', main: true }),
  R(-255, -250, -160, 205, { name: 'Lange Gasse', main: true }),
  R(250, 255, -160, 205, { name: 'Hochstraße', main: true }),
  ...crossStreets(-292, -205, 'w'),
  ...crossStreets(205, 292, 'o'),
  R(-292, 292, 205, 211, { name: 'Kaistraße', main: true }),
  R(-292, 292, 232, CITY.QUAY, { name: 'Kai', main: true }),
  R(-5, 5, 205, 232, { name: 'Hafenstraße', main: true }),
  ...QUAY_LANES.map(([a, c]) => R(a, c, 211, 232, {})),
];

// ------------------------------------------------------------------ Generator
const MARKET_CFG = { block: 30, alley: [2, 3.2], alleyWide: [3, 4], deep: 14, minW: 6.5, maxW: 11, minD: 6.5 };
const OUTER_DISTRICTS = [
  {
    id: 'weiss',
    cfg: { block: 44, alley: [3, 4.5], alleyWide: [3.5, 5], deep: 15, minW: 9, maxW: 14, minD: 8 },
    xs: [[-292, -255], [-250, -211]], zs: ROWS,
  },
  {
    id: 'turm',
    cfg: { block: 30, alley: [4, 6], alleyWide: [5, 7], deep: 16, minW: 11, maxW: 16, minD: 10 },
    xs: [[211, 250], [255, 292]], zs: ROWS,
  },
  {
    id: 'hafen',
    cfg: { block: 40, alley: [3, 4.5], alleyWide: [4, 6], deep: 13, minW: 9, maxW: 16, minD: 8 },
    xs: [[-292, -210], [-205, -150], [-145, -90], [-85, -5], [5, 85], [90, 145], [150, 205], [210, 292]], zs: [[211, 232]],
  },
];
const WALD_CFG = { block: 38, alley: [2.5, 3.5], alleyWide: [3, 4.5], deep: 16, minW: 10, maxW: 15, minD: 9 };

function bsp(r, rnd, cfg, out, depth = 0) {
  const w = r.x1 - r.x0, d = r.z1 - r.z0;
  if (w <= cfg.block && d <= cfg.block) { out.blocks.push({ ...r }); return; }
  const alongX = w >= d;
  const len = alongX ? w : d;
  const g = depth === 0 ? cfg.alleyWide : cfg.alley;
  const gap = snap(g[0] + rnd() * (g[1] - g[0]));
  const cut = snap((alongX ? r.x0 : r.z0) + len * (0.36 + rnd() * 0.28));
  let a, b, alley;
  if (alongX) {
    a = R(r.x0, cut - gap / 2, r.z0, r.z1); b = R(cut + gap / 2, r.x1, r.z0, r.z1);
    alley = R(cut - gap / 2, cut + gap / 2, r.z0, r.z1);
  } else {
    a = R(r.x0, r.x1, r.z0, cut - gap / 2); b = R(r.x0, r.x1, cut + gap / 2, r.z1);
    alley = R(r.x0, r.x1, cut - gap / 2, cut + gap / 2);
  }
  out.alleys.push(alley);
  bsp(a, rnd, cfg, out, depth + 1);
  bsp(b, rnd, cfg, out, depth + 1);
}

// Teilt einen Block in Häuserreihen und Grundstücke
function blockLots(b, rnd, cfg, out) {
  const w = b.x1 - b.x0, d = b.z1 - b.z0;
  if (w < cfg.minD || d < cfg.minD) { out.yards.push({ ...b }); return; }
  const alongX = w >= d; // Grundstücke reihen sich entlang der langen Seite
  const L = alongX ? w : d, S = alongX ? d : w;
  const s0 = alongX ? b.z0 : b.x0;
  const bands = [];
  if (S > cfg.deep * 2 + 4) {
    // Innenhof zwischen zwei Reihen
    const da = snap(cfg.deep * (0.75 + rnd() * 0.2)), db = snap(cfg.deep * (0.75 + rnd() * 0.2));
    bands.push([s0, s0 + da], [s0 + S - db, s0 + S]);
    const yard = alongX ? R(b.x0, b.x1, s0 + da, s0 + S - db) : R(s0 + da, s0 + S - db, b.z0, b.z1);
    out.yards.push({ ...yard, court: true });
  } else if (S > cfg.deep) {
    const m = snap(S * (0.42 + rnd() * 0.16));
    bands.push([s0, s0 + m], [s0 + m, s0 + S]);
  } else bands.push([s0, s0 + S]);

  const l0 = alongX ? b.x0 : b.z0;
  for (const [a0, a1] of bands) {
    let p = l0;
    while (p < l0 + L - 0.01) {
      let wid = snap(cfg.minW + rnd() * (cfg.maxW - cfg.minW));
      if (l0 + L - (p + wid) < cfg.minW) wid = l0 + L - p;
      const lot = alongX ? R(p, p + wid, a0, a1) : R(a0, a1, p, p + wid);
      out.lots.push(lot);
      p += wid;
    }
  }
}

export function generateLayout() {
  const rnd = mulberry32(1871);
  const out = { blocks: [], alleys: [], lots: [], yards: [] };
  const reserved = []; // Flächen ohne Häuser

  // Freiflächen für Aufzüge, Riesenbäume, Felsensteig
  for (const e of EXTRACTS) reserved.push(R(e.x - 7, e.x + 7, e.z - 7, e.z + 7, { kind: 'extract' }));
  for (const t of GIANT_TREES) reserved.push(R(t.x - t.r - 1, t.x + t.r + 1, t.z - t.r - 1, t.z + t.r + 1, { kind: 'tree' }));
  reserved.push(R(STEIG.tower.x0 - 3, STEIG.tower.x1 + 3, STEIG.tower.z0 - 2, STEIG.tower.z1 + 4, { kind: 'steig' }));
  // Brunnenplatz in den Südgassen und kleiner Platz im Zunftviertel
  const squares = [
    R(-70, -44, 164, 186, { name: 'Brunnenplatz' }), R(104, 128, 56, 80, { name: 'Zunftplatz' }), R(-132, -108, 110, 134, { name: 'Gerberhof' }),
    R(-268, -237, 21, 53, { name: 'Turmplatz' }), R(237, 268, 21, 53, { name: 'Energieplatz' }), R(-30, 30, 211, 232, { name: 'Fischmarkt' }),
  ];
  for (const s of squares) reserved.push({ ...s, kind: 'square' });

  // --- Kreuzmarkt-Viertel
  const market = { blocks: [], alleys: [] };
  for (const b of MARKET_BLOCKS) bsp(b, rnd, MARKET_CFG, market);
  for (const b of market.blocks) blockLots(b, rnd, MARKET_CFG, out);
  const marketLots = out.lots.length;
  out.alleys.push(...market.alleys);

  // --- Waldsiedlung: Terrassenbänder, geteilt durch Gassen an den Treppen
  const waldAlleys = [];
  TERRACES.forEach((t, k) => {
    const zs = t.z1; // Südrand
    const north = k < 3 ? t.z0 + 2 : t.z0 + 4; // Stützmauer bzw. Felsfuß
    const south = k === 0 ? zs : zs - 6.5; // Promenade an der eigenen Stützmauer
    // Gassen an allen Treppen, die diese Terrasse berühren
    const xs = [];
    for (const s of STAIRS) {
      if (s.z === t.z1 || s.z === t.z0) for (const [cx, w] of s.list) xs.push([cx, Math.max(w, 4)]);
    }
    xs.sort((a, b) => a[0] - b[0]);
    // zusätzliche Gassen, damit Blöcke nicht zu breit werden
    const lanes = [];
    let prev = -197;
    for (const [cx, w] of xs) {
      if (cx - w / 2 - prev > 52) lanes.push([snap((prev + cx - w / 2) / 2), 3]);
      lanes.push([cx, w]);
      prev = cx + w / 2;
    }
    if (197 - prev > 52) lanes.push([snap((prev + 197) / 2), 3]);
    // doppelte/überlappende Gassen zusammenfassen
    lanes.sort((a, b) => a[0] - b[0]);
    const merged = [];
    for (const l of lanes) {
      const last = merged[merged.length - 1];
      if (last && l[0] - l[1] / 2 < last[0] + last[1] / 2 + 6) {
        const x0 = Math.min(last[0] - last[1] / 2, l[0] - l[1] / 2), x1 = Math.max(last[0] + last[1] / 2, l[0] + l[1] / 2);
        last[0] = (x0 + x1) / 2; last[1] = x1 - x0;
      } else merged.push([...l]);
    }
    let x = -197;
    for (const [cx, w] of [...merged, [197 + 1.5, 3]]) {
      const x0 = x, x1 = Math.min(197, cx - w / 2);
      if (x1 - x0 > 6) {
        const block = R(x0, x1, north, south);
        const sub = { blocks: [], alleys: [] };
        bsp(block, rnd, WALD_CFG, sub, 1);
        for (const bb of sub.blocks) blockLots(bb, rnd, WALD_CFG, out);
        waldAlleys.push(...sub.alleys);
      }
      if (cx < 197) waldAlleys.push(R(cx - w / 2, cx + w / 2, north, south, { lane: true }));
      x = cx + w / 2;
    }
  });
  out.alleys.push(...waldAlleys);

  // --- Grundstücke bewerten
  const lots = [];
  out.lots.forEach((l, i) => {
    const wald = i >= marketLots;
    const w = l.x1 - l.x0, d = l.z1 - l.z0;
    const cfg = wald ? WALD_CFG : MARKET_CFG;
    if (reserved.some((r) => overlap(l, r))) {
      if (!reserved.some((r) => r.kind === 'tree' && overlap(l, r))) out.yards.push({ ...l });
      return;
    }
    if (Math.min(w, d) < 4.6 || (Math.min(w, d) < cfg.minW - 0.5 && Math.max(w, d) < 7)) { out.yards.push({ ...l, passage: true }); return; }
    const cx = (l.x0 + l.x1) / 2, cz = (l.z0 + l.z1) / 2;
    const lot = { ...l, cx, cz, w, d, id: lots.length, district: wald ? 'wald' : 'markt', level: levelAt(cz) };
    const r = rnd();
    if (wald) {
      const t = TERRACES.find((tt) => cz <= tt.z1 && cz > tt.z0) || TERRACES[0];
      lot.terrace = t.id;
      if (r < 0.14) { out.yards.push({ ...l, garden: true }); return; }
      if (r < 0.2) lot.kind = 'ruin';
      else lot.kind = 'house';
      lot.style = rnd() < 0.24 ? 'turm' : 'schwarzwald';
      lot.floors = lot.style === 'turm' ? 3 + Math.floor(rnd() * 3) : 2 + Math.floor(rnd() * 2);
      lot.tier = t.id === 't3' || t.id === 't2' ? 2 : 1;
      lot.poi = t.id;
    } else {
      const dPlaza = Math.hypot(Math.max(PLAZA.x0 - cx, 0, cx - PLAZA.x1), Math.max(PLAZA.z0 - cz, 0, cz - PLAZA.z1));
      if (r < 0.09 && dPlaza > 4) { out.yards.push({ ...l, garden: true }); return; }
      lot.kind = r < 0.16 && dPlaza > 4 ? 'ruin' : 'house';
      const sr = rnd();
      lot.style = dPlaza < 4 ? (sr < 0.7 ? 'stein' : 'fachwerk') : sr < 0.58 ? 'fachwerk' : sr < 0.88 ? 'stein' : 'backstein';
      let f = dPlaza < 4 ? 4 + Math.floor(rnd() * 2) : dPlaza < 60 ? 3 + Math.floor(rnd() * 2) : 2 + Math.floor(rnd() * 3);
      if (rnd() < 0.035 && w >= 6 && d >= 6) { f = 6 + Math.floor(rnd() * 2); lot.style = 'stein'; lot.tall = true; }
      lot.floors = f;
      lot.tier = dPlaza < 40 ? 2 : 1;
      lot.poi = cx < -60 ? 'gerber' : cx > 60 ? 'zunft' : cz > 150 ? 'brunnen' : 'markt';
    }
    lots.push(lot);
  });

  // Ränder der Hauptstraßen und des Platzes: Läden im Erdgeschoss
  const mains = [...STREETS.filter((s) => s.main), PLAZA];
  for (const l of lots) l.shop = l.district === 'markt' && mains.some((s) => overlap(l, s, 0.6));

  // --- Viertel vor der Mauer. Eigener Zufall, damit Altstadt und Waldsiedlung gleich bleiben.
  const outerRnd = mulberry32(1872);
  const outerMains = OUTER_STREETS.filter((s) => s.main);
  for (const dist of OUTER_DISTRICTS) {
    const parts = { lots: [], yards: out.yards };
    for (const [x0, x1] of dist.xs) {
      for (const [z0, z1] of dist.zs) {
        const sub = { blocks: [], alleys: [] };
        bsp(R(x0, x1, z0, z1), outerRnd, dist.cfg, sub);
        for (const bb of sub.blocks) blockLots(bb, outerRnd, dist.cfg, parts);
        out.alleys.push(...sub.alleys);
      }
    }
    for (const l of parts.lots) {
      const w = l.x1 - l.x0, d = l.z1 - l.z0;
      if (reserved.some((r) => overlap(l, r))) { out.yards.push({ ...l }); continue; }
      if (Math.min(w, d) < 5) { out.yards.push({ ...l, passage: true }); continue; }
      const cx = (l.x0 + l.x1) / 2, cz = (l.z0 + l.z1) / 2;
      const lot = { ...l, cx, cz, w, d, id: lots.length, district: dist.id, level: CITY.GROUND, kind: 'house', poi: dist.id };
      const r = outerRnd();
      if (dist.id === 'weiss') {
        if (r < 0.2) { out.yards.push({ ...l, garden: true }); continue; }
        if (r < 0.26) lot.kind = 'ruin';
        lot.style = 'weiss';
        // Musterabschnitt mit überarbeiteten Häusern (Ablauf: docs/hausstandard.md)
        lot.pilot = cz < -94;
        lot.floors = 2 + Math.floor(outerRnd() * 2);
        lot.tier = 2;
        lot.shop = outerMains.some((s) => overlap(l, s, 0.6)) && outerRnd() < 0.5;
      } else if (dist.id === 'turm') {
        if (r < 0.08) { out.yards.push({ ...l }); continue; }
        if (r < 0.18) lot.kind = 'ruin';
        lot.style = 'hochhaus';
        lot.floors = outerRnd() < 0.2 ? 11 + Math.floor(outerRnd() * 4) : 5 + Math.floor(outerRnd() * 5);
        lot.tier = Math.hypot(cx - ENERGY_TOWER.x, cz - ENERGY_TOWER.z) < 70 ? 3 : 2;
      } else {
        if (r < 0.07) { out.yards.push({ ...l }); continue; }
        if (r < 0.15) lot.kind = 'ruin';
        lot.style = 'speicher';
        lot.floors = 2 + Math.floor(outerRnd() * 3);
        lot.tier = 1;
      }
      lots.push(lot);
    }
  }

  return {
    lots,
    yards: out.yards,
    alleys: out.alleys,
    squares,
    reserved,
  };
}

// Belegungsraster (0,5 m) für "Was liegt neben dieser Wand?"
export class Occupancy {
  constructor(half = CITY.OUTER + 2, cell = 0.5) {
    this.half = half;
    this.cell = cell;
    this.n = Math.ceil((half * 2) / cell);
    this.grid = new Int32Array(this.n * this.n).fill(-1);
  }

  mark(r, id) {
    const c = this.cell, h = this.half;
    const i0 = Math.max(0, Math.floor((r.x0 + h) / c + 0.01)), i1 = Math.min(this.n - 1, Math.ceil((r.x1 + h) / c - 0.01) - 1);
    const j0 = Math.max(0, Math.floor((r.z0 + h) / c + 0.01)), j1 = Math.min(this.n - 1, Math.ceil((r.z1 + h) / c - 0.01) - 1);
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) this.grid[j * this.n + i] = id;
  }

  at(x, z) {
    const i = Math.floor((x + this.half) / this.cell), j = Math.floor((z + this.half) / this.cell);
    if (i < 0 || j < 0 || i >= this.n || j >= this.n) return -2;
    return this.grid[j * this.n + i];
  }
}

export { overlap, inside };
