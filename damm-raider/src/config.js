// Layout-Daten der Karte "Der Damm". Alles in Metern; +X = Osten, -Z = Norden.

export const MAP = {
  HALF: 320, // Terrain reicht von -320 bis +320
  RES: 321, // Stützpunkte pro Achse (2 m Raster)
  BOUND: 292, // spielbarer Bereich
  VALLEY: 2,
  PLATEAU: 26,
  LAKE_FLOOR: 9,
  LAKE_LEVEL: 24,
  VALLEY_WATER: 0.8,
  SWAMP_WATER: 0.95,
  DAM_Z: -80,
  DAM_THICK: 16,
  DAM_HALF_X: 142,
  DAM_TOP: 34,
  LAKE: { cx: 0, cz: -180, hx: 150, hz: 104, r: 40 },
  SWAMP: { x: -165, z: 178, r: 62, blend: 34 },
  RIVER: [[0, -74], [8, -30], [-12, 30], [15, 100], [-8, 170], [12, 240], [0, 330]],
};

// Straßen (nur Optik): Polylinien
export const ROADS = [
  [[-268, 40], [-150, 12], [-60, -8], [4, -24], [60, -26], [150, 18], [268, 60]],
  [[150, 18], [118, 95], [70, 165], [20, 220], [-40, 262]],
  [[-60, -8], [-62, 60], [-60, 110], [-110, 150], [-165, 178]],
  [[60, -26], [60, -36]],
  [[150, 18], [196, -40], [214, -100], [230, -140], [236, -190]],
  [[-150, 12], [-200, -46], [-218, -110], [-226, -160]],
  [[118, 95], [185, 165]],
];

// Planierflächen für Bauplätze. h:'auto' = Höhe am Mittelpunkt übernehmen.
export const PADS = [
  { x: 60, z: -50, r: 40, blend: 20, h: 2.5 }, // Kraftwerk
  { x: -95, z: -58, r: 14, blend: 16, h: 2.5 }, // Kontrollturm
  { x: 158, z: -80, r: 14, blend: 44, h: 34 }, // Widerlager Ost
  { x: -158, z: -80, r: 14, blend: 44, h: 34 }, // Widerlager West
  { x: -150, z: 10, r: 42, blend: 22, h: 3 }, // Forschungsanlage
  { x: 150, z: 20, r: 46, blend: 22, h: 2.5 }, // Wasseraufbereitung
  { x: -60, z: 110, r: 26, blend: 18, h: 3 }, // Umspannwerk
  { x: 70, z: 168, r: 46, blend: 22, h: 3 }, // Wohnblöcke
  { x: 190, z: 160, r: 30, blend: 30, h: 'auto' }, // Schlachtfeld
  { x: 234, z: -165, r: 36, blend: 20, h: 27 }, // Hydrokuppeln
  { x: -228, z: -162, r: 26, blend: 20, h: 27 }, // Pumpstation
  { x: -268, z: 40, r: 12, blend: 14, h: 'auto' }, // Aufzug West
  { x: -40, z: 262, r: 12, blend: 14, h: 'auto' }, // Aufzug Süd
  { x: 268, z: 60, r: 12, blend: 14, h: 'auto' }, // Aufzug Ost
  { x: 5, z: -24, r: 10, blend: 12, h: 3 }, // Brückenköpfe
];

// Orte für Karte und Kompass
export const POIS = [
  { id: 'krone', name: 'Dammkrone', x: 0, z: -80, tier: 2 },
  { id: 'kraftwerk', name: 'Kraftwerk', x: 60, z: -50, tier: 3 },
  { id: 'turm', name: 'Kontrollturm', x: -95, z: -58, tier: 2 },
  { id: 'forschung', name: 'Forschungsanlage', x: -150, z: 10, tier: 3 },
  { id: 'wasser', name: 'Wasseraufbereitung', x: 150, z: 20, tier: 2 },
  { id: 'umspann', name: 'Umspannwerk', x: -60, z: 110, tier: 1 },
  { id: 'wohn', name: 'Blasse Wohnblöcke', x: 70, z: 168, tier: 2 },
  { id: 'sumpf', name: 'Giftsumpf', x: -165, z: 178, tier: 2 },
  { id: 'schlacht', name: 'Altes Schlachtfeld', x: 190, z: 160, tier: 3 },
  { id: 'kuppeln', name: 'Hydrokuppeln', x: 234, z: -165, tier: 3 },
  { id: 'pumpe', name: 'Pumpstation', x: -228, z: -162, tier: 2 },
  { id: 'bruecke', name: 'Alte Brücke', x: 5, z: -24, tier: 1 },
];

export const EXTRACTS = [
  { id: 'west', name: 'Aufzug West', x: -268, z: 40 },
  { id: 'sued', name: 'Aufzug Süd', x: -40, z: 262 },
  { id: 'ost', name: 'Aufzug Ost', x: 268, z: 60 },
  { id: 'damm', name: 'Aufzug Damm', x: 160, z: -88 },
];

export const PLAYER_SPAWNS = [
  [-255, 120], [-240, -30], [255, 130], [250, -30], [120, 255], [-120, 255], [-280, 200], [40, 262],
];

export const RAID_TIME = 20 * 60; // Sekunden

export const CONDITIONS = [
  { id: 'tag', name: 'Klarer Tag', desc: 'Gute Sicht, normale Beute.', lootMul: 1, weight: 4 },
  { id: 'daemmerung', name: 'Dämmerung', desc: 'Tiefe Sonne, lange Schatten.', lootMul: 1.1, weight: 3 },
  { id: 'sturm', name: 'Elektrosturm', desc: 'Regen, Blitze, kaum Sicht. Mehr und bessere Beute.', lootMul: 1.35, weight: 2 },
];
