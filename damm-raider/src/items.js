// Gegenstands-Datenbank. value = Verkaufswert in Credits pro Stück.

export const RARITY = [
  { name: 'Gewöhnlich', color: '#9da3a6' },
  { name: 'Ungewöhnlich', color: '#5dbb63' },
  { name: 'Selten', color: '#3f95f2' },
  { name: 'Episch', color: '#b566f2' },
  { name: 'Legendär', color: '#f2b53f' },
];

export const CATEGORY = {
  weapon: 'Waffe',
  ammo: 'Munition',
  heal: 'Heilung',
  shieldcell: 'Schildladung',
  grenade: 'Wurfobjekt',
  shield: 'Schild',
  material: 'Material',
  valuable: 'Wertsache',
  arc: 'ARC-Teil',
  key: 'Schlüssel',
};

// Waffenwerte: dmg pro Kugel, rpm, mag, reload (s), spread (Grad), range (m ohne Abfall),
// pen = Panzerdurchschlag 0..1, mode: semi/auto/burst/pump, zoom = ADS-FOV-Faktor
export const WEAPONS = {
  w_nadel: { ammo: 'ammo_light', dmg: 15, rpm: 320, mag: 12, reload: 1.4, spread: 1.1, range: 30, pen: 0.1, mode: 'semi', sound: 'pistol', recoil: 1.2, zoom: 0.8, noise: 55, model: 'pistol' },
  w_stichler: { ammo: 'ammo_light', dmg: 10, rpm: 780, mag: 30, reload: 2.0, spread: 2.4, range: 25, pen: 0.1, mode: 'auto', sound: 'smg', recoil: 0.7, zoom: 0.8, noise: 60, model: 'smg' },
  w_klapper: { ammo: 'ammo_medium', dmg: 16, rpm: 560, mag: 25, reload: 2.3, spread: 1.6, range: 45, pen: 0.25, mode: 'auto', sound: 'rifle', recoil: 1.0, zoom: 0.72, noise: 80, model: 'rifle' },
  w_kessel: { ammo: 'ammo_medium', dmg: 32, rpm: 260, mag: 12, reload: 2.2, spread: 0.4, range: 70, pen: 0.35, mode: 'semi', sound: 'marksman', recoil: 2.0, zoom: 0.5, noise: 85, model: 'dmr' },
  w_stier: { ammo: 'ammo_shell', dmg: 12, pellets: 9, rpm: 75, mag: 6, reload: 0.55, perShell: true, spread: 5.5, range: 12, pen: 0.1, mode: 'pump', sound: 'shotgun', recoil: 3.2, zoom: 0.85, noise: 90, model: 'shotgun' },
  w_amboss: { ammo: 'ammo_heavy', dmg: 52, rpm: 150, mag: 6, reload: 2.6, spread: 0.7, range: 45, pen: 0.55, mode: 'semi', sound: 'heavy', recoil: 3.0, zoom: 0.72, noise: 95, model: 'revolver' },
  w_donner: { ammo: 'ammo_medium', dmg: 18, rpm: 620, mag: 60, reload: 4.2, spread: 2.0, range: 50, pen: 0.3, mode: 'auto', sound: 'lmg', recoil: 0.9, zoom: 0.72, noise: 95, model: 'lmg', slow: 0.9 },
  w_eisen: { ammo: 'ammo_heavy', dmg: 105, rpm: 55, mag: 1, reload: 1.5, spread: 0.15, range: 120, pen: 0.85, mode: 'semi', sound: 'heavy', recoil: 4.5, zoom: 0.35, noise: 110, model: 'sniper' },
  w_sturmwind: { ammo: 'ammo_heavy', dmg: 36, rpm: 900, burst: 3, burstDelay: 0.42, mag: 18, reload: 2.6, spread: 0.6, range: 80, pen: 0.6, mode: 'burst', sound: 'rifle', recoil: 1.6, zoom: 0.55, noise: 90, model: 'rifle' },
};

export const ITEMS = {
  // Waffen
  w_nadel: { name: 'Nadel', cat: 'weapon', rarity: 0, value: 280, stack: 1, desc: 'Handliche Pistole. Leichte Munition.' },
  w_stichler: { name: 'Stichler', cat: 'weapon', rarity: 1, value: 850, stack: 1, desc: 'Maschinenpistole mit hoher Feuerrate. Leichte Munition.' },
  w_klapper: { name: 'Klapper', cat: 'weapon', rarity: 1, value: 1100, stack: 1, desc: 'Robustes Sturmgewehr. Mittlere Munition.' },
  w_kessel: { name: 'Kessel', cat: 'weapon', rarity: 2, value: 1500, stack: 1, desc: 'Halbautomatisches Präzisionsgewehr. Mittlere Munition.' },
  w_stier: { name: 'Stier', cat: 'weapon', rarity: 2, value: 1400, stack: 1, desc: 'Pumpflinte. Verheerend auf kurze Distanz.' },
  w_amboss: { name: 'Amboss', cat: 'weapon', rarity: 2, value: 1650, stack: 1, desc: 'Schwerer Revolver, durchschlägt Panzerung. Schwere Munition.' },
  w_donner: { name: 'Donner', cat: 'weapon', rarity: 3, value: 3200, stack: 1, desc: 'Leichtes MG mit großem Magazin. Macht dich langsamer.' },
  w_eisen: { name: 'Eisen', cat: 'weapon', rarity: 3, value: 2900, stack: 1, desc: 'Einzellader-Gewehr. Ein Schuss, ein Treffer – knackt ARC-Panzer.' },
  w_sturmwind: { name: 'Sturmwind', cat: 'weapon', rarity: 4, value: 7200, stack: 1, desc: 'Legendäres Salvengewehr. Schwere Munition, drei Schuss pro Abzug.' },

  // Munition
  ammo_light: { name: 'Leichte Munition', cat: 'ammo', rarity: 0, value: 2, stack: 120, desc: 'Für Nadel und Stichler.' },
  ammo_medium: { name: 'Mittlere Munition', cat: 'ammo', rarity: 0, value: 3, stack: 90, desc: 'Für Klapper, Kessel und Donner.' },
  ammo_heavy: { name: 'Schwere Munition', cat: 'ammo', rarity: 1, value: 7, stack: 40, desc: 'Für Amboss, Eisen und Sturmwind.' },
  ammo_shell: { name: 'Schrotpatronen', cat: 'ammo', rarity: 0, value: 6, stack: 30, desc: 'Für den Stier.' },

  // Verbrauchsgüter
  bandage: { name: 'Verband', cat: 'heal', rarity: 0, value: 35, stack: 5, desc: 'Heilt 30 Lebenspunkte. Anwenden: 1,6 s.', heal: 30, use: 1.6 },
  medkit: { name: 'Sanitätskit', cat: 'heal', rarity: 2, value: 160, stack: 3, desc: 'Heilt 80 Lebenspunkte. Anwenden: 3,5 s.', heal: 80, use: 3.5 },
  shieldcell: { name: 'Schildlader', cat: 'shieldcell', rarity: 1, value: 60, stack: 5, desc: 'Lädt 40 Schildpunkte. Anwenden: 2 s.', charge: 40, use: 2.0 },
  grenade: { name: 'Splittergranate', cat: 'grenade', rarity: 1, value: 90, stack: 3, desc: 'Explodiert nach 2,5 s. Hoher Flächenschaden.' },

  // Schilde
  shield_light: { name: 'Leichter Schild', cat: 'shield', rarity: 1, value: 300, stack: 1, desc: 'Kapazität 40, fängt 40 % Schaden ab.', cap: 40, absorb: 0.4 },
  shield_medium: { name: 'Mittlerer Schild', cat: 'shield', rarity: 2, value: 850, stack: 1, desc: 'Kapazität 70, fängt 55 % Schaden ab.', cap: 70, absorb: 0.55 },
  shield_heavy: { name: 'Schwerer Schild', cat: 'shield', rarity: 3, value: 1900, stack: 1, desc: 'Kapazität 100, fängt 70 % Schaden ab. 8 % langsamer.', cap: 100, absorb: 0.7, slow: 0.92 },

  // Materialien
  scrap: { name: 'Schrott', cat: 'material', rarity: 0, value: 8, stack: 20, desc: 'Verbogenes Metall. Nicht viel wert.' },
  metal: { name: 'Metallteile', cat: 'material', rarity: 0, value: 16, stack: 20, desc: 'Brauchbare Bleche und Streben.' },
  rubber: { name: 'Gummi', cat: 'material', rarity: 0, value: 12, stack: 20, desc: 'Dichtungen und Schläuche.' },
  fabric: { name: 'Stoff', cat: 'material', rarity: 0, value: 10, stack: 20, desc: 'Planen und Kleidungsreste.' },
  plastic: { name: 'Plastikteile', cat: 'material', rarity: 0, value: 12, stack: 20, desc: 'Gehäuse, Deckel, Kanister.' },
  chem: { name: 'Chemikalien', cat: 'material', rarity: 1, value: 32, stack: 10, desc: 'Aus der Wasseraufbereitung.' },
  electronics: { name: 'Elektronik', cat: 'material', rarity: 1, value: 45, stack: 10, desc: 'Platinen und Kabel.' },
  battery: { name: 'Batterie', cat: 'material', rarity: 1, value: 55, stack: 10, desc: 'Hält sogar noch Ladung.' },
  mech: { name: 'Mechanische Teile', cat: 'material', rarity: 1, value: 50, stack: 10, desc: 'Zahnräder, Lager, Federn.' },
  adv_elec: { name: 'Hochleistungselektronik', cat: 'material', rarity: 2, value: 150, stack: 5, desc: 'Vorkriegs-Technik. Begehrt in Speranza.' },
  sensors: { name: 'Sensoren', cat: 'material', rarity: 2, value: 125, stack: 5, desc: 'Optische und thermische Sensoren.' },

  // Wertsachen
  vinyl: { name: 'Schallplatte', cat: 'valuable', rarity: 1, value: 140, stack: 5, desc: 'Musik von Oben. Kaum Kratzer.' },
  vase: { name: 'Keramikvase', cat: 'valuable', rarity: 1, value: 170, stack: 3, desc: 'Überraschend heil geblieben.' },
  watch: { name: 'Alte Taschenuhr', cat: 'valuable', rarity: 2, value: 320, stack: 3, desc: 'Tickt nicht mehr, glänzt aber.' },
  camera: { name: 'Alte Kamera', cat: 'valuable', rarity: 2, value: 390, stack: 3, desc: 'Sammler zahlen gut dafür.' },
  goldring: { name: 'Goldring', cat: 'valuable', rarity: 3, value: 680, stack: 3, desc: 'Echtes Gold.' },
  statuette: { name: 'Bronzefigur', cat: 'valuable', rarity: 3, value: 920, stack: 2, desc: 'Ein Reiter aus Bronze. Schwer.' },
  datacore: { name: 'Vorkriegs-Datenkern', cat: 'valuable', rarity: 4, value: 2600, stack: 1, desc: 'Verschlüsselte Daten aus der Zeit vor den ARC.' },

  // ARC-Teile
  arc_alloy: { name: 'ARC-Legierung', cat: 'arc', rarity: 1, value: 75, stack: 10, desc: 'Panzerplatten-Fragmente.' },
  arc_circuit: { name: 'ARC-Schaltkreis', cat: 'arc', rarity: 2, value: 170, stack: 5, desc: 'Fremdartige Leiterbahnen.' },
  tick_pod: { name: 'Zecken-Kapsel', cat: 'arc', rarity: 0, value: 55, stack: 10, desc: 'Aus einer zerstörten Zecke.' },
  wasp_driver: { name: 'Wespen-Treiber', cat: 'arc', rarity: 1, value: 120, stack: 5, desc: 'Antrieb einer Wespe.' },
  pop_trigger: { name: 'Knaller-Zünder', cat: 'arc', rarity: 1, value: 95, stack: 5, desc: 'Vorsicht, empfindlich.' },
  hornet_driver: { name: 'Hornissen-Treiber', cat: 'arc', rarity: 2, value: 230, stack: 5, desc: 'Starker Antrieb einer Hornisse.' },
  snitch_scanner: { name: 'Späher-Scanner', cat: 'arc', rarity: 2, value: 270, stack: 3, desc: 'Damit rufen Späher Verstärkung.' },
  turret_optic: { name: 'Wächter-Optik', cat: 'arc', rarity: 2, value: 310, stack: 3, desc: 'Präzisionslinse eines Wächters.' },
  rocket_thruster: { name: 'Raketier-Düse', cat: 'arc', rarity: 3, value: 520, stack: 3, desc: 'Schubdüse eines Raketiers.' },
  leaper_pulse: { name: 'Springer-Impulseinheit', cat: 'arc', rarity: 3, value: 640, stack: 2, desc: 'Speichert Sprungenergie.' },
  arc_core: { name: 'ARC-Motorkern', cat: 'arc', rarity: 3, value: 820, stack: 2, desc: 'Das Herz einer großen ARC-Maschine.' },
  bastion_cell: { name: 'Bastion-Zelle', cat: 'arc', rarity: 4, value: 1500, stack: 1, desc: 'Energiezelle einer Bastion. Sehr wertvoll.' },
};

export function item(id) {
  return ITEMS[id];
}

export function itemColor(id) {
  return RARITY[ITEMS[id]?.rarity ?? 0].color;
}

export const HEAL_ORDER = ['bandage', 'medkit'];

// Kleine SVG-Symbole je Kategorie
export const ICONS = {
  weapon: '<path d="M3 13h13l2-2h3v3h-2l-1 1h-4l-1 4h-3l1-4H3z"/>',
  ammo: '<path d="M6 20V9l2-5 2 5v11zM11 20V9l2-5 2 5v11zM16 20V9l2-5 2 5v11z"/>',
  heal: '<path d="M9 3h6v6h6v6h-6v6H9v-6H3V9h6z"/>',
  shieldcell: '<path d="M8 2h8v3h2v17H6V5h2zM11 8l-3 7h3l-1 5 5-8h-3l2-4z"/>',
  grenade: '<path d="M10 3h4v3h-4zM12 7a7 7 0 1 1 0 14a7 7 0 1 1 0-14z"/>',
  shield: '<path d="M12 2l8 3v6c0 5-3.5 9-8 11c-4.5-2-8-6-8-11V5z"/>',
  material: '<path d="M12 2l1.8 3.2 3.6-.6.4 3.6 3.2 1.8-1.8 3.2 1.8 3.2-3.2 1.8-.4 3.6-3.6-.6L12 22l-1.8-3.2-3.6.6-.4-3.6-3.2-1.8L4.8 12 3 8.8 6.2 7l.4-3.6 3.6.6zM12 8a4 4 0 1 0 0 8a4 4 0 1 0 0-8z"/>',
  valuable: '<path d="M6 3h12l4 6-10 12L2 9zM7.5 9h9L12 17z"/>',
  arc: '<path d="M12 2l9 5v10l-9 5-9-5V7zM12 8a4 4 0 1 0 0 8a4 4 0 1 0 0-8z"/>',
};

export function iconSvg(cat, color = 'currentColor') {
  return `<svg viewBox="0 0 24 24" fill="${color}" aria-hidden="true">${ICONS[cat] || ICONS.material}</svg>`;
}
