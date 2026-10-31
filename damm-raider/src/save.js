// Spielstand in localStorage (mit Fallback, falls Speicher blockiert ist).
const KEY = 'damm-raider.save.v1';

export const BACKPACK_LEVELS = [18, 22, 26, 30];
export const STASH_LEVELS = [60, 80, 100, 130];
export const UPGRADE_COST = { backpack: [2500, 6000, 12000], stash: [2000, 5000, 10000] };

export function starterLoadout() {
  const backpack = new Array(BACKPACK_LEVELS[0]).fill(null);
  backpack[0] = { id: 'ammo_medium', qty: 75 };
  backpack[1] = { id: 'ammo_light', qty: 48 };
  backpack[2] = { id: 'bandage', qty: 3 };
  backpack[3] = { id: 'shieldcell', qty: 2 };
  backpack[4] = { id: 'grenade', qty: 1 };
  return {
    weapons: [{ id: 'w_klapper', mag: 25 }, { id: 'w_nadel', mag: 12 }],
    shield: { id: 'shield_light', charge: 40 },
    backpack,
    safe: [null],
  };
}

export function freeLoadout() {
  const guns = [['w_nadel', 'ammo_light', 60], ['w_stichler', 'ammo_light', 90], ['w_klapper', 'ammo_medium', 60], ['w_stier', 'ammo_shell', 18]];
  const [gid, ammo, n] = guns[Math.floor(Math.random() * guns.length)];
  const backpack = new Array(BACKPACK_LEVELS[0]).fill(null);
  backpack[0] = { id: ammo, qty: n };
  backpack[1] = { id: 'bandage', qty: 2 };
  if (Math.random() < 0.5) backpack[2] = { id: 'shieldcell', qty: 1 };
  return {
    weapons: [{ id: gid, mag: null }, null],
    shield: Math.random() < 0.6 ? { id: 'shield_light', charge: 40 } : null,
    backpack,
    safe: [null],
    free: true,
  };
}

export function defaultSave() {
  const stash = new Array(STASH_LEVELS[0]).fill(null);
  stash[0] = { id: 'bandage', qty: 3 };
  stash[1] = { id: 'ammo_medium', qty: 60 };
  stash[2] = { id: 'w_stier', qty: 1 };
  stash[3] = { id: 'ammo_shell', qty: 24 };
  return {
    v: 1,
    credits: 800,
    stash,
    loadout: starterLoadout(),
    upgrades: { backpack: 0, stash: 0 },
    stats: { raids: 0, extracted: 0, died: 0, kills: 0, earned: 0, bestHaul: 0, arcKills: {} },
    settings: { sens: 1, volume: 0.7, quality: 'mittel' },
  };
}

export function loadSave() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return defaultSave();
    const data = JSON.parse(raw);
    const def = defaultSave();
    const merged = { ...def, ...data, stats: { ...def.stats, ...(data.stats || {}) }, settings: { ...def.settings, ...(data.settings || {}) }, upgrades: { ...def.upgrades, ...(data.upgrades || {}) } };
    if (!Array.isArray(merged.stash) || !merged.loadout) return def;
    return merged;
  } catch {
    return defaultSave();
  }
}

export function writeSave(save) {
  try {
    localStorage.setItem(KEY, JSON.stringify(save));
    return true;
  } catch {
    return false;
  }
}

export function wipeSave() {
  try { localStorage.removeItem(KEY); } catch { /* egal */ }
  return defaultSave();
}
