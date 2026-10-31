// Slot-Inventar mit Stapeln. Ein Slot ist null oder {id, qty}.
import { ITEMS } from './items.js';

export class Inventory {
  constructor(size, slots = null) {
    this.slots = new Array(size).fill(null);
    if (slots) {
      for (let i = 0; i < Math.min(size, slots.length); i++) this.slots[i] = slots[i] ? { ...slots[i] } : null;
    }
  }

  get size() { return this.slots.length; }

  resize(n) {
    if (n > this.slots.length) while (this.slots.length < n) this.slots.push(null);
  }

  // Legt Gegenstände ab; gibt die Menge zurück, die nicht gepasst hat.
  add(id, qty = 1) {
    const def = ITEMS[id];
    if (!def) return qty;
    const max = def.stack;
    for (const s of this.slots) {
      if (qty <= 0) break;
      if (s && s.id === id && s.qty < max) {
        const n = Math.min(max - s.qty, qty);
        s.qty += n;
        qty -= n;
      }
    }
    for (let i = 0; i < this.slots.length && qty > 0; i++) {
      if (!this.slots[i]) {
        const n = Math.min(max, qty);
        this.slots[i] = { id, qty: n };
        qty -= n;
      }
    }
    return qty;
  }

  canFit(id, qty = 1) {
    const def = ITEMS[id];
    let room = 0;
    for (const s of this.slots) {
      if (!s) room += def.stack;
      else if (s.id === id) room += def.stack - s.qty;
      if (room >= qty) return true;
    }
    return room >= qty;
  }

  count(id) {
    let n = 0;
    for (const s of this.slots) if (s && s.id === id) n += s.qty;
    return n;
  }

  // Entfernt bis zu qty; gibt tatsächlich entfernte Menge zurück (hinten zuerst).
  remove(id, qty = 1) {
    let removed = 0;
    for (let i = this.slots.length - 1; i >= 0 && removed < qty; i--) {
      const s = this.slots[i];
      if (!s || s.id !== id) continue;
      const n = Math.min(s.qty, qty - removed);
      s.qty -= n;
      removed += n;
      if (s.qty <= 0) this.slots[i] = null;
    }
    return removed;
  }

  takeSlot(i) {
    const s = this.slots[i];
    this.slots[i] = null;
    return s;
  }

  freeSlots() {
    return this.slots.filter((s) => !s).length;
  }

  isEmpty() {
    return this.slots.every((s) => !s);
  }

  value() {
    let v = 0;
    for (const s of this.slots) if (s) v += (ITEMS[s.id]?.value || 0) * s.qty;
    return v;
  }

  clear() {
    this.slots.fill(null);
  }

  // Zusammenfassen und sortieren (Seltenheit, dann Wert)
  sort() {
    const counts = new Map();
    for (const s of this.slots) if (s) counts.set(s.id, (counts.get(s.id) || 0) + s.qty);
    this.clear();
    const ids = [...counts.keys()].sort((a, b) => {
      const A = ITEMS[a], B = ITEMS[b];
      const cats = ['weapon', 'shield', 'ammo', 'heal', 'shieldcell', 'grenade', 'valuable', 'arc', 'material'];
      return cats.indexOf(A.cat) - cats.indexOf(B.cat) || B.rarity - A.rarity || B.value - A.value;
    });
    for (const id of ids) this.add(id, counts.get(id));
  }

  toJSON() {
    return this.slots.map((s) => (s ? { id: s.id, qty: s.qty } : null));
  }
}
