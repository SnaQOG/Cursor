// Waffen-Laufzeitlogik und einfache Waffenmodelle.
import * as THREE from 'three';
import { WEAPONS, ITEMS } from './items.js';
import { assets, Assets } from './assets.js';

export class Gun {
  constructor(id, mag = null) {
    this.id = id;
    this.def = WEAPONS[id];
    this.name = ITEMS[id].name;
    this.mag = mag ?? this.def.mag;
    this.cooldown = 0;
    this.reloadT = 0;
    this.reloading = false;
    this.burstLeft = 0;
    this.burstT = 0;
    this.bloom = 0;
    this.triggerLatch = false;
  }

  get interval() { return 60 / this.def.rpm; }

  startReload(inv) {
    if (this.reloading || this.mag >= this.def.mag) return false;
    if (inv.count(this.def.ammo) <= 0) return false;
    this.reloading = true;
    this.reloadT = this.def.perShell ? this.def.reload : this.def.reload;
    this.burstLeft = 0;
    return true;
  }

  cancelReload() {
    this.reloading = false;
    this.reloadT = 0;
  }

  // Liefert true, wenn eine Patrone nachgeladen wurde (für Sound)
  updateReload(dt, inv) {
    if (!this.reloading) return false;
    this.reloadT -= dt;
    if (this.reloadT > 0) return false;
    if (this.def.perShell) {
      if (inv.remove(this.def.ammo, 1) > 0) this.mag++;
      if (this.mag >= this.def.mag || inv.count(this.def.ammo) <= 0) this.reloading = false;
      else this.reloadT = this.def.reload;
      return true;
    }
    const need = this.def.mag - this.mag;
    this.mag += inv.remove(this.def.ammo, need);
    this.reloading = false;
    return true;
  }

  // Abzug auswerten. Gibt Anzahl der Schüsse in diesem Frame zurück (0/1).
  trigger(dt, held, pressed) {
    this.cooldown -= dt;
    this.bloom = Math.max(0, this.bloom - dt * 3.5);
    const d = this.def;
    if (d.mode === 'burst') {
      if (this.burstLeft > 0) {
        this.burstT -= dt;
        if (this.burstT <= 0 && this.mag > 0) {
          this.burstLeft--;
          this.burstT = this.interval;
          if (this.burstLeft === 0) this.cooldown = d.burstDelay;
          return 1;
        }
        if (this.mag <= 0) this.burstLeft = 0;
        return 0;
      }
      if (pressed && this.cooldown <= 0 && this.mag > 0) {
        this.burstLeft = d.burst - 1;
        this.burstT = this.interval;
        if (this.burstLeft === 0) this.cooldown = d.burstDelay;
        return 1;
      }
      return 0;
    }
    if (this.cooldown > 0 || this.mag <= 0) return 0;
    if (d.mode === 'auto' ? held : pressed) {
      this.cooldown = this.interval;
      return 1;
    }
    return 0;
  }
}

// Waffenmodell: aus Blender, sonst kleine Boxen (Lauf zeigt nach -Z)
export function makeGunModel(id) {
  if (assets.has(id)) {
    const g = assets.clone(id);
    const n = Assets.nodes(g, id);
    g.userData.muzzle = n.Muzzle ? n.Muzzle.position.clone() : new THREE.Vector3(0, 0.05, -0.5);
    g.traverse((o) => { if (o.isMesh) o.castShadow = true; });
    return g;
  }
  const def = WEAPONS[id];
  const g = new THREE.Group();
  const dark = new THREE.MeshStandardMaterial({ color: 0x2b2d2f, roughness: 0.5, metalness: 0.6 });
  const accentCol = { 0: 0x8d9296, 1: 0x5aa860, 2: 0x3f8fe0, 3: 0xa060e0, 4: 0xe8a93a }[ITEMS[id].rarity];
  const accent = new THREE.MeshStandardMaterial({ color: accentCol, roughness: 0.6, metalness: 0.2 });
  const wood = new THREE.MeshStandardMaterial({ color: 0x7a5534, roughness: 0.8 });
  const box = (w, h, d, x, y, z, m = dark) => {
    const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m);
    b.position.set(x, y, z);
    b.castShadow = true;
    g.add(b);
    return b;
  };
  let muzzleZ = -0.5;
  switch (def.model) {
    case 'pistol':
    case 'revolver':
      box(0.05, 0.08, 0.28, 0, 0.02, -0.1);
      box(0.045, 0.13, 0.06, 0, -0.07, 0.01, accent);
      if (def.model === 'revolver') box(0.07, 0.07, 0.07, 0, 0.02, -0.02, accent);
      muzzleZ = -0.26;
      break;
    case 'smg':
      box(0.06, 0.09, 0.45, 0, 0, -0.15);
      box(0.04, 0.16, 0.05, 0, -0.1, -0.12, accent);
      box(0.04, 0.06, 0.2, 0, -0.01, 0.15);
      muzzleZ = -0.4;
      break;
    case 'shotgun':
      box(0.07, 0.08, 0.7, 0, 0.02, -0.25);
      box(0.07, 0.06, 0.28, 0, -0.04, -0.3, wood);
      box(0.06, 0.1, 0.25, 0, -0.03, 0.2, wood);
      muzzleZ = -0.62;
      break;
    case 'sniper':
      box(0.05, 0.07, 0.95, 0, 0.02, -0.35);
      box(0.06, 0.06, 0.26, 0, 0.1, -0.12, accent);
      box(0.07, 0.11, 0.35, 0, -0.03, 0.22, wood);
      muzzleZ = -0.84;
      break;
    case 'lmg':
      box(0.09, 0.12, 0.8, 0, 0.01, -0.25);
      box(0.12, 0.12, 0.14, 0.02, -0.1, -0.12, accent);
      box(0.07, 0.1, 0.25, 0, -0.01, 0.25);
      muzzleZ = -0.7;
      break;
    default:
      box(0.06, 0.09, 0.62, 0, 0.01, -0.2);
      box(0.045, 0.16, 0.06, 0, -0.1, -0.15, accent);
      box(0.06, 0.1, 0.26, 0, -0.02, 0.2);
      if (def.model === 'dmr') box(0.05, 0.05, 0.2, 0, 0.09, -0.12, accent);
      muzzleZ = -0.54;
  }
  g.userData.muzzle = new THREE.Vector3(0, 0.02, muzzleZ);
  return g;
}
