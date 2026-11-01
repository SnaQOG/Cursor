// Beutebehälter, Beutetabellen und ARC-Wracks.
import * as THREE from 'three';
import { Inventory } from './inventory.js';
import { ITEMS } from './items.js';
import { weighted, rand, randInt } from './util.js';
import { assets } from './assets.js';

// Einträge: [id, gewicht, [min,max]]
const TABLES = {
  crate: { rolls: [2, 4], search: 1.4, items: [
    ['scrap', 10, [2, 6]], ['metal', 8, [1, 4]], ['rubber', 6, [1, 4]], ['plastic', 6, [1, 4]], ['fabric', 5, [1, 4]],
    ['ammo_light', 5, [15, 40]], ['ammo_medium', 5, [12, 30]], ['ammo_shell', 2, [4, 10]], ['ammo_heavy', 1.5, [4, 10]],
    ['mech', 3, [1, 2]], ['battery', 2, [1, 2]], ['bandage', 2, [1, 2]], ['vinyl', 1, [1, 1]], ['grenade', 0.8, [1, 1]],
  ] },
  locker: { rolls: [1, 3], search: 1.6, items: [
    ['fabric', 6, [1, 4]], ['bandage', 6, [1, 2]], ['shieldcell', 4, [1, 2]], ['ammo_light', 5, [15, 40]], ['ammo_medium', 5, [12, 30]],
    ['plastic', 4, [1, 3]], ['vinyl', 2, [1, 1]], ['watch', 1, [1, 1]], ['shield_light', 1, [1, 1]], ['grenade', 1.5, [1, 1]], ['camera', 0.7, [1, 1]],
  ] },
  toolbox: { rolls: [2, 4], search: 1.8, items: [
    ['mech', 8, [1, 3]], ['metal', 8, [2, 5]], ['electronics', 6, [1, 3]], ['battery', 5, [1, 2]], ['rubber', 5, [1, 4]],
    ['adv_elec', 1.5, [1, 1]], ['sensors', 1.2, [1, 1]], ['scrap', 5, [2, 6]],
  ] },
  medical: { rolls: [2, 3], search: 1.8, items: [
    ['bandage', 10, [1, 3]], ['medkit', 3, [1, 1]], ['shieldcell', 7, [1, 2]], ['chem', 5, [1, 3]], ['plastic', 3, [1, 2]],
  ] },
  weapon: { rolls: [2, 3], search: 2.4, items: [
    ['w_nadel', 3, [1, 1]], ['w_stichler', 3, [1, 1]], ['w_klapper', 3, [1, 1]], ['w_kessel', 2, [1, 1]], ['w_stier', 2, [1, 1]],
    ['w_amboss', 1.5, [1, 1]], ['w_donner', 0.6, [1, 1]], ['w_eisen', 0.6, [1, 1]], ['w_sturmwind', 0.15, [1, 1]],
    ['ammo_light', 4, [30, 60]], ['ammo_medium', 4, [25, 50]], ['ammo_heavy', 3, [8, 18]], ['ammo_shell', 3, [8, 16]],
    ['grenade', 3, [1, 2]], ['shield_medium', 0.8, [1, 1]], ['shield_heavy', 0.2, [1, 1]],
  ] },
  drawer: { rolls: [1, 3], search: 1.2, items: [
    ['plastic', 5, [1, 3]], ['electronics', 4, [1, 2]], ['vinyl', 3, [1, 2]], ['vase', 2, [1, 1]], ['watch', 2, [1, 1]],
    ['camera', 1.2, [1, 1]], ['goldring', 0.8, [1, 1]], ['statuette', 0.4, [1, 1]], ['datacore', 0.12, [1, 1]], ['battery', 3, [1, 2]],
  ] },
  arcprobe: { rolls: [3, 5], search: 3.0, items: [
    ['arc_alloy', 6, [1, 3]], ['arc_circuit', 4, [1, 2]], ['adv_elec', 4, [1, 2]], ['sensors', 4, [1, 2]], ['battery', 4, [1, 3]],
    ['arc_core', 1.2, [1, 1]], ['datacore', 0.5, [1, 1]], ['shield_medium', 0.8, [1, 1]], ['medkit', 1.5, [1, 1]],
  ] },
  arcwreck: { rolls: [2, 4], search: 2.2, items: [
    ['arc_alloy', 8, [1, 4]], ['arc_circuit', 4, [1, 2]], ['scrap', 6, [2, 6]], ['metal', 5, [2, 4]], ['arc_core', 0.8, [1, 1]], ['sensors', 2, [1, 1]],
  ] },
};

// Wrack-Beute je ARC-Typ
const ENEMY_LOOT = {
  tick: [['tick_pod', 1, [1, 1]], ['arc_alloy', 0.4, [1, 1]]],
  pop: [['pop_trigger', 1, [1, 1]], ['scrap', 0.6, [1, 3]]],
  wasp: [['wasp_driver', 1, [1, 1]], ['arc_alloy', 0.7, [1, 2]], ['ammo_light', 0.3, [10, 20]]],
  hornet: [['hornet_driver', 1, [1, 1]], ['arc_alloy', 0.9, [1, 3]], ['arc_circuit', 0.4, [1, 1]]],
  snitch: [['snitch_scanner', 1, [1, 1]], ['sensors', 0.5, [1, 1]]],
  rocketeer: [['rocket_thruster', 1, [1, 2]], ['arc_alloy', 1, [2, 4]], ['arc_circuit', 0.8, [1, 2]], ['arc_core', 0.35, [1, 1]]],
  leaper: [['leaper_pulse', 1, [1, 1]], ['arc_alloy', 1, [2, 4]], ['arc_core', 0.4, [1, 1]], ['arc_circuit', 0.6, [1, 2]]],
  bastion: [['bastion_cell', 1, [1, 1]], ['arc_core', 1, [1, 2]], ['arc_alloy', 1, [3, 6]], ['arc_circuit', 1, [2, 3]]],
  turret: [['turret_optic', 1, [1, 1]], ['arc_alloy', 0.8, [1, 2]], ['ammo_heavy', 0.4, [5, 10]]],
};

export const CONTAINER_NAMES = {
  crate: 'Kiste', locker: 'Spind', toolbox: 'Werkzeugkiste', medical: 'Medizinschrank', weapon: 'Waffenkiste',
  drawer: 'Schreibtisch', arcprobe: 'ARC-Sonde', arcwreck: 'ARC-Wrack', bag: 'Beutel',
};

function rollTable(table, tier, lootMul) {
  const inv = new Inventory(10);
  const t = TABLES[table];
  const rolls = randInt(t.rolls[0], t.rolls[1]) + (tier >= 3 && Math.random() < 0.5 ? 1 : 0);
  const entries = t.items.map(([id, w, q]) => {
    const r = ITEMS[id].rarity;
    // höhere Stufe / Sturm = seltene Dinge wahrscheinlicher
    const boost = r >= 2 ? Math.pow(tier * 0.55 * lootMul, r - 1) : 1;
    return [[id, q], w * boost];
  });
  for (let i = 0; i < rolls; i++) {
    const [id, q] = weighted(entries);
    inv.add(id, randInt(q[0], q[1]));
  }
  return inv;
}

// Instanzierte Behälter-Optik
const TYPE_LOOK = {
  crate: { size: [1.1, 0.8, 0.8], color: 0x5f6b45, lid: 0xc79a3a },
  locker: { size: [0.8, 1.9, 0.55], color: 0x4e6a70, lid: 0xd9d2c0 },
  toolbox: { size: [0.9, 0.5, 0.45], color: 0xb2422e, lid: 0x2c2c2c },
  medical: { size: [0.9, 1.1, 0.4], color: 0xdad6cc, lid: 0xc23b30 },
  weapon: { size: [1.6, 0.6, 0.7], color: 0x39413a, lid: 0xe0a632 },
  drawer: { size: [1.4, 0.8, 0.7], color: 0x6c5238, lid: 0x9b8c70 },
  arcprobe: { size: [1.2, 1.4, 1.2], color: 0x3a3d40, lid: 0xff6a20 },
  arcwreck: { size: [1.6, 0.6, 1.2], color: 0x2d2e30, lid: 0xff6a20 },
};

export class LootSystem {
  constructor(game) {
    this.game = game;
    this.containers = [];
    this.wrecks = [];
    this.bags = [];
    this.inst = {};
    this.lampInst = {};
  }

  // Einmal beim Start: Positionen bestimmen und Instanzen anlegen
  setup(mapData) {
    const defs = [];
    for (const c of mapData.containers) defs.push({ ...c });
    // Zusätzlich zufällig an Beute-Spots in Gebäuden
    const types = [['drawer', 4], ['locker', 4], ['crate', 3], ['medical', 2], ['toolbox', 2], ['weapon', 1]];
    for (const s of mapData.spots) {
      if (Math.random() > (s.indoor ? (mapData.spotChance ?? 0.62) : 0.5)) continue;
      const type = s.indoor ? weighted(types) : weighted([['crate', 3], ['toolbox', 1], ['weapon', 0.5]]);
      defs.push({ type, x: s.x, y: s.y, z: s.z, tier: s.tier || 1, ry: Math.random() < 0.5 ? 0 : Math.PI / 2 });
    }
    const byType = {};
    for (const d of defs) (byType[d.type] ||= []).push(d);
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), p = new THREE.Vector3(), s = new THREE.Vector3(1, 1, 1);
    const Y = new THREE.Vector3(0, 1, 0);
    for (const [type, list] of Object.entries(byType)) {
      const look = TYPE_LOOK[type];
      const bodies = [];
      let lamps = null;
      let size = look.size;
      let base = new THREE.Color(look.color), lampCol = new THREE.Color(look.lid);
      const skin = mapData.containerSkin?.[type];
      const asset = 'c_' + (skin && assets.has('c_' + skin) ? skin : type);
      if (assets.has(asset)) {
        const bb = assets.bounds(asset);
        size = [bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z];
        base = new THREE.Color(1, 1, 1);
        for (const bm of assets.bakedMeshes(asset)) {
          const isLamp = bm.name.includes('__Lamp');
          const mat = isLamp ? new THREE.MeshBasicMaterial({ color: 0xffffff }) : bm.material;
          const im = new THREE.InstancedMesh(bm.geometry, mat, list.length);
          im.castShadow = !isLamp;
          im.receiveShadow = true;
          if (isLamp) {
            lamps = im;
            lampCol = bm.material.emissive.clone().multiplyScalar(Math.max(1, bm.material.emissiveIntensity * 0.5));
          } else bodies.push(im);
        }
      } else {
        const geo = new THREE.BoxGeometry(...look.size);
        geo.translate(0, look.size[1] / 2, 0);
        const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.7, metalness: type.startsWith('arc') ? 0.6 : 0.1 });
        const im = new THREE.InstancedMesh(geo, mat, list.length);
        im.castShadow = true;
        im.receiveShadow = true;
        bodies.push(im);
        const lampGeo = new THREE.BoxGeometry(look.size[0] * 0.5, 0.06, 0.06);
        lampGeo.translate(0, look.size[1] + 0.03, 0);
        lamps = new THREE.InstancedMesh(lampGeo, new THREE.MeshBasicMaterial({ color: 0xffffff }), list.length);
      }
      list.forEach((d, i) => {
        q.setFromAxisAngle(Y, d.ry || 0);
        p.set(d.x, d.y, d.z);
        m.compose(p, q, s);
        for (const im of bodies) { im.setMatrixAt(i, m); im.setColorAt(i, base); }
        if (lamps) { lamps.setMatrixAt(i, m); lamps.setColorAt(i, lampCol); }
        this.containers.push({
          type, name: (asset !== 'c_' + type && mapData.containerNames?.[type]) || CONTAINER_NAMES[type], pos: new THREE.Vector3(d.x, d.y + size[1] * 0.5, d.z),
          tier: d.tier || 1, bodies, lamps, idx: i, searched: false, items: null, base, lampCol,
        });
      });
      for (const im of lamps ? [...bodies, lamps] : bodies) {
        im.instanceMatrix.needsUpdate = true;
        im.instanceColor.needsUpdate = true;
        im.computeBoundingSphere();
        this.game.scene.add(im);
      }
      this.inst[type] = bodies;
      // Kollision (Oberseite begehbar)
      for (const d of list) {
        const [w, h, dd] = size;
        const rot = Math.abs(Math.sin(d.ry || 0)) > 0.5;
        const hw = (rot ? dd : w) / 2, hd = (rot ? w : dd) / 2;
        this.game.col.addBox(d.x - hw, d.y, d.z - hd, d.x + hw, d.y + h, d.z + hd);
      }
    }
  }

  // Zu Raid-Beginn: alle Behälter zurücksetzen, Wracks und Beutel entfernen
  reset(lootMul) {
    this.lootMul = lootMul;
    for (const c of this.containers) {
      c.searched = false;
      c.items = null;
      this.paint(c, false);
    }
    for (const w of this.wrecks) {
      if (w.mesh) this.game.scene.remove(w.mesh);
    }
    for (const b of this.bags) this.game.scene.remove(b.mesh);
    this.wrecks.length = 0;
    this.bags.length = 0;
  }

  paint(c, searched) {
    if (!c.bodies) return;
    const body = searched ? c.base.clone().multiplyScalar(0.6) : c.base;
    for (const im of c.bodies) {
      im.setColorAt(c.idx, body);
      im.instanceColor.needsUpdate = true;
    }
    if (c.lamps) {
      c.lamps.setColorAt(c.idx, searched ? new THREE.Color(0x1a1a1a) : c.lampCol);
      c.lamps.instanceColor.needsUpdate = true;
    }
  }

  markSearched(c) {
    c.searched = true;
    this.paint(c, true);
  }

  generate(c) {
    if (c.items) return;
    if (c.enemyType) {
      const inv = new Inventory(8);
      for (const [id, chance, q] of ENEMY_LOOT[c.enemyType]) {
        if (Math.random() < chance) inv.add(id, randInt(q[0], q[1]));
      }
      c.items = inv;
      return;
    }
    c.items = rollTable(c.type, c.tier, this.lootMul || 1);
  }

  searchTime(c) {
    if (c.searched) return 0;
    if (c.enemyType) return 1.2;
    return TABLES[c.type] ? TABLES[c.type].search : 0.5;
  }

  addWreck(enemy) {
    let mesh = null;
    if (!enemy.root.visible) {
      // Knaller: nur Trümmer
      mesh = new THREE.Mesh(new THREE.DodecahedronGeometry(0.35, 0), new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.9 }));
      mesh.position.set(enemy.pos.x, enemy.game.col.groundAt(enemy.pos.x, enemy.pos.z, 0.3, enemy.pos.y + 1, 0.5) + 0.2, enemy.pos.z);
      this.game.scene.add(mesh);
    }
    const w = {
      type: 'arcwreck', name: `${enemy.name} (Wrack)`, enemyType: enemy.type, pos: enemy.pos.clone().add(new THREE.Vector3(0, 0.5, 0)),
      searched: false, items: null, mesh, enemy,
    };
    this.wrecks.push(w);
  }

  addBag(pos, inv) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.4, 0.4), new THREE.MeshStandardMaterial({ color: 0xc8502c, roughness: 0.8 }));
    const g = this.game.col.groundAt(pos.x, pos.z, 0.2, pos.y + 1, 0.5);
    mesh.position.set(pos.x, g + 0.2, pos.z);
    mesh.castShadow = true;
    this.game.scene.add(mesh);
    const b = { type: 'bag', name: 'Beutel', pos: mesh.position.clone(), searched: true, items: inv, mesh };
    this.bags.push(b);
    return b;
  }

  // Nächster Behälter im Blickfeld
  nearest(pos, fwd, maxDist = 2.4) {
    let best = null, bestScore = Infinity;
    const check = (c) => {
      const dx = c.pos.x - pos.x, dy = c.pos.y - pos.y, dz = c.pos.z - pos.z;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d > maxDist + 0.6) return;
      const hd = Math.hypot(dx, dz) || 1;
      const facing = (dx * fwd.x + dz * fwd.z) / hd;
      if (d > 1.3 && facing < 0.3) return;
      const score = d - facing;
      if (score < bestScore) { bestScore = score; best = c; }
    };
    for (const c of this.containers) check(c);
    for (const w of this.wrecks) check(w);
    for (const b of this.bags) {
      if (b.items && b.items.isEmpty()) continue;
      check(b);
    }
    return best;
  }

  // Für die Karte: offene Behälter in der Nähe zählen etc.
  count() {
    return this.containers.length;
  }
}

export function lootValue(inv) {
  return inv.value();
}

export { rollTable, rand };
