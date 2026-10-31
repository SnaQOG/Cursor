// Einfaches, prozedural animiertes Raider-Modell.
import * as THREE from 'three';
import { makeGunModel } from './weapons.js';

export class Humanoid {
  constructor(colors = {}) {
    const c = {
      jacket: 0xd9a33a, pants: 0x3c4b4a, pack: 0xc8502c, helmet: 0xe6dfcf, visor: 0x1c2a30, skin: 0xc49a78, boots: 0x2c2522,
      ...colors,
    };
    const mat = (color, rough = 0.8) => new THREE.MeshStandardMaterial({ color, roughness: rough, metalness: 0.05 });
    this.root = new THREE.Group();
    this.body = new THREE.Group(); // wird für Rolle/Ducken bewegt
    this.root.add(this.body);
    const mesh = (geo, m, parent, x, y, z) => {
      const o = new THREE.Mesh(geo, m);
      o.position.set(x, y, z);
      o.castShadow = true;
      parent.add(o);
      return o;
    };
    const mJ = mat(c.jacket), mP = mat(c.pants), mPack = mat(c.pack), mH = mat(c.helmet, 0.5), mV = mat(c.visor, 0.2), mB = mat(c.boots);

    this.hips = new THREE.Group();
    this.hips.position.y = 0.95;
    this.body.add(this.hips);
    this.legs = [];
    for (const s of [-1, 1]) {
      const pivot = new THREE.Group();
      pivot.position.set(s * 0.12, 0, 0);
      this.hips.add(pivot);
      const upper = mesh(new THREE.BoxGeometry(0.17, 0.5, 0.19), mP, pivot, 0, -0.25, 0);
      const knee = new THREE.Group();
      knee.position.set(0, -0.48, 0);
      pivot.add(knee);
      mesh(new THREE.BoxGeometry(0.15, 0.45, 0.17), mP, knee, 0, -0.22, 0);
      mesh(new THREE.BoxGeometry(0.16, 0.1, 0.26), mB, knee, 0, -0.44, -0.04);
      upper.userData.side = s;
      this.legs.push({ pivot, knee });
    }
    this.torso = new THREE.Group();
    this.torso.position.y = 0.95;
    this.body.add(this.torso);
    mesh(new THREE.BoxGeometry(0.46, 0.58, 0.28), mJ, this.torso, 0, 0.33, 0);
    mesh(new THREE.BoxGeometry(0.48, 0.1, 0.3), mat(0x3a3430), this.torso, 0, 0.05, 0);
    // Rucksack
    mesh(new THREE.BoxGeometry(0.4, 0.5, 0.24), mPack, this.torso, 0, 0.38, 0.26);
    mesh(new THREE.BoxGeometry(0.42, 0.14, 0.26), mat(0x6b6a5c), this.torso, 0, 0.66, 0.26);
    mesh(new THREE.BoxGeometry(0.06, 0.4, 0.05), mat(0x2a2a2a), this.torso, -0.14, 0.4, 0.4);
    // Kopf
    this.head = new THREE.Group();
    this.head.position.y = 0.7;
    this.torso.add(this.head);
    mesh(new THREE.SphereGeometry(0.13, 12, 10), mat(c.skin), this.head, 0, 0.09, 0);
    const helm = mesh(new THREE.SphereGeometry(0.155, 14, 10, 0, Math.PI * 2, 0, Math.PI * 0.55), mH, this.head, 0, 0.12, 0.01);
    helm.scale.set(1, 0.9, 1.08);
    mesh(new THREE.BoxGeometry(0.22, 0.07, 0.06), mV, this.head, 0, 0.1, -0.12);
    // Arme
    this.arms = [];
    for (const s of [-1, 1]) {
      const pivot = new THREE.Group();
      pivot.position.set(s * 0.29, 0.56, 0);
      pivot.rotation.order = 'YXZ';
      this.torso.add(pivot);
      mesh(new THREE.BoxGeometry(0.13, 0.52, 0.14), mJ, pivot, 0, -0.24, 0);
      mesh(new THREE.BoxGeometry(0.1, 0.1, 0.1), mat(0x2c2c2c), pivot, 0, -0.52, 0);
      this.arms.push(pivot);
    }
    // Waffenhalter (Schulterhöhe rechts)
    this.gunPivot = new THREE.Group();
    this.gunPivot.position.set(0.24, 0.42, -0.3);
    this.torso.add(this.gunPivot);
    this.gun = null;
    this.gunId = null;
    this.phase = 0;
    this.muzzleWorld = new THREE.Vector3();
  }

  setGun(id) {
    if (this.gunId === id) return;
    if (this.gun) this.gunPivot.remove(this.gun);
    this.gun = id ? makeGunModel(id) : null;
    this.gunId = id;
    if (this.gun) this.gunPivot.add(this.gun);
  }

  getMuzzle(out) {
    if (!this.gun) {
      this.gunPivot.getWorldPosition(out);
      return out;
    }
    out.copy(this.gun.userData.muzzle);
    this.gun.localToWorld(out);
    return out;
  }

  // speed: horizontale Geschwindigkeit, aim: Waffe hoch, pitch: Blickneigung
  animate(dt, { speed = 0, crouch = false, aim = false, pitch = 0, air = false, roll = 0, action = null, sprint = false }) {
    this.phase += dt * (speed > 0.3 ? 2.2 + speed * 1.1 : 0);
    const sw = Math.min(1, speed / 4);
    const s = Math.sin(this.phase);
    const crouchK = crouch ? 1 : 0;
    this.hips.position.y = 0.95 - crouchK * 0.32 + Math.abs(Math.cos(this.phase)) * 0.04 * sw;
    this.torso.position.y = this.hips.position.y;
    this.torso.rotation.x = crouchK * 0.25 + (sprint ? 0.2 : 0);
    for (let i = 0; i < 2; i++) {
      const L = this.legs[i];
      const sgn = i === 0 ? 1 : -1;
      const swing = air ? -0.4 * sgn * 0.2 - 0.3 : s * 0.7 * sw * sgn;
      L.pivot.rotation.x = swing - crouchK * 0.9;
      L.knee.rotation.x = Math.max(0, -swing) * 0.9 + crouchK * 1.5 + (air ? 0.6 : 0);
    }
    // Arme / Waffe (Arm hängt entlang -Y; +PI/2 um X zeigt nach vorne)
    const hold = aim || !sprint;
    const armPitch = hold ? Math.PI / 2 + pitch : -0.3 + s * 0.5 * sw;
    this.gunPivot.rotation.x = hold ? pitch : -0.9;
    this.gunPivot.rotation.y = hold ? 0 : 0.6;
    this.arms[1].rotation.set(armPitch, hold ? 0.05 : 0, 0);
    this.arms[0].rotation.set(hold ? armPitch - 0.05 : 0.3 - s * 0.5 * sw, hold ? -0.75 : 0, 0);
    if (action === 'heal' || action === 'shield') {
      this.arms[0].rotation.set(1.1, -0.9, 0);
      this.gunPivot.rotation.x = -1.0;
    }
    this.head.rotation.x = -pitch * 0.5;
    // Vorwärtsrolle um den Körpermittelpunkt
    const th = -roll * Math.PI * 2;
    this.body.rotation.x = th;
    this.body.position.set(0, 0.6 - 0.6 * Math.cos(th), -0.6 * Math.sin(th));
  }
}
