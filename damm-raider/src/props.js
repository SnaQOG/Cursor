// Extraktions-Aufzüge und explosive Fässer.
import * as THREE from 'three';
import { EXTRACTS } from './config.js';
import { makeHazardTexture } from './textures.js';
import { assets, Assets } from './assets.js';

const CALL_TIME = 25;
const BOARD_TIME = 5;
const OPEN_TIME = 60;

export class Elevator {
  constructor(game, def) {
    this.game = game;
    this.def = def;
    this.name = def.name;
    this.id = def.id;
    const gy = game.col.groundAt(def.x, def.z, 3, 999, 0);
    this.base = new THREE.Vector3(def.x, gy, def.z);
    // Zur Kartenmitte ausrichten (Tür zeigt zur Mitte)
    this.yaw = Math.atan2(-def.x, -def.z) + Math.PI;
    const g = new THREE.Group();
    g.position.copy(this.base);
    g.rotation.y = this.yaw;
    game.scene.add(g);
    this.group = g;
    if (assets.has('l_elevator')) this.buildModel(game, g);
    else this.buildPrims(game, g);
    this.beam = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 0.8, 200, 12, 1, true),
      new THREE.MeshBasicMaterial({ color: 0x9fffd0, transparent: true, opacity: 0.18, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
    this.beam.position.set(0, 100, 0);
    this.beam.visible = false;
    g.add(this.beam);
    this.consolePos = new THREE.Vector3(2.8, 1, -2.6).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.yaw).add(this.base);
    this.cabinPos = this.base.clone();
    this.reset();
  }

  // Kollisionsbox in lokalen Koordinaten (Tür zeigt nach -Z)
  colBox(w, h, d, x, y, z) {
    const wp = new THREE.Vector3(x, y, z).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.yaw).add(this.base);
    const c = Math.abs(Math.cos(this.yaw)), s = Math.abs(Math.sin(this.yaw));
    const hw = (w / 2) * c + (d / 2) * s, hd = (w / 2) * s + (d / 2) * c;
    return this.game.col.addBox(wp.x - hw, wp.y, wp.z - hd, wp.x + hw, wp.y + h, wp.z + hd);
  }

  buildModel(game, g) {
    const m = assets.clone('l_elevator');
    g.add(m);
    const n = Assets.nodes(m, 'l_elevator');
    this.doors = [n.Door0, n.Door1];
    this.doorBase = this.doors.map((d) => d.position.x);
    const lamp = n.Lamp;
    this.lampMat = lamp.material.clone();
    this.lampMat.emissiveIntensity = 1.6;
    lamp.material = this.lampMat;
    this.lampIsEmissive = true;
    // Kollision wie beim alten Aufbau
    this.colBox(8, 0.35, 8, 0, -0.2, 0);
    for (const [x, z] of [[-1.8, -1.6], [1.8, -1.6], [-1.8, 1.6], [1.8, 1.6]]) this.colBox(0.4, 5, 0.4, x, 0.15, z);
    this.colBox(4, 0.3, 3.6, 0, 5.15, 0);
    this.colBox(3.6, 5, 0.2, 0, 0.15, 1.7);
    this.colBox(0.2, 5, 3.2, -1.9, 0.15, 0);
    this.colBox(0.2, 5, 3.2, 1.9, 0.15, 0);
    this.colBox(0.6, 1.1, 0.5, 2.8, 0.15, -2.6);
    this.doorCol = this.colBox(3.2, 4.6, 0.2, 0, 0, -1.75);
  }

  buildPrims(game, g) {
    const concrete = new THREE.MeshStandardMaterial({ color: 0x9d998d, roughness: 0.9 });
    const metal = new THREE.MeshStandardMaterial({ color: 0x3f4447, roughness: 0.5, metalness: 0.6 });
    const hazard = new THREE.MeshStandardMaterial({ map: makeHazardTexture(), roughness: 0.7 });
    const box = (w, h, d, x, y, z, m, collide = true) => {
      const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m);
      b.position.set(x, y + h / 2, z);
      b.castShadow = b.receiveShadow = true;
      g.add(b);
      if (collide) {
        const wp = new THREE.Vector3(x, y, z).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.yaw).add(this.base);
        const c = Math.abs(Math.cos(this.yaw)), s = Math.abs(Math.sin(this.yaw));
        const hw = (w / 2) * c + (d / 2) * s, hd = (w / 2) * s + (d / 2) * c;
        game.col.addBox(wp.x - hw, wp.y, wp.z - hd, wp.x + hw, wp.y + h, wp.z + hd);
      }
      return b;
    };
    // Plattform
    box(8, 0.35, 8, 0, -0.2, 0, concrete);
    box(8.02, 0.05, 0.6, 0, 0.15, -3.7, hazard, false);
    // Schacht (Tür zeigt nach -Z lokal)
    box(0.4, 5, 0.4, -1.8, 0.15, -1.6, metal);
    box(0.4, 5, 0.4, 1.8, 0.15, -1.6, metal);
    box(0.4, 5, 0.4, -1.8, 0.15, 1.6, metal);
    box(0.4, 5, 0.4, 1.8, 0.15, 1.6, metal);
    box(4, 0.3, 3.6, 0, 5.15, 0, concrete);
    box(3.6, 5, 0.2, 0, 0.15, 1.7, metal);
    box(0.2, 5, 3.2, -1.9, 0.15, 0, metal);
    box(0.2, 5, 3.2, 1.9, 0.15, 0, metal);
    box(3.2, 0.12, 3.2, 0, 0.15, 0, hazard, false);
    // Türflügel
    const doorMat = new THREE.MeshStandardMaterial({ color: 0xd8a52e, roughness: 0.6, metalness: 0.3 });
    this.doors = [box(1.6, 4.4, 0.15, -0.8, 0.2, -1.75, doorMat, false), box(1.6, 4.4, 0.15, 0.8, 0.2, -1.75, doorMat, false)];
    // Türkollision (wird beim Öffnen deaktiviert)
    const dp = new THREE.Vector3(0, 0, -1.75).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.yaw).add(this.base);
    const c = Math.abs(Math.cos(this.yaw)), s = Math.abs(Math.sin(this.yaw));
    const hw = 1.6 * c + 0.1 * s, hd = 1.6 * s + 0.1 * c;
    this.doorCol = game.col.addBox(dp.x - hw, dp.y, dp.z - hd, dp.x + hw, dp.y + 4.6, dp.z + hd);
    // Konsole
    box(0.6, 1.1, 0.5, 2.8, 0.15, -2.6, metal);
    this.lampMat = new THREE.MeshBasicMaterial({ color: 0xff8a2a });
    const lamp = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.15, 0.3), this.lampMat);
    lamp.position.set(2.8, 1.35, -2.6);
    g.add(lamp);
    const beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.25, 0.5, 10), this.lampMat);
    beacon.position.set(0, 5.7, 0);
    g.add(beacon);
    this.doorBase = [-0.8, 0.8];
  }

  setLamp(hex) {
    if (this.lampIsEmissive) this.lampMat.emissive.set(hex);
    this.lampMat.color.set(hex);
  }

  reset() {
    this.state = 'idle';
    this.t = 0;
    this.board = 0;
    this.doorOpen = 0;
    this.doorCol.active = true;
    this.beam.visible = false;
    this.setLamp(0xff8a2a);
  }

  get label() {
    if (this.state === 'idle') return `${this.name} rufen`;
    return null;
  }

  interact() {
    if (this.state !== 'idle') return;
    this.state = 'calling';
    this.t = CALL_TIME;
    this.game.sfx.play('elevator', this.consolePos);
    this.game.hud.toast(`${this.name} kommt in ${CALL_TIME} s. Halte die Stellung!`, 'info');
    // Lärm lockt ARC an
    this.game.makeNoise(this.base, 90, null);
  }

  playerInside() {
    const p = this.game.player;
    const local = p.pos.clone().sub(this.base).applyAxisAngle(new THREE.Vector3(0, 1, 0), -this.yaw);
    return Math.abs(local.x) < 1.7 && Math.abs(local.z) < 1.55 && Math.abs(p.pos.y - this.base.y) < 1.5;
  }

  update(dt) {
    const g = this.game;
    if (this.state === 'calling') {
      this.t -= dt;
      this.setLamp(Math.sin(this.t * 8) > 0 ? 0xffd040 : 0x553300);
      if (Math.floor(this.t + dt) !== Math.floor(this.t) && Math.floor(this.t) % 8 === 0) g.makeNoise(this.base, 70, null);
      if (this.t <= 0) {
        this.state = 'open';
        this.t = OPEN_TIME;
        this.doorCol.active = false;
        this.beam.visible = true;
        this.setLamp(0x40ff90);
        g.sfx.play('elevator', this.consolePos, { vol: 1.2 });
        g.hud.toast(`${this.name} ist offen! Steig ein.`, 'good');
      }
    } else if (this.state === 'open') {
      this.t -= dt;
      this.doorOpen = Math.min(1, this.doorOpen + dt);
      if (g.player.alive && this.playerInside()) {
        this.board += dt;
        g.hud.setExtractProgress(this.board / BOARD_TIME, BOARD_TIME - this.board);
        if (this.board >= BOARD_TIME) g.extract(this);
      } else {
        if (this.board > 0) g.hud.setExtractProgress(0, 0);
        this.board = 0;
      }
      if (this.t <= 0) {
        this.state = 'closing';
        this.t = 30;
        this.doorCol.active = true;
        this.beam.visible = false;
        this.setLamp(0x802010);
      }
    } else if (this.state === 'closing') {
      this.doorOpen = Math.max(0, this.doorOpen - dt);
      this.t -= dt;
      if (this.t <= 0) this.reset();
    }
    this.doors[0].position.x = this.doorBase[0] - this.doorOpen * 1.5;
    this.doors[1].position.x = this.doorBase[1] + this.doorOpen * 1.5;
  }
}

export function createElevators(game) {
  return EXTRACTS.map((d) => new Elevator(game, d));
}

export class Barrel {
  constructor(game, x, z) {
    this.game = game;
    const gy = game.col.groundAt(x, z, 0.4, 999, 0);
    this.pos = new THREE.Vector3(x, gy, z);
    let g;
    if (assets.has('p_barrel')) {
      g = assets.clone('p_barrel');
      g.rotation.y = Math.random() * Math.PI * 2;
    } else {
      g = new THREE.Group();
      const body = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.42, 1.2, 12), new THREE.MeshStandardMaterial({ color: 0xb8352a, roughness: 0.6, metalness: 0.3 }));
      body.position.y = 0.6;
      body.castShadow = true;
      const band = new THREE.Mesh(new THREE.CylinderGeometry(0.43, 0.43, 0.18, 12), new THREE.MeshStandardMaterial({ color: 0xe0b030, roughness: 0.6 }));
      band.position.y = 0.75;
      g.add(body, band);
    }
    g.position.copy(this.pos);
    game.scene.add(g);
    this.mesh = g;
    this.shape = game.col.addCyl(x, z, 0.42, gy, gy + 1.2, { barrel: this });
    this.reset();
  }

  reset() {
    this.hp = 25;
    this.alive = true;
    this.fuse = -1;
    this.mesh.visible = true;
    this.shape.active = true;
  }

  hit(dmg) {
    if (!this.alive) return;
    this.hp -= dmg;
    if (this.hp <= 0 && this.fuse < 0) this.fuse = 0.12;
  }

  explode() {
    if (!this.alive) return;
    this.alive = false;
    this.mesh.visible = false;
    this.shape.active = false;
    this.game.explode(this.pos.clone().add(new THREE.Vector3(0, 0.7, 0)), 6.5, 80, null);
  }

  update(dt) {
    if (this.fuse >= 0 && this.alive) {
      this.fuse -= dt;
      this.game.effects.fire(this.pos.clone().add(new THREE.Vector3(0, 1.2, 0)));
      if (this.fuse <= 0) this.explode();
    }
  }
}
