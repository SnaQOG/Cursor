// ARC-Maschinen: Wahrnehmung, Bewegung und Kampfverhalten.
import * as THREE from 'three';
import * as AM from './arcmodels.js';
import { rand, clamp, angleDiff, turnTowards, lerp, pick, TAU } from './util.js';
import { MAP } from './config.js';

export const ENEMY_DEFS = {
  tick: { name: 'Zecke', hp: 28, sight: 18, fov: 3.5, react: 0.2, radius: 0.4, height: 0.6, air: false, speed: 8.5, loot: 'tick', big: 0.4 },
  pop: { name: 'Knaller', hp: 20, sight: 36, fov: 3.2, react: 0.4, radius: 0.5, height: 1.0, air: false, speed: 7.5, loot: 'pop', big: 0.6 },
  wasp: { name: 'Wespe', hp: 60, sight: 58, fov: 2.4, react: 0.6, radius: 0.8, height: 0.8, air: true, speed: 10, loot: 'wasp', big: 0.8, crashAt: 2 },
  hornet: { name: 'Hornisse', hp: 200, sight: 62, fov: 2.2, react: 0.7, radius: 1.0, height: 1.0, air: true, speed: 7.5, loot: 'hornet', big: 1, crashAt: 1 },
  snitch: { name: 'Späher', hp: 50, sight: 75, fov: TAU, react: 1.4, radius: 0.8, height: 0.8, air: true, speed: 6, loot: 'snitch', big: 0.7 },
  rocketeer: { name: 'Raketier', hp: 500, sight: 80, fov: 2.4, react: 0.8, radius: 1.8, height: 1.6, air: true, speed: 6, loot: 'rocketeer', big: 1.8, crashAt: 2 },
  leaper: { name: 'Springer', hp: 680, sight: 60, fov: 2.6, react: 0.6, radius: 1.7, height: 3.4, air: false, speed: 6.8, loot: 'leaper', big: 2 },
  bastion: { name: 'Bastion', hp: 2100, sight: 75, fov: 2.4, react: 0.8, radius: 2.6, height: 6.5, air: false, speed: 2.5, loot: 'bastion', big: 3 },
  turret: { name: 'Wächter', hp: 240, sight: 85, fov: TAU, react: 0.5, radius: 0.9, height: 1.6, air: false, speed: 0, loot: 'turret', big: 1 },
};

const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _c = new THREE.Vector3();

export class Enemy {
  constructor(game, type, spawn) {
    this.game = game;
    this.type = type;
    this.def = ENEMY_DEFS[type];
    this.name = this.def.name;
    this.hp = this.def.hp;
    this.maxHp = this.def.hp;
    this.alive = true;
    this.crashing = false;
    let y;
    if (this.def.air) y = Math.max(spawn.y || 0, game.col.groundAt(spawn.x, spawn.z, 0.5, 999, 0) + rand(7, 11));
    else if (spawn.y != null) y = spawn.y;
    else y = game.col.groundAt(spawn.x, spawn.z, 0.5, game.terrain.height(spawn.x, spawn.z) + 1, 0.6);
    this.pos = new THREE.Vector3(spawn.x, y, spawn.z);
    if (!this.def.air && type !== 'turret') game.col.pushOut(this.pos, this.def.radius, this.def.height, 0.5);
    this.home = this.pos.clone();
    this.roam = !!spawn.roam;
    this.vel = new THREE.Vector3();
    this.yaw = rand(0, TAU);
    this.state = 'idle';
    this.aware = 0;
    this.lastSeen = new THREE.Vector3();
    this.seesTarget = false;
    this.lostT = 0;
    this.senseT = rand(0, 0.3);
    this.waypoint = null;
    this.wpT = 0;
    this.stuckT = 0;
    this.detourT = 0;
    this.detourYaw = 0;
    this.flashT = 0;
    this.t = rand(0, 10);
    this.searchT = 0;
    this.farT = 0;
    this.build();
    this.game.scene.add(this.root);
    this.syncModel();
  }

  build() {
    const builders = {
      tick: AM.buildTick, pop: AM.buildPop, wasp: AM.buildWasp, hornet: AM.buildHornet, snitch: AM.buildSnitch,
      rocketeer: AM.buildRocketeer, leaper: AM.buildLeaper, bastion: AM.buildBastion, turret: AM.buildTurret,
    };
    this.model = AM.buildArcModel(this.type) || builders[this.type]();
    this.root = this.model.root;
    this.parts = this.model.parts;
    this.flashMats = [];
    // Eigene Materialkopien für Treffer-Aufblitzen
    this.root.traverse((o) => {
      if (o.isMesh) {
        o.material = o.material.clone();
        this.flashMats.push(o.material);
      }
    });
    this.boundR = this.def.radius * 2 + 1;
  }

  get center() {
    return _c.set(this.pos.x, this.pos.y + (this.def.air ? 0 : this.def.height * 0.5), this.pos.z);
  }

  eye(out) {
    return out.set(this.pos.x, this.pos.y + (this.def.air ? 0 : this.def.height * 0.8), this.pos.z);
  }

  partCenter(p, out) {
    if (p.offset) return this.root.localToWorld(out.copy(p.offset));
    return p.obj.getWorldPosition(out);
  }

  syncModel() {
    this.root.position.copy(this.pos);
    this.root.rotation.y = this.yaw;
  }

  // ------------------------------------------------ Wahrnehmung
  targetInfo() {
    const p = this.game.player;
    if (!p.alive) return null;
    return p;
  }

  perceive(dt) {
    this.senseT -= dt;
    if (this.senseT > 0) return;
    const interval = 0.25;
    this.senseT = interval;
    const p = this.targetInfo();
    this.seesTarget = false;
    if (!p) { this.aware = 0; return; }
    const tc = p.chest(_b);
    const e = this.eye(_a);
    const dist = e.distanceTo(tc);
    let sight = this.def.sight;
    if (p.crouch && p.lastShot > 2) sight *= 0.6;
    if (p.sprinting) sight *= 1.15;
    if (this.game.condition === 'sturm') sight *= 0.75;
    if (this.state === 'attack' || this.state === 'search') sight *= 1.3;
    if (dist > sight) { this.decay(interval); return; }
    const toYaw = Math.atan2(-(tc.x - e.x), -(tc.z - e.z));
    const inFov = this.def.fov >= TAU - 0.01 || this.state === 'attack' || dist < 6 || Math.abs(angleDiff(this.yaw, toYaw)) < this.def.fov / 2;
    if (!inFov) { this.decay(interval); return; }
    if (!this.game.col.lineOfSight(e, tc)) { this.decay(interval); return; }
    this.seesTarget = true;
    this.lastSeen.copy(p.pos);
    this.lostT = 0;
    const gain = dist < 10 ? 4 : 1 - dist / (sight * 1.4);
    this.aware += interval * Math.max(0.35, gain) / this.def.react;
    if (this.aware >= 1 && this.state !== 'attack') this.engage(true);
  }

  decay(dt) {
    if (this.state !== 'attack') this.aware = Math.max(0, this.aware - dt * 0.3);
  }

  engage(alertOthers) {
    const was = this.state;
    this.state = 'attack';
    this.aware = 1;
    this.onEngage?.();
    if (was !== 'attack') this.game.hud.arcAlert(this);
    if (alertOthers) {
      for (const o of this.game.enemies) {
        if (o === this || !o.alive || o.state === 'attack') continue;
        if (o.pos.distanceToSquared(this.pos) < 32 * 32) o.hear(this.lastSeen, true);
      }
    }
  }

  // Geräusch oder Alarm in der Nähe
  hear(pos, strong = false) {
    if (!this.alive || this.state === 'attack') return;
    this.state = 'search';
    this.lastSeen.copy(pos);
    this.searchT = strong ? 10 : 7;
    this.aware = Math.max(this.aware, strong ? 0.7 : 0.4);
  }

  // ------------------------------------------------ Schaden
  hit(part, dmg, pen, point, dir, source) {
    if (!this.alive) return null;
    const armor = (part.armor || 0) * (1 - pen);
    let dealt = dmg * (part.mult || 1) * (1 - armor);
    const kind = part.weak ? 'weak' : armor > 0.3 ? 'armor' : 'normal';
    this.flashT = 0.08;
    if (part.hp != null && !part.destroyed) {
      part.hp -= dealt;
      if (part.hp <= 0) this.destroyPart(part);
    }
    this.hp -= dealt;
    if (source === this.game.player && this.state !== 'attack') {
      this.lastSeen.copy(source.pos);
      this.engage(true);
    }
    this.onHit?.(dealt, source);
    if (this.hp <= 0) this.die(source);
    return { dealt, kind, killed: !this.alive };
  }

  destroyPart(part) {
    part.destroyed = true;
    const p = this.partCenter(part, _a).clone();
    part.obj.visible = false;
    this.game.effects.sparks(p, new THREE.Vector3(0, 1, 0), 18, [1, 0.7, 0.3], 8);
    this.game.effects.smokePuff(p);
    this.game.sfx.play('explosion', p, { vol: 0.35 });
    if (part.thruster) {
      const lost = this.parts.filter((q) => q.thruster && q.destroyed).length;
      if (this.def.crashAt && lost >= this.def.crashAt && !this.crashing) this.startCrash();
    }
  }

  startCrash() {
    this.crashing = true;
    this.state = 'crash';
    this.vel.y = Math.max(this.vel.y, 1);
    this.game.hud.toast(`${this.name} stürzt ab!`, 'good');
  }

  die(source, silent = false) {
    if (!this.alive) return;
    this.alive = false;
    this.state = 'dead';
    const c = this.center.clone();
    if (!silent) {
      this.game.effects.explosion(c, 2 + this.def.big * 1.4);
      this.game.sfx.play('explosion', c, { big: this.def.big > 1.5 });
    }
    this.game.onEnemyKilled(this, source);
    this.toWreck();
  }

  toWreck() {
    // Wrack bleibt liegen und kann durchsucht werden
    const wreckMat = AM.ARC_MATS.wreck;
    this.root.traverse((o) => {
      if (o.isMesh) {
        const base = o.material.userData.baseE ?? (o.material.emissive ? o.material.emissive.getHex() : 0);
        if (base > 0) o.visible = false;
        else o.material = wreckMat;
      }
    });
    const g = this.game.col.groundAt(this.pos.x, this.pos.z, 0.5, this.pos.y + 1, 0.5);
    this.pos.y = g + (this.def.air ? this.def.radius * 0.35 : 0);
    this.root.position.copy(this.pos);
    this.root.rotation.set(rand(-0.4, 0.4), this.yaw, rand(-0.5, 0.5));
    if (this.type === 'leaper' || this.type === 'bastion') {
      const hull = this.model.hull;
      hull.position.y *= 0.4;
    }
    this.smokeT = 25;
    this.game.loot.addWreck(this);
  }

  // ------------------------------------------------ Bewegung
  flyTo(dt, tx, ty, tz, speed, accel = 3) {
    const col = this.game.col;
    _a.set(tx - this.pos.x, ty - this.pos.y, tz - this.pos.z);
    const d = _a.length();
    if (d > 0.01) _a.multiplyScalar(Math.min(speed, d * 1.2) / d);
    // Hindernis voraus? Dann steigen.
    this.probeT = (this.probeT || 0) - dt;
    if (this.probeT <= 0) {
      this.probeT = 0.25;
      const hs = Math.hypot(this.vel.x, this.vel.z);
      if (hs > 0.5) {
        const hit = col.raycastShapes(this.pos.x, this.pos.y, this.pos.z, this.vel.x / hs, 0, this.vel.z / hs, 6);
        this.blocked = !!hit;
      } else this.blocked = false;
    }
    if (this.blocked) _a.y = Math.max(_a.y, speed * 0.8);
    this.vel.lerp(_a, Math.min(1, accel * dt));
    this.pos.addScaledVector(this.vel, dt);
    const floor = col.groundAt(this.pos.x, this.pos.z, this.def.radius, this.pos.y + 60, 0) + this.def.radius + 1.2;
    if (this.pos.y < floor) {
      this.pos.y = lerp(this.pos.y, floor, Math.min(1, dt * 6));
      if (this.vel.y < 0) this.vel.y = 0;
    }
    const lim = MAP.BOUND + 10;
    this.pos.x = clamp(this.pos.x, -lim, lim);
    this.pos.z = clamp(this.pos.z, -lim, lim);
  }

  walkTo(dt, tx, tz, speed, turnRate = 8) {
    const col = this.game.col;
    const T = this.game.terrain;
    let want = Math.atan2(-(tx - this.pos.x), -(tz - this.pos.z));
    if (this.detourT > 0) {
      this.detourT -= dt;
      want = this.detourYaw;
    }
    this.yaw = turnTowards(this.yaw, want, turnRate * dt);
    const ox = this.pos.x, oz = this.pos.z;
    const fx = -Math.sin(this.yaw), fz = -Math.cos(this.yaw);
    const facing = Math.abs(angleDiff(this.yaw, want)) < 1.2 ? 1 : 0.3;
    this.pos.x += fx * speed * facing * dt;
    this.pos.z += fz * speed * facing * dt;
    const w = T.waterAt(this.pos.x, this.pos.z);
    if (w && w.level - T.height(this.pos.x, this.pos.z) > 1.4 && this.type !== 'bastion') {
      this.pos.x = ox; this.pos.z = oz;
    }
    col.pushOut(this.pos, this.def.radius, this.def.height, 0.5);
    const B = MAP.BOUND;
    this.pos.x = clamp(this.pos.x, -B, B);
    this.pos.z = clamp(this.pos.z, -B, B);
    const g = col.groundAt(this.pos.x, this.pos.z, this.def.radius * 0.5, this.pos.y, 0.6);
    this.pos.y = g > this.pos.y ? lerp(this.pos.y, g, Math.min(1, dt * 12)) : Math.max(g, this.pos.y - 9 * dt);
    // festgefahren?
    const moved = Math.hypot(this.pos.x - ox, this.pos.z - oz);
    if (speed > 0.5 && moved < speed * dt * 0.25) {
      this.stuckT += dt;
      if (this.stuckT > 0.6) {
        this.stuckT = 0;
        this.detourT = rand(0.8, 1.6);
        this.detourYaw = this.yaw + (Math.random() < 0.5 ? 1 : -1) * rand(1.2, 2.2);
      }
    } else this.stuckT = Math.max(0, this.stuckT - dt);
    return moved;
  }

  wanderPoint(radius) {
    if (this.roam && Math.random() < 0.5) {
      const poi = pick(this.game.poiList);
      return new THREE.Vector3(poi.x + rand(-20, 20), 0, poi.z + rand(-20, 20));
    }
    const a = rand(0, TAU), r = rand(radius * 0.3, radius);
    return new THREE.Vector3(this.home.x + Math.cos(a) * r, 0, this.home.z + Math.sin(a) * r);
  }

  faceTowards(dt, x, z, rate) {
    const want = Math.atan2(-(x - this.pos.x), -(z - this.pos.z));
    this.yaw = turnTowards(this.yaw, want, rate * dt);
    return Math.abs(angleDiff(this.yaw, want));
  }

  // ------------------------------------------------ Hauptschleife
  update(dt) {
    this.t += dt;
    if (!this.alive) {
      if (this.smokeT > 0) {
        this.smokeT -= dt;
        if (Math.random() < dt * 6) this.game.effects.smokePuff(_a.set(this.pos.x, this.pos.y + 0.5, this.pos.z));
        if (this.smokeT > 18 && Math.random() < dt * 8) this.game.effects.fire(_a.set(this.pos.x, this.pos.y + 0.4, this.pos.z));
      }
      return;
    }
    if (this.crashing) {
      this.vel.y -= 14 * dt;
      this.vel.x *= 0.98; this.vel.z *= 0.98;
      this.pos.addScaledVector(this.vel, dt);
      this.yaw += dt * 8;
      if (Math.random() < dt * 20) this.game.effects.smokePuff(this.pos);
      const g = this.game.col.groundAt(this.pos.x, this.pos.z, 0.5, this.pos.y + 1, 0.5);
      if (this.pos.y <= g + 0.4) this.die(this.game.player);
      this.syncModel();
      return;
    }
    if (this.flashT > 0) {
      this.flashT -= dt;
      const k = this.flashT > 0 ? 0.9 : 0;
      for (const m of this.flashMats) if (m.emissive) {
        if (m.userData.baseE == null) m.userData.baseE = m.emissive.getHex();
        if (k) m.emissive.setRGB(0.9, 0.9, 0.9);
        else m.emissive.setHex(m.userData.baseE);
      }
    }
    this.perceive(dt);
    if (this.state === 'attack' && !this.seesTarget) {
      this.lostT += dt;
      if (this.lostT > 5) {
        this.state = 'search';
        this.searchT = 8;
      }
    }
    if (this.state === 'search') {
      this.searchT -= dt;
      if (this.searchT <= 0) {
        this.state = 'idle';
        this.aware = 0;
      }
    }
    this.think(dt);
    this.syncModel();
  }

  think() {}

  // Projektil in Richtung Ziel mit Streuung
  shootAt(from, target, speed, dmg, spreadDeg, kind = 'bullet') {
    _a.subVectors(target, from).normalize();
    const s = (spreadDeg * Math.PI) / 180;
    _a.x += rand(-s, s); _a.y += rand(-s, s) * 0.6; _a.z += rand(-s, s);
    _a.normalize();
    this.game.projectiles.enemyShot(from.clone(), _a.clone().multiplyScalar(speed), dmg, kind, this);
  }

  // Vorhalt: wohin muss man zielen?
  leadTarget(from, speed, out) {
    const p = this.game.player;
    p.chest(out);
    const t = from.distanceTo(out) / speed;
    out.x += p.vel.x * t * 0.7;
    out.z += p.vel.z * t * 0.7;
    return out;
  }
}

// =============================================================== Zecke
class Tick extends Enemy {
  constructor(g, t, s) {
    super(g, t, s);
    this.mode = 'crawl';
    this.cd = 0;
  }
  think(dt) {
    const p = this.game.player;
    const legs = this.model.legs;
    const moving = this.mode === 'crawl' && this.state !== 'idle';
    for (let i = 0; i < legs.length; i++) legs[i].rotation.y = moving ? Math.sin(this.t * 22 + i) * 0.4 : 0;
    if (this.mode === 'latched') {
      // hängt am Rücken des Spielers
      const a = p.yaw;
      this.pos.set(p.pos.x + Math.sin(a) * 0.3, p.pos.y + 1.2, p.pos.z + Math.cos(a) * 0.3);
      this.yaw = a;
      if (!p.alive) this.detach(false);
      return;
    }
    if (this.mode === 'leap') {
      this.vel.y -= 20 * dt;
      this.pos.addScaledVector(this.vel, dt);
      const pc = p.chest(_a);
      if (p.alive && this.pos.distanceTo(pc) < 1.1) {
        this.mode = 'latched';
        p.latched.push(this);
        this.game.hud.toast('Zecke hängt an dir! [V] Nahkampf', 'bad');
        this.game.sfx.play('hurt', null);
        return;
      }
      const g = this.game.col.groundAt(this.pos.x, this.pos.z, 0.3, this.pos.y + 0.3, 0.3);
      if (this.pos.y <= g) {
        this.pos.y = g;
        this.mode = 'crawl';
        this.cd = 1.2;
      }
      return;
    }
    this.cd -= dt;
    if (this.state === 'attack') {
      const d = this.pos.distanceTo(p.pos);
      if (Math.random() < dt * 4) this.game.sfx.play('tick', this.pos);
      if (d < 6 && this.cd <= 0 && this.seesTarget) {
        const pc = p.chest(_a);
        const T = 0.42;
        this.vel.set((pc.x - this.pos.x) / T + p.vel.x * 0.5, (pc.y - this.pos.y) / T + 0.5 * 20 * T, (pc.z - this.pos.z) / T + p.vel.z * 0.5);
        this.mode = 'leap';
        this.game.sfx.play('leap', this.pos, { vol: 0.5 });
      } else {
        const tgt = this.seesTarget ? p.pos : this.lastSeen;
        this.walkTo(dt, tgt.x, tgt.z, this.def.speed, 12);
      }
    } else if (this.state === 'search') {
      if (this.pos.distanceTo(this.lastSeen) > 2) this.walkTo(dt, this.lastSeen.x, this.lastSeen.z, this.def.speed * 0.6);
    } else {
      // lauert
      this.yaw += Math.sin(this.t * 0.7) * dt * 0.5;
    }
  }
  detach(kill) {
    const p = this.game.player;
    const i = p.latched.indexOf(this);
    if (i >= 0) p.latched.splice(i, 1);
    this.mode = 'crawl';
    this.pos.set(p.pos.x + rand(-1.5, 1.5), p.pos.y, p.pos.z + rand(-1.5, 1.5));
    this.cd = 1.5;
    if (kill) this.die(p);
  }
}

// =============================================================== Knaller
class Pop extends Enemy {
  constructor(g, t, s) {
    super(g, t, s);
    this.armed = -1;
    this.roll = 0;
    this.beepT = 0;
  }
  think(dt) {
    const p = this.game.player;
    const ball = this.model.ball;
    if (this.armed >= 0) {
      this.armed += dt;
      this.beepT -= dt;
      if (this.beepT <= 0) {
        this.beepT = Math.max(0.08, 0.3 - this.armed * 0.25);
        this.game.sfx.play('beep', this.pos, { f: 1400 + this.armed * 900 });
        this.model.light.visible = !this.model.light.visible;
      }
      ball.scale.setScalar(1 + this.armed * 0.25);
      if (this.armed > 0.95) this.explodeNow(null);
      return;
    }
    let moved = 0;
    if (this.state === 'attack') {
      const d = this.pos.distanceTo(p.pos);
      if (d < 3.2) { this.armed = 0; this.beepT = 0; return; }
      const tgt = this.seesTarget ? p.pos : this.lastSeen;
      moved = this.walkTo(dt, tgt.x, tgt.z, this.def.speed, 10);
    } else if (this.state === 'search') {
      moved = this.walkTo(dt, this.lastSeen.x, this.lastSeen.z, this.def.speed * 0.5);
    } else {
      this.wpT -= dt;
      if (!this.waypoint || this.wpT <= 0) { this.waypoint = this.wanderPoint(14); this.wpT = rand(4, 9); }
      if (this.pos.distanceTo(_a.set(this.waypoint.x, this.pos.y, this.waypoint.z)) > 2) moved = this.walkTo(dt, this.waypoint.x, this.waypoint.z, 2);
    }
    this.roll += moved / 0.5;
    ball.rotation.x = -this.roll;
    ball.position.y = (ball.userData.restY ?? 0.5) + Math.abs(Math.sin(this.roll * 0.5)) * 0.12;
  }
  onHit() {
    if (this.hp <= 0) return;
    if (this.armed < 0 && this.hp < this.maxHp * 0.5) { this.armed = 0.5; }
  }
  explodeNow(source = null) {
    if (!this.alive) return;
    this.hp = 0;
    this.alive = false;
    this.state = 'dead';
    this.root.visible = false;
    const c = this.center.clone();
    this.game.onEnemyKilled(this, source);
    this.game.explode(c, 5.5, 55, this);
    this.game.loot.addWreck(this);
  }
  die(source) {
    if (!this.alive) return;
    this.explodeNow(source);
  }
}

// =============================================================== Wespe & Hornisse
class Drone extends Enemy {
  constructor(g, t, s) {
    super(g, t, s);
    this.orbit = rand(0, TAU);
    this.orbitDir = Math.random() < 0.5 ? 1 : -1;
    this.fireT = rand(1, 2.5);
    this.burst = 0;
    this.burstT = 0;
    this.bob = rand(0, 6);
  }
  think(dt) {
    const p = this.game.player;
    const heavy = this.type === 'hornet';
    const sp = this.def.speed;
    this.bob += dt;
    if (this.state === 'attack' && p.alive) {
      const dist = heavy ? 13 : 16;
      this.orbit += this.orbitDir * dt * (heavy ? 0.25 : 0.45);
      if (Math.random() < dt * 0.15) this.orbitDir *= -1;
      const tgt = this.seesTarget ? p.pos : this.lastSeen;
      const tx = tgt.x + Math.cos(this.orbit) * dist, tz = tgt.z + Math.sin(this.orbit) * dist;
      const ty = tgt.y + (heavy ? 5 : 6.5) + Math.sin(this.bob * 1.3) * 1.2;
      this.flyTo(dt, tx, ty, tz, sp, 2.5);
      this.faceTowards(dt, tgt.x, tgt.z, heavy ? 2.2 : 5);
      // Feuern
      this.fireT -= dt;
      if (this.burst > 0) {
        this.burstT -= dt;
        if (this.burstT <= 0) {
          this.burst--;
          this.burstT = heavy ? 0.22 : 0.11;
          const from = this.eye(_b).clone();
          from.y -= 0.2;
          const aim = this.leadTarget(from, heavy ? 55 : 70, _c);
          this.shootAt(from, aim, heavy ? 55 : 70, heavy ? 11 : 5, heavy ? 1.8 : 3.4, heavy ? 'bolt' : 'bullet');
          this.game.sfx.play(heavy ? 'bolt' : 'arcShot', from);
          this.game.effects.muzzle(from, 0.6);
        }
      } else if (this.fireT <= 0 && this.seesTarget) {
        this.burst = heavy ? 2 : 3;
        this.burstT = 0;
        this.fireT = heavy ? rand(1.8, 2.6) : rand(1.6, 2.4);
      }
    } else if (this.state === 'search') {
      this.flyTo(dt, this.lastSeen.x, this.lastSeen.y + 8, this.lastSeen.z, sp * 0.6);
      this.yaw += dt * 0.8;
    } else {
      this.wpT -= dt;
      if (!this.waypoint || this.wpT <= 0) {
        this.waypoint = this.wanderPoint(22);
        const g = this.game.terrain.height(this.waypoint.x, this.waypoint.z);
        this.waypoint.y = Math.max(g, this.home.y - 8) + rand(7, 12);
        this.wpT = rand(4, 8);
      }
      this.flyTo(dt, this.waypoint.x, this.waypoint.y + Math.sin(this.bob) * 0.8, this.waypoint.z, sp * 0.35, 1.5);
      if (Math.hypot(this.vel.x, this.vel.z) > 0.5) this.yaw = turnTowards(this.yaw, Math.atan2(-this.vel.x, -this.vel.z), dt * 2);
    }
    // leichte Neigung + Düsenglühen + drehende Propeller
    this.root.rotation.x = clamp(-this.vel.z * 0.03, -0.3, 0.3);
    if (this.model.fans) for (const f of this.model.fans) f.rotation.y += dt * 40;
    if (Math.random() < dt * 10) {
      const t = pick(this.model.thrusters);
      if (t.visible) this.game.effects.thrusterTrail(t.getWorldPosition(_a));
    }
  }
  syncModel() {
    super.syncModel();
    this.root.rotation.z = clamp(this.vel.x * 0.02, -0.3, 0.3);
  }
}

// =============================================================== Späher
class Snitch extends Enemy {
  constructor(g, t, s) {
    super(g, t, s);
    this.alarmCd = 0;
    this.charge = 0;
    this.circle = rand(0, TAU);
    this.pos.y = Math.max(this.pos.y, this.game.terrain.height(this.pos.x, this.pos.z) + 18);
  }
  think(dt) {
    const p = this.game.player;
    this.alarmCd -= dt;
    this.model.ring.rotation[this.model.glb ? 'y' : 'z'] += dt * (this.charge > 0 ? 12 : 2);
    this.model.beacon.visible = this.charge > 0 ? Math.sin(this.t * 20) > 0 : Math.sin(this.t * 3) > 0.6;
    if (this.state === 'attack' && p.alive) {
      if (this.seesTarget && this.alarmCd <= 0) {
        if (this.charge === 0) {
          this.game.sfx.play('scan', this.pos);
          this.game.hud.toast('Ein Späher hat dich entdeckt!', 'bad');
        }
        this.charge += dt;
        if (this.charge > 2.2) {
          this.charge = 0;
          this.alarmCd = 40;
          this.game.sfx.play('alarm', this.pos, { maxDist: 200, vol: 1.2 });
          this.game.callReinforcements(p.pos, this);
        }
      } else if (!this.seesTarget) this.charge = Math.max(0, this.charge - dt);
      this.circle += dt * 0.3;
      const tx = p.pos.x + Math.cos(this.circle) * 28, tz = p.pos.z + Math.sin(this.circle) * 28;
      this.flyTo(dt, tx, this.game.terrain.height(tx, tz) + 20, tz, this.def.speed);
      this.faceTowards(dt, p.pos.x, p.pos.z, 3);
    } else {
      this.charge = 0;
      this.circle += dt * 0.12;
      const r = this.roam ? 70 : 35;
      const tx = this.home.x + Math.cos(this.circle) * r, tz = this.home.z + Math.sin(this.circle) * r;
      this.flyTo(dt, tx, this.game.terrain.height(tx, tz) + 20, tz, this.def.speed * 0.6, 1.5);
      this.yaw += dt * 0.6;
    }
  }
}

// =============================================================== Raketier
class Rocketeer extends Enemy {
  constructor(g, t, s) {
    super(g, t, s);
    this.volleyT = rand(3, 5);
    this.lockT = -1;
    this.salvo = 0;
    this.salvoT = 0;
    this.orbit = rand(0, TAU);
    this.laser = game_laser(this.game, 0xff3020);
  }
  think(dt) {
    const p = this.game.player;
    const sp = this.def.speed;
    if (this.model.fans) for (const f of this.model.fans) f.rotation.y += dt * 30;
    this.laser.visible = false;
    if (this.state === 'attack' && p.alive) {
      this.orbit += dt * 0.12;
      const tgt = this.seesTarget ? p.pos : this.lastSeen;
      const tx = tgt.x + Math.cos(this.orbit) * 30, tz = tgt.z + Math.sin(this.orbit) * 30;
      this.flyTo(dt, tx, tgt.y + 15, tz, sp, 1.5);
      this.faceTowards(dt, tgt.x, tgt.z, 1.5);
      if (this.salvo > 0) {
        this.salvoT -= dt;
        if (this.salvoT <= 0) {
          this.salvo--;
          this.salvoT = 0.28;
          const pod = this.model.pods[this.salvo % 2];
          const from = pod.getWorldPosition(new THREE.Vector3());
          const aim = p.chest(new THREE.Vector3());
          aim.x += rand(-2.5, 2.5); aim.z += rand(-2.5, 2.5);
          const dir = aim.sub(from).normalize();
          this.game.projectiles.rocket(from, dir.multiplyScalar(30), this);
          this.game.sfx.play('rocketLaunch', from);
        }
      } else if (this.lockT >= 0) {
        this.lockT += dt;
        const from = this.eye(_a);
        this.laser.visible = true;
        setLaser(this.laser, from, p.chest(_b));
        if (this.lockT > 1.3) {
          this.lockT = -1;
          this.salvo = 4;
          this.salvoT = 0;
        }
      } else {
        this.volleyT -= dt;
        if (this.volleyT <= 0 && this.seesTarget) {
          this.volleyT = rand(5, 7);
          this.lockT = 0;
          this.game.sfx.play('lock', this.pos, { maxDist: 120 });
        }
      }
    } else if (this.state === 'search') {
      this.flyTo(dt, this.lastSeen.x, this.lastSeen.y + 15, this.lastSeen.z, sp * 0.5);
      this.yaw += dt * 0.5;
    } else {
      this.orbit += dt * 0.08;
      const tx = this.home.x + Math.cos(this.orbit) * 30, tz = this.home.z + Math.sin(this.orbit) * 30;
      this.flyTo(dt, tx, Math.max(this.home.y, this.game.terrain.height(tx, tz) + 14), tz, sp * 0.4, 1);
      this.yaw = turnTowards(this.yaw, Math.atan2(-this.vel.x, -this.vel.z), dt);
    }
    if (Math.random() < dt * 14) {
      const t = pick(this.model.thrusters);
      if (t.visible) this.game.effects.thrusterTrail(t.getWorldPosition(_a));
    }
  }
  die(source, silent) {
    this.laser.visible = false;
    super.die(source, silent);
  }
}

// =============================================================== Springer
class Leaper extends Enemy {
  constructor(g, t, s) {
    super(g, t, s);
    this.mode = 'walk';
    this.modeT = 0;
    this.leapCd = rand(2, 4);
    this.clawCd = 0;
    this.leapFrom = new THREE.Vector3();
    this.leapTo = new THREE.Vector3();
  }
  think(dt) {
    const p = this.game.player;
    const hull = this.model.hull;
    const hb = hull.userData.restY ?? 2.2;
    const legs = this.model.legs;
    this.leapCd -= dt;
    this.clawCd -= dt;
    if (this.mode === 'windup') {
      this.modeT += dt;
      hull.position.y = hb - Math.min(1, this.modeT / 0.7) * 0.9;
      this.faceTowards(dt, p.pos.x, p.pos.z, 4);
      if (this.modeT > 0.7) {
        this.mode = 'leap';
        this.modeT = 0;
        this.leapFrom.copy(this.pos);
        this.leapTo.set(p.pos.x + p.vel.x * 0.5, 0, p.pos.z + p.vel.z * 0.5);
        this.leapTo.y = this.game.col.groundAt(this.leapTo.x, this.leapTo.z, 1, p.pos.y + 1, 0.5);
        this.game.sfx.play('leap', this.pos);
      }
      return;
    }
    if (this.mode === 'leap') {
      this.modeT += dt;
      const T = 1.0;
      const k = Math.min(1, this.modeT / T);
      this.pos.lerpVectors(this.leapFrom, this.leapTo, k);
      this.pos.y += Math.sin(k * Math.PI) * 9;
      hull.position.y = hb + Math.sin(k * Math.PI) * 0.4;
      for (const l of legs) l.rotation.x = -0.5;
      if (k >= 1) {
        this.mode = 'recover';
        this.modeT = 0;
        this.game.col.pushOut(this.pos, this.def.radius, 3, 0.5);
        this.game.stomp(this.pos, 5, 38, this);
      }
      return;
    }
    if (this.mode === 'recover') {
      this.modeT += dt;
      hull.position.y = lerp(hb - 0.9, hb, Math.min(1, this.modeT / 1.3));
      for (const l of legs) l.rotation.x = 0;
      if (this.modeT > 1.4) { this.mode = 'walk'; this.leapCd = rand(2.5, 4.5); }
      return;
    }
    if (this.mode === 'claw') {
      this.modeT += dt;
      legs[0].rotation.x = -Math.sin(this.modeT * 8) * 0.8;
      legs[1].rotation.x = -Math.sin(this.modeT * 8) * 0.8;
      if (this.modeT > 0.45 && !this.clawDone) {
        this.clawDone = true;
        if (p.alive && this.pos.distanceTo(p.pos) < 4.2) p.damage(24, this.pos, 'claw');
      }
      if (this.modeT > 0.8) this.mode = 'walk';
      return;
    }
    let moved = 0;
    if (this.state === 'attack' && p.alive) {
      const d = this.pos.distanceTo(p.pos);
      if (d < 3.8 && this.clawCd <= 0) {
        this.mode = 'claw'; this.modeT = 0; this.clawCd = 1.6; this.clawDone = false;
        this.game.sfx.play('servo', this.pos);
        return;
      }
      if (d > 7 && d < 28 && this.leapCd <= 0 && this.seesTarget) {
        this.mode = 'windup'; this.modeT = 0;
        this.game.sfx.play('servo', this.pos);
        return;
      }
      const tgt = this.seesTarget ? p.pos : this.lastSeen;
      moved = this.walkTo(dt, tgt.x, tgt.z, this.def.speed, 3);
    } else if (this.state === 'search') {
      moved = this.walkTo(dt, this.lastSeen.x, this.lastSeen.z, this.def.speed * 0.5, 2);
    } else {
      this.wpT -= dt;
      if (!this.waypoint || this.wpT <= 0) { this.waypoint = this.wanderPoint(25); this.wpT = rand(6, 12); }
      if (Math.hypot(this.pos.x - this.waypoint.x, this.pos.z - this.waypoint.z) > 3) moved = this.walkTo(dt, this.waypoint.x, this.waypoint.z, 2.2, 1.5);
    }
    this.gait = (this.gait || 0) + moved * 1.2;
    for (let i = 0; i < legs.length; i++) legs[i].rotation.x = Math.sin(this.gait + (i % 2) * Math.PI) * 0.35;
    hull.position.y = hb + Math.abs(Math.sin(this.gait)) * 0.1;
  }
}

// =============================================================== Bastion
class Bastion extends Enemy {
  constructor(g, t, s) {
    super(g, t, s);
    this.cycle = 'idle';
    this.cycleT = rand(0, 1);
    this.shotT = 0;
    this.gait = 0;
    this.stompCd = 0;
  }
  think(dt) {
    const p = this.game.player;
    const M = this.model;
    this.stompCd -= dt;
    let moved = 0;
    if (this.state === 'attack' && p.alive) {
      const d = this.pos.distanceTo(p.pos);
      const tgt = this.seesTarget ? p.pos : this.lastSeen;
      const off = this.faceTowards(dt, tgt.x, tgt.z, 0.75);
      if (d < 7 && this.stompCd <= 0) {
        this.stompCd = 4;
        this.cycle = 'stomp';
        this.cycleT = 0;
      }
      if (this.cycle === 'stomp') {
        this.cycleT += dt;
        M.hull.position.y = (M.hull.userData.restY ?? 4.2) + Math.sin(Math.min(1, this.cycleT) * Math.PI) * 1.2;
        if (this.cycleT > 1) {
          this.game.stomp(this.pos, 7, 45, this);
          this.cycle = 'cool';
          this.cycleT = 0;
        }
      } else if (d > 18) moved = this.walkTo(dt, tgt.x, tgt.z, this.def.speed, 0.75);
      // Feuerzyklus: hochdrehen → Salve → abkühlen
      this.cycleT += dt;
      if (this.cycle === 'idle' || this.cycle === 'cool') {
        if (this.cycleT > (this.cycle === 'cool' ? 2.2 : 0.5) && this.seesTarget && off < 0.5) {
          this.cycle = 'spin'; this.cycleT = 0;
          this.game.sfx.play('servo', this.pos, { vol: 1.5 });
        }
      } else if (this.cycle === 'spin') {
        if (this.cycleT > 0.9) { this.cycle = 'fire'; this.cycleT = 0; }
      } else if (this.cycle === 'fire') {
        this.shotT -= dt;
        while (this.shotT <= 0) {
          this.shotT += 1 / 11;
          const g = M.guns[Math.random() < 0.5 ? 0 : 1];
          const from = g.localToWorld(_a.set(0, 0, -2.2)).clone();
          const aim = this.leadTarget(from, 110, _b);
          this.shootAt(from, aim, 110, 4.5, 2.6, 'heavy');
          this.game.sfx.play('minigun', from);
          if (Math.random() < 0.5) this.game.effects.muzzle(from, 1.2);
        }
        if (this.cycleT > 2.6) { this.cycle = 'cool'; this.cycleT = 0; }
      }
      for (const g of M.guns) g.rotation.z += (this.cycle === 'fire' || this.cycle === 'spin' ? 25 : 0) * dt;
    } else if (this.state === 'search') {
      moved = this.walkTo(dt, this.lastSeen.x, this.lastSeen.z, this.def.speed * 0.7, 0.6);
    } else {
      this.wpT -= dt;
      if (!this.waypoint || this.wpT <= 0) { this.waypoint = this.wanderPoint(30); this.wpT = rand(10, 18); }
      if (Math.hypot(this.pos.x - this.waypoint.x, this.pos.z - this.waypoint.z) > 4) moved = this.walkTo(dt, this.waypoint.x, this.waypoint.z, 1.6, 0.5);
    }
    this.gait += moved * 0.8;
    for (let i = 0; i < 4; i++) M.legs[i].rotation.x = Math.sin(this.gait + (i === 0 || i === 3 ? 0 : Math.PI)) * 0.3;
    if (moved > 0 && Math.sin(this.gait) * Math.sin(this.gait - moved * 0.8) < 0) {
      this.game.sfx.play('stomp', this.pos, { vol: 0.35 });
      const d = this.pos.distanceTo(p.pos);
      if (d < 40) this.game.effects.addShake(0.15 * (1 - d / 40));
    }
  }
}

// =============================================================== Wächter (Geschütz)
class Turret extends Enemy {
  constructor(g, t, s) {
    super(g, t, s);
    this.lockT = -1;
    this.cd = rand(1, 2);
    this.aimPoint = new THREE.Vector3();
    this.laser = game_laser(this.game, 0xff2010);
    this.scan = rand(0, TAU);
  }
  think(dt) {
    const p = this.game.player;
    const head = this.model.head;
    this.laser.visible = false;
    if (this.state === 'attack' && p.alive && this.seesTarget) {
      const pc = p.chest(_b);
      if (this.lockT < 0) {
        this.cd -= dt;
        if (this.cd <= 0) { this.lockT = 0; this.game.sfx.play('lock', this.pos, { maxDist: 120 }); }
      }
      if (this.lockT >= 0) {
        this.lockT += dt;
        // folgt dem Ziel mit Verzögerung, friert kurz vor dem Schuss ein
        if (this.lockT < 1.25) this.aimPoint.lerp(pc, Math.min(1, dt * 6));
        const from = this.eye(_a);
        this.laser.visible = true;
        setLaser(this.laser, from, this.aimPoint, this.lockT > 1.25 ? 0.06 : 0.025);
        if (this.lockT > 1.45) {
          this.lockT = -1;
          this.cd = 2.2;
          this.fire(from.clone());
        }
      } else this.aimPoint.copy(pc);
      const want = Math.atan2(-(this.aimPoint.x - this.pos.x), -(this.aimPoint.z - this.pos.z));
      this.yaw = turnTowards(this.yaw, want, dt * 4);
    } else {
      this.lockT = -1;
      this.scan += dt * 0.5;
      this.yaw = this.scan;
    }
    head.rotation.y = 0;
  }
  fire(from) {
    const dir = this.aimPoint.clone().sub(from).normalize();
    const hit = this.game.col.raycast(from, dir, from.distanceTo(this.aimPoint) + 2, 1);
    const p = this.game.player;
    const end = hit ? from.clone().addScaledVector(dir, hit.t) : this.aimPoint.clone();
    this.game.effects.tracer(from, end, 0xff6040, 0.09, 0.15);
    this.game.effects.muzzle(from, 1.5);
    this.game.sfx.play('sniper', from, { maxDist: 250 });
    const pc = p.chest(_b);
    // Treffer, wenn der Spieler noch nahe am Zielpunkt ist
    if (!hit || hit.t >= from.distanceTo(this.aimPoint) - 0.5) {
      if (pc.distanceTo(this.aimPoint) < 1.0) p.damage(32, this.pos, 'sniper');
    }
    if (hit) this.game.effects.sparks(end, new THREE.Vector3(hit.nx, hit.ny, hit.nz), 10);
  }
  eye(out) {
    return out.set(this.pos.x, this.pos.y + 1.12, this.pos.z);
  }
  die(source, silent) {
    this.laser.visible = false;
    super.die(source, silent);
  }
}

// Laserlinie (Zielerfassung)
function game_laser(game, color) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending, depthWrite: false }));
  m.visible = false;
  m.frustumCulled = false;
  game.scene.add(m);
  game.transient.push(m);
  return m;
}

function setLaser(m, from, to, w = 0.025) {
  const len = from.distanceTo(to);
  m.position.copy(from).lerp(to, 0.5);
  m.scale.set(w, w, len);
  m.lookAt(to);
}

const CLASSES = { tick: Tick, pop: Pop, wasp: Drone, hornet: Drone, snitch: Snitch, rocketeer: Rocketeer, leaper: Leaper, bastion: Bastion, turret: Turret };

export function createEnemy(game, type, spawn) {
  const C = CLASSES[type] || Enemy;
  return new C(game, type, spawn);
}
