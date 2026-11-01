// Spielfigur: Bewegung, Kamera, Waffen, Verbrauchsgüter, Schaden.
import * as THREE from 'three';
import { Humanoid } from './humanoid.js';
import { Gun } from './weapons.js';
import { ITEMS } from './items.js';
import { Inventory } from './inventory.js';
import { clamp, lerp, rand, angleDiff } from './util.js';

const RADIUS = 0.38;
const STEP = 0.45;
const GRAVITY = 17;
const _v = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _dir = new THREE.Vector3();

export class Player {
  constructor(game) {
    this.game = game;
    this.model = new Humanoid();
    game.scene.add(this.model.root);
    this.pos = new THREE.Vector3();
    this.vel = new THREE.Vector3();
    this.yaw = 0;
    this.camYaw = 0;
    this.camPitch = -0.1;
    this.camDist = 3.2;
    this.fov = 70;
    this.inv = new Inventory(18);
    this.safe = new Inventory(1);
    this.guns = [null, null];
    this.active = 0;
    this.shield = null;
    this.reset();
  }

  reset() {
    this.hp = 100;
    this.maxHp = 100;
    this.stamina = 100;
    this.staminaDelay = 0;
    this.alive = true;
    this.onGround = true;
    this.crouch = false;
    this.sprinting = false;
    this.aiming = false;
    this.action = null;
    this.rollT = 0;
    this.rollDir = new THREE.Vector3();
    this.switchT = 0;
    this.lastShot = 10;
    this.stepT = 0;
    this.toxicT = 0;
    this.latched = [];
    this.meleeCd = 0;
    this.airTime = 0;
    this.recoilPitch = 0;
    this.inWater = 0;
    this.vel.set(0, 0, 0);
    this.hurtFlash = 0;
    this.lastDamageFrom = null;
    this.swimming = false;
  }

  get gun() { return this.guns[this.active]; }
  get height() { return this.crouch ? 1.2 : 1.8; }
  get eye() { return _v.set(this.pos.x, this.pos.y + (this.crouch ? 1.15 : 1.55), this.pos.z); }

  chest(out = new THREE.Vector3()) {
    return out.set(this.pos.x, this.pos.y + (this.crouch ? 0.8 : 1.2), this.pos.z);
  }

  loadout(data) {
    this.inv = new Inventory(data.backpackSize, data.backpack);
    this.safe = new Inventory(1, data.safe);
    this.guns = data.weapons.map((w) => (w ? new Gun(w.id, w.mag) : null));
    this.active = this.guns[0] ? 0 : this.guns[1] ? 1 : 0;
    this.shield = data.shield ? { id: data.shield.id, charge: data.shield.charge ?? ITEMS[data.shield.id].cap } : null;
    this.model.setGun(this.gun ? this.gun.id : null);
  }

  exportLoadout() {
    return {
      weapons: this.guns.map((g) => (g ? { id: g.id, mag: g.mag } : null)),
      shield: this.shield ? { id: this.shield.id, charge: this.shield.charge } : null,
      backpack: this.inv.toJSON(),
      safe: this.safe.toJSON(),
    };
  }

  spawn(x, z) {
    const g = this.game.col.groundAt(x, z, RADIUS, 999, 0);
    this.pos.set(x, g + 0.05, z);
    this.vel.set(0, 0, 0);
    // Blick zur Kartenmitte
    this.camYaw = Math.atan2(x, z);
    this.yaw = this.camYaw;
    this.camPitch = -0.08;
  }

  forward(out) {
    return out.set(-Math.sin(this.camYaw) * Math.cos(this.camPitch), Math.sin(this.camPitch), -Math.cos(this.camYaw) * Math.cos(this.camPitch));
  }

  update(dt) {
    const game = this.game;
    const input = game.input;
    const uiOpen = game.ui.modalOpen;
    this.lastShot += dt;
    this.meleeCd -= dt;
    this.hurtFlash = Math.max(0, this.hurtFlash - dt);

    if (!this.alive) {
      this.model.root.visible = true;
      return;
    }

    // --- Blick
    if (!uiOpen) {
      const zoom = this.aiming && this.gun ? this.gun.def.zoom : 1;
      const sens = 0.0022 * input.sensitivity * zoom;
      this.camYaw -= input.mouseDX * sens;
      this.camPitch -= input.mouseDY * sens;
      this.camPitch = clamp(this.camPitch, -1.3, 1.25);
    }
    // Rückstoß erholt sich
    if (this.recoilPitch > 0) {
      const r = Math.min(this.recoilPitch, dt * 2.2);
      this.camPitch -= r * 0.6;
      this.recoilPitch -= r;
    }

    const can = !uiOpen;
    const gun = this.gun;
    this.aiming = can && input.rmb && this.rollT <= 0 && !!gun;

    // --- Aktionen (Heilen etc.)
    if (this.action) {
      this.action.t += dt;
      if (this.action.t >= this.action.dur) this.finishAction();
      else if (can && input.lmbPressed) this.cancelAction();
    }

    // --- Tasten
    if (can) {
      if (input.hit('KeyC')) this.crouch = !this.crouch;
      if (input.hit('Digit1')) this.switchTo(0);
      if (input.hit('Digit2')) this.switchTo(1);
      if (input.wheel) this.switchTo(this.active === 0 ? 1 : 0);
      if (input.hit('KeyR') && gun) {
        if (gun.startReload(this.inv)) game.sfx.play('reload', this.pos, { dur: gun.def.reload });
      }
      if (input.hit('Digit3')) this.useItem('bandage');
      if (input.hit('Digit4')) this.useItem('medkit');
      if (input.hit('Digit5')) this.useItem('shieldcell');
      if (input.hit('KeyG') || input.hit('Digit6')) this.throwGrenade();
      if (input.hit('KeyV')) this.melee();
      if (input.hit('KeyQ')) this.startRoll();
    }

    // --- Bewegung
    let mx = 0, mz = 0;
    if (can) {
      if (input.down('KeyW')) mz -= 1;
      if (input.down('KeyS')) mz += 1;
      if (input.down('KeyA')) mx -= 1;
      if (input.down('KeyD')) mx += 1;
    }
    const cy = Math.cos(this.camYaw), sy = Math.sin(this.camYaw);
    // Kamera-Vorwärts = (-sin, -cos), rechts = (cos, -sin)
    let wx = mx * cy + mz * sy;
    let wz = -mx * sy + mz * cy;
    const wl = Math.hypot(wx, wz);
    if (wl > 0) { wx /= wl; wz /= wl; }

    const wantSprint = can && input.down('ShiftLeft') && mz < 0 && !this.aiming && !this.action && this.stamina > 1;
    this.sprinting = wantSprint && this.onGround;
    if (this.sprinting) this.crouch = false;
    let speed = this.crouch ? 2.5 : this.sprinting ? 7.2 : 4.7;
    if (this.aiming) speed = Math.min(speed, 3.0);
    if (this.action) speed *= 0.5;
    if (gun && gun.def.slow) speed *= gun.def.slow;
    if (this.shield && ITEMS[this.shield.id].slow) speed *= ITEMS[this.shield.id].slow;
    if (this.inWater > 0.4) speed *= 0.62;
    if (this.swimming) { speed = 2.6; this.sprinting = false; this.crouch = false; }
    if (this.latched.length) speed *= 0.75;

    if (this.rollT > 0) {
      this.rollT -= dt;
      const k = 9.5 * (0.4 + this.rollT / 0.55);
      this.vel.x = this.rollDir.x * k;
      this.vel.z = this.rollDir.z * k;
    } else {
      const accel = this.onGround ? 14 : 3;
      const tx = wx * speed, tz = wz * speed;
      this.vel.x = lerp(this.vel.x, tx, Math.min(1, accel * dt));
      this.vel.z = lerp(this.vel.z, tz, Math.min(1, accel * dt));
    }

    if (this.sprinting) {
      this.stamina -= 13 * dt;
      this.staminaDelay = 0.8;
    } else {
      this.staminaDelay -= dt;
      if (this.staminaDelay <= 0) this.stamina = Math.min(100, this.stamina + 22 * dt);
    }

    if (can && input.hit('Space') && this.onGround && this.stamina > 8 && this.rollT <= 0 && !this.swimming) {
      this.vel.y = this.crouch ? 4 : 5.6;
      this.onGround = false;
      this.stamina -= 8;
      this.staminaDelay = 0.6;
      this.crouch = false;
    }

    this.vel.y -= GRAVITY * dt;
    this.integrate(dt);

    // --- Körperausrichtung
    const hspeed = Math.hypot(this.vel.x, this.vel.z);
    if (this.rollT > 0) this.yaw = Math.atan2(-this.rollDir.x, -this.rollDir.z);
    else if (this.aiming || this.lastShot < 1.2 || (gun && gun.reloading) || this.action) this.yaw = this.camYaw;
    else if (hspeed > 0.5) {
      const target = Math.atan2(-this.vel.x, -this.vel.z);
      this.yaw += angleDiff(this.yaw, target) * Math.min(1, dt * 10);
    }

    // --- Waffe
    this.switchT -= dt;
    if (gun) {
      if (gun.updateReload(dt, this.inv) && gun.def.perShell) game.sfx.play('shell', this.pos);
      const ready = !this.sprinting && this.rollT <= 0 && !this.action && this.switchT <= 0 && !this.swimming;
      const held = can && input.lmb && ready;
      const pressed = can && input.lmbPressed && ready;
      if (held && gun.reloading && gun.def.perShell && gun.mag > 0) gun.cancelReload();
      if (pressed && gun.mag <= 0 && !gun.reloading) {
        game.sfx.play('empty', this.pos);
        if (gun.startReload(this.inv)) game.sfx.play('reload', this.pos, { dur: gun.def.reload });
      }
      if (!gun.reloading) {
        const shots = gun.trigger(dt, held, pressed);
        if (shots) this.fire();
        if (gun.mag <= 0 && gun.burstLeft === 0 && this.lastShot < 0.3 && !gun.reloading) {
          if (gun.startReload(this.inv)) game.sfx.play('reload', this.pos, { dur: gun.def.reload });
        }
      }
    }

    // --- Ausdauer, Wasser, Gift
    const w = game.terrain.waterAt(this.pos.x, this.pos.z);
    this.inWater = w ? clamp(w.level - this.pos.y, 0, 2) : 0;
    if (w && w.toxic && this.inWater > 0.05) {
      this.toxicT += dt;
      if (this.toxicT > 0.5) {
        this.toxicT = 0;
        this.damage(4, null, 'toxic');
      }
    }

    // --- Schritte (Geräusch für ARC)
    if (this.onGround && hspeed > 1) {
      this.stepT -= dt * hspeed;
      if (this.stepT <= 0) {
        this.stepT = 2.2;
        const loud = this.sprinting ? 28 : this.crouch ? 0 : 12;
        if (loud) game.makeNoise(this.pos, loud, this);
        game.sfx.play('footstep', this.pos, { vol: this.crouch ? 0.3 : 0.6, water: this.inWater > 0.1 });
        if (this.inWater > 0.1) game.effects.splash(_v2.set(this.pos.x, w.level, this.pos.z), w.toxic);
      }
    }

    // --- Zecken, die am Spieler hängen
    for (const t of this.latched) {
      this.damage(7 * dt, t.pos, 'tick', true);
    }

    // --- Animation
    this.model.root.position.copy(this.pos);
    this.model.root.rotation.y = this.yaw;
    const pitchForGun = this.aiming || this.lastShot < 1.2 ? this.camPitch : this.camPitch * 0.5;
    this.model.animate(dt, {
      speed: hspeed, crouch: this.crouch, aim: this.aiming || this.lastShot < 0.8, pitch: pitchForGun,
      air: !this.onGround && this.airTime > 0.15, roll: this.rollT > 0 ? 1 - this.rollT / 0.55 : 0,
      action: this.action ? this.action.kind : null, sprint: this.sprinting,
    });
  }

  integrate(dt) {
    const col = this.game.col;
    const T = this.game.terrain;
    const ox = this.pos.x, oz = this.pos.z;
    const oldGround = T.height(ox, oz);
    this.pos.x += this.vel.x * dt;
    this.pos.z += this.vel.z * dt;

    // Kartenrand
    const B = this.game.map.bound;
    this.pos.x = clamp(this.pos.x, -B, B);
    this.pos.z = clamp(this.pos.z, -B, B);

    // Zu steile Hänge blockieren
    const nh = T.height(this.pos.x, this.pos.z);
    const tooSteep = nh > oldGround + 0.02 && T.slope(this.pos.x, this.pos.z) > 1.25 && nh > this.pos.y + 0.1;
    if (tooSteep) {
      this.pos.x = ox;
      this.pos.z = oz;
      this.vel.x *= 0.2;
      this.vel.z *= 0.2;
    }

    const h = this.height;
    col.pushOut(this.pos, RADIUS, h, STEP);

    // Vertikal
    this.pos.y += this.vel.y * dt;
    const ground = col.groundAt(this.pos.x, this.pos.z, RADIUS, this.pos.y - this.vel.y * dt, STEP);
    if (this.vel.y > 0) {
      const ceil = col.ceilingAt(this.pos.x, this.pos.z, RADIUS, this.pos.y + h - 0.5, this.pos.y + h + 0.3);
      if (ceil < this.pos.y + h) {
        this.pos.y = ceil - h;
        this.vel.y = 0;
      }
    }
    const wasGround = this.onGround;
    if (this.pos.y <= ground + 0.001) {
      if (!wasGround && this.airTime > 0.4) {
        this.game.sfx.play('jumpLand', this.pos, { vol: 0.6 });
        const fall = -this.vel.y;
        if (fall > 14) this.damage((fall - 14) * 6, null, 'fall');
      }
      this.pos.y = ground;
      this.vel.y = 0;
      this.onGround = true;
      this.airTime = 0;
    } else if (wasGround && this.vel.y <= 0 && this.pos.y - ground < 0.5) {
      // an Treppen/Hängen am Boden kleben
      this.pos.y = ground;
      this.vel.y = 0;
      this.airTime = 0;
    } else {
      this.onGround = false;
      this.airTime += dt;
    }
    // Schwimmen: in tiefem Wasser an der Oberfläche treiben
    const w = T.waterAt(this.pos.x, this.pos.z);
    this.swimming = false;
    if (w && w.level - 1.3 > ground && this.pos.y < w.level - 1.3) {
      this.pos.y = w.level - 1.3;
      if (this.vel.y < 0) this.vel.y = 0;
      this.onGround = true;
      this.airTime = 0;
      this.swimming = true;
    }
    // Nicht in Decken stecken bleiben (Ducken unter Hindernissen)
    if (!this.crouch) {
      const c = col.ceilingAt(this.pos.x, this.pos.z, RADIUS * 0.8, this.pos.y + 1.0, this.pos.y + 1.8);
      if (c < this.pos.y + 1.8 && this.onGround) this.crouch = true;
    }
  }

  switchTo(i) {
    if (i === this.active || !this.guns[i] || this.action) return;
    if (this.gun) this.gun.cancelReload();
    this.active = i;
    this.switchT = 0.45;
    this.model.setGun(this.gun.id);
    this.game.sfx.play('reload', this.pos, { dur: 0.3, vol: 0.6 });
  }

  fire() {
    const game = this.game;
    const gun = this.gun;
    const def = gun.def;
    const cam = game.camera;
    gun.mag--;
    this.lastShot = 0;
    const moving = Math.hypot(this.vel.x, this.vel.z) > 1;
    let spread = def.spread * (this.aiming ? 0.4 : 1) * (moving ? 1.5 : 1) * (this.crouch ? 0.8 : 1) * (this.onGround ? 1 : 2) + gun.bloom;
    spread = (spread * Math.PI) / 180;
    const fwd = this.forward(_dir);
    const camPos = cam.position;
    // Strahl beginnt auf Höhe des Spielers (nicht hinter ihm)
    const pivot = this.eye;
    const t0 = Math.max(0, (pivot.x - camPos.x) * fwd.x + (pivot.y - camPos.y) * fwd.y + (pivot.z - camPos.z) * fwd.z);
    const origin = new THREE.Vector3().copy(camPos).addScaledVector(fwd, t0);
    const muzzle = this.model.getMuzzle(new THREE.Vector3());
    const pellets = def.pellets || 1;
    const right = new THREE.Vector3(Math.cos(this.camYaw), 0, -Math.sin(this.camYaw));
    const up = new THREE.Vector3().crossVectors(right, fwd).normalize();
    for (let p = 0; p < pellets; p++) {
      const a = Math.random() * Math.PI * 2, r = Math.sqrt(Math.random()) * spread;
      const d = new THREE.Vector3().copy(fwd).addScaledVector(right, Math.cos(a) * r).addScaledVector(up, Math.sin(a) * r).normalize();
      const aimHit = game.raycastAll(origin, d, 450, this);
      const target = aimHit ? aimHit.point : new THREE.Vector3().copy(origin).addScaledVector(d, 450);
      const d2 = new THREE.Vector3().subVectors(target, muzzle);
      const len = d2.length();
      d2.divideScalar(len || 1);
      // Mündung hinter Deckung? Dann trifft der Schuss die Deckung.
      let hit = aimHit;
      if (len > 0.5) {
        const block = game.raycastAll(muzzle, d2, len - 0.2, this);
        if (block) hit = block;
      }
      const end = hit ? hit.point : target;
      if (p < 3 || Math.random() < 0.3) game.effects.tracer(muzzle, end, 0xffe0a0, pellets > 1 ? 0.025 : 0.035);
      if (hit) game.applyBulletHit(hit, def, muzzle.distanceTo(hit.point), this, d2);
    }
    game.effects.muzzle(muzzle, def.pellets ? 1.4 : 1);
    game.sfx.play('shot', this.pos, { kind: def.sound });
    game.makeNoise(this.pos, def.noise, this);
    const rk = this.aiming ? 0.6 : 1;
    this.camPitch += def.recoil * 0.011 * rk;
    this.recoilPitch += def.recoil * 0.011 * rk;
    this.camYaw += rand(-1, 1) * def.recoil * 0.004 * rk;
    gun.bloom = Math.min(4, gun.bloom + def.recoil * 0.35);
    game.effects.addShake(def.recoil * 0.03);
    game.hud.pulseCrosshair();
  }

  startRoll() {
    if (!this.onGround || this.rollT > 0 || this.stamina < 20 || this.action || this.swimming) return;
    const input = this.game.input;
    let mx = 0, mz = 0;
    if (input.down('KeyW')) mz -= 1;
    if (input.down('KeyS')) mz += 1;
    if (input.down('KeyA')) mx -= 1;
    if (input.down('KeyD')) mx += 1;
    if (!mx && !mz) mz = -1;
    const cy = Math.cos(this.camYaw), sy = Math.sin(this.camYaw);
    this.rollDir.set(mx * cy + mz * sy, 0, -mx * sy + mz * cy).normalize();
    this.rollT = 0.55;
    this.stamina -= 22;
    this.staminaDelay = 0.8;
    this.crouch = false;
    if (this.gun) this.gun.cancelReload();
    this.game.sfx.play('footstep', this.pos, { vol: 1 });
    // Zecken abschütteln (mit etwas Glück)
    if (this.latched.length && Math.random() < 0.5) this.shakeOffTicks(false);
  }

  melee() {
    if (this.meleeCd > 0 || this.action) return;
    this.meleeCd = 0.6;
    this.game.sfx.play('melee', this.pos);
    if (this.latched.length) {
      this.shakeOffTicks(true);
      return;
    }
    this.game.meleeHit(this);
  }

  shakeOffTicks(kill) {
    for (const t of this.latched.splice(0)) t.detach(kill);
  }

  useItem(id) {
    if (this.action || this.rollT > 0) return;
    const def = ITEMS[id];
    if (this.inv.count(id) <= 0) {
      this.game.hud.toast(`Kein ${def.name} im Rucksack`, 'warn');
      return;
    }
    if (def.cat === 'heal' && this.hp >= this.maxHp) {
      this.game.hud.toast('Gesundheit ist voll', 'warn');
      return;
    }
    if (def.cat === 'shieldcell') {
      if (!this.shield) { this.game.hud.toast('Kein Schild ausgerüstet', 'warn'); return; }
      if (this.shield.charge >= ITEMS[this.shield.id].cap) { this.game.hud.toast('Schild ist voll', 'warn'); return; }
    }
    if (this.gun) this.gun.cancelReload();
    this.action = { kind: def.cat === 'heal' ? 'heal' : 'shield', id, t: 0, dur: def.use };
    this.game.sfx.play(def.cat === 'heal' ? 'heal' : 'shieldCharge', this.pos);
  }

  finishAction() {
    const a = this.action;
    this.action = null;
    const def = ITEMS[a.id];
    if (this.inv.remove(a.id, 1) <= 0) return;
    if (def.cat === 'heal') {
      this.hp = Math.min(this.maxHp, this.hp + def.heal);
      this.game.hud.toast(`+${def.heal} Gesundheit`, 'good');
    } else if (def.cat === 'shieldcell' && this.shield) {
      const cap = ITEMS[this.shield.id].cap;
      this.shield.charge = Math.min(cap, this.shield.charge + def.charge);
      this.game.hud.toast(`+${def.charge} Schild`, 'good');
    }
  }

  cancelAction() {
    this.action = null;
  }

  throwGrenade() {
    if (this.action || this.rollT > 0) return;
    if (this.inv.count('grenade') <= 0) {
      this.game.hud.toast('Keine Granate im Rucksack', 'warn');
      return;
    }
    this.inv.remove('grenade', 1);
    const fwd = this.forward(new THREE.Vector3());
    const start = this.chest().add(new THREE.Vector3(fwd.x * 0.6, 0.3, fwd.z * 0.6));
    const vel = fwd.clone().multiplyScalar(19);
    vel.y += 4.5;
    vel.x += this.vel.x * 0.5;
    vel.z += this.vel.z * 0.5;
    this.game.projectiles.grenade(start, vel, this);
    this.game.sfx.play('melee', this.pos, { vol: 0.5 });
  }

  // Schaden mit Schildabsorption. silent = ohne Treffereffekt (Dauerschaden)
  damage(amount, fromPos, kind = 'hit', silent = false) {
    if (!this.alive || this.game.state !== 'raid') return;
    let absorbed = 0;
    if (this.shield && this.shield.charge > 0 && kind !== 'toxic' && kind !== 'fall') {
      const sdef = ITEMS[this.shield.id];
      absorbed = Math.min(amount * sdef.absorb, this.shield.charge);
      this.shield.charge -= absorbed;
      if (this.shield.charge <= 0.01) {
        this.shield.charge = 0;
        this.game.sfx.play('shieldBreak', null);
        this.game.hud.toast('Schild gebrochen!', 'bad');
      } else if (!silent) this.game.sfx.play('shieldHit', null);
    }
    this.hp -= amount - absorbed;
    if (!silent) {
      this.hurtFlash = 0.35;
      this.game.sfx.play('hurt', null, { vol: 0.6 });
      this.game.effects.addShake(Math.min(0.6, amount * 0.02));
    }
    if (fromPos) this.game.hud.damageFrom(fromPos);
    // Heilen wird durch Treffer nicht abgebrochen, aber verlangsamt: nichts weiter
    if (this.hp <= 0) {
      this.hp = 0;
      this.die(kind);
    }
  }

  die(kind) {
    this.alive = false;
    this.action = null;
    this.shakeOffTicks(false);
    this.model.body.rotation.z = Math.PI / 2;
    this.model.body.position.set(0.6, 0.2, 0);
    this.game.onPlayerDeath(kind);
  }

  updateCamera(dt) {
    const cam = this.game.camera;
    const aimK = this.aiming ? 1 : 0;
    const targetFov = this.aiming && this.gun ? 70 * this.gun.def.zoom : this.sprinting ? 76 : 70;
    this.fov = lerp(this.fov, targetFov, Math.min(1, dt * 12));
    if (Math.abs(cam.fov - this.fov) > 0.01) {
      cam.fov = this.fov;
      cam.updateProjectionMatrix();
      this.game.effects.setViewport(this.game.viewH);
    }
    const col = this.game.col;
    const pivotY = this.pos.y + (this.crouch ? 1.2 : 1.62) + (this.alive ? 0 : -0.8);
    // In Innenräumen (niedrige Decke) näher heranrücken
    const ceil = col.ceilingAt(this.pos.x, this.pos.z, 0.2, this.pos.y + 1.0, this.pos.y + 6);
    const indoor = ceil < this.pos.y + 4.2;
    const dist = indoor ? lerp(2.1, 1.3, aimK) : lerp(3.3, 1.6, aimK);
    this.camDist = lerp(this.camDist, dist, Math.min(1, dt * 10));
    const cp = Math.cos(this.camPitch), sp = Math.sin(this.camPitch);
    const fx = -Math.sin(this.camYaw) * cp, fy = sp, fz = -Math.cos(this.camYaw) * cp;
    const rx = Math.cos(this.camYaw), rz = -Math.sin(this.camYaw);
    // Schulterversatz nicht in die Wand schieben
    const head = new THREE.Vector3(this.pos.x, pivotY, this.pos.z);
    const side = new THREE.Vector3(rx, 0, rz);
    let shoulder = lerp(indoor ? 0.45 : 0.62, 0.5, aimK);
    const hs = col.raycast(head, side, shoulder + 0.3, 1);
    if (hs) shoulder = Math.max(0, Math.min(shoulder, hs.t - 0.3));
    const pivot = head.clone().addScaledVector(side, shoulder);
    // Kamera hinter dem Pivot, gegen Wände abgesichert. Mehrere Strahlen,
    // damit sie nicht durch Fenster und Türspalten nach draußen rutscht.
    const back = new THREE.Vector3(-fx, -fy + 0.12, -fz).normalize();
    const up = new THREE.Vector3(0, 1, 0);
    const right = new THREE.Vector3().crossVectors(back, up).normalize();
    let d = this.camDist;
    const o = new THREE.Vector3();
    for (const [a, b] of [[0, 0], [0.22, 0], [-0.22, 0], [0, 0.2], [0, -0.2]]) {
      o.copy(pivot).addScaledVector(right, a).addScaledVector(up, b);
      const hit = col.raycast(o, back, this.camDist + 0.3, 1);
      if (hit) d = Math.min(d, hit.t - 0.3);
    }
    d = Math.max(0.3, d);
    cam.position.copy(pivot).addScaledVector(back, d);
    // Kamerawackeln
    const sh = this.game.effects.shake;
    if (sh > 0) {
      cam.position.x += (Math.random() - 0.5) * sh * 0.15;
      cam.position.y += (Math.random() - 0.5) * sh * 0.15;
    }
    cam.lookAt(pivot.x + fx * 50, pivot.y + fy * 50, pivot.z + fz * 50);
    // Spielermodell ausblenden, wenn die Kamera im Kopf steckt
    this.model.root.visible = d > 0.7 || !this.alive;
  }
}
