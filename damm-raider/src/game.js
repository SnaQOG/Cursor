// Spielablauf: Hub ↔ Raid, Treffer, Explosionen, Extraktion, Speichern.
import * as THREE from 'three';
import { World } from './world.js';
import { Player } from './player.js';
import { Input } from './input.js';
import { Sfx } from './audio.js';
import { Effects } from './effects.js';
import { Hud } from './hud.js';
import { UI } from './ui.js';
import { LootSystem } from './loot.js';
import { Projectiles } from './projectiles.js';
import { createEnemy } from './enemies.js';
import { createElevators, Barrel } from './props.js';
import { Inventory } from './inventory.js';
import { ITEMS } from './items.js';
import { MAP, POIS, PLAYER_SPAWNS, RAID_TIME, CONDITIONS } from './config.js';
import { loadSave, writeSave, BACKPACK_LEVELS } from './save.js';
import { weighted, pick, rand, clamp, TAU } from './util.js';

const _v = new THREE.Vector3();
const _w = new THREE.Vector3();

function raySphere(o, d, c, r) {
  const ox = o.x - c.x, oy = o.y - c.y, oz = o.z - c.z;
  const b = ox * d.x + oy * d.y + oz * d.z;
  const cc = ox * ox + oy * oy + oz * oz - r * r;
  const disc = b * b - cc;
  if (disc < 0) return null;
  const t = -b - Math.sqrt(disc);
  if (t < 0) return cc < 0 ? 0 : null;
  return t;
}

const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));

export class Game {
  constructor(canvas) {
    this.canvas = canvas;
    this.save = loadSave();
    this.state = 'boot';
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.camera = new THREE.PerspectiveCamera(70, 1, 0.1, 800);
    this.input = new Input(canvas);
    this.sfx = new Sfx();
    this.enemies = [];
    this.transient = [];
    this.barrels = [];
    this.poiList = POIS;
    this.marker = null;
    this.raidTime = RAID_TIME;
    this.stats = { kills: 0, looted: 0 };
    this.interact = null;
    this.holdT = 0;
    this.holdTarget = null;
    this.hubT = 0;
    this.reinforcements = 0;
    this.ui = new UI(this);
    this.hud = new Hud(this);
    this.input.onLockChange = (locked) => {
      if (!locked && this.state === 'raid' && !this.ui.open && this.player.alive) this.ui.openPause();
    };
    window.addEventListener('resize', () => this.resize());
  }

  async init() {
    this.ui.bootProgress('Staumauer wird gegossen …');
    await nextFrame();
    this.world = new World(this.renderer, this.save.settings.quality);
    this.scene = this.world.scene;
    this.col = this.world.col;
    this.terrain = this.world.terrain;
    this.ui.bootProgress('ARC-Signale werden kalibriert …');
    await nextFrame();
    this.effects = new Effects(this.scene, this.camera, document.getElementById('dmgnums'));
    this.projectiles = new Projectiles(this);
    this.loot = new LootSystem(this);
    this.loot.setup(this.world.mapData);
    this.elevators = createElevators(this);
    for (const b of this.world.mapData.barrels) this.barrels.push(new Barrel(this, b.x, b.z));
    this.player = new Player(this);
    this.player.model.root.visible = false;
    this.ui.bootProgress('Karte wird gezeichnet …');
    await nextFrame();
    this.mapImage = this.renderMapImage(640);
    this.world.applyCondition('tag');
    this.applySettings();
    this.resize();
    // Shader vorkompilieren, damit der erste Raid-Frame nicht ruckelt
    this.renderer.compile(this.scene, this.camera);
    this.ui.hideBoot();
    this.toHub();
    this.last = performance.now();
    requestAnimationFrame((t) => this.loop(t));
  }

  applySettings() {
    const o = this.save.settings;
    this.input.sensitivity = o.sens;
    this.sfx.setVolume(o.volume);
    const q = o.quality;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, q === 'hoch' ? 2 : q === 'mittel' ? 1.25 : 1));
    if (this.world) {
      const shadows = q !== 'niedrig';
      this.renderer.shadowMap.enabled = shadows;
      this.world.sun.castShadow = shadows;
      const size = q === 'hoch' ? 2048 : 1024;
      const sh = this.world.sun.shadow;
      if (sh.mapSize.x !== size) {
        sh.mapSize.set(size, size);
        if (sh.map) { sh.map.dispose(); sh.map = null; }
      }
      this.camera.far = q === 'niedrig' ? 450 : 800;
      this.camera.updateProjectionMatrix();
    }
    this.resize();
  }

  resize() {
    const w = window.innerWidth, h = window.innerHeight;
    this.width = w;
    this.height = h;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.viewH = h * this.renderer.getPixelRatio();
    if (this.effects) this.effects.setViewport(this.viewH);
  }

  get conditionName() {
    return CONDITIONS.find((c) => c.id === this.condition)?.name || '';
  }

  // =========================================================== Übersichtskarte
  renderMapImage(size) {
    const T = this.terrain;
    const c = T.renderMapImage(size, (x, z, h) => {
      const w = T.waterAt(x, z);
      if (w && h < w.level) {
        const depth = Math.min(1, (w.level - h) / 6);
        return w.toxic ? [96, 120, 40] : [Math.round(60 - depth * 30), Math.round(96 - depth * 30), Math.round(110 - depth * 20)];
      }
      const e = 0.8;
      const shade = clamp(0.75 + (T.height(x - e, z - e) - T.height(x + e, z + e)) * 0.25, 0.45, 1.25);
      const k = clamp((h - 0) / 60, 0, 1);
      const base = [128 - k * 30, 132 - k * 20, 100 - k * 10];
      return base.map((v) => Math.round(clamp(v * shade, 0, 255)));
    });
    const g = c.getContext('2d');
    const toPx = (v) => ((v + MAP.HALF) / (MAP.HALF * 2)) * size;
    const k = size / (MAP.HALF * 2);
    for (const s of this.col.shapes) {
      if (s.maxY - s.minY < 2.2) continue;
      if (s.kind === 'box') {
        const w = (s.maxX - s.minX) * k, d = (s.maxZ - s.minZ) * k;
        if (w * d < 1.5) continue;
        g.fillStyle = 'rgba(214,206,186,0.9)';
        g.fillRect(toPx(s.minX), toPx(s.minZ), Math.max(1, w), Math.max(1, d));
      } else if (s.r > 2.5 && !s.tag) {
        g.fillStyle = 'rgba(190,184,170,0.85)';
        g.beginPath();
        g.arc(toPx(s.x), toPx(s.z), s.r * k, 0, TAU);
        g.fill();
      }
    }
    // Raster
    g.strokeStyle = 'rgba(20,24,26,0.18)';
    g.lineWidth = 1;
    for (let v = -MAP.HALF; v <= MAP.HALF; v += 80) {
      g.beginPath(); g.moveTo(toPx(v), 0); g.lineTo(toPx(v), size); g.stroke();
      g.beginPath(); g.moveTo(0, toPx(v)); g.lineTo(size, toPx(v)); g.stroke();
    }
    return c;
  }

  // =========================================================== Hub / Raid
  cleanupRaidObjects() {
    for (const e of this.enemies) this.scene.remove(e.root);
    this.enemies = [];
    for (const m of this.transient) this.scene.remove(m);
    this.transient = [];
    this.projectiles.clear();
    this.effects.clear();
  }

  toHub() {
    this.state = 'hub';
    this.ui.hideResults();
    this.hud.show(false);
    this.input.enabled = false;
    this.input.exitLock();
    if (this.player) {
      this.player.model.root.visible = false;
      this.cleanupRaidObjects();
      this.loot.reset(1);
    }
    this.world.applyCondition('daemmerung');
    this.sfx.setAmbient(0, 0, 0);
    this.ui.showHub();
  }

  startRaid() {
    const s = this.save;
    if (!s.loadout.weapons.some(Boolean)) {
      this.ui.hubToast('Ohne Waffe geht es nicht nach oben. Nimm die Gratis-Ausrüstung.');
      return;
    }
    this.sfx.init();
    const cond = weighted(CONDITIONS.map((c) => [c, c.weight]));
    this.condition = cond.id;
    this.world.applyCondition(cond.id);
    this.cleanupRaidObjects();
    this.loot.reset(cond.lootMul);
    for (const b of this.barrels) b.reset();
    for (const e of this.elevators) e.reset();
    this.reinforcements = 0;

    for (const sp of this.world.mapData.enemySpawns) {
      if (Math.random() > (sp.roam ? 0.75 : 0.88)) continue;
      this.enemies.push(createEnemy(this, sp.type, sp));
    }
    for (const t of this.world.mapData.turrets) this.enemies.push(createEnemy(this, 'turret', { x: t.x, y: t.y, z: t.z }));
    if (cond.id === 'sturm') {
      for (let i = 0; i < 5; i++) {
        const poi = pick(POIS);
        this.enemies.push(createEnemy(this, pick(['wasp', 'hornet', 'wasp']), { x: poi.x + rand(-30, 30), z: poi.z + rand(-30, 30), roam: true }));
      }
    }

    const p = this.player;
    p.reset();
    p.loadout({ ...s.loadout, backpackSize: BACKPACK_LEVELS[s.upgrades.backpack] });
    for (const g of p.guns) {
      if (!g) continue;
      const need = g.def.mag - g.mag;
      if (need > 0) g.mag += p.inv.remove(g.def.ammo, need);
    }
    this.startValue = p.inv.value();
    const [sx, sz] = pick(PLAYER_SPAWNS);
    p.spawn(sx + rand(-4, 4), sz + rand(-4, 4));
    p.model.root.visible = true;
    this.raidTime = RAID_TIME;
    this.stats = { kills: 0, looted: 0, start: this.raidTime };
    this.marker = null;
    this.deathT = 0;
    this.holdT = 0;
    this.holdTarget = null;
    this.state = 'raid';
    this.ui.hideHub();
    this.hud.show(true);
    this.input.reset();
    this.input.enabled = true;
    this.input.requestLock();
    this.hud.toast(`Der Damm · ${cond.name}. ${cond.desc}`, 'info');
    this.hud.toast('Finde Beute und erreiche einen Aufzug (grüne Pfeile im Kompass).', 'info');
  }

  endRaid(outcome, via = '') {
    if (this.state !== 'raid') return;
    const s = this.save;
    const p = this.player;
    this.state = 'results';
    this.input.enabled = false;
    this.input.exitLock();
    this.ui.closePanel(false);
    this.hud.show(false);
    this.sfx.setAmbient(0, 0, 0);
    const stash = new Inventory(s.stash.length, s.stash);
    const safeItems = p.safe.slots.filter(Boolean).map((sl) => ({ ...sl }));
    const bpItems = p.inv.slots.filter(Boolean).map((sl) => ({ ...sl }));
    const eqValue = p.guns.reduce((v, g) => v + (g ? ITEMS[g.id].value : 0), 0) + (p.shield ? ITEMS[p.shield.id].value : 0);
    const bpValue = p.inv.value();
    s.stats.raids++;
    if (outcome === 'extrahiert') {
      const L = p.exportLoadout();
      if (L.shield) L.shield.charge = ITEMS[L.shield.id].cap;
      s.loadout = { ...L, safe: [null] };
      s.stats.extracted++;
      const gain = Math.max(0, bpValue - this.startValue);
      s.stats.earned += gain;
      s.stats.bestHaul = Math.max(s.stats.bestHaul, gain);
    } else {
      s.loadout = { weapons: [null, null], shield: null, backpack: new Array(BACKPACK_LEVELS[s.upgrades.backpack]).fill(null), safe: [null] };
      s.stats.died++;
    }
    for (const sl of safeItems) stash.add(sl.id, sl.qty);
    s.stash = stash.toJSON();
    writeSave(s);
    const reasons = {
      gefallen: 'Die ARC haben dich erwischt. Deine Ausrüstung bleibt oben liegen.',
      verschollen: 'Die Zeit ist abgelaufen. Kein Aufzug hat dich mehr geholt.',
      aufgegeben: 'Du hast den Einsatz abgebrochen.',
    };
    this.ui.showResults({
      outcome, via, reason: reasons[outcome] || '',
      time: RAID_TIME - this.raidTime, kills: this.stats.kills, looted: this.stats.looted,
      value: outcome === 'extrahiert' ? bpValue : bpValue + eqValue,
      items: outcome === 'extrahiert' ? bpItems : [], safe: safeItems,
    });
  }

  extract(elevator) {
    this.sfx.play('elevator', null, { vol: 1 });
    this.endRaid('extrahiert', elevator.name);
  }

  onPlayerDeath(kind) {
    this.deathT = 3;
    this.ui.closePanel(false);
    const why = { toxic: 'Giftiges Wasser', fall: 'Absturz', explosion: 'Explosion' }[kind];
    this.hud.toast(why ? `Du bist gefallen (${why})` : 'Du bist gefallen', 'bad');
    this.input.exitLock();
  }

  onEnemyKilled(e, source) {
    if (source === this.player) {
      this.stats.kills++;
      const k = this.save.stats.arcKills;
      k[e.type] = (k[e.type] || 0) + 1;
      this.hud.toast(`${e.name} zerstört`, 'good');
    }
    if (this.player.latched.includes(e)) this.player.latched.splice(this.player.latched.indexOf(e), 1);
  }

  // =========================================================== Treffer
  raycastAll(origin, dir, maxDist, ignore = null) {
    let best = null;
    const wh = this.col.raycast(origin, dir, maxDist, 1);
    if (wh) best = { t: wh.t, nx: wh.nx, ny: wh.ny, nz: wh.nz, terrain: !!wh.terrain, shape: wh.shape };
    const lim = () => (best ? best.t : maxDist);
    for (const e of this.enemies) {
      if (!e.alive || e === ignore || e.mode === 'latched') continue;
      const tb = raySphere(origin, dir, e.center, e.boundR);
      if (tb == null || tb > lim()) continue;
      for (const part of e.parts) {
        if (part.destroyed || !part.obj.visible) continue;
        const pc = e.partCenter(part, _v);
        const t = raySphere(origin, dir, pc, part.r);
        if (t != null && t < lim()) best = { t, enemy: e, part };
      }
    }
    if (best) best.point = new THREE.Vector3().copy(origin).addScaledVector(dir, best.t);
    return best;
  }

  applyBulletHit(hit, def, dist, shooter, dir) {
    const fall = dist > def.range ? Math.max(0.45, 1 - (dist - def.range) / (def.range * 2)) : 1;
    const dmg = def.dmg * fall;
    if (hit.enemy) {
      const r = hit.enemy.hit(hit.part, dmg, def.pen, hit.point, dir, shooter);
      if (!r) return;
      const n = _w.copy(dir).multiplyScalar(-1);
      this.effects.sparks(hit.point, n, r.kind === 'weak' ? 10 : 6, r.kind === 'weak' ? [1, 0.8, 0.3] : r.kind === 'armor' ? [0.9, 0.9, 1] : [1, 0.6, 0.3], 5);
      if (r.kind !== 'armor') this.effects.blood(hit.point);
      this.hud.hitmarker(r.killed ? 'kill' : r.kind);
      this.effects.number(hit.point, r.dealt, r.kind);
      this.sfx.play(r.kind === 'weak' ? 'hitWeak' : r.kind === 'armor' ? 'hitArmor' : 'hit', null, { vol: 0.8 });
      if (r.killed) this.sfx.play('kill', null);
      return;
    }
    const n = _w.set(hit.nx || 0, hit.ny || 1, hit.nz || 0);
    const barrel = hit.shape && hit.shape.tag && hit.shape.tag.barrel;
    if (barrel) barrel.hit(dmg);
    const w = this.terrain.waterAt(hit.point.x, hit.point.z);
    if (w && hit.point.y < w.level + 0.05 && hit.terrain) {
      this.effects.splash(_v.set(hit.point.x, w.level, hit.point.z), w.toxic);
    } else {
      this.effects.sparks(hit.point, n, 4, [1, 0.75, 0.4], 4);
      this.effects.dust(hit.point, n, hit.terrain ? [0.45, 0.4, 0.3] : [0.6, 0.58, 0.54], 3);
    }
    this.sfx.play(Math.random() < 0.15 ? 'ricochet' : 'impact', hit.point, { vol: 0.5, maxDist: 40 });
  }

  meleeHit(player) {
    const fwd = player.forward(_v).setY(0).normalize();
    let hitAny = false;
    for (const e of this.enemies) {
      if (!e.alive) continue;
      const c = e.center;
      const dx = c.x - player.pos.x, dz = c.z - player.pos.z;
      const d = Math.hypot(dx, dz) - e.def.radius;
      if (d > 1.8 || Math.abs(c.y - player.pos.y - 1) > 2.5) continue;
      if ((dx * fwd.x + dz * fwd.z) / (Math.hypot(dx, dz) || 1) < 0.2) continue;
      const r = e.hit({ mult: 1, armor: 0.2 }, 40, 0.3, c, fwd, player);
      if (r) {
        this.hud.hitmarker(r.killed ? 'kill' : 'normal');
        this.effects.number(c, r.dealt, 'normal');
        this.effects.sparks(c, _w.set(0, 1, 0), 8);
        hitAny = true;
      }
    }
    if (hitAny) this.sfx.play('hitArmor', null);
  }

  makeNoise(pos, radius, source) {
    for (const e of this.enemies) {
      if (!e.alive) continue;
      if (e.pos.distanceToSquared(pos) < radius * radius) e.hear(pos, radius > 60);
    }
    void source;
  }

  explode(pos, radius, dmg, source) {
    this.effects.explosion(pos, radius);
    this.sfx.play('explosion', pos, { big: radius > 6 });
    const p = this.player;
    const dp = p.chest(_v).distanceTo(pos);
    this.effects.addShake(clamp(1.2 - dp / 40, 0, 1));
    if (p.alive && dp < radius) {
      const up = _w.copy(pos);
      up.y += 0.3;
      if (this.col.lineOfSight(up, p.chest(new THREE.Vector3()))) p.damage(dmg * Math.pow(1 - dp / radius, 0.6), pos, 'explosion');
    }
    for (const e of this.enemies) {
      if (!e.alive || e === source) continue;
      const d = e.center.distanceTo(pos) - e.def.radius;
      if (d >= radius) continue;
      const k = Math.pow(1 - Math.max(0, d) / radius, 0.6);
      const r = e.hit({ mult: 1, armor: 0.3 }, dmg * k * 1.2, 0.5, pos, _w.set(0, 1, 0), source === p ? p : source);
      if (r && source === p) {
        this.hud.hitmarker(r.killed ? 'kill' : 'normal');
        this.effects.number(e.center, r.dealt, 'normal');
      }
    }
    for (const b of this.barrels) {
      if (b.alive && b.pos.distanceTo(pos) < radius) b.hit(999);
    }
    this.makeNoise(pos, 90, source);
  }

  stomp(pos, radius, dmg, source) {
    this.sfx.play('stomp', pos, { vol: 1.2 });
    for (let i = 0; i < 16; i++) {
      const a = (i / 16) * TAU;
      this.effects.dust(_v.set(pos.x + Math.cos(a) * radius * 0.6, pos.y + 0.2, pos.z + Math.sin(a) * radius * 0.6), _w.set(Math.cos(a), 0.4, Math.sin(a)), [0.5, 0.46, 0.38], 2);
    }
    const p = this.player;
    const d = Math.hypot(p.pos.x - pos.x, p.pos.z - pos.z);
    this.effects.addShake(clamp(1 - d / 30, 0, 1));
    if (p.alive && d < radius && Math.abs(p.pos.y - pos.y) < 2.5) p.damage(dmg * (1 - (d / radius) * 0.6), pos, 'stomp');
    void source;
  }

  spawnEnemy(type, spawn) {
    const e = createEnemy(this, type, spawn);
    this.enemies.push(e);
    return e;
  }

  callReinforcements(target) {
    if (this.reinforcements >= 14) return;
    const n = this.condition === 'sturm' ? 4 : 3;
    const a = rand(0, TAU);
    for (let i = 0; i < n; i++) {
      const type = i === 0 && Math.random() < 0.55 ? 'hornet' : 'wasp';
      const x = clamp(target.x + Math.cos(a) * 95 + rand(-10, 10), -MAP.BOUND, MAP.BOUND);
      const z = clamp(target.z + Math.sin(a) * 95 + rand(-10, 10), -MAP.BOUND, MAP.BOUND);
      const e = this.spawnEnemy(type, { x, z, y: this.terrain.height(x, z) + 25 });
      e.hear(target, true);
      e.searchT = 25;
      this.reinforcements++;
    }
    this.hud.toast('ARC-Verstärkung im Anflug!', 'bad');
  }

  // =========================================================== Schleife
  loop(now) {
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    try {
      if (this.state === 'raid') this.updateRaid(dt);
      else this.updateMenu(dt);
      this.renderer.render(this.scene, this.camera);
    } catch (err) {
      console.error(err);
    }
    this.input.endFrame();
    requestAnimationFrame((t) => this.loop(t));
  }

  // Für automatisierte Tests: Simulation ohne Rendern vorspulen
  debugStep(seconds, dt = 1 / 30) {
    for (let t = 0; t < seconds && this.state === 'raid'; t += dt) {
      this.updateRaid(dt);
      this.input.endFrame();
    }
  }

  updateMenu(dt) {
    this.hubT += dt;
    const a = this.hubT * 0.035 + 0.6;
    const cx = 0, cz = -30;
    this.camera.position.set(cx + Math.cos(a) * 200, 85 + Math.sin(this.hubT * 0.1) * 8, cz + Math.sin(a) * 200);
    this.camera.lookAt(cx, 18, cz);
    if (this.camera.fov !== 60) { this.camera.fov = 60; this.camera.updateProjectionMatrix(); }
    this.world.update(dt, _v.set(cx, 20, cz), null);
    this.effects.update(dt, this.width, this.height);
  }

  updateRaid(dt) {
    const p = this.player;
    const input = this.input;
    const ui = this.ui;

    // Fenster
    if (p.alive) {
      if (input.hit('Tab') || input.hit('KeyI')) {
        if (ui.open === 'inv' || ui.open === 'loot') ui.closePanel();
        else if (!ui.open) ui.openInventory();
      }
      if (input.hit('KeyM')) {
        if (ui.open === 'map') ui.closePanel();
        else if (!ui.open) ui.openMap();
      }
      if (input.hit('Escape') && ui.open) ui.closePanel();
      if (ui.open === 'loot' && input.hit('KeyE')) { ui.closePanel(); input.pressed.delete('KeyE'); }
      if (ui.open === 'loot' && ui.lootTarget && ui.lootTarget.pos.distanceTo(p.pos) > 4) ui.closePanel();
      if (ui.open === 'map' && Math.random() < dt * 4) ui.drawMap();
    }

    this.raidTime -= dt;
    if (this.raidTime <= 0 && p.alive) {
      this.raidTime = 0;
      this.endRaid('verschollen');
      return;
    }

    p.update(dt);
    this.updateInteraction(dt);

    for (const e of this.enemies) {
      const d2 = e.pos.distanceToSquared(p.pos);
      e.root.visible = d2 < 300 * 300 && (e.alive || e.type !== 'pop');
      if (d2 > 240 * 240 && e.state !== 'attack' && e.alive) {
        e.farT += dt;
        if (e.farT < 0.5) continue;
        e.update(e.farT);
        e.farT = 0;
        continue;
      }
      e.update(dt);
    }
    this.projectiles.update(dt);
    for (const b of this.barrels) b.update(dt);
    for (const el of this.elevators) el.update(dt);
    if (this.state !== 'raid') return;

    this.effects.update(dt, this.width, this.height);
    p.updateCamera(dt);
    this.world.update(dt, p.pos, this.sfx);

    // Audio
    this.sfx.setListener(this.camera.position.x, this.camera.position.y, this.camera.position.z, p.camYaw);
    let drone = 0;
    for (const e of this.enemies) {
      if (!e.alive || !e.def.air) continue;
      const d = e.pos.distanceTo(p.pos);
      if (d < 45) drone = Math.max(drone, (1 - d / 45) * (e.type === 'rocketeer' ? 0.2 : 0.12));
    }
    this.sfx.setAmbient(this.condition === 'sturm' ? 0.12 : 0.07, this.condition === 'sturm' ? 0.12 : 0, drone);

    this.hud.update(dt);

    if (!p.alive) {
      this.deathT -= dt;
      if (this.deathT <= 0) this.endRaid('gefallen');
    }
  }

  updateInteraction(dt) {
    const p = this.player;
    const input = this.input;
    this.interact = null;
    if (!p.alive || this.ui.open || p.rollT > 0) { this.holdT = 0; return; }
    const fwd = p.forward(_v);
    let target = null, label = '', hold = 0;
    for (const e of this.elevators) {
      if (e.state === 'idle' && e.consolePos.distanceTo(p.pos) < 3) {
        target = e; label = `${e.name} rufen`; hold = 0.6;
      }
    }
    if (!target) {
      const c = this.loot.nearest(p.chest(_w), fwd);
      if (c) {
        target = c;
        hold = this.loot.searchTime(c);
        label = hold > 0 ? `${c.name} durchsuchen` : `${c.name} öffnen`;
        if (c.searched && c.items && c.items.isEmpty()) label = `${c.name} (leer)`;
      }
    }
    if (target !== this.holdTarget) { this.holdT = 0; this.holdTarget = target; }
    if (!target) return;
    const pressed = input.hit('KeyE');
    const down = input.down('KeyE');
    if (hold > 0 && down) {
      if (this.holdT === 0) this.sfx.play('search', p.pos, { vol: 0.6 });
      this.holdT += dt;
      if (this.holdT >= hold) {
        this.holdT = 0;
        this.activate(target);
      }
    } else if (hold === 0 && pressed) this.activate(target);
    else if (!down) this.holdT = 0;
    this.interact = { label: `[E] ${label}`, hold: hold > 0, progress: hold > 0 ? this.holdT / hold : 0 };
  }

  activate(target) {
    if (target.interact) { target.interact(); return; }
    this.loot.generate(target);
    if (!target.searched) {
      this.loot.markSearched(target);
      this.sfx.play('open', target.pos);
    }
    this.ui.openLoot(target);
  }
}
