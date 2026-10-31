// HUD während eines Raids (DOM + ein kleiner Kompass-Canvas).
import { ITEMS } from './items.js';
import { formatTime } from './util.js';
import { POIS, EXTRACTS } from './config.js';

const $ = (id) => document.getElementById(id);

export class Hud {
  constructor(game) {
    this.game = game;
    this.root = $('hud');
    this.compass = $('compass');
    this.cctx = this.compass.getContext('2d');
    this.cache = {};
    this.toasts = $('toasts');
    this.hitT = 0;
    this.pulse = 0;
    this.dirs = [];
    this.alertT = 0;
    for (let i = 0; i < 6; i++) {
      const el = document.createElement('div');
      el.className = 'dmgdir';
      el.hidden = true;
      $('dmgdirs').appendChild(el);
      this.dirs.push({ el, t: 0, pos: null });
    }
    this.dirIdx = 0;
  }

  set(id, value, prop = 'textContent') {
    const key = id + prop;
    if (this.cache[key] === value) return;
    this.cache[key] = value;
    const el = $(id);
    if (prop === 'width') el.style.width = value;
    else if (prop === 'hidden') el.hidden = value;
    else if (prop === 'class') el.className = value;
    else el[prop] = value;
  }

  show(v) { this.root.hidden = !v; }

  toast(text, kind = 'info') {
    const el = document.createElement('div');
    el.className = `toast ${kind}`;
    el.textContent = text;
    this.toasts.prepend(el);
    while (this.toasts.children.length > 5) this.toasts.lastChild.remove();
    setTimeout(() => el.classList.add('fade'), 3200);
    setTimeout(() => el.remove(), 3800);
  }

  lootToast(id, qty) {
    const d = ITEMS[id];
    this.toast(`+ ${qty > 1 ? qty + '× ' : ''}${d.name}`, `loot r${d.rarity}`);
  }

  hitmarker(kind) {
    const el = $('hitmarker');
    el.className = 'show ' + kind;
    this.hitT = kind === 'kill' ? 0.35 : 0.15;
  }

  pulseCrosshair() { this.pulse = 1; }

  damageFrom(pos) {
    const D = this.dirs[this.dirIdx];
    this.dirIdx = (this.dirIdx + 1) % this.dirs.length;
    D.pos = pos.clone ? pos.clone() : pos;
    D.t = 1.2;
    D.el.hidden = false;
  }

  arcAlert(enemy) {
    if (this.alertT > 0) return;
    this.alertT = 6;
    this.toast(`${enemy.name} hat dich entdeckt`, 'bad');
  }

  setExtractProgress(k, remain) {
    const bar = $('extract');
    if (k <= 0) { bar.hidden = true; return; }
    bar.hidden = false;
    $('extract-fill').style.width = `${Math.min(100, k * 100)}%`;
    $('extract-text').textContent = `Extraktion in ${Math.ceil(remain)} s – bleib im Aufzug`;
  }

  update(dt) {
    const g = this.game;
    const p = g.player;
    this.alertT -= dt;

    // Vitalwerte
    this.set('hp-fill', `${(p.hp / p.maxHp) * 100}%`, 'width');
    this.set('hp-num', `${Math.ceil(p.hp)}`);
    if (p.shield) {
      const cap = ITEMS[p.shield.id].cap;
      this.set('sh-row', false, 'hidden');
      this.set('sh-fill', `${(p.shield.charge / cap) * 100}%`, 'width');
      this.set('sh-num', `${Math.ceil(p.shield.charge)}`);
    } else this.set('sh-row', true, 'hidden');
    this.set('st-fill', `${p.stamina}%`, 'width');

    // Waffe
    const gun = p.gun;
    if (gun) {
      this.set('w-name', gun.name);
      this.set('w-mag', gun.reloading ? '…' : `${gun.mag}`);
      this.set('w-res', `${p.inv.count(gun.def.ammo)}`);
      this.set('w-ammo', ITEMS[gun.def.ammo].name);
    } else {
      this.set('w-name', 'Keine Waffe');
      this.set('w-mag', '–');
      this.set('w-res', '');
      this.set('w-ammo', '');
    }
    for (let i = 0; i < 2; i++) {
      const w = p.guns[i];
      this.set(`slot${i + 1}`, w ? w.name : '—');
      this.set(`slot${i + 1}`, `wslot${i === p.active ? ' on' : ''}${w ? '' : ' empty'}`, 'class');
    }
    this.set('q-bandage', `${p.inv.count('bandage')}`);
    this.set('q-medkit', `${p.inv.count('medkit')}`);
    this.set('q-shieldcell', `${p.inv.count('shieldcell')}`);
    this.set('q-grenade', `${p.inv.count('grenade')}`);

    // Timer
    this.set('timer', formatTime(g.raidTime));
    this.set('timer', g.raidTime < 120 ? 'timer low' : 'timer', 'class');

    // Aktion
    if (p.action) {
      this.set('action', false, 'hidden');
      this.set('action-text', `${ITEMS[p.action.id].name} wird angewendet…`);
      $('action-fill').style.width = `${(p.action.t / p.action.dur) * 100}%`;
    } else this.set('action', true, 'hidden');

    // Zecken-Warnung
    this.set('tickwarn', p.latched.length === 0, 'hidden');

    // Interaktion
    const it = g.interact;
    if (it && !g.ui.modalOpen) {
      this.set('prompt', false, 'hidden');
      this.set('prompt-text', it.label);
      const ring = $('prompt-ring');
      ring.style.setProperty('--p', `${Math.round(it.progress * 100)}`);
      this.set('prompt-ring', it.hold ? 'ring' : 'ring hidden', 'class');
    } else this.set('prompt', true, 'hidden');

    // Fadenkreuz
    this.pulse = Math.max(0, this.pulse - dt * 8);
    let spread = 10;
    if (gun) {
      const moving = Math.hypot(p.vel.x, p.vel.z) > 1;
      spread = 6 + (gun.def.spread * (p.aiming ? 0.4 : 1) * (moving ? 1.5 : 1) + gun.bloom) * 9 + this.pulse * 6;
    }
    $('crosshair').style.setProperty('--s', `${spread.toFixed(1)}px`);
    this.set('crosshair', p.sprinting || !gun ? 'dim' : '', 'class');
    if (this.hitT > 0) {
      this.hitT -= dt;
      if (this.hitT <= 0) $('hitmarker').className = '';
    }

    // Vignette
    const low = 1 - p.hp / p.maxHp;
    const vig = Math.max(p.hurtFlash * 2, low > 0.6 ? (low - 0.6) * 2 : 0);
    $('vignette').style.opacity = Math.min(0.9, vig).toFixed(2);
    const w = g.terrain.waterAt(p.pos.x, p.pos.z);
    $('toxic').style.opacity = w && w.toxic && p.inWater > 0.05 ? '0.55' : '0';

    // Schadensrichtung
    for (const D of this.dirs) {
      if (D.el.hidden) continue;
      D.t -= dt;
      if (D.t <= 0) { D.el.hidden = true; continue; }
      const ang = Math.atan2(D.pos.x - p.pos.x, -(D.pos.z - p.pos.z)) + p.camYaw;
      D.el.style.transform = `rotate(${ang}rad)`;
      D.el.style.opacity = Math.min(1, D.t).toFixed(2);
    }

    this.drawCompass();
  }

  drawCompass() {
    const c = this.compass;
    const ctx = this.cctx;
    const W = c.width, H = c.height;
    const p = this.game.player;
    ctx.clearRect(0, 0, W, H);
    const fov = Math.PI * 0.75;
    // Kompassrichtung: Norden = -Z. Heading in Grad (0 = N, 90 = O)
    const heading = -p.camYaw;
    const toX = (bearing) => {
      let d = bearing - heading;
      while (d > Math.PI) d -= Math.PI * 2;
      while (d < -Math.PI) d += Math.PI * 2;
      return { x: W / 2 + (d / (fov / 2)) * (W / 2), vis: Math.abs(d) < fov / 2 };
    };
    ctx.font = '600 13px "Barlow Semi Condensed", sans-serif';
    ctx.textAlign = 'center';
    const labels = { 0: 'N', 45: 'NO', 90: 'O', 135: 'SO', 180: 'S', 225: 'SW', 270: 'W', 315: 'NW' };
    for (let deg = 0; deg < 360; deg += 15) {
      const { x, vis } = toX((deg * Math.PI) / 180);
      if (!vis) continue;
      const fade = 1 - Math.abs(x - W / 2) / (W / 2);
      ctx.globalAlpha = 0.25 + fade * 0.75;
      ctx.fillStyle = '#efe6d2';
      if (labels[deg] != null) {
        ctx.fillStyle = deg === 0 ? '#ff8a3a' : '#efe6d2';
        ctx.fillText(labels[deg], x, 18);
      } else {
        ctx.fillRect(x - 0.5, 8, 1, 7);
      }
    }
    // Orte & Aufzüge
    // Nur die Markierung, die der Bildmitte am nächsten ist, bekommt eine Beschriftung
    let focus = null;
    const mark = (x, z, color, text, shape) => {
      const b = Math.atan2(x - p.pos.x, -(z - p.pos.z));
      const r = toX(b);
      if (!r.vis) return;
      const dist = Math.hypot(x - p.pos.x, z - p.pos.z);
      const fade = 1 - Math.abs(r.x - W / 2) / (W / 2);
      ctx.globalAlpha = 0.35 + fade * 0.65;
      ctx.fillStyle = color;
      ctx.beginPath();
      if (shape === 'up') { ctx.moveTo(r.x, 24); ctx.lineTo(r.x + 5, 32); ctx.lineTo(r.x - 5, 32); }
      else { ctx.moveTo(r.x, 24); ctx.lineTo(r.x + 4, 28); ctx.lineTo(r.x, 32); ctx.lineTo(r.x - 4, 28); }
      ctx.fill();
      const off = Math.abs(r.x - W / 2);
      if (off < 70 && (!focus || off < focus.off)) focus = { off, x: r.x, color, label: `${text} · ${Math.round(dist)} m` };
    };
    for (const poi of POIS) {
      if (Math.hypot(poi.x - p.pos.x, poi.z - p.pos.z) < 25) continue;
      mark(poi.x, poi.z, '#d9d2bf', poi.name, 'dia');
    }
    for (const e of this.game.elevators) {
      mark(e.base.x, e.base.z, e.state === 'open' ? '#5dff9a' : '#6fd3c0', e.name, 'up');
    }
    if (this.game.marker) mark(this.game.marker.x, this.game.marker.z, '#ffd23a', 'Markierung', 'dia');
    ctx.globalAlpha = 1;
    if (focus) {
      ctx.font = '600 12px "Barlow Semi Condensed", sans-serif';
      ctx.fillStyle = focus.color;
      ctx.fillText(focus.label, Math.max(70, Math.min(W - 70, focus.x)), 45);
    }
    ctx.fillStyle = '#ff8a3a';
    ctx.fillRect(W / 2 - 1, 0, 2, 6);
  }
}

export { EXTRACTS };
