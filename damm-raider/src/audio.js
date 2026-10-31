// Alle Sounds werden zur Laufzeit mit WebAudio synthetisiert – keine Dateien nötig.
import { clamp } from './util.js';

export class Sfx {
  constructor() {
    this.ctx = null;
    this.master = null;
    this.volume = 0.7;
    this.listener = { x: 0, y: 0, z: 0, yaw: 0 };
    this.lastPlay = new Map();
  }

  init() {
    if (this.ctx) {
      if (this.ctx.state === 'suspended') this.ctx.resume();
      return;
    }
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    this.ctx = new AC();
    this.master = this.ctx.createGain();
    this.master.gain.value = this.volume;
    const comp = this.ctx.createDynamicsCompressor();
    comp.threshold.value = -14;
    comp.ratio.value = 4;
    this.master.connect(comp);
    comp.connect(this.ctx.destination);

    const len = this.ctx.sampleRate * 2;
    this.noiseBuf = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
    const d = this.noiseBuf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;

    this.setupAmbient();
  }

  setVolume(v) {
    this.volume = v;
    if (this.master) this.master.gain.value = v;
  }

  setListener(x, y, z, yaw) {
    const l = this.listener;
    l.x = x; l.y = y; l.z = z; l.yaw = yaw;
  }

  // Lautstärke + Panorama aus einer Weltposition
  spatial(pos, maxDist = 120) {
    if (!pos) return { vol: 1, pan: 0 };
    const l = this.listener;
    const dx = pos.x - l.x, dy = pos.y - l.y, dz = pos.z - l.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dist > maxDist) return null;
    const vol = Math.pow(1 - dist / maxDist, 1.6) * (8 / (8 + dist * 0.25));
    // Blickrichtung: -Z bei yaw 0
    const ang = Math.atan2(dx, -dz) - (-l.yaw);
    const pan = clamp(Math.sin(ang), -1, 1) * clamp(dist / 6, 0, 0.85);
    return { vol, pan };
  }

  out(vol, pan) {
    const g = this.ctx.createGain();
    g.gain.value = vol;
    if (pan && this.ctx.createStereoPanner) {
      const p = this.ctx.createStereoPanner();
      p.pan.value = pan;
      g.connect(p);
      p.connect(this.master);
    } else {
      g.connect(this.master);
    }
    return g;
  }

  noise(dest, t0, dur, { type = 'lowpass', f0 = 2000, f1 = 300, q = 0.8, gain = 1, attack = 0.002 } = {}) {
    const ctx = this.ctx;
    const src = ctx.createBufferSource();
    src.buffer = this.noiseBuf;
    src.playbackRate.value = 0.8 + Math.random() * 0.4;
    const filt = ctx.createBiquadFilter();
    filt.type = type;
    filt.Q.value = q;
    filt.frequency.setValueAtTime(f0, t0);
    filt.frequency.exponentialRampToValueAtTime(Math.max(20, f1), t0 + dur);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(gain, t0 + attack);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    src.connect(filt); filt.connect(g); g.connect(dest);
    src.start(t0, Math.random() * 1.5);
    src.stop(t0 + dur + 0.05);
  }

  tone(dest, t0, dur, { type = 'sine', f0 = 440, f1 = null, gain = 0.5, attack = 0.005 } = {}) {
    const ctx = this.ctx;
    const o = ctx.createOscillator();
    o.type = type;
    o.frequency.setValueAtTime(f0, t0);
    if (f1) o.frequency.exponentialRampToValueAtTime(Math.max(20, f1), t0 + dur);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(gain, t0 + attack);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    o.connect(g); g.connect(dest);
    o.start(t0);
    o.stop(t0 + dur + 0.05);
  }

  // Drosselt identische Sounds, damit z.B. Minigun-Salven nicht übersteuern
  throttle(key, minGap) {
    const now = this.ctx.currentTime;
    const last = this.lastPlay.get(key) || 0;
    if (now - last < minGap) return false;
    this.lastPlay.set(key, now);
    return true;
  }

  play(name, pos = null, opts = {}) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const maxDist = opts.maxDist || (name.startsWith('shot') || name === 'explosion' ? 260 : 90);
    const sp = this.spatial(pos, maxDist);
    if (!sp) return;
    const vol = sp.vol * (opts.vol ?? 1);
    if (vol < 0.01) return;
    const t = this.ctx.currentTime + 0.001;
    const out = this.out(vol, sp.pan);
    const fn = SOUNDS[name];
    if (fn) fn(this, out, t, opts);
  }

  setupAmbient() {
    const ctx = this.ctx;
    const mk = (type, freq, gainVal) => {
      const src = ctx.createBufferSource();
      src.buffer = this.noiseBuf;
      src.loop = true;
      const f = ctx.createBiquadFilter();
      f.type = type;
      f.frequency.value = freq;
      const g = ctx.createGain();
      g.gain.value = gainVal;
      src.connect(f); f.connect(g); g.connect(this.master);
      src.start();
      return g;
    };
    this.windGain = mk('lowpass', 380, 0);
    this.rainGain = mk('highpass', 2500, 0);

    // Brummen von ARC-Drohnen in der Nähe
    this.droneGain = ctx.createGain();
    this.droneGain.gain.value = 0;
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = 900;
    for (const f of [118, 121.5, 236]) {
      const o = ctx.createOscillator();
      o.type = 'sawtooth';
      o.frequency.value = f;
      o.connect(lp);
      o.start();
    }
    lp.connect(this.droneGain);
    this.droneGain.connect(this.master);
  }

  setAmbient(wind, rain, drone) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    this.windGain.gain.setTargetAtTime(wind, t, 0.5);
    this.rainGain.gain.setTargetAtTime(rain, t, 0.5);
    this.droneGain.gain.setTargetAtTime(drone, t, 0.15);
  }
}

const SHOT_PROFILES = {
  pistol: { f0: 3800, f1: 500, dur: 0.16, thump: 150, g: 0.9 },
  smg: { f0: 4200, f1: 700, dur: 0.1, thump: 170, g: 0.7 },
  rifle: { f0: 3000, f1: 400, dur: 0.2, thump: 120, g: 0.95 },
  marksman: { f0: 2600, f1: 260, dur: 0.32, thump: 95, g: 1 },
  heavy: { f0: 2000, f1: 160, dur: 0.45, thump: 70, g: 1.1 },
  shotgun: { f0: 1800, f1: 180, dur: 0.42, thump: 80, g: 1.1 },
  lmg: { f0: 2800, f1: 380, dur: 0.18, thump: 110, g: 0.9 },
};

const SOUNDS = {
  shot(s, out, t, o) {
    const p = SHOT_PROFILES[o.kind] || SHOT_PROFILES.rifle;
    s.noise(out, t, p.dur, { f0: p.f0, f1: p.f1, gain: p.g, q: 0.6 });
    s.noise(out, t, p.dur * 2.2, { type: 'bandpass', f0: 900, f1: 200, gain: p.g * 0.25, q: 0.4, attack: 0.01 });
    s.tone(out, t, p.dur * 0.8, { f0: p.thump * 2, f1: p.thump * 0.5, gain: 0.6 });
  },
  arcShot(s, out, t) {
    if (!s.throttle('arcShot', 0.03)) return;
    s.tone(out, t, 0.09, { type: 'square', f0: 1300, f1: 260, gain: 0.25 });
    s.noise(out, t, 0.06, { type: 'highpass', f0: 3000, f1: 1500, gain: 0.3 });
  },
  bolt(s, out, t) {
    s.tone(out, t, 0.18, { type: 'sawtooth', f0: 700, f1: 140, gain: 0.3 });
    s.tone(out, t, 0.12, { type: 'square', f0: 1900, f1: 900, gain: 0.12 });
  },
  minigun(s, out, t) {
    if (!s.throttle('minigun', 0.05)) return;
    s.noise(out, t, 0.08, { f0: 2600, f1: 500, gain: 0.8 });
    s.tone(out, t, 0.06, { f0: 180, f1: 70, gain: 0.4 });
  },
  sniper(s, out, t) {
    s.noise(out, t, 0.5, { f0: 5000, f1: 200, gain: 1.1 });
    s.tone(out, t, 0.4, { type: 'sawtooth', f0: 1800, f1: 60, gain: 0.35 });
  },
  rocketLaunch(s, out, t) {
    s.noise(out, t, 0.5, { type: 'bandpass', f0: 600, f1: 2000, gain: 0.7, q: 1.2, attack: 0.03 });
  },
  explosion(s, out, t, o) {
    const big = o.big ? 1.5 : 1;
    s.noise(out, t, 1.4 * big, { f0: 1400, f1: 60, gain: 1.3, q: 0.5, attack: 0.005 });
    s.tone(out, t, 0.9 * big, { f0: 90, f1: 30, gain: 1.0 });
    s.noise(out, t + 0.05, 0.4, { type: 'highpass', f0: 2500, f1: 800, gain: 0.25 });
  },
  hit(s, out, t) {
    if (!s.throttle('hit', 0.025)) return;
    s.tone(out, t, 0.05, { type: 'triangle', f0: 1900, f1: 1500, gain: 0.35 });
  },
  hitWeak(s, out, t) {
    if (!s.throttle('hitw', 0.025)) return;
    s.tone(out, t, 0.08, { type: 'triangle', f0: 2800, f1: 2400, gain: 0.4 });
  },
  hitArmor(s, out, t) {
    if (!s.throttle('hita', 0.025)) return;
    s.tone(out, t, 0.06, { type: 'square', f0: 900, f1: 700, gain: 0.12 });
    s.noise(out, t, 0.05, { type: 'highpass', f0: 5000, f1: 3000, gain: 0.2 });
  },
  kill(s, out, t) {
    s.tone(out, t, 0.1, { type: 'triangle', f0: 1400, gain: 0.35 });
    s.tone(out, t + 0.08, 0.18, { type: 'triangle', f0: 2100, gain: 0.35 });
  },
  impact(s, out, t) {
    if (!s.throttle('impact', 0.02)) return;
    s.noise(out, t, 0.07, { type: 'bandpass', f0: 2400, f1: 900, gain: 0.3, q: 1.5 });
  },
  ricochet(s, out, t) {
    if (!s.throttle('ric', 0.05)) return;
    s.tone(out, t, 0.25, { type: 'sine', f0: 3200, f1: 1800, gain: 0.12 });
  },
  empty(s, out, t) {
    s.tone(out, t, 0.03, { type: 'square', f0: 1200, gain: 0.15 });
  },
  reload(s, out, t, o) {
    const d = o.dur || 1.5;
    s.noise(out, t + 0.05, 0.06, { type: 'highpass', f0: 3000, f1: 2000, gain: 0.3 });
    s.noise(out, t + d * 0.45, 0.08, { type: 'bandpass', f0: 1800, f1: 1200, gain: 0.35, q: 2 });
    s.noise(out, t + d * 0.9, 0.05, { type: 'highpass', f0: 4000, f1: 2500, gain: 0.35 });
  },
  shell(s, out, t) {
    s.noise(out, t, 0.08, { type: 'bandpass', f0: 1500, f1: 900, gain: 0.3, q: 2 });
  },
  footstep(s, out, t, o) {
    s.noise(out, t, 0.09, { f0: o.water ? 1400 : 700, f1: 150, gain: o.water ? 0.35 : 0.22, q: 0.7 });
  },
  jumpLand(s, out, t) {
    s.noise(out, t, 0.15, { f0: 500, f1: 80, gain: 0.4 });
  },
  beep(s, out, t, o) {
    s.tone(out, t, 0.07, { type: 'square', f0: o.f || 1600, gain: 0.25 });
  },
  alarm(s, out, t) {
    const o = s.ctx.createOscillator();
    o.type = 'sawtooth';
    const g = s.ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.35, t + 0.05);
    g.gain.setValueAtTime(0.35, t + 2.2);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 2.6);
    for (let i = 0; i < 6; i++) {
      o.frequency.setValueAtTime(700, t + i * 0.42);
      o.frequency.linearRampToValueAtTime(1150, t + i * 0.42 + 0.21);
      o.frequency.linearRampToValueAtTime(700, t + i * 0.42 + 0.42);
    }
    const f = s.ctx.createBiquadFilter();
    f.type = 'lowpass';
    f.frequency.value = 2200;
    o.connect(f); f.connect(g); g.connect(out);
    o.start(t); o.stop(t + 2.7);
  },
  scan(s, out, t) {
    s.tone(out, t, 0.6, { type: 'sine', f0: 400, f1: 1200, gain: 0.2, attack: 0.1 });
  },
  lock(s, out, t) {
    s.tone(out, t, 0.12, { type: 'square', f0: 2200, gain: 0.15 });
    s.tone(out, t + 0.15, 0.12, { type: 'square', f0: 2200, gain: 0.15 });
  },
  leap(s, out, t) {
    s.noise(out, t, 0.35, { type: 'bandpass', f0: 300, f1: 1200, gain: 0.5, q: 1 });
    s.tone(out, t, 0.3, { type: 'sawtooth', f0: 90, f1: 220, gain: 0.25 });
  },
  stomp(s, out, t) {
    s.tone(out, t, 0.5, { f0: 70, f1: 30, gain: 1 });
    s.noise(out, t, 0.4, { f0: 600, f1: 60, gain: 0.8 });
  },
  servo(s, out, t) {
    if (!s.throttle('servo', 0.25)) return;
    s.tone(out, t, 0.25, { type: 'sawtooth', f0: 180, f1: 260, gain: 0.08, attack: 0.05 });
  },
  tick(s, out, t) {
    if (!s.throttle('tick', 0.06)) return;
    s.noise(out, t, 0.03, { type: 'highpass', f0: 6000, f1: 4000, gain: 0.25 });
  },
  pickup(s, out, t) {
    s.tone(out, t, 0.08, { type: 'triangle', f0: 700, f1: 1100, gain: 0.25 });
  },
  search(s, out, t) {
    s.noise(out, t, 0.25, { type: 'bandpass', f0: 800, f1: 1600, gain: 0.25, q: 1.2, attack: 0.05 });
  },
  open(s, out, t) {
    s.noise(out, t, 0.2, { type: 'bandpass', f0: 500, f1: 1400, gain: 0.35, q: 1.5 });
    s.tone(out, t + 0.05, 0.12, { type: 'triangle', f0: 520, f1: 780, gain: 0.15 });
  },
  ui(s, out, t) {
    s.tone(out, t, 0.04, { type: 'triangle', f0: 1100, gain: 0.15 });
  },
  hurt(s, out, t) {
    if (!s.throttle('hurt', 0.12)) return;
    s.tone(out, t, 0.18, { f0: 160, f1: 70, gain: 0.5 });
    s.noise(out, t, 0.12, { f0: 900, f1: 200, gain: 0.3 });
  },
  shieldHit(s, out, t) {
    if (!s.throttle('shieldHit', 0.1)) return;
    s.tone(out, t, 0.12, { type: 'sine', f0: 1400, f1: 900, gain: 0.18 });
  },
  shieldBreak(s, out, t) {
    s.noise(out, t, 0.5, { type: 'highpass', f0: 6000, f1: 1500, gain: 0.5 });
    s.tone(out, t, 0.5, { type: 'triangle', f0: 1600, f1: 200, gain: 0.3 });
  },
  heal(s, out, t) {
    s.tone(out, t, 0.35, { type: 'sine', f0: 500, f1: 900, gain: 0.2, attack: 0.05 });
  },
  shieldCharge(s, out, t) {
    s.tone(out, t, 0.5, { type: 'sawtooth', f0: 300, f1: 1400, gain: 0.12, attack: 0.08 });
  },
  elevator(s, out, t) {
    s.tone(out, t, 0.5, { type: 'sine', f0: 660, gain: 0.3 });
    s.tone(out, t + 0.25, 0.5, { type: 'sine', f0: 880, gain: 0.3 });
    s.tone(out, t + 0.5, 0.8, { type: 'sine', f0: 1320, gain: 0.25 });
  },
  thunder(s, out, t) {
    s.noise(out, t, 3.5, { f0: 500, f1: 40, gain: 1.2, attack: 0.15 });
  },
  melee(s, out, t) {
    s.noise(out, t, 0.12, { type: 'bandpass', f0: 1200, f1: 300, gain: 0.5, q: 1 });
  },
  grenadeBounce(s, out, t) {
    s.tone(out, t, 0.05, { type: 'triangle', f0: 900, f1: 600, gain: 0.2 });
  },
};
