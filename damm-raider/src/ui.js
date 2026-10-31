// Menüs: Hub (Speranza), Beutefenster, Inventar, Karte, Pause, Auswertung.
import { ITEMS, RARITY, CATEGORY, WEAPONS, iconSvg } from './items.js';
import { Inventory } from './inventory.js';
import { POIS, EXTRACTS, MAP } from './config.js';
import { BACKPACK_LEVELS, STASH_LEVELS, UPGRADE_COST, freeLoadout, writeSave, wipeSave } from './save.js';
import { formatNum, formatTime } from './util.js';
import { ENEMY_DEFS } from './enemies.js';
import { Gun } from './weapons.js';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const SHOP = [
  ['w_nadel', 1, 700], ['w_stichler', 1, 2100], ['w_klapper', 1, 2600], ['w_stier', 1, 3100], ['w_kessel', 1, 3600], ['w_amboss', 1, 4100],
  ['ammo_light', 60, 150], ['ammo_medium', 45, 170], ['ammo_heavy', 20, 190], ['ammo_shell', 12, 95],
  ['bandage', 1, 90], ['shieldcell', 1, 150], ['medkit', 1, 420], ['grenade', 1, 230],
  ['shield_light', 1, 700], ['shield_medium', 1, 2100],
];

const CONTROLS = [
  ['W A S D', 'Bewegen'], ['Maus', 'Umsehen'], ['Linke Maustaste', 'Schießen'], ['Rechte Maustaste', 'Zielen'],
  ['Shift', 'Sprinten'], ['Leertaste', 'Springen'], ['C', 'Ducken'], ['Q', 'Ausweichrolle'], ['V', 'Nahkampf / Zecke abschütteln'],
  ['R', 'Nachladen'], ['1 / 2 / Mausrad', 'Waffe wechseln'], ['E (halten)', 'Durchsuchen / Benutzen'], ['3', 'Verband'],
  ['4', 'Sanitätskit'], ['5', 'Schildlader'], ['G', 'Granate'], ['Tab', 'Inventar'], ['M', 'Karte'], ['Esc', 'Pause'],
];

export function tileHTML(slot, extra = '') {
  if (!slot) return `<button class="tile empty" ${extra} tabindex="-1"></button>`;
  const d = ITEMS[slot.id];
  const col = RARITY[d.rarity].color;
  return `<button class="tile" style="--rc:${col}" data-id="${slot.id}" ${extra}>
    <span class="ic">${iconSvg(d.cat, col)}</span>
    <span class="nm">${esc(d.name)}</span>
    ${slot.qty > 1 ? `<span class="qty">${slot.qty}</span>` : ''}
  </button>`;
}

export class UI {
  constructor(game) {
    this.game = game;
    this.open = null; // 'loot' | 'inv' | 'map' | 'pause' | null
    this.tab = 'einsatz';
    this.lootTarget = null;
    this.selected = null;
    this.tooltip = $('tooltip');
    document.addEventListener('mouseover', (e) => {
      const t = e.target.closest('[data-id]');
      if (!t) { this.tooltip.hidden = true; return; }
      this.showTip(t.dataset.id, t);
    });
    document.addEventListener('mousemove', (e) => {
      if (this.tooltip.hidden) return;
      const x = Math.min(window.innerWidth - 270, e.clientX + 16);
      const y = Math.min(window.innerHeight - this.tooltip.offsetHeight - 8, e.clientY + 12);
      this.tooltip.style.transform = `translate(${x}px, ${y}px)`;
    });
    this.bindHub();
    this.bindRaidPanels();
  }

  get modalOpen() { return !!this.open; }

  showTip(id, el) {
    const d = ITEMS[id];
    if (!d) return;
    const w = WEAPONS[id];
    let stats = '';
    if (w) {
      stats = `<dl class="tstats">
        <dt>Schaden</dt><dd>${w.dmg}${w.pellets ? ' × ' + w.pellets : ''}</dd>
        <dt>Feuerrate</dt><dd>${w.rpm} /min</dd>
        <dt>Magazin</dt><dd>${w.mag}</dd>
        <dt>Durchschlag</dt><dd>${Math.round(w.pen * 100)} %</dd>
        <dt>Munition</dt><dd>${ITEMS[w.ammo].name}</dd></dl>`;
    }
    this.tooltip.innerHTML = `<div class="tname" style="color:${RARITY[d.rarity].color}">${esc(d.name)}</div>
      <div class="tmeta">${RARITY[d.rarity].name} · ${CATEGORY[d.cat]}</div>
      <p>${esc(d.desc)}</p>${stats}
      <div class="tval">Wert: ${formatNum(d.value)} Cr${d.stack > 1 ? ' / Stück' : ''}</div>`;
    this.tooltip.hidden = false;
    void el;
  }

  // ================================================================ Laden
  bootProgress(text) {
    $('boot-text').textContent = text;
  }

  hideBoot() {
    $('boot').hidden = true;
  }

  // ================================================================ Hub
  bindHub() {
    for (const b of document.querySelectorAll('#hub-nav button')) {
      b.addEventListener('click', () => {
        this.game.sfx.init();
        this.game.sfx.play('ui');
        this.tab = b.dataset.tab;
        this.selected = null;
        this.renderHub();
      });
    }
    $('hub-body').addEventListener('click', (e) => this.onHubClick(e));
    $('hub-body').addEventListener('input', (e) => this.onHubInput(e));
    $('hub-body').addEventListener('change', (e) => this.onHubInput(e));
  }

  showHub() {
    $('hub').hidden = false;
    this.renderHub();
  }

  hideHub() {
    $('hub').hidden = true;
  }

  persist() {
    writeSave(this.game.save);
  }

  renderHub() {
    const s = this.game.save;
    $('hub-credits').textContent = formatNum(s.credits);
    for (const b of document.querySelectorAll('#hub-nav button')) b.classList.toggle('on', b.dataset.tab === this.tab);
    const body = $('hub-body');
    const fn = {
      einsatz: () => this.renderDeploy(),
      lager: () => this.renderStash(),
      haendler: () => this.renderTrader(),
      ausbau: () => this.renderUpgrades(),
      statistik: () => this.renderStats(),
      optionen: () => this.renderOptions(),
    }[this.tab];
    const same = this.lastTab === this.tab;
    const scroll = body.scrollTop;
    body.innerHTML = fn();
    body.scrollTop = same ? scroll : 0;
    this.lastTab = this.tab;
  }

  loadoutValue() {
    const L = this.game.save.loadout;
    let v = 0;
    for (const w of L.weapons) if (w) v += ITEMS[w.id].value;
    if (L.shield) v += ITEMS[L.shield.id].value;
    for (const sl of L.backpack) if (sl) v += ITEMS[sl.id].value * sl.qty;
    return v;
  }

  loadoutHTML(interactive = true) {
    const L = this.game.save.loadout;
    const w = (i) => {
      const g = L.weapons[i];
      if (!g) return `<div class="eqslot empty"><span class="lbl">Waffe ${i + 1}</span><span class="val">leer</span></div>`;
      const d = ITEMS[g.id];
      return `<button class="eqslot" style="--rc:${RARITY[d.rarity].color}" data-id="${g.id}" ${interactive ? `data-act="unequip-w" data-i="${i}"` : ''}>
        <span class="lbl">Waffe ${i + 1}</span><span class="ic">${iconSvg('weapon', RARITY[d.rarity].color)}</span><span class="val">${esc(d.name)}</span></button>`;
    };
    const sh = L.shield
      ? `<button class="eqslot" style="--rc:${RARITY[ITEMS[L.shield.id].rarity].color}" data-id="${L.shield.id}" ${interactive ? 'data-act="unequip-s"' : ''}>
          <span class="lbl">Schild</span><span class="ic">${iconSvg('shield', RARITY[ITEMS[L.shield.id].rarity].color)}</span><span class="val">${esc(ITEMS[L.shield.id].name)}</span></button>`
      : '<div class="eqslot empty"><span class="lbl">Schild</span><span class="val">keiner</span></div>';
    const bp = L.backpack.map((sl, i) => tileHTML(sl, interactive ? `data-act="to-stash" data-i="${i}"` : '')).join('');
    return `<div class="eqrow">${w(0)}${w(1)}${sh}</div>
      <div class="gridhead"><span>Rucksack</span><span class="muted">${L.backpack.filter(Boolean).length}/${L.backpack.length} Plätze</span></div>
      <div class="grid">${bp}</div>`;
  }

  renderDeploy() {
    const s = this.game.save;
    const L = s.loadout;
    const hasGun = L.weapons.some(Boolean);
    const risk = this.loadoutValue();
    const pois = POIS.map((p) => `<li><span class="tier t${p.tier}">${'◆'.repeat(p.tier)}</span>${esc(p.name)}</li>`).join('');
    return `<div class="deploy">
      <section class="panel loadout">
        <h2>Ausrüstung</h2>
        ${this.loadoutHTML(true)}
        <div class="riskline"><span>Einsatz-Risiko</span><strong>${formatNum(risk)} Cr</strong></div>
        <p class="hint">Klick auf einen Gegenstand legt ihn ins Lager. Ausrüsten kannst du im Lager.</p>
      </section>
      <section class="panel mission">
        <div class="kicker">Oberfläche · Einsatzgebiet</div>
        <h2 class="mapname">Der Damm</h2>
        <p class="lead">Talsperre 7 staut den Fluss noch immer. Unter der Mauer liegen Kraftwerk, Forschungsanlage und die Wohnblöcke – bewacht von ARC-Maschinen. Hol dir, was du tragen kannst, und schaff es zu einem Aufzug, bevor die Zeit abläuft.</p>
        <ul class="poilist">${pois}</ul>
        <div class="facts">
          <div><span>Einsatzzeit</span><strong>20:00</strong></div>
          <div><span>Aufzüge</span><strong>${EXTRACTS.length}</strong></div>
          <div><span>Bedingung</span><strong>zufällig</strong></div>
        </div>
        ${!hasGun ? `<div class="warnbox">Du hast keine Waffe dabei. Nimm eine aus dem Lager oder die Gratis-Ausrüstung.
          <button class="btn ghost" data-act="free">Gratis-Ausrüstung nehmen</button></div>` : ''}
        <button class="btn primary big" data-act="deploy">Raid starten</button>
        <p class="hint">Stirbst du, ist alles weg, was du dabei hattest – außer der sicheren Tasche.</p>
      </section>
    </div>`;
  }

  renderStash() {
    const s = this.game.save;
    const stash = s.stash.map((sl, i) => tileHTML(sl, `data-act="from-stash" data-i="${i}"`)).join('');
    return `<div class="twocol">
      <section class="panel">
        <div class="gridhead"><h2>Lager</h2><span class="muted">${s.stash.filter(Boolean).length}/${s.stash.length} Plätze</span>
          <button class="btn ghost small" data-act="sort-stash">Sortieren</button></div>
        <div class="grid stash">${stash}</div>
        <p class="hint">Klick: in die Ausrüstung legen. Waffen und Schilde werden direkt ausgerüstet, wenn ein Platz frei ist.</p>
      </section>
      <section class="panel">
        <div class="gridhead"><h2>Ausrüstung</h2><button class="btn ghost small" data-act="bp-to-stash">Rucksack einlagern</button></div>
        ${this.loadoutHTML(true)}
      </section>
    </div>`;
  }

  renderTrader() {
    const s = this.game.save;
    const shop = SHOP.map(([id, qty, price], i) => {
      const d = ITEMS[id];
      const can = s.credits >= price;
      return `<div class="shoprow" style="--rc:${RARITY[d.rarity].color}">
        <span class="ic" data-id="${id}">${iconSvg(d.cat, RARITY[d.rarity].color)}</span>
        <span class="sname">${esc(d.name)}${qty > 1 ? ` <em>×${qty}</em>` : ''}</span>
        <span class="price">${formatNum(price)} Cr</span>
        <button class="btn small" data-act="buy" data-i="${i}" ${can ? '' : 'disabled'}>Kaufen</button></div>`;
    }).join('');
    const sel = this.selected;
    let sellbar = '<div class="sellbar muted">Wähle rechts einen Gegenstand aus dem Lager zum Verkaufen.</div>';
    if (sel != null && s.stash[sel]) {
      const sl = s.stash[sel];
      const d = ITEMS[sl.id];
      sellbar = `<div class="sellbar"><span>${esc(d.name)} ×${sl.qty}</span>
        <button class="btn small" data-act="sell" data-n="1">1 verkaufen · ${formatNum(d.value)} Cr</button>
        ${sl.qty > 1 ? `<button class="btn small" data-act="sell" data-n="${sl.qty}">Alle · ${formatNum(d.value * sl.qty)} Cr</button>` : ''}</div>`;
    }
    const junk = s.stash.reduce((v, sl) => (sl && ['valuable', 'arc'].includes(ITEMS[sl.id].cat) ? v + ITEMS[sl.id].value * sl.qty : v), 0);
    const stash = s.stash.map((sl, i) => tileHTML(sl, `data-act="select-sell" data-i="${i}"${sel === i ? ' aria-pressed="true"' : ''}`)).join('');
    return `<div class="twocol">
      <section class="panel">
        <h2>Händlerin Celeste</h2>
        <p class="hint">„Munition, Verbände, Eisen. Bezahlt wird in Credits.“</p>
        <div class="shop">${shop}</div>
      </section>
      <section class="panel">
        <div class="gridhead"><h2>Verkaufen</h2>
          <button class="btn ghost small" data-act="sell-junk" ${junk ? '' : 'disabled'}>Wertsachen &amp; ARC-Teile · ${formatNum(junk)} Cr</button></div>
        ${sellbar}
        <div class="grid stash">${stash}</div>
      </section>
    </div>`;
  }

  renderUpgrades() {
    const s = this.game.save;
    const row = (key, title, levels, unit) => {
      const lvl = s.upgrades[key];
      const next = UPGRADE_COST[key][lvl];
      return `<div class="upg">
        <div><h3>${title}</h3><p class="muted">Stufe ${lvl + 1} von ${levels.length} · ${levels[lvl]} ${unit}</p>
        <div class="pips">${levels.map((_, i) => `<i class="${i <= lvl ? 'on' : ''}"></i>`).join('')}</div></div>
        ${next != null ? `<button class="btn" data-act="upgrade" data-k="${key}" ${s.credits >= next ? '' : 'disabled'}>Auf ${levels[lvl + 1]} ${unit} · ${formatNum(next)} Cr</button>` : '<span class="muted">Maximal ausgebaut</span>'}
      </div>`;
    };
    return `<section class="panel narrow">
      <h2>Ausbau</h2>
      ${row('backpack', 'Rucksack', BACKPACK_LEVELS, 'Plätze')}
      ${row('stash', 'Lager in Speranza', STASH_LEVELS, 'Plätze')}
    </section>`;
  }

  renderStats() {
    const st = this.game.save.stats;
    const rate = st.raids ? Math.round((st.extracted / st.raids) * 100) : 0;
    const kills = Object.entries(st.arcKills || {}).sort((a, b) => b[1] - a[1])
      .map(([t, n]) => `<li><span>${esc(ENEMY_DEFS[t]?.name || t)}</span><strong>${n}</strong></li>`).join('') || '<li class="muted">Noch keine ARC zerstört.</li>';
    return `<div class="twocol">
      <section class="panel">
        <h2>Raider-Akte</h2>
        <div class="statgrid">
          <div><span>Raids</span><strong>${st.raids}</strong></div>
          <div><span>Extraktionen</span><strong>${st.extracted}</strong></div>
          <div><span>Gefallen</span><strong>${st.died}</strong></div>
          <div><span>Überlebensquote</span><strong>${rate} %</strong></div>
          <div><span>Beute extrahiert</span><strong>${formatNum(st.earned)} Cr</strong></div>
          <div><span>Beste Beute</span><strong>${formatNum(st.bestHaul)} Cr</strong></div>
        </div>
      </section>
      <section class="panel"><h2>Zerstörte ARC</h2><ul class="killlist">${kills}</ul></section>
    </div>`;
  }

  renderOptions() {
    const o = this.game.save.settings;
    const ctr = CONTROLS.map(([k, v]) => `<li><kbd>${k}</kbd><span>${v}</span></li>`).join('');
    return `<div class="twocol">
      <section class="panel">
        <h2>Optionen</h2>
        <label class="opt" for="opt-sens">Mausempfindlichkeit <output>${o.sens.toFixed(2)}</output></label>
        <input type="range" id="opt-sens" min="0.3" max="2.5" step="0.05" value="${o.sens}">
        <label class="opt" for="opt-vol">Lautstärke <output>${Math.round(o.volume * 100)} %</output></label>
        <input type="range" id="opt-vol" min="0" max="1" step="0.05" value="${o.volume}">
        <label class="opt" for="opt-q">Grafikqualität</label>
        <select id="opt-q">
          ${['niedrig', 'mittel', 'hoch'].map((q) => `<option value="${q}" ${o.quality === q ? 'selected' : ''}>${q[0].toUpperCase() + q.slice(1)}</option>`).join('')}
        </select>
        <div class="danger">
          <h3>Spielstand</h3>
          ${this.confirmWipe
            ? '<p>Wirklich alles löschen? Lager, Credits und Statistik gehen verloren.</p><button class="btn danger" data-act="wipe-yes">Ja, zurücksetzen</button> <button class="btn ghost" data-act="wipe-no">Abbrechen</button>'
            : '<button class="btn ghost" data-act="wipe">Spielstand zurücksetzen</button>'}
        </div>
      </section>
      <section class="panel"><h2>Steuerung</h2><ul class="controls">${ctr}</ul></section>
    </div>`;
  }

  onHubInput(e) {
    const o = this.game.save.settings;
    if (e.target.id === 'opt-sens') {
      o.sens = parseFloat(e.target.value);
      e.target.previousElementSibling.querySelector('output').textContent = o.sens.toFixed(2);
    } else if (e.target.id === 'opt-vol') {
      o.volume = parseFloat(e.target.value);
      e.target.previousElementSibling.querySelector('output').textContent = `${Math.round(o.volume * 100)} %`;
    } else if (e.target.id === 'opt-q' && e.type === 'change') {
      o.quality = e.target.value;
    } else return;
    this.game.applySettings();
    this.persist();
  }

  onHubClick(e) {
    const b = e.target.closest('[data-act]');
    if (!b || b.disabled) return;
    const s = this.game.save;
    const L = s.loadout;
    const act = b.dataset.act;
    const i = parseInt(b.dataset.i, 10);
    const stash = new Inventory(s.stash.length, s.stash);
    const bp = new Inventory(L.backpack.length, L.backpack);
    this.game.sfx.init();
    let msg = null;
    switch (act) {
      case 'deploy':
        this.game.startRaid();
        return;
      case 'free':
        s.loadout = freeLoadout();
        s.loadout.backpack.length = BACKPACK_LEVELS[s.upgrades.backpack];
        for (let k = 0; k < s.loadout.backpack.length; k++) s.loadout.backpack[k] ??= null;
        msg = 'Gratis-Ausrüstung erhalten.';
        break;
      case 'from-stash': {
        const sl = stash.slots[i];
        if (!sl) return;
        const d = ITEMS[sl.id];
        if (d.cat === 'weapon' && L.weapons.some((w) => !w)) {
          L.weapons[L.weapons[0] ? 1 : 0] = { id: sl.id, mag: 0 };
          stash.remove(sl.id, 1);
        } else if (d.cat === 'shield' && !L.shield) {
          L.shield = { id: sl.id, charge: d.cap };
          stash.remove(sl.id, 1);
        } else {
          const left = bp.add(sl.id, sl.qty);
          if (left === sl.qty) msg = 'Rucksack ist voll.';
          stash.slots[i] = left > 0 ? { id: sl.id, qty: left } : null;
        }
        break;
      }
      case 'to-stash': {
        const sl = bp.slots[i];
        if (!sl) return;
        const left = stash.add(sl.id, sl.qty);
        if (left === sl.qty) msg = 'Lager ist voll.';
        bp.slots[i] = left > 0 ? { id: sl.id, qty: left } : null;
        break;
      }
      case 'bp-to-stash':
        for (let k = 0; k < bp.slots.length; k++) {
          const sl = bp.slots[k];
          if (!sl) continue;
          const left = stash.add(sl.id, sl.qty);
          bp.slots[k] = left > 0 ? { id: sl.id, qty: left } : null;
        }
        break;
      case 'unequip-w': {
        const w = L.weapons[i];
        if (stash.add(w.id, 1) > 0) { msg = 'Lager ist voll.'; break; }
        if (w.mag > 0) stash.add(WEAPONS[w.id].ammo, w.mag);
        L.weapons[i] = null;
        break;
      }
      case 'unequip-s':
        if (stash.add(L.shield.id, 1) > 0) { msg = 'Lager ist voll.'; break; }
        L.shield = null;
        break;
      case 'sort-stash':
        stash.sort();
        break;
      case 'buy': {
        const [id, qty, price] = SHOP[i];
        if (s.credits < price) return;
        if (!stash.canFit(id, qty)) { msg = 'Kein Platz im Lager.'; break; }
        stash.add(id, qty);
        s.credits -= price;
        this.game.sfx.play('pickup');
        msg = `${ITEMS[id].name} gekauft.`;
        break;
      }
      case 'select-sell':
        this.selected = stash.slots[i] ? i : null;
        break;
      case 'sell': {
        const sl = stash.slots[this.selected];
        if (!sl) return;
        const n = Math.min(sl.qty, parseInt(b.dataset.n, 10));
        s.credits += ITEMS[sl.id].value * n;
        sl.qty -= n;
        if (sl.qty <= 0) { stash.slots[this.selected] = null; this.selected = null; }
        this.game.sfx.play('pickup');
        break;
      }
      case 'sell-junk': {
        let sum = 0;
        stash.slots.forEach((sl, k) => {
          if (sl && ['valuable', 'arc'].includes(ITEMS[sl.id].cat)) {
            sum += ITEMS[sl.id].value * sl.qty;
            stash.slots[k] = null;
          }
        });
        s.credits += sum;
        msg = `${formatNum(sum)} Credits erhalten.`;
        this.selected = null;
        this.game.sfx.play('pickup');
        break;
      }
      case 'upgrade': {
        const k = b.dataset.k;
        const cost = UPGRADE_COST[k][s.upgrades[k]];
        if (cost == null || s.credits < cost) return;
        s.credits -= cost;
        s.upgrades[k]++;
        if (k === 'backpack') while (L.backpack.length < BACKPACK_LEVELS[s.upgrades.backpack]) L.backpack.push(null);
        else stash.resize(STASH_LEVELS[s.upgrades.stash]);
        msg = 'Ausgebaut!';
        break;
      }
      case 'wipe': this.confirmWipe = true; break;
      case 'wipe-no': this.confirmWipe = false; break;
      case 'wipe-yes':
        this.confirmWipe = false;
        this.game.save = wipeSave();
        this.game.applySettings();
        this.persist();
        this.renderHub();
        return;
      default: return;
    }
    if (!['buy', 'sell', 'sell-junk'].includes(act)) this.game.sfx.play('ui');
    s.stash = stash.toJSON();
    if (act !== 'free') L.backpack = bp.toJSON();
    this.persist();
    this.renderHub();
    if (msg) this.hubToast(msg);
  }

  hubToast(msg) {
    const el = $('hub-toast');
    el.textContent = msg;
    el.hidden = false;
    clearTimeout(this.hubToastT);
    this.hubToastT = setTimeout(() => { el.hidden = true; }, 2200);
  }

  // ================================================================ Raid-Fenster
  bindRaidPanels() {
    $('loot').addEventListener('click', (e) => this.onLootClick(e));
    $('inv').addEventListener('click', (e) => this.onInvClick(e));
    $('pause').addEventListener('click', (e) => this.onPauseClick(e));
    $('mapview').addEventListener('click', (e) => this.onMapClick(e));
    $('results').addEventListener('click', (e) => {
      if (e.target.closest('[data-act="back"]')) this.game.toHub();
    });
    $('pause').addEventListener('input', (e) => {
      const o = this.game.save.settings;
      if (e.target.id === 'p-sens') o.sens = parseFloat(e.target.value);
      if (e.target.id === 'p-vol') o.volume = parseFloat(e.target.value);
      this.game.applySettings();
      this.persist();
    });
  }

  openPanel(name) {
    if (this.open && this.open !== name) this.closePanel();
    this.open = name;
    this.game.input.exitLock();
    this.tooltip.hidden = true;
  }

  closePanel(relock = true) {
    const was = this.open;
    this.open = null;
    for (const id of ['loot', 'inv', 'mapview', 'pause']) $(id).hidden = true;
    this.tooltip.hidden = true;
    this.lootTarget = null;
    if (relock && was && this.game.state === 'raid' && this.game.player.alive) this.game.input.requestLock();
  }

  // --- Beute
  openLoot(c) {
    this.openPanel('loot');
    this.lootTarget = c;
    $('loot').hidden = false;
    this.renderLoot();
  }

  renderLoot() {
    const c = this.lootTarget;
    const p = this.game.player;
    if (!c) return;
    $('loot-title').textContent = c.name;
    $('loot-items').innerHTML = c.items.slots.map((sl, i) => tileHTML(sl, `data-act="take" data-i="${i}"`)).join('');
    $('loot-bp').innerHTML = p.inv.slots.map((sl, i) => tileHTML(sl, `data-act="put" data-i="${i}"`)).join('');
    $('loot-cap').textContent = `${p.inv.slots.filter(Boolean).length}/${p.inv.size}`;
    $('loot-value').textContent = `${formatNum(p.inv.value())} Cr`;
  }

  onLootClick(e) {
    const b = e.target.closest('[data-act]');
    if (!b) return;
    const c = this.lootTarget;
    const p = this.game.player;
    const act = b.dataset.act;
    const i = parseInt(b.dataset.i, 10);
    if (act === 'close') { this.closePanel(); return; }
    if (!c) return;
    if (act === 'take') this.take(c, i);
    else if (act === 'takeall') {
      for (let k = 0; k < c.items.size; k++) if (c.items.slots[k]) this.take(c, k, true);
    } else if (act === 'put') {
      const sl = p.inv.slots[i];
      if (!sl) return;
      c.items.resize(Math.max(c.items.size, c.items.slots.filter(Boolean).length + 1));
      const left = c.items.add(sl.id, sl.qty);
      p.inv.slots[i] = left > 0 ? { id: sl.id, qty: left } : null;
      this.game.sfx.play('ui');
    }
    this.renderLoot();
  }

  take(c, i, quiet = false) {
    const p = this.game.player;
    const sl = c.items.slots[i];
    if (!sl) return;
    const left = p.inv.add(sl.id, sl.qty);
    const got = sl.qty - left;
    if (got > 0) {
      this.game.sfx.play('pickup');
      this.game.hud.lootToast(sl.id, got);
      this.game.stats.looted += ITEMS[sl.id].value * got;
    } else if (!quiet) this.game.hud.toast('Rucksack ist voll', 'warn');
    c.items.slots[i] = left > 0 ? { id: sl.id, qty: left } : null;
  }

  // --- Inventar
  openInventory() {
    this.openPanel('inv');
    $('inv').hidden = false;
    this.invSel = null;
    this.renderInventory();
  }

  renderInventory() {
    const p = this.game.player;
    const eq = (label, g, idx) => {
      if (!g) return `<div class="eqslot empty"><span class="lbl">${label}</span><span class="val">leer</span></div>`;
      const d = ITEMS[g.id];
      return `<div class="eqslot${idx === p.active ? ' on' : ''}" style="--rc:${RARITY[d.rarity].color}" data-id="${g.id}">
        <span class="lbl">${label}</span><span class="ic">${iconSvg('weapon', RARITY[d.rarity].color)}</span>
        <span class="val">${esc(d.name)} · ${g.mag}/${g.def.mag}</span></div>`;
    };
    const sh = p.shield
      ? `<div class="eqslot" style="--rc:${RARITY[ITEMS[p.shield.id].rarity].color}" data-id="${p.shield.id}"><span class="lbl">Schild</span>
          <span class="ic">${iconSvg('shield', RARITY[ITEMS[p.shield.id].rarity].color)}</span>
          <span class="val">${esc(ITEMS[p.shield.id].name)} · ${Math.ceil(p.shield.charge)}/${ITEMS[p.shield.id].cap}</span></div>`
      : '<div class="eqslot empty"><span class="lbl">Schild</span><span class="val">keiner</span></div>';
    $('inv-eq').innerHTML = eq('Waffe 1', p.guns[0], 0) + eq('Waffe 2', p.guns[1], 1) + sh;
    $('inv-bp').innerHTML = p.inv.slots.map((sl, i) => tileHTML(sl, `data-act="sel" data-i="${i}"${this.invSel === i ? ' aria-pressed="true"' : ''}`)).join('');
    $('inv-safe').innerHTML = p.safe.slots.map((sl, i) => tileHTML(sl, `data-act="unsafe" data-i="${i}"`)).join('');
    $('inv-cap').textContent = `${p.inv.slots.filter(Boolean).length}/${p.inv.size}`;
    $('inv-value').textContent = `${formatNum(p.inv.value() + p.safe.value())} Cr`;
    const act = $('inv-actions');
    const sl = this.invSel != null ? p.inv.slots[this.invSel] : null;
    if (!sl) {
      act.innerHTML = '<p class="muted">Wähle einen Gegenstand aus.</p>';
      return;
    }
    const d = ITEMS[sl.id];
    const btns = [];
    if (d.cat === 'weapon') btns.push('<button class="btn small" data-act="equip-w" data-slot="0">Als Waffe 1</button>', '<button class="btn small" data-act="equip-w" data-slot="1">Als Waffe 2</button>');
    if (d.cat === 'shield') btns.push('<button class="btn small" data-act="equip-s">Ausrüsten</button>');
    if (d.cat === 'heal' || d.cat === 'shieldcell') btns.push('<button class="btn small" data-act="use">Benutzen</button>');
    btns.push('<button class="btn small" data-act="safe">In sichere Tasche</button>');
    btns.push(`<button class="btn small ghost" data-act="drop">Wegwerfen${sl.qty > 1 ? ' (alle)' : ''}</button>`);
    act.innerHTML = `<div class="selname" style="color:${RARITY[d.rarity].color}">${esc(d.name)} ×${sl.qty}</div>
      <p class="muted">${esc(d.desc)}</p><div class="btnrow">${btns.join('')}</div>`;
  }

  onInvClick(e) {
    const b = e.target.closest('[data-act]');
    if (!b) return;
    const p = this.game.player;
    const act = b.dataset.act;
    if (act === 'close') { this.closePanel(); return; }
    if (act === 'sel') { this.invSel = parseInt(b.dataset.i, 10); this.renderInventory(); return; }
    if (act === 'unsafe') {
      const i = parseInt(b.dataset.i, 10);
      const sl = p.safe.slots[i];
      if (!sl) return;
      const left = p.inv.add(sl.id, sl.qty);
      p.safe.slots[i] = left > 0 ? { id: sl.id, qty: left } : null;
      this.renderInventory();
      return;
    }
    const i = this.invSel;
    const sl = i != null ? p.inv.slots[i] : null;
    if (!sl) return;
    const d = ITEMS[sl.id];
    if (act === 'equip-w') {
      const slot = parseInt(b.dataset.slot, 10);
      const old = p.guns[slot];
      p.inv.slots[i] = null;
      if (old) {
        p.inv.add(old.id, 1);
        if (old.mag > 0) p.inv.add(old.def.ammo, old.mag);
      }
      p.guns[slot] = new Gun(sl.id, 0);
      p.active = slot;
      p.model.setGun(sl.id);
    } else if (act === 'equip-s') {
      p.inv.slots[i] = null;
      if (p.shield) p.inv.add(p.shield.id, 1);
      p.shield = { id: sl.id, charge: 0 };
      this.game.hud.toast('Schild ausgerüstet – leer. Lade ihn mit Schildladern.', 'info');
    } else if (act === 'use') {
      this.closePanel();
      p.useItem(sl.id);
      return;
    } else if (act === 'safe') {
      const one = { id: sl.id, qty: sl.qty };
      const left = p.safe.add(one.id, one.qty);
      if (left === one.qty) this.game.hud.toast('Sichere Tasche ist voll', 'warn');
      p.inv.slots[i] = left > 0 ? { id: sl.id, qty: left } : null;
    } else if (act === 'drop') {
      const bag = new Inventory(Math.max(6, 1));
      bag.add(sl.id, sl.qty);
      p.inv.slots[i] = null;
      this.game.loot.addBag(p.pos.clone(), bag);
      this.game.hud.toast(`${d.name} weggeworfen`, 'info');
      this.invSel = null;
    }
    this.game.sfx.play('ui');
    this.renderInventory();
  }

  // --- Karte
  openMap() {
    this.openPanel('map');
    $('mapview').hidden = false;
    this.drawMap();
  }

  drawMap() {
    const cv = $('map-canvas');
    const ctx = cv.getContext('2d');
    const S = cv.width;
    const g = this.game;
    ctx.drawImage(g.mapImage, 0, 0, S, S);
    const toPx = (x, z) => [((x + MAP.HALF) / (MAP.HALF * 2)) * S, ((z + MAP.HALF) / (MAP.HALF * 2)) * S];
    ctx.textAlign = 'center';
    for (const poi of POIS) {
      const [px, py] = toPx(poi.x, poi.z);
      ctx.fillStyle = 'rgba(20,24,26,0.75)';
      ctx.font = '600 13px "Barlow Semi Condensed", sans-serif';
      const w = ctx.measureText(poi.name).width + 10;
      ctx.fillRect(px - w / 2, py - 22, w, 17);
      ctx.fillStyle = ['#d9d2bf', '#d9d2bf', '#f2c96a', '#ff9a5a'][poi.tier];
      ctx.fillText(poi.name, px, py - 9);
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fill();
    }
    for (const e of g.elevators) {
      const [px, py] = toPx(e.base.x, e.base.z);
      ctx.fillStyle = e.state === 'open' ? '#5dff9a' : '#6fd3c0';
      ctx.beginPath();
      ctx.moveTo(px, py - 9); ctx.lineTo(px + 8, py + 6); ctx.lineTo(px - 8, py + 6);
      ctx.fill();
      ctx.font = '600 12px "Barlow Semi Condensed", sans-serif';
      ctx.fillText(e.name, px, py + 20);
    }
    if (g.marker) {
      const [px, py] = toPx(g.marker.x, g.marker.z);
      ctx.strokeStyle = '#ffd23a';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px, py, 7, 0, Math.PI * 2);
      ctx.stroke();
    }
    // Spieler
    const p = g.player;
    const [px, py] = toPx(p.pos.x, p.pos.z);
    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(-p.camYaw);
    ctx.fillStyle = '#ff8a3a';
    ctx.strokeStyle = '#1a1a1a';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, -10); ctx.lineTo(7, 8); ctx.lineTo(0, 4); ctx.lineTo(-7, 8); ctx.closePath();
    ctx.stroke();
    ctx.fill();
    ctx.restore();
    $('map-info').textContent = `Restzeit ${formatTime(g.raidTime)} · Bedingung: ${g.conditionName} · Klick setzt eine Markierung`;
  }

  onMapClick(e) {
    if (e.target.closest('[data-act="close"]')) { this.closePanel(); return; }
    const cv = $('map-canvas');
    if (e.target !== cv) return;
    const r = cv.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * MAP.HALF * 2 - MAP.HALF;
    const z = ((e.clientY - r.top) / r.height) * MAP.HALF * 2 - MAP.HALF;
    const g = this.game;
    g.marker = g.marker && Math.hypot(g.marker.x - x, g.marker.z - z) < 10 ? null : { x, z };
    this.drawMap();
  }

  // --- Pause
  openPause() {
    if (this.open === 'pause') return;
    this.openPanel('pause');
    $('pause').hidden = false;
    const o = this.game.save.settings;
    $('p-sens').value = o.sens;
    $('p-vol').value = o.volume;
    this.confirmQuit = false;
    $('p-quit').textContent = 'Raid aufgeben';
  }

  onPauseClick(e) {
    const b = e.target.closest('[data-act]');
    if (!b) return;
    if (b.dataset.act === 'resume') this.closePanel();
    if (b.dataset.act === 'quit') {
      if (!this.confirmQuit) {
        this.confirmQuit = true;
        b.textContent = 'Sicher? Deine Ausrüstung ist dann weg.';
        return;
      }
      this.closePanel(false);
      this.game.endRaid('aufgegeben');
    }
  }

  // --- Auswertung
  showResults(r) {
    $('results').hidden = false;
    const titles = { extrahiert: 'Extrahiert', gefallen: 'Gefallen', verschollen: 'Verschollen', aufgegeben: 'Aufgegeben' };
    const good = r.outcome === 'extrahiert';
    $('res-title').textContent = titles[r.outcome];
    $('res-title').className = good ? 'good' : 'bad';
    $('res-sub').textContent = good ? `Über ${r.via} zurück nach Speranza.` : r.reason;
    const items = r.items.length
      ? r.items.map((sl) => tileHTML(sl)).join('')
      : '<p class="muted">Nichts mitgebracht.</p>';
    $('res-stats').innerHTML = `
      <div><span>Zeit</span><strong>${formatTime(r.time)}</strong></div>
      <div><span>ARC zerstört</span><strong>${r.kills}</strong></div>
      <div><span>Gesammelte Beute</span><strong>${formatNum(r.looted)} Cr</strong></div>
      <div><span>${good ? 'Rucksackwert' : 'Verloren'}</span><strong>${formatNum(r.value)} Cr</strong></div>`;
    $('res-items').innerHTML = items;
    $('res-safe').hidden = !r.safe.length;
    $('res-safe-items').innerHTML = r.safe.map((sl) => tileHTML(sl)).join('');
  }

  hideResults() {
    $('results').hidden = true;
  }
}
