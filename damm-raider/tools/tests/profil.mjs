// CPU-Profil des Stadt-Ladens (teuerste Funktionen)
import { open } from './common.mjs';
const { browser, page, errors } = await open();
const cdp = await page.context().newCDPSession(page);
await cdp.send('Profiler.enable');
await cdp.send('Profiler.setSamplingInterval', { interval: 2000 });
await cdp.send('Profiler.start');
const t1 = Date.now();
await page.evaluate(() => window.__game.selectMap('weltenbrueck'));
await page.waitForFunction(() => window.__game.mapId === 'weltenbrueck' && !window.__game.loadingMap, null, { timeout: 300000 });
console.log('city load', (Date.now() - t1) / 1000, 's');
const { profile } = await cdp.send('Profiler.stop');
const self = new Map();
const byId = new Map(profile.nodes.map((n) => [n.id, n]));
const dt = profile.timeDeltas; let tot = 0;
const counts = new Map();
profile.samples.forEach((id, i) => counts.set(id, (counts.get(id) || 0) + (dt[i] || 0)));
for (const [id, t] of counts) {
  const n = byId.get(id); const k = `${n.callFrame.functionName || '(anon)'}:${n.callFrame.lineNumber}`;
  self.set(k, (self.get(k) || 0) + t); tot += t;
}
console.log('total ms', (tot / 1000).toFixed(0));
console.log([...self.entries()].sort((a, b) => b[1] - a[1]).slice(0, 25).map(([k, t]) => `${(t / 1000).toFixed(0)}ms ${k}`).join('\n'));
// inclusive by function name via parents
const parent = new Map(); for (const n of profile.nodes) for (const c of n.children || []) parent.set(c, n.id);
const incl = new Map();
for (const [id, t] of counts) { const seen = new Set(); let cur = id; while (cur) { const n = byId.get(cur); const k = `${n.callFrame.functionName || '(anon)'}:${n.callFrame.lineNumber}`; if (!seen.has(k)) { incl.set(k, (incl.get(k) || 0) + t); seen.add(k); } cur = parent.get(cur); } }
console.log('--- inclusive');
console.log([...incl.entries()].sort((a, b) => b[1] - a[1]).slice(0, 40).map(([k, t]) => `${(t / 1000).toFixed(0)}ms ${k}`).join('\n'));
console.log('errors:', errors.slice(0, 10));
await browser.close();
