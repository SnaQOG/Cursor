// Prüfbilder der Ausstattung: je Szenenart ein Beispiel aus der Nähe, dazu die Stückzahlen.
//   node tools/tests/ausstattung.mjs w02 [damm]     Bilder: tools/tests/shots/ausstattung-<version>-<karte>-<szene>.png
import { open, SHOTS } from './common.mjs';
const version = process.argv[2] || 'w00';
const { browser, page, errors } = await open();
const prepare = () => page.evaluate(() => {
  const g = window.__game;
  g.startRaid();
  for (const e of g.enemies) { e.alive = false; e.root.visible = false; }
  g.building = true;
  g.player.model.root.visible = false;
  document.getElementById('hud').style.display = 'none';
  const B = g.world.mapData.builder;
  return { stats: g.world.mapData.stats?.dressed, samples: B.dressSamples };
});
const shoot = async (prefix, samples) => {
  for (const [kind, list] of Object.entries(samples)) {
    const [x, z, y, ox, oz] = list[0];
    const far = { lager: 8, barrikade: 8, geruest: 9, leine: 2.6, banner: 3.4, kanone: 6 }[kind] ?? 4;
    const off = kind === 'leine' ? 0 : 1.6, dir = kind === 'kanone' ? -1 : 1;
    // Von schräg vorn auf den Wandfuß schauen, die Kanone von der Wasserseite
    const pos = [x + ox * far * dir + oz * off, y + (kind === 'leine' || kind === 'banner' ? 1.7 : 2), z + oz * far * dir + ox * off];
    const tgt = [x + ox * 0.5, y + (kind === 'leine' ? 5 : kind === 'banner' ? 6 : kind === 'geruest' ? 4 : 0.5), z + oz * 0.5];
    await page.evaluate(([pos, tgt]) => {
      const g = window.__game, c = g.camera;
      g.world.applyCondition('tag');
      c.fov = 66; c.updateProjectionMatrix();
      c.position.set(...pos); c.lookAt(...tgt);
      for (let i = 0; i < 2; i++) { g.world.update(0.016, c.position, null); g.renderFrame(0.016); }
    }, [pos, tgt]);
    await page.screenshot({ path: `${SHOTS}ausstattung-${version}-${prefix}-${kind}.png`, timeout: 240000 });
  }
};
const map = process.argv[3] || 'weltenbrueck';
if (map !== 'damm') {
  await page.evaluate((m) => window.__game.selectMap(m), map);
  await page.waitForFunction((m) => window.__game.mapId === m && !window.__game.loadingMap, map, { timeout: 300000 });
}
const res = await prepare();
console.log(map, JSON.stringify(res.stats ?? {}), Object.entries(res.samples).map(([k, v]) => `${k}:${v.length}`).join(' '));
await shoot(map, res.samples);
console.log('Fehler:', errors.filter((e) => !/fonts|ERR_|GPU stall|GL Driver/.test(e)).slice(0, 8));
await browser.close();
