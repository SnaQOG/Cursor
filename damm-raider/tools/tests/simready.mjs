// Vorher-nachher-Bilder der SimReady-Requisiten: feste Kameras auf beiden Karten.
//   GAME_URL=file:///.../alt.html   node tools/tests/simready.mjs vorher
//   GAME_URL=file:///.../neu.html   node tools/tests/simready.mjs nachher
// Bilder: tools/tests/shots/<prefix>-<ort>.png. Dazu Draw-Calls, Dreiecke und Bildzeit je Karte.
import { open, SHOTS } from './common.mjs';
const prefix = process.argv[2] || 'nachher';

// Name, Kamera, Ziel, Blickwinkel. 'g1.7' = 1,7 m über dem Gelände an dieser Stelle.
const DAMM = [
  ['kraftwerk-west', [48.3, 4.3, -29.2], [45.5, 3, -36], 62],
  ['kraftwerk-ost', [72, 4.3, -29.4], [72.6, 3, -36], 62],
  ['kompressor', [47.5, 4.2, -49.5], [51, 3.2, -52.5], 55],
  ['sprengfass', [58.2, 4.2, -31.2], [55, 3, -34], 50],
  ['dammkrone', [-21, 36, -71.5], [-28, 34.5, -76], 60],
  ['aufbereitung', [150.5, 4.8, 35], [144.5, 3.2, 28.5], 60],
  ['schieber', [131.2, 'g2', 3.2], [135, 'g0.9', 7], 55],
  ['pumpstation', [-212, 'g1.8', -175.5], [-217, 'g0.5', -169.5], 55],
  ['bunker', [152, 'g1.8', 153.5], [160, 'g0.5', 160], 58],
];
const STADT = [
  ['markt', [-20, 4.4, 84], [-22.8, 3.5, 80], 55],
  ['markt2', [21, 4.4, 84.5], [22.6, 3.5, 80], 55],
  ['ruine', [-132.5, 5.6, 3.2], [-127.8, 3.2, 9.4], 62],
  ['hof', [-129.8, 4.6, 127], [-129.8, 3.4, 131], 62],
  ['stube', [-136.5, 4.8, 10.5], [-138.3, 3.6, 7.6], 75],
  ['schlafzimmer', [-59, 7.9, 18], [-62.3, 6.6, 20.2], 75],
];
const LOOK = { damm: [[60, 4, -20], [60, 4, -60]], weltenbrueck: [[0, 5, 118], [0, 5, 70]] };

async function run(mapId, list) {
  const { browser, page, errors } = await open();
  if (mapId !== 'damm') {
    await page.evaluate((id) => window.__game.selectMap(id), mapId);
    await page.waitForFunction((id) => window.__game.mapId === id && !window.__game.loadingMap, mapId, { timeout: 300000 });
  }
  await page.evaluate(() => {
    const g = window.__game;
    g.startRaid();
    for (const e of g.enemies) { e.alive = false; e.root.visible = false; }
    g.building = true;
    g.player.model.root.visible = false;
    document.getElementById('hud').style.display = 'none';
  });
  const frame = () => {
    const g = window.__game;
    g.world.update(0.016, g.camera.position, null);
    g.renderFrame(0.016);
  };
  const shot = async (name, pos, target, fov) => {
    await page.evaluate(([pos, target, fov, frameSrc]) => {
      const g = window.__game, c = g.camera;
      g.world.applyCondition('tag');
      c.fov = fov; c.updateProjectionMatrix();
      const h = (v, x, z) => (typeof v === 'string' ? g.world.terrain.height(x, z) + parseFloat(v.slice(1)) : v);
      c.position.set(pos[0], h(pos[1], pos[0], pos[2]), pos[2]);
      c.lookAt(target[0], h(target[1], target[0], target[2]), target[2]);
      const f = new Function('return ' + frameSrc)();
      f(); f();
    }, [pos, target, fov, frame.toString()]);
    await page.screenshot({ path: `${SHOTS}${prefix}-${name}.png`, timeout: 240000 });
  };
  for (const [n, p, t, f] of list) await shot(n, p, t, f);
  const stats = await page.evaluate(([look, frameSrc]) => {
    const g = window.__game, c = g.camera, i = g.renderer.info;
    const f = new Function('return ' + frameSrc)();
    c.fov = 70; c.updateProjectionMatrix();
    c.position.set(...look[0]); c.lookAt(...look[1]);
    f();
    i.autoReset = false; i.reset();
    f();
    const r = { calls: i.render.calls, tris: i.render.triangles, tex: i.memory.textures, geos: i.memory.geometries };
    i.autoReset = true;
    const gl = g.renderer.getContext(), t0 = performance.now();
    for (let k = 0; k < 20; k++) { f(); gl.finish(); }
    r.msPerFrame = +((performance.now() - t0) / 20).toFixed(1);
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    r.gpu = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : '?';
    return r;
  }, [LOOK[mapId], frame.toString()]);
  console.log(mapId, JSON.stringify(stats));
  console.log('fehler:', errors.filter((e) => !/fonts|ERR_|GPU stall|GL Driver Message/.test(e)).slice(0, 12));
  await browser.close();
}

await run('damm', DAMM);
await run('weltenbrueck', STADT);
