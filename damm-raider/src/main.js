// Einstiegspunkt
import { Game } from './game.js';

const canvas = document.getElementById('view');

function fail(err) {
  console.error(err);
  const t = document.getElementById('boot-text');
  if (t) t.textContent = `Fehler beim Start: ${err && err.message ? err.message : err}. WebGL wird benötigt.`;
}

try {
  const game = new Game(canvas);
  window.__game = game;
  game.init().catch(fail);
} catch (err) {
  fail(err);
}
