// Baut eine einzelne, eigenständige HTML-Datei (three.js eingebettet), die ohne
// Webserver per Doppelklick läuft: dist/damm-raider.html
// Mit "--fragment <pfad>" zusätzlich eine Variante ohne <html>/<head>/<body>.
import { build } from 'esbuild';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const fragIdx = args.indexOf('--fragment');
const fragmentOut = fragIdx >= 0 ? resolve(args[fragIdx + 1]) : null;

// Blender-Assets einbetten (models.glb als Base64, Icon-Atlas als data-URI)
const embed = {
  models: (await readFile(resolve(root, 'assets/models.glb'))).toString('base64'),
  icons: 'data:image/webp;base64,' + (await readFile(resolve(root, 'assets/icons.webp'))).toString('base64'),
  iconMeta: JSON.parse(await readFile(resolve(root, 'assets/icons.json'), 'utf8')),
};
const embedPlugin = {
  name: 'embed-assets',
  setup(b) {
    b.onLoad({ filter: /assetdata\.js$/ }, () => ({ contents: `export const EMBED = ${JSON.stringify(embed)};`, loader: 'js' }));
  },
};

const result = await build({
  plugins: [embedPlugin],
  entryPoints: [resolve(root, 'src/main.js')],
  bundle: true,
  format: 'esm',
  minify: true,
  target: 'es2020',
  write: false,
  legalComments: 'none',
});
const js = result.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
const css = await readFile(resolve(root, 'css/style.css'), 'utf8');
let html = await readFile(resolve(root, 'index.html'), 'utf8');

html = html
  .replace(/<!-- @build:importmap -->[\s\S]*?<!-- @build:end -->\n?/, '')
  .replace(/<link rel="stylesheet" href="css\/style.css">/, () => `<style>\n${css}</style>`)
  .replace(/<!-- @build:script -->[\s\S]*?<!-- @build:end -->/, () => `<script type="module">\n${js}</script>`);

await mkdir(resolve(root, 'dist'), { recursive: true });
const out = resolve(root, 'dist/damm-raider.html');
await writeFile(out, html);
console.log(`geschrieben: ${out} (${(html.length / 1024).toFixed(0)} KB)`);

if (fragmentOut) {
  const frag = html
    .replace(/<!doctype html>\s*/i, '')
    .replace(/<html[^>]*>\s*/i, '')
    .replace(/<\/html>\s*/i, '')
    .replace(/<head>\s*/i, '')
    .replace(/<\/head>\s*/i, '')
    .replace(/<body>\s*/i, '')
    .replace(/<\/body>\s*/i, '')
    .replace(/<meta charset="utf-8">\s*/i, '')
    .replace(/<meta name="viewport"[^>]*>\s*/i, '');
  await mkdir(dirname(fragmentOut), { recursive: true });
  await writeFile(fragmentOut, frag);
  console.log(`Fragment: ${fragmentOut}`);
}
