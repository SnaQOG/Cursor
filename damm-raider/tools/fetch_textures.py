# Lädt CC0-Texturen von Poly Haven und speichert sie verkleinert unter assets/tex/.
#   python -m venv .venv-hf && .venv-hf/bin/pip install pillow
#   .venv-hf/bin/python -I tools/fetch_textures.py
# Lizenz: CC0, https://polyhaven.com/license
import io
import json
import os
import sys
import urllib.request

from PIL import Image, ImageEnhance, ImageStat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Name im Spiel: (Poly-Haven-ID, Kantenlänge, WebP-Qualität, Sättigung, mittlere Helligkeit)
# Die Grundfarbe wird angeglichen, damit die Hausfarben aus dem Spiel stimmen (weißer Putz bleibt weiß).
TEXTURES = {
    'plaster': ('white_plaster_02', 768, 80, 0.0, 0.86),
    'rooftile': ('clay_roof_tiles_02', 768, 80, 0.62, 0.42),
    'cobble': ('cobblestone_floor_08', 1024, 78, 0.8, None),
    'cliff': ('cliff_side', 1024, 78, 0.22, 0.46),
}
UA = {'User-Agent': 'damm-raider-texture-fetch'}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def main():
    out = os.path.join(ROOT, 'assets', 'tex')
    os.makedirs(out, exist_ok=True)
    for name, (asset, size, q, sat, level) in TEXTURES.items():
        files = json.loads(get(f'https://api.polyhaven.com/files/{asset}'))
        for key, suffix in (('Diffuse', 'c'), ('nor_gl', 'n')):
            im = Image.open(io.BytesIO(get(files[key]['1k']['jpg']['url']))).convert('RGB').resize((size, size), Image.LANCZOS)
            if suffix == 'c':
                im = ImageEnhance.Color(im).enhance(sat)
                if level:
                    mean = sum(ImageStat.Stat(im).mean) / 3 / 255
                    im = im.point(lambda v: min(255, v * level / mean))
                if name == 'plaster':
                    im = ImageEnhance.Contrast(im).enhance(0.55)  # Flecken nur andeuten, sie wiederholen sich sonst
            path = os.path.join(out, f'{name}_{suffix}.webp')
            im.save(path, 'WEBP', quality=q, method=6)
            print(path, os.path.getsize(path) // 1024, 'KB', flush=True)


if __name__ == '__main__':
    sys.exit(main())
