# Lädt die CC0-Modelle für die Bodenpflanzen von Poly Haven (ohne Blender):
#   python3 -I blender/ground_fetch.py --out <downloads>
import json
import os
import sys
import urllib.request

ASSETS = ['grass_medium_01', 'grass_medium_02', 'grass_bermuda_01', 'dandelion_01', 'nettle_plant', 'weed_plant_02', 'shrub_sorrel_01', 'flower_ursinia', 'flower_gazania', 'flower_empodium', 'moss_01', 'fern_02']
UA = {'User-Agent': 'damm-raider-ground'}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:
        return r.read()


def main(out):
    for src in ASSETS:
        info = json.loads(get(f'https://api.polyhaven.com/files/{src}'))['gltf']['1k']['gltf']
        base = os.path.join(out, src)
        os.makedirs(base, exist_ok=True)
        files = {f'{src}.gltf': info['url'], **{k: v['url'] for k, v in info['include'].items()}}
        for rel, url in files.items():
            path = os.path.join(base, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                with open(path, 'wb') as fh:
                    fh.write(get(url))
        print(src, flush=True)


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--out') + 1])
