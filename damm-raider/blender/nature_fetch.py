# Lädt die CC0-Naturmodelle von Poly Haven (glTF, 1k-Texturen) für blender/nature_build.py.
#   python3 -I blender/nature_fetch.py --out <downloads>
# Lizenz: CC0, https://polyhaven.com/license
import argparse
import json
import os
import urllib.request

ASSETS = ['rock_moss_set_01', 'namaqualand_boulder_05', 'namaqualand_cliff_01', 'fir_sapling_medium', 'island_tree_02', 'searsia_lucida']
UA = {'User-Agent': 'damm-raider-nature-fetch'}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    for aid in ASSETS:
        info = json.loads(get(f'https://api.polyhaven.com/files/{aid}'))['gltf']['1k']['gltf']
        base = os.path.join(args.out, aid)
        os.makedirs(base, exist_ok=True)
        files = {f'{aid}.gltf': info['url'], **{k: v['url'] for k, v in info['include'].items()}}
        for rel, url in files.items():
            path = os.path.join(base, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                with open(path, 'wb') as fh:
                    fh.write(get(url))
        print(aid, len(files), 'Dateien', flush=True)


if __name__ == '__main__':
    main()
