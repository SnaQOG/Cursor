# Lädt nur die in simready_assets.json eingetragenen SimReady-Pakete von Hugging Face.
#
#   python -m venv .venv-hf && .venv-hf/bin/pip install huggingface_hub
#   .venv-hf/bin/python -I blender/simready_fetch.py --out <ordner> [id ...]
#
# Jedes Asset landet in einem eigenen Ordner <ordner>/<id>/. Die Downloads sind fremde Daten:
# Der Ordner gehört nicht ins Repo, Skripte aus den Paketen (*.py) werden gar nicht erst geladen.
import argparse
import json
import os

from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
# source_assets: nur MDL-Shader und Kopien der Texturen, die Materialien nutzen Textures/
IGNORE = ['*.py', '*/.thumbs/*', '*/physics/*', '*/runnables/*', '*_physx*', '*_physics*', '*/guides/*', '*/source_assets/*', '*.mp4', '*.gif']


def patterns(a):
    f = a['folder']
    allow = [f'{f}/com.nvidia.simready.packaging.json', f'{f}/simready_usd/*']
    # Scans: nur die reduzierte Stufe, die volle hat bis zu 2,6 Mio. Dreiecke und 8K-Texturen
    ignore = IGNORE + (['*_lod00*'] if a.get('lod1') else [])
    return allow, ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('ids', nargs='*')
    args = ap.parse_args()
    with open(os.path.join(HERE, 'simready_assets.json'), encoding='utf-8') as fh:
        assets = json.load(fh)['assets']
    for a in assets:
        if args.ids and a['id'] not in args.ids:
            continue
        allow, ignore = patterns(a)
        print(a['id'], '<-', a['repo'], a['folder'], '@', a['rev'][:8], flush=True)
        if args.dry_run:
            continue
        snapshot_download(a['repo'], repo_type='dataset', revision=a['rev'], allow_patterns=allow, ignore_patterns=ignore,
                          local_dir=os.path.join(args.out, a['id']), max_workers=6)


if __name__ == '__main__':
    main()
