"""Welt-Szene bauen und als .blend speichern (Render-Einstellungen bleiben im blend).

  python build_scene.py <code_dir> <welt_modul> <exr_out_dir> <blend_path> <BREITExHÖHE> <samples>

<code_dir> = Ordner drohnen_3d_welten des Repos; <welt_modul> = z. B. world_dragonball (Modul mit build(args)).
Modelle unter <code_dir>/assets/models/... (oder NIDO_ASSETS setzen).
"""
import argparse
import os
import sys

import importlib

code, modname, out, blend = sys.argv[1:5]
res = tuple(int(v) for v in sys.argv[5].split("x"))
spp = int(sys.argv[6])
sys.path.insert(0, os.path.abspath(code))
sys.argv = ["x"]
import bpy  # noqa: E402
W = importlib.import_module(modname)

os.makedirs(out, exist_ok=True)
W.build(argparse.Namespace(out=out, res=res, samples=spp, no_mblur=False, frames=None, save_blend=None))
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(blend))
print("SAVED", blend, flush=True)
