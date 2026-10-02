"""Vorschau: Szene in Vorschau-Qualität bauen (oder vorhandenes scene.blend öffnen), dann Standbilder oder eine
Sequenz rendern. Alle Qualitätswerte kommen als Argumente (pro Video festlegen und im Werte-Archiv notieren).

  python preview.py <code_dir> <welt_modul> <params.py> <out_dir> stills <t1,t2,...> <BxH> <samples>
  python preview.py <code_dir> <welt_modul> <params.py> <out_dir> seq <f0> <f1> <BxH> <samples> <AUSGABE_BxH> <out.mp4>

PREVIEW_BITRATE (z. B. 6M) setzt die Vorschau-Bitrate.
"""
import argparse
import importlib
import os
import sys
import time

code, modname, params, out, mode = sys.argv[1:6]
rest = sys.argv[6:]
sys.path.insert(0, os.path.abspath(code))
sys.argv = ["x"]
import bpy  # noqa: E402
import fpv  # noqa: E402
import post  # noqa: E402

W = importlib.import_module(modname)
os.makedirs(out, exist_ok=True)
ns = {}
exec(open(params).read(), ns)
P = ns["POST"]
blend = os.path.join(out, "scene.blend")
if os.path.exists(blend):
    bpy.ops.wm.open_mainfile(filepath=blend)
    sc = bpy.context.scene
else:
    res = tuple(int(v) for v in (rest[1] if mode == "stills" else rest[2]).split("x"))
    spp = int(rest[2] if mode == "stills" else rest[3])
    sc = W.build(argparse.Namespace(out=out, res=res, samples=spp, no_mblur=False, frames=None, save_blend=None))
    bpy.ops.wm.save_as_mainfile(filepath=blend)
fps = sc.render.fps
if mode == "stills":
    res, spp = tuple(int(v) for v in rest[1].split("x")), int(rest[2])
    for t in [float(x) for x in rest[0].split(",")]:
        sc.render.resolution_x, sc.render.resolution_y = res
        sc.cycles.samples = spp
        f = int(round(t * fps)) + 1
        sc.frame_set(f)
        t0 = time.time()
        exr = fpv.render_still(out, f"still_t{t:05.2f}")
        post.still(exr, os.path.join(out, f"still_t{t:05.2f}_{f:04d}.png"), P, ev=P.get("ev", 0))
        print("SHOT", t, round(time.time() - t0, 1), flush=True)
else:
    f0, f1 = int(rest[0]), int(rest[1])
    res, spp = tuple(int(v) for v in rest[2].split("x")), int(rest[3])
    size, mp4 = tuple(int(v) for v in rest[4].split("x")), rest[5]
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.cycles.samples = spp
    seq = os.path.join(out, "seq")
    fpv.set_output(seq, "f_####")
    fpv.render_frames(seq, range(f0, f1 + 1))
    P["bitrate"] = os.environ.get("PREVIEW_BITRATE", P.get("bitrate"))
    post.process(seq, mp4, P, size=size, workers=4, crf=20)
    print("MP4", mp4, flush=True)
