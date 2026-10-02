"""Fortsetzbarer Final-Render: rendert nur fehlende Frames, danach Grade + H.264 (Ausgabegröße, Bitrate aus params).

  python render_final.py <code_dir> <blend> <params.py> <exr_dir> <out.mp4> <BREITExHÖHE> <samples> <AUSGABE_BxH>

Frames gelten als fertig, wenn f_####Image.exr existiert (fpv.frame_done). Nach einem Neustart einfach erneut
aufrufen. Log-Zeilen: FTIME <frame> <s>, SEQTIME, MP4.
"""
import os
import sys
import time

code, blend, params, outdir, mp4 = sys.argv[1:6]
res = tuple(int(v) for v in sys.argv[6].split("x"))
spp = int(sys.argv[7])
size = tuple(int(v) for v in sys.argv[8].split("x"))
sys.path.insert(0, os.path.abspath(code))
sys.argv = ["x"]
import bpy  # noqa: E402
import fpv  # noqa: E402
import post  # noqa: E402

bpy.ops.wm.open_mainfile(filepath=blend)
sc = bpy.context.scene
for ob in bpy.data.objects:                      # Wind-/Flatter-Modifier: keine Deform-Motion-Blur-Kosten
    if any(m.name in ("gust", "flutter") for m in ob.modifiers):
        ob.cycles.use_deform_motion = False
sc.render.resolution_x, sc.render.resolution_y = res
sc.render.resolution_percentage = 100
sc.cycles.samples = spp
fpv.set_output(outdir, "f_####")
t0 = time.time()
for f in range(sc.frame_start, sc.frame_end + 1):
    if fpv.frame_done(outdir, f):
        continue
    t1 = time.time()
    fpv.render_frames(outdir, [f])
    print("FTIME", f, round(time.time() - t1, 1), flush=True)
print("SEQTIME", round(time.time() - t0, 1), flush=True)
ns = {}
exec(open(params).read(), ns)
post.process(outdir, mp4, ns["POST"], size=size, workers=4)
print("MP4", mp4, flush=True)
