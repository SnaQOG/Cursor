"""Kampfplan, Kamerabahn und Ereignisse eines Welt-Moduls als JSON ins Werte-Archiv schreiben (ohne Szene zu bauen).

  python werte_dump.py <code_dir> <welt_modul> <ziel.json>

Erwartet im Welt-Modul (wie world_dragonball): FPS, SECONDS, CAM_KEYS, fight_plan() -> (G, Z, shots, ev),
camera_path(frames, G, Z, shots), frame_at(tau), sound_markers(info), HITS, SHAKES, CAM_HOLDS.
Fehlt etwas, wird der Teil übersprungen.
"""
import importlib
import json
import os
import sys

code, modname, ziel = sys.argv[1:4]
sys.path.insert(0, os.path.abspath(code))
sys.argv = ["x"]
import numpy as np  # noqa: E402

W = importlib.import_module(modname)


def r(v, n=3):
    return [round(float(x), n) for x in v]


def keys(L):
    rows = []
    for (t, pose, loc, look, air, o) in L:
        oo = {k: v for k, v in o.items() if v not in (None, 0.0)}
        if "rot" in oo:
            oo["rot"] = {j: r(x, 1) for j, x in oo["rot"].items()}
        rows.append({"t": round(t, 3), "pose": pose if isinstance(pose, str) else "custom", "ort": r(loc, 2),
                     "blick": r(look, 2), "luft": air, **oo})
    return rows


d = {}
if hasattr(W, "CAM_KEYS"):
    d["kamera_wegpunkte"] = [{"t": t, "pos": r(p, 2)} for t, p in W.CAM_KEYS]
if hasattr(W, "fight_plan"):
    G, Z, shots, ev = W.fight_plan()
    d["figur_a_keys"], d["figur_b_keys"] = keys(G), keys(Z)
    d["strahlen"] = [{"t": t, "von": r(a, 2), "einschlag": r(p, 2)} for t, a, p in shots]
    d["treffer"] = [{"t": t, "ort": r(c, 2), "art": k, "achse": r(ax, 3)} for t, c, k, ax in ev.get("hits", [])]
    d["ki_spuren"] = [{"wer": w, "t0": t0, "t1": t1, "radius": rr} for w, t0, t1, rr in ev.get("trails", [])]
    if hasattr(W, "frame_at"):
        d["zanzoken"] = [{"wer": w, "weg_bis_wieder": list(s), "frame_weg": W.frame_at(s[0]),
                          "frame_wieder": W.frame_at(s[1]), "von": r(a, 2), "nach": r(b, 2)}
                         for w, s, a, b in ev.get("zan", [])]
    if hasattr(W, "camera_path"):
        frames = W.FPS * W.SECONDS
        pos, quats, info = W.camera_path(frames, G, Z, shots)
        samp = []
        for tt in np.arange(0, W.SECONDS + 0.01, 0.5):
            i = int(np.argmin(abs(info["t"] - tt)))
            samp.append({"t": round(float(tt), 2), "pos": r(pos[i], 2),
                         "tempo_ms": round(float(info["v"][min(i, len(info["v"]) - 1)]), 1),
                         "blick_gewicht": round(float(info["look_w"][i]), 2)})
        d["kamera_bahn_alle_0_5s"] = samp
        if hasattr(W, "sound_markers"):
            d["sound_marker"] = [{"name": n, "t": round(t, 2), "frame": int(round(t * W.FPS)) + 1}
                                 for n, t in W.sound_markers({**info, "pos": pos})]
for nm in ("HITS", "SHAKES", "CAM_HOLDS"):
    if hasattr(W, nm):
        d[nm] = getattr(W, nm)
os.makedirs(os.path.dirname(os.path.abspath(ziel)), exist_ok=True)
json.dump(d, open(ziel, "w"), indent=1, ensure_ascii=False)
print("ok", ziel, {k: len(v) if isinstance(v, list) else "" for k, v in d.items()})
