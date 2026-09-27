"""Nachbearbeitung: EXR-Passes -> fertiges MP4 im Action-Cam-Look.

Schritte je Frame (linear):
  1. Objekte (Combined, Alpha) + Himmel (Env-Pass) zusammensetzen
  2. Atmosphärischer Dunst aus Mist-Pass (exponentiell, Sonnen-Streulicht)
  3. Auto-Belichtung wie eine Action-Cam (geglättet, ~0,4 s, teilweise Kompensation)
  4. AgX-Tonemapping (Blender OCIO-Config) + dezente Farbkorrektur
  5. dezentes Sensorrauschen + Vignette ("realismus_grade dezent")
Danach ffmpeg: Lanczos auf 1080x1920, leichtes Nachschärfen, H.264 High, CRF 17 (max. 24 Mbit/s).
"""
import argparse
import glob
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import OpenEXR
import PyOpenColorIO as ocio
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def ocio_config():
    import importlib.util
    spec = importlib.util.find_spec("bpy")
    bases = list(spec.submodule_search_locations or []) + [os.path.dirname(spec.origin or "")]
    cands = []
    for base in bases:
        cands += glob.glob(os.path.join(base, "*", "datafiles", "colormanagement", "config.ocio"))
        cands += glob.glob(os.path.join(base, "bpy", "*", "datafiles", "colormanagement", "config.ocio"))
    if os.environ.get("NIDO_OCIO"):
        cands.insert(0, os.environ["NIDO_OCIO"])
    return ocio.Config.CreateFromFile(cands[0])


def read_exr(path):
    with OpenEXR.File(path) as f:
        ch = f.channels()
        out = {}
        for name, c in ch.items():
            out[name] = np.asarray(c.pixels, dtype=np.float32)
    return out


def _first(ch, n):
    """Kanäle einer Einzel-EXR als Array mit n Komponenten."""
    keys = sorted(ch)
    if len(keys) == 1 and ch[keys[0]].ndim == 3:
        return ch[keys[0]][..., :n]
    order = {"R": 0, "G": 1, "B": 2, "A": 3, "V": 0, "Y": 0, "Z": 0}
    arrs = sorted(((order.get(k.split(".")[-1], 9), ch[k]) for k in keys), key=lambda x: x[0])
    return np.stack([a for _, a in arrs][:n], -1)


def load_frame(image_path):
    base = image_path[: -len("Image.exr")]
    img = read_exr(image_path)
    rgba = _first(img, 4)
    if rgba.shape[-1] == 3:
        rgba = np.concatenate([rgba, np.ones(rgba.shape[:2] + (1,), np.float32)], -1)
    mist = _first(read_exr(base + "Mist.exr"), 1)[..., 0]
    env = _first(read_exr(base + "Env.exr"), 3)
    return rgba[..., :3], rgba[..., 3], mist, env


def composite(path, P):
    obj, alpha, mist, env = load_frame(path)
    dist = mist * P["mist_depth"]
    # Dunst: Transmission + Einstreuung
    L = P["haze_dist"]
    T = np.exp(-np.maximum(dist - P.get("haze_start", 0.0), 0) / L)
    haze = np.array(P["haze_color"], np.float32)
    a3 = alpha[..., None]
    T3 = T[..., None]
    obj_h = obj * T3 + haze * (1 - T3) * a3
    img = obj_h + env
    return img.astype(np.float32)


def luminance(img):
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def _stage1(args):
    path, P = args
    img = composite(path, P)
    Y = luminance(img)
    h, w = Y.shape
    # zentrumsgewichtete Messung wie eine Action-Cam
    yy, xx = np.mgrid[0:h, 0:w]
    wgt = np.exp(-(((xx - w / 2) / (0.45 * w)) ** 2 + ((yy - h * 0.55) / (0.45 * h)) ** 2))
    logavg = float(np.sum(wgt * np.log(np.maximum(Y, 1e-4))) / np.sum(wgt))
    return logavg


def tonemap(img, look, proc_holder={}):
    key = look or ""
    if key not in proc_holder:
        cfg = ocio_config()
        dv = ocio.DisplayViewTransform()
        dv.setSrc("Linear Rec.709")
        dv.setDisplay("sRGB")
        dv.setView("AgX")
        if look:
            ctx = cfg.getCurrentContext()
            lvp = ocio.LegacyViewingPipeline()
            lvp.setDisplayViewTransform(dv)
            lvp.setLooksOverrideEnabled(True)
            lvp.setLooksOverride(look)
            proc = lvp.getProcessor(cfg, ctx)
        else:
            proc = cfg.getProcessor(dv)
        proc_holder[key] = proc.getDefaultCPUProcessor()
    cpu = proc_holder[key]
    buf = np.ascontiguousarray(img.astype(np.float32))
    cpu.applyRGB(buf)
    return buf


def _blur(img, sigma):
    """Separabler Gauß (numpy) für kleine Bilder."""
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-x * x / (2 * sigma * sigma))
    k /= k.sum()
    out = img
    for axis in (0, 1):
        pad = [(0, 0)] * img.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        acc = np.zeros_like(out)
        for i, wgt in enumerate(k):
            sl = [slice(None)] * img.ndim
            sl[axis] = slice(i, i + out.shape[axis])
            acc += wgt * p[tuple(sl)]
        out = acc
    return out


def bloom(img, thr=2.5, strength=0.35, f=8):
    """Leuchten heller Quellen (Energieattacken, Blitze, Sonnenglanz): Hochpass über thr, verkleinert
    weichgezeichnet (zwei Radien), bilinear zurück."""
    h, w, _ = img.shape
    hh, ww = h // f, w // f
    b = np.maximum(img - thr, 0)[:hh * f, :ww * f].reshape(hh, f, ww, f, 3).mean((1, 3))
    if b.max() <= 0:
        return img
    bb = 0.55 * _blur(b, 1.5) + 0.45 * _blur(b, 6.0)
    up = np.stack([np.asarray(Image.fromarray(bb[..., c].astype(np.float32), mode="F").resize((w, h), Image.BILINEAR))
                   for c in range(3)], -1)
    return img + strength * up


def _stage2(args):
    i, path, P, ev, outdir = args
    img = composite(path, P)
    img *= 2.0 ** ev
    # dezente Farbkorrektur vor Tonemapping (linear)
    wb = np.array(P.get("wb", (1, 1, 1)), np.float32)
    img *= wb
    Y = luminance(img)[..., None]
    sat = P.get("sat", 1.0)
    img = Y + (img - Y) * sat
    img = np.maximum(img, 0)
    if P.get("bloom"):
        img = bloom(img, thr=P.get("bloom_thr", 2.5), strength=P["bloom"])
    out = tonemap(img, P.get("look"))
    out = np.clip(out, 0, 1)
    h, w = out.shape[:2]
    # Vignette
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 * 0.8 + ((yy - h / 2) / (h / 2)) ** 2 * 0.6
    vig = 1 - P.get("vignette", 0.18) * np.clip(r2, 0, 1.6) ** 1.5 / 1.9
    out *= vig[..., None]
    # Sensorrauschen (luminanzabhängig, fein)
    rng = np.random.default_rng(1000 + i)
    g = P.get("grain", 0.012)
    if g:
        lum = out.mean(-1, keepdims=True)
        n = rng.normal(0, 1, out.shape[:2] + (1,)).astype(np.float32)
        nc = rng.normal(0, 1, out.shape).astype(np.float32) * 0.35
        out += (n + nc) * g * (0.5 + 0.8 * np.sqrt(np.clip(lum, 0, 1))) * (1.1 - lum)
    out = np.clip(out, 0, 1)
    # Frames liegen unten-links bei EXR? OpenEXR liefert oben-links -> direkt speichern
    im = Image.fromarray((out * 255 + 0.5).astype(np.uint8))
    im.save(os.path.join(outdir, f"p_{i:04d}.png"), compress_level=1)
    return i


def process(exr_dir, out_mp4, P, fps=24, size=(1080, 1920), workers=4, crf=17):
    files = sorted(glob.glob(os.path.join(exr_dir, "f_*Image.exr")))
    if not files:
        raise SystemExit("keine EXR-Frames gefunden")
    with ProcessPoolExecutor(workers) as ex:
        logs = list(ex.map(_stage1, [(f, P) for f in files], chunksize=4))
    logs = np.array(logs)
    # Auto-Belichtung: exponentielle Glättung (tau), teilweise Kompensation
    tau = P.get("ae_tau", 0.45) * fps
    sm = np.zeros_like(logs)
    sm[0] = logs[0]
    a = 1 - math.exp(-1 / tau)
    for k in range(1, len(logs)):
        sm[k] = sm[k - 1] + a * (logs[k] - sm[k - 1])
    ref = P.get("ae_ref", float(np.median(logs)))
    k_ae = P.get("ae_strength", 0.55)
    evs = P.get("ev", 0.0) - k_ae * (sm - ref) / math.log(2)
    pngdir = os.path.join(exr_dir, "png")
    os.makedirs(pngdir, exist_ok=True)
    for f in glob.glob(os.path.join(pngdir, "*.png")):
        os.remove(f)
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_stage2, [(i, f, P, float(evs[i]), pngdir) for i, f in enumerate(files)], chunksize=2))
    ff = ffmpeg_bin()
    vf = (f"scale={size[0]}:{size[1]}:flags=lanczos,"
          "unsharp=5:5:0.35:5:5:0.0,format=yuv420p")
    cmd = [ff, "-y", "-framerate", str(fps), "-i", os.path.join(pngdir, "p_%04d.png"),
           "-vf", vf, "-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", str(crf),
           "-maxrate", "24M", "-bufsize", "48M",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709", out_mp4]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", out_mp4, "EV range", evs.min(), evs.max())


def ffmpeg_bin():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def still(exr_path, png_path, P, ev=0.0):
    img = composite(exr_path, P) * 2.0 ** ev
    img = np.maximum(img * np.array(P.get("wb", (1, 1, 1)), np.float32), 0)
    Y = luminance(img)[..., None]
    img = Y + (img - Y) * P.get("sat", 1.0)
    out = np.clip(tonemap(np.maximum(img, 0), P.get("look")), 0, 1)
    Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save(png_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exr_dir")
    ap.add_argument("out")
    ap.add_argument("--params", required=True, help="Python-Datei mit POST = dict(...)")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()
    ns = {}
    exec(open(a.params).read(), ns)
    process(a.exr_dir, a.out, ns["POST"], fps=a.fps)
