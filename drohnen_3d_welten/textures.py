"""Selbst gezeichnete Texturen (PIL): Jolly Roger, Kanji-Schilder, Dragon-Ball-Sterne."""
import math
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = os.environ.get("NIDO_TEX", os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "gen"))
os.makedirs(OUT, exist_ok=True)
JP_FONT = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"


def _skull(d, cx, cy, s, fill, hole, hat=True, hat_fill=None, band_fill=None, outline=None):
    # Knochen (gekreuzt)
    bw = int(0.16 * s)
    for ang in (35, -35):
        a = math.radians(ang)
        dx, dy = math.cos(a) * 1.25 * s, math.sin(a) * 1.25 * s
        d.line([(cx - dx, cy + 0.35 * s - dy), (cx + dx, cy + 0.35 * s + dy)], fill=fill, width=bw)
        for sx, sy in ((cx - dx, cy + 0.35 * s - dy), (cx + dx, cy + 0.35 * s + dy)):
            for off in (-1, 1):
                ox, oy = -math.sin(a) * 0.1 * s * off, math.cos(a) * 0.1 * s * off
                r = 0.13 * s
                d.ellipse([sx + ox - r, sy + oy - r, sx + ox + r, sy + oy + r], fill=fill)
    # Schädel
    d.ellipse([cx - 0.62 * s, cy - 0.75 * s, cx + 0.62 * s, cy + 0.45 * s], fill=fill)
    d.rounded_rectangle([cx - 0.36 * s, cy + 0.1 * s, cx + 0.36 * s, cy + 0.72 * s], radius=int(0.12 * s), fill=fill)
    # Augen, Nase, Zähne
    for ex in (-0.26, 0.26):
        d.ellipse([cx + (ex - 0.17) * s, cy - 0.22 * s, cx + (ex + 0.17) * s, cy + 0.14 * s], fill=hole)
    d.polygon([(cx, cy + 0.18 * s), (cx - 0.08 * s, cy + 0.34 * s), (cx + 0.08 * s, cy + 0.34 * s)], fill=hole)
    for tx in (-0.2, -0.07, 0.07, 0.2):
        d.line([(cx + tx * s, cy + 0.48 * s), (cx + tx * s, cy + 0.70 * s)], fill=hole, width=max(2, int(0.035 * s)))
    if hat:
        hf = hat_fill or fill
        bf = band_fill or hole
        # Krempe
        d.ellipse([cx - 1.0 * s, cy - 0.78 * s, cx + 1.0 * s, cy - 0.48 * s], fill=hf, outline=outline,
                  width=max(1, int(0.03 * s)) if outline else 0)
        # Kuppel
        d.pieslice([cx - 0.6 * s, cy - 1.35 * s, cx + 0.6 * s, cy - 0.35 * s], 180, 360, fill=hf, outline=outline,
                   width=max(1, int(0.03 * s)) if outline else 0)
        # Band
        d.rectangle([cx - 0.6 * s, cy - 0.78 * s, cx + 0.6 * s, cy - 0.66 * s], fill=bf)


def jolly_roger_flag(path=None, W=1536, H=1024):
    """Schwarze Flagge: weißer Schädel, gelber Strohhut mit rotem Band."""
    path = path or os.path.join(OUT, "jolly_flag.png")
    im = Image.new("RGB", (W, H), (8, 8, 9))
    d = ImageDraw.Draw(im)
    _skull(d, W / 2, H / 2 + 60, 300, fill=(236, 232, 222), hole=(8, 8, 9), hat=True,
           hat_fill=(228, 182, 58), band_fill=(170, 22, 20))
    im = im.filter(ImageFilter.GaussianBlur(1.2))
    im.save(path)
    return path


def _skull_outlined(d, cx, cy, s, ow, white, black, hat_fill, band_fill):
    """Jolly Roger wie auf dem Model Sheet: jedes Teil erst schwarz (Kontur, Breite ow), dann farbig."""
    bw = int(0.16 * s)
    bones = []
    for ang in (35, -35):
        a = math.radians(ang)
        dx, dy = math.cos(a) * 1.25 * s, math.sin(a) * 1.25 * s
        p0, p1 = (cx - dx, cy + 0.35 * s - dy), (cx + dx, cy + 0.35 * s + dy)
        knobs = []
        for sx, sy in (p0, p1):
            for off in (-1, 1):
                knobs.append((sx - math.sin(a) * 0.1 * s * off, sy + math.cos(a) * 0.1 * s * off))
        bones.append((p0, p1, knobs))
    r = 0.13 * s
    for grow, fill in ((ow, black), (0, white)):
        for p0, p1, knobs in bones:
            d.line([p0, p1], fill=fill, width=bw + 2 * grow)
            for (kx, ky) in knobs:
                d.ellipse([kx - r - grow, ky - r - grow, kx + r + grow, ky + r + grow], fill=fill)
    for grow, fill in ((ow, black), (0, white)):
        d.ellipse([cx - 0.62 * s - grow, cy - 0.75 * s - grow, cx + 0.62 * s + grow, cy + 0.45 * s + grow], fill=fill)
        d.rounded_rectangle([cx - 0.36 * s - grow, cy + 0.1 * s - grow, cx + 0.36 * s + grow, cy + 0.72 * s + grow],
                            radius=int(0.12 * s), fill=fill)
    for ex in (-0.26, 0.26):
        d.ellipse([cx + (ex - 0.17) * s, cy - 0.22 * s, cx + (ex + 0.17) * s, cy + 0.14 * s], fill=black)
    d.polygon([(cx, cy + 0.18 * s), (cx - 0.08 * s, cy + 0.34 * s), (cx + 0.08 * s, cy + 0.34 * s)], fill=black)
    for tx in (-0.2, -0.07, 0.07, 0.2):
        d.line([(cx + tx * s, cy + 0.48 * s), (cx + tx * s, cy + 0.70 * s)], fill=black, width=max(2, int(0.035 * s)))
    # Strohhut: Krempe, Kuppel, rotes Band – mit Kontur
    for grow, fill in ((ow, black), (0, hat_fill)):
        d.ellipse([cx - 1.0 * s - grow, cy - 0.78 * s - grow, cx + 1.0 * s + grow, cy - 0.48 * s + grow], fill=fill)
        d.pieslice([cx - 0.6 * s - grow, cy - 1.35 * s - grow, cx + 0.6 * s + grow, cy - 0.35 * s + grow], 180, 360,
                   fill=fill)
    d.rectangle([cx - 0.6 * s, cy - 0.78 * s, cx + 0.6 * s, cy - 0.64 * s], fill=band_fill)
    d.line([(cx - 0.6 * s, cy - 0.78 * s), (cx + 0.6 * s, cy - 0.78 * s)], fill=black, width=max(2, ow // 2))
    d.line([(cx - 0.6 * s, cy - 0.64 * s), (cx + 0.6 * s, cy - 0.64 * s)], fill=black, width=max(2, ow // 2))


def jolly_roger_sail(path=None, W=2048, H=2048):
    """Segel-Aufdruck nach dem Model Sheet (決定稿): weißer Schädel und Knochen mit schwarzer Kontur,
    schwarze Augenhöhlen, gelber Strohhut mit rotem Band – auf transparentem Grund (Alpha)."""
    path = path or os.path.join(OUT, "jolly_sail_color.png")
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    _skull_outlined(d, W / 2, H * 0.42 + 90, 500, 26, white=(244, 240, 230, 255), black=(14, 12, 11, 255),
                    hat_fill=(232, 184, 52, 255), band_fill=(184, 26, 22, 255))
    im = im.filter(ImageFilter.GaussianBlur(1.5))
    im.save(path)
    return path


def number_decal(text, path, W=512, fg=(16, 14, 12)):
    """Schwarze Ziffer (z. B. die „1“ des Soldier Dock) auf transparentem Grund."""
    im = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(JP_FONT, int(W * 0.78))
    bb = d.textbbox((0, 0), text, font=font)
    d.text(((W - (bb[2] - bb[0])) / 2 - bb[0], (W - (bb[3] - bb[1])) / 2 - bb[1]), text, font=font, fill=(*fg, 255))
    im = im.filter(ImageFilter.GaussianBlur(1.0))
    im.save(path)
    return path


def kanji_disc(char, path, W=1024, fg=(180, 20, 16), bg=(236, 230, 214), ring=None):
    im = Image.new("RGB", (W, W), bg)
    d = ImageDraw.Draw(im)
    if ring:
        d.ellipse([W * 0.04, W * 0.04, W * 0.96, W * 0.96], outline=ring, width=int(W * 0.05))
    font = ImageFont.truetype(JP_FONT, int(W * 0.66))
    bb = d.textbbox((0, 0), char, font=font)
    d.text(((W - (bb[2] - bb[0])) / 2 - bb[0], (W - (bb[3] - bb[1])) / 2 - bb[1]), char, font=font, fill=fg)
    im = im.filter(ImageFilter.GaussianBlur(1.0))
    im.save(path)
    return path


def kanji_alpha(char, path, W=1024, fg=(20, 20, 20)):
    im = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(JP_FONT, int(W * 0.8))
    bb = d.textbbox((0, 0), char, font=font)
    d.text(((W - (bb[2] - bb[0])) / 2 - bb[0], (W - (bb[3] - bb[1])) / 2 - bb[1]), char, font=font, fill=(*fg, 255))
    im = im.filter(ImageFilter.GaussianBlur(1.0))
    im.save(path)
    return path


def star_map(n, path, W=1024, H=512):
    """Equirect-Textur für eine Dragon-Ball-Kugel mit n roten Sternen auf der Vorderseite."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    layouts = {
        1: [(0, 0)], 2: [(-1, 0), (1, 0)], 3: [(-1, 0.6), (1, 0.6), (0, -0.9)],
        4: [(-1, -1), (1, -1), (-1, 1), (1, 1)], 5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)],
        6: [(-1, -1.1), (1, -1.1), (-1.3, 0.2), (1.3, 0.2), (-0.6, 1.3), (0.6, 1.3)],
        7: [(0, 0), (-1.2, -0.7), (1.2, -0.7), (-1.2, 0.7), (1.2, 0.7), (0, -1.4), (0, 1.4)],
    }
    cx, cy = W * 0.5, H * 0.5
    sp = W * 0.028
    r = W * 0.022
    for (ox, oy) in layouts[n]:
        x, y = cx + ox * sp * 1.6, cy + oy * sp * 1.6
        pts = []
        for k in range(10):
            a = -math.pi / 2 + k * math.pi / 5
            rr = r if k % 2 == 0 else r * 0.42
            pts.append((x + rr * math.cos(a), y + rr * math.sin(a) * 1.0))
        d.polygon(pts, fill=(200, 18, 12, 255))
    im = im.filter(ImageFilter.GaussianBlur(0.8))
    im.save(path)
    return path
