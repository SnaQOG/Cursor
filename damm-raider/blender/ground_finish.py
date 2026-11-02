# Atlas der Bodenpflanzen nachbearbeiten und als assets/tex/cover_c.webp speichern (nur Pillow):
#   python -I blender/ground_finish.py <atlas.png> <ziel.webp>
# Die Samenstände der Gräser rendern schwarz: sie werden strohfarben, das Grün wird heller und frischer.
import sys

from PIL import Image

im = Image.open(sys.argv[1]).convert('RGBA')
px = im.load()
for y in range(im.height):
    for x in range(im.width):
        r, g, b, a = px[x, y]
        lum = (r + g + b) / 3
        spread = max(r, g, b) - min(r, g, b)
        if lum < 62 and spread < 22:
            t = 0.55 + lum / 62 * 0.75
            r, g, b = 150 * t, 128 * t, 84 * t
        else:
            k = 1.55
            r, g, b = min(255, r * k), min(255, g * k * 1.04), min(255, b * k * 0.92)
        px[x, y] = (int(r), int(g), int(b), a)
im.save(sys.argv[2], 'WEBP', quality=86, method=6, exact=True)
print(sys.argv[2], im.size)
