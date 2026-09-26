# NIDO – 3D-Welten mit FPV-Drohnenflug (One Piece · Naruto · Dragon Ball)

Drei selbst gebaute, fotorealistisch gerenderte 3D-Welten (Blender 5.0 / Cycles), jeweils mit
einem durchgehenden 20-Sekunden-FPV-Flug im Hochformat 1080×1920 – nach den NIDO-Hausregeln:
eine einzige Aufnahme ohne Schnitt, konstantes Tempo, Manöver nur an festen Objekten,
Wow-Motiv hinter einer Verdeckung, stabilisierte Weitwinkel-Action-Cam (kein Fisheye).

| Video | Welt | Flug |
|---|---|---|
| `videos/01_OnePiece_ThousandSunny.mp4` | Grand Line: Karstfelsen im Meer, Thousand Sunny | tief über den Wellen durch ein Felsentor, Linkskurve um die große Felsnadel → Reveal der Thousand Sunny (Löwenkopf, Jolly-Roger-Segel), Vorbeiflug am Bug |
| `videos/02_Naruto_Konoha.mp4` | Konohagakure mit Hokage-Felsen | Waldstraße → Durchflug durch das あ/ん-Tor → Hauptstraße auf Dachhöhe (Laternen, Wassertanks) → S-Kurve um den alten Baum → Steigflug an der 火-Residenz vorbei → die fünf Hokage-Gesichter füllen das Bild |
| `videos/03_DragonBall_Namek.mp4` | Planet Namek | türkiser Himmel mit drei Sonnen, grünes Meer, Pilzfelsen → Steigflug an der Tafelberg-Wand → Reveal über der Kante: Namekianer-Dorf, sieben leuchtende Dragon Balls, Friezas Raumschiff |

## Aufbau

| Datei | Inhalt |
|---|---|
| `fpv.py` | Render-Setup, Himmel + Wolken, **FPV-Kamera** (konstantes Bahn-Tempo, Schräglage aus der Kurvenkrümmung, Piloten-/Mikrokorrekturen, Bewegungsunschärfe 180°) |
| `post.py` | Action-Cam-Look: Dunst aus Tiefenpass, geglättete Auto-Belichtung, AgX-Tonemapping, dezentes Sensorrauschen/Vignette, Lanczos auf 1080×1920, H.264 |
| `ocean.py`, `nature.py` | FFT-Ozean mit Schaum/Kielwasser, Fels, Gras, Bäume, Ajisa-Bäume |
| `sunny.py`, `konoha.py` | Thousand Sunny; Konoha-Gebäude, Tor, Hokage-Residenz, Hokage-Felsen |
| `world_*.py` | die drei Welten inkl. Flugroute |
| `params/*.py` | Farbkorrektur/Dunst je Welt |

## Vorlagen aus dem Internet

- **Kopf-Scan „Lee Perry-Smith“** (Infinite-Realities, CC BY 3.0, aus den three.js-Beispielen) als Basis
  der in Stein gehauenen Hokage-Gesichter; Haarformen (Hashirama, Tobirama, Hiruzen, Minato, Tsunade)
  sind selbst modelliert.
- Holz- und Grastexturen aus den three.js-Beispiel-Assets (`fetch_assets.sh` lädt alles nach `assets/`).
- Design-Referenzen per Websuche: Thousand Sunny (Brigantine, Löwen-Galionsfigur mit Sonnenmähne,
  Rasendeck), Konoha (Hokage-Felsen über dem Dorf, Residenz mittig darunter), Namek (türkiser Himmel,
  grünes Wasser, blau-grünes Gras, Ajisa-Bäume, weiße Kuppelhäuser, drei Sonnen).
- Jolly Roger, あ/ん, 火 und Dragon-Ball-Sterne sind selbst gezeichnet (`textures.py`).

## Selbst rendern (z. B. auf dem Mac mit GPU – deutlich schneller)

```bash
pip install bpy==5.0.1 numpy pillow imageio-ffmpeg OpenEXR opencolorio
./fetch_assets.sh
python world_naruto.py --out /tmp/na --res 1080x1920 --samples 48   # nutzt Metal/GPU automatisch
python post.py /tmp/na videos/02_Naruto_Konoha.mp4 --params params/naruto.py --fps 24
```

Die Cloud-Renderings wurden aus Zeitgründen (4 CPU-Kerne, keine GPU) mit 720×1280 und 10–12 Samples
+ KI-Entrauschen gerendert und im Post auf 1080×1920 skaliert. Mit GPU lohnt `--res 1080x1920 --samples 48`.
`--frames 1,120,240` rendert nur einzelne Vorschaubilder; das Rendern ist fortsetzbar (fertige Frames
werden übersprungen).
