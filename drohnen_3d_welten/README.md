# NIDO – 3D-Welten mit FPV-Drohnenflug (One Piece · Naruto · Dragon Ball)

Drei selbst gebaute, fotorealistisch gerenderte 3D-Welten (Blender 5.0 / Cycles), jeweils mit
einem durchgehenden 20-Sekunden-FPV-Flug im Hochformat 1080×1920 – nach den NIDO-Hausregeln:
eine einzige Aufnahme ohne Schnitt, konstantes Tempo, Manöver nur an festen Objekten,
Wow-Motiv hinter einer Verdeckung, stabilisierte Weitwinkel-Action-Cam (kein Fisheye).

| Video | Welt | Flug |
|---|---|---|
| `videos/01_OnePiece_ThousandSunny.mp4` | Grand Line: tiefblaues Meer, Karstfelsen, Thousand Sunny | tief über den Wellen durch ein Felsentor, Linkskurve um die große Felsnadel → Reveal der Thousand Sunny, die frontal entgegenkommt (Sonnen-Löwe, gekreuzte Knochen, Jolly-Roger-Segel) → Endbild dicht vor dem Bug |
| `videos/02_Naruto_Konoha.mp4` | Konohagakure mit Hokage-Felsen | Waldstraße → Durchflug durch das あ/ん-Tor → sandfarbene Hauptstraße mit Pastellhäusern, bunten Dächern, Wassertanks und Laternen → S-Kurve um den alten Baum → Steigflug über das Flachdach der 火-Residenz, auf dem Naruto und Sasuke zum Felsen hinaufschauen → die fünf Hokage-Gesichter im Zickzack füllen das Bild |
| `videos/03_DragonBall_Namek.mp4` | Planet Namek | grasgrüner Himmel mit drei Sonnen, grünes Meer, beige Felsnadeln mit blauen Ajisa-Bäumen → Steigflug an der Tafelberg-Wand → Reveal über der Kante: blaues Gras mit Sandflecken und roten Pilzen, Namekianer-Dorf, sieben Dragon Balls auf einem Steinsockel, Friezas Raumschiff → Flug über die Nordkante aufs Meer |

## Aufbau

| Datei | Inhalt |
|---|---|
| `fpv.py` | Render-Setup, Himmel + Wolken, **FPV-Kamera** (konstantes Bahn-Tempo, Schräglage aus der Kurvenkrümmung, Piloten-/Mikrokorrekturen, Bewegungsunschärfe 180°) |
| `post.py` | Action-Cam-Look: Dunst aus Tiefenpass, geglättete Auto-Belichtung, AgX-Tonemapping, dezentes Sensorrauschen/Vignette, Lanczos auf 1080×1920, H.264 |
| `ocean.py` | FFT-Ozean (Ocean-Modifier) + Geometry Nodes: Bugwelle, Kelvin-Heckwelle (19,47°), Rumpf-Schaum, Brandung über ein Küsten-Abstandsfeld |
| `nature.py` | Fels (Karst, Schichtbänke, Kavität), Palmen, Lianen, Bäume |
| `sunny.py` | Thousand Sunny nach den Model Sheets (三面図/決定稿): bauchiger Planken-Rumpf, rote U-Bordwand mit Voluten und Bullaugen, Soldier-Dock-Ring „1“, Bugschild, Sonnen-Löwe mit gekreuzten Knochen, Vorschiff mit Steuerrad, Achterkastell mit Bogenfenstern, rot-gelbe Kuppeln, Heckkanone, Rasendeck mit Mandarinenbäumen, 20-m-Jolly-Roger-Segel, gestreiftes Gaffelsegel |
| `konoha.py` | Konoha: Pastell-Putzfassaden, bunte Ziegel-/Blechdächer, Stufentürme, Sandstraße, Fenster, Rohre, Stromleitungen, 火-Residenz nach Anime-Vorlage (Fensterreihen, Ziegelkragen, weiße Hörner), Hokage-Felsen aus ockerfarbenem Sandstein (Scan-Köpfe + Haar per Voxel-Remesh verschmolzen, Zickzack-Anordnung, Treppen, Felshütten, Kuppelbauten) |
| `ninja.py` | Naruto und Sasuke (Shippuden) nach den Model Sheets: Skin-Modifier-Körper, Kleidung mit Farbzonen, Stachelhaar, Stirnband, Uzumaki-Spirale und Uchiha-Wappen als Rücken-Decals, Seilgürtel, Kusanagi |
| `namek.py` | Namek: Lehm-Kuppelhäuser (Boolean + Normalen-Transfer), schlanke Ajisa-Bäume mit blauen Kugelkronen, blaue Grashalme, Sandflecken, rote Pilze, Dragon Balls, Friezas Raumschiff |
| `world_*.py` | die drei Welten inkl. Flugroute |
| `params/*.py` | Farbkorrektur/Dunst je Welt |

## Vorlagen aus dem Internet

- **Kopf-Scan „Lee Perry-Smith“** (Infinite-Realities, CC BY 3.0, aus den three.js-Beispielen) als Basis
  der in Stein gehauenen Hokage-Gesichter; Haarformen (Hashirama, Tobirama, Hiruzen, Minato, Tsunade)
  sind selbst modelliert.
- Texturen (`fetch_assets.sh` lädt alles nach `assets/`):
  - three.js-Beispielassets: Gras (MIT)
  - [BabylonJS/Assets](https://github.com/BabylonJS/Assets): Gras, Erde, steiniger Boden (CC BY 4.0)
  - [ambientCG](https://ambientcg.com) über das Repo `ubyjvovk/asciicity`: Putz, Pflaster, Metallplatten (CC0)
  - [Godot TPS-Demo](https://github.com/godotengine/tps-demo): Nietenplatten für Friezas Raumschiff
    (J. Linietsky, F. M. Calabró, CC BY 3.0)
- Farb- und Formreferenzen vom Auftraggeber: Thousand-Sunny-Model-Sheets (Seiten-, Front-, Heck- und
  Deckansicht), One-Piece-Meer/Himmel, Namek-Anime- und Manga-Bilder (Farben per Pixelmessung übernommen), Konoha-Standbilder aus Anime und
  Spiel, Model Sheets von Naruto und Sasuke.
- Design-Referenzen per Websuche: Thousand Sunny (Brigantine, Löwen-Galionsfigur mit Sonnenmähne,
  Rasendeck), Konoha (Hokage-Felsen über dem Dorf, Residenz mittig darunter), Namek (grüner Himmel,
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
