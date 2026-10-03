# NIDO – 3D-Welten mit FPV-Drohnenflug (One Piece · Naruto · Dragon Ball)

Drei selbst gebaute, fotorealistisch gerenderte 3D-Welten (Blender 5.0 / Cycles), jeweils mit
einem durchgehenden 20-Sekunden-FPV-Flug im Hochformat 1080×1920 – nach den NIDO-Hausregeln:
eine einzige Aufnahme ohne Schnitt, konstantes Tempo, Manöver nur an festen Objekten,
Wow-Motiv hinter einer Verdeckung, stabilisierte Weitwinkel-Action-Cam (kein Fisheye).

| Video | Welt | Flug |
|---|---|---|
| `videos/01_OnePiece_ThousandSunny.mp4` | Grand Line: tiefblaues Meer, Karstfelsen, Thousand Sunny mit der ganzen Strohhutbande | tief über den Wellen durch ein Felsentor, Linkskurve um die große Felsnadel → Reveal der Sunny in Breitseite → Anflug quer auf die Steuerbordseite: Ruffy auf dem Löwenkopf, Jinbei am Steuer, Brook mit Geige, Lysop und Chopper winken, Nami an den Mandarinen, Zorro schläft am Mast, Sanji, Robin liest, Franky („SUPER!“) auf dem Achterkastell → Überflug ~4 m über dem Rasendeck zwischen Fockmast und Achterkastell → hinaus aufs offene Meer |
| `videos/02_Naruto_Konoha.mp4` | Konohagakure mit Hokage-Felsen | Waldstraße → Durchflug durch das あ/ん-Tor → Naruto und Sasuke liefern sich vor der Drohne ein Duell über die Hauptstraße (Wandsprünge, Zusammenstöße in der Luft) → S-Kurve um den alten Baum, beide springen aufs Dach der 火-Residenz → Rasengan gegen Chidori: Aufladen, Ansturm, Zusammenprall mit Lichtexplosion und Druckwelle → die fünf Hokage-Gesichter im Zickzack füllen das Bild |
| `videos/03_DragonBall_Namek.mp4` | Planet Namek | grasgrüner Himmel mit drei Sonnen, grünes Meer, beige Felsnadeln → über dem Tafelberg blitzen schon Zusammenstöße → Steigflug an der Wand → über der Kante kämpfen Super-Saiyajin Goku und Freezer direkt vor der Drohne (Schockwellen bei jedem Treffer) → Freezers Todesstrahlen sprengen das Plateau (Staub, Brocken), eine abgelenkte Ki-Kugel trifft eine ferne Felsnadel → die Drohne fliegt unter Goku durch, während er das Kamehameha abfeuert → Strahlenduell, Durchbruch, große Explosion über der Nordkante → weiter aufs Meer |

## Aufbau

| Datei | Inhalt |
|---|---|
| `fpv.py` | Render-Setup, Himmel + Wolken, **FPV-Kamera** (konstantes Bahn-Tempo, Schräglage aus der Kurvenkrümmung, Piloten-/Mikrokorrekturen, Bewegungsunschärfe 180°) |
| `post.py` | Action-Cam-Look: Dunst aus Tiefenpass (für halbtransparente Effekte korrigiert), geglättete Auto-Belichtung, AgX-Tonemapping, Bloom für Energie-Effekte, dezentes Sensorrauschen/Vignette, Lanczos auf 1080×1920, H.264 |
| `ocean.py` | FFT-Ozean (Ocean-Modifier) + Geometry Nodes: Bugwelle, Kelvin-Heckwelle (19,47°), Rumpf-Schaum, Brandung über ein Küsten-Abstandsfeld |
| `nature.py` | Fels (Karst, Schichtbänke, Kavität), Palmen, Lianen, Bäume |
| `sunny.py` | Thousand Sunny nach den Model Sheets (三面図/決定稿): bauchiger Planken-Rumpf, rote U-Bordwand mit Voluten und Bullaugen, Soldier-Dock-Ring „1“, Bugschild, Sonnen-Löwe mit gekreuzten Knochen, Vorschiff mit Steuerrad, Achterkastell mit Bogenfenstern, rot-gelbe Kuppeln, Heckkanone, Rasendeck mit Mandarinenbäumen, 20-m-Jolly-Roger-Segel, gestreiftes Gaffelsegel |
| `konoha.py` | Konoha: Pastell-Putzfassaden, bunte Ziegel-/Blechdächer, Stufentürme, Sandstraße, Fenster, Rohre, Stromleitungen, 火-Residenz nach Anime-Vorlage (Fensterreihen, Ziegelkragen, weiße Hörner), Hokage-Felsen aus ockerfarbenem Sandstein (Scan-Köpfe + Haar per Voxel-Remesh verschmolzen, Zickzack-Anordnung, Treppen, Felshütten, Kuppelbauten) |
| `figures.py` | Gelenk-Rig für alle Figuren: Empty-Hierarchie (Wurzel, Rumpf, Kopf, Schultern/Ellbogen/Handgelenke, Hüften/Knie/Knöchel), Skin-Modifier-Segmente an den Gelenken, Posenbibliothek mit Spiegelung, Bodenkontakt per Vorwärtskinematik |
| `ninja.py` | Naruto und Sasuke (Shippuden) nach den Model Sheets auf dem Rig: Kleidung mit Farbzonen, Stachelhaar, Stirnband, Uzumaki-Spirale und Uchiha-Wappen als Rücken-Decals, Seilgürtel, Kusanagi |
| `dbz.py` | Super-Saiyajin Goku (orange-blauer Gi, goldenes Stachelhaar) und Freezer (Endform, weiß mit lila Panzerteilen, Schwanz) |
| `crew.py` | die Strohhutbande (Ruffy, Zorro, Nami, Lysop, Sanji, Chopper, Robin, Franky, Brook, Jinbei) in Kanon-Größen mit Posen/Animationen und Platz an Bord |
| `vfx.py` | Energie-Effekte: Rasengan/Kamehameha-Ladung (Wirbelkugel), Chidori (flackernde Blitzvarianten), Strahlen mit Längenverlauf fürs Strahlenduell, Explosionen mit Druckwellenring, Ki-Kugeln, Super-Saiyajin-Aura, Staubwolken, Gesteinsbrocken (ballistisch) – alles über Keyframes gesteuert und mit Punktlichtern, die die Umgebung beleuchten |
| `tripo_chars.py` | Tripo-3D-Modelle (Goku SSJ, Freezer Endform) als animierbare Figuren: Gelenkpunkte aus der Mittellinie des voxelisierten Modells, Armature in der Modellpose mit Copy Transforms von den Rig-Empties (kein Zurückbiegen in eine Grundpose), Gewichte über Abstände entlang der Mesh-Oberfläche, Freezers Schwanz als Knochenkette, Toon-Material aus der Farbtextur + Kontur. GLB-Meshes sind an den UV-Nähten aufgetrennt; für die Gewichte werden die Nähte zusammengelegt (sonst reißt Kleidung) |
| `tripo_crew.py` | die Strohhutbande aus Tripo-GLB-Modellen für die Thousand Sunny (Standard, `NIDO_CREW=proc` = selbst gebaute Figuren aus `crew.py`): Mixamo-Gelenke, Franky von Hand vermessen, figurenspezifische Posen (`POSE_MAP`), Robins Buch |
| `repair_glb.py` | repariert unvollständige GLB-Downloads (`.glb.download`): Binärteil auffüllen, Bind-Matrizen aus der Knochenhierarchie neu berechnen, abgeschnittene Bilder entfernen |
| `dbz_fx.py` | Dragon-Ball-Bewegungseffekte: Ki-Spuren hinter schnellen Vorstößen, Zanzoken (Verschwinden, flackerndes Nachbild, Luftring beim Auftauchen) |
| `anime_chars.py` | selbst gebaute Anime-Figuren (Fallback, `NIDO_CHARS=anime`): Toon-Shading, Konturen, Gesichtsausdrücke als Shape Keys |
| `namek.py` | Namek: Lehm-Kuppelhäuser (Boolean + Normalen-Transfer), schlanke Ajisa-Bäume mit blauen Kugelkronen, blaue Grashalme, Sandflecken, rote Pilze, Dragon Balls, Friezas Raumschiff |
| `world_*.py` | die drei Welten inkl. Flugroute |
| `params/*.py` | Farbkorrektur/Dunst je Welt |

## Werte-Archiv und Skill

- `werte/<NN_name>/`: alle Parameterwerte eines fertigen Videos für spätere Auswertungen. `werte.json` nutzt die IDs
  aus dem Parameter-Katalog, `kampfplan_und_kamera.json` enthält alle Schlüssel aus dem Code, `werte.md` ist die
  lesbare Fassung. Einträge: `01_onepiece`, `02_naruto`, `03_namek`. Alle drei füllen den vollständigen Katalog mit
  332 IDs (Stadt und Kino aus dem Skill blender-stadt-vfx plus die gemeinsamen IDs); „entfällt“ heißt, dass es den
  Parameter im Video nicht gibt, „nicht erfasst“, dass er nicht gemessen wurde.
- Skill `.claude/skills/anime-fpv-kampfvideo/` (im Repo-Hauptordner): Ablauf, Parameter-Katalog ohne Werte,
  Werte-Vorlage, Pipeline-Skripte (Vorschau, fortsetzbares Final, Watchdog, Webversion, Werte-Dump).

## Figuren-Modelle (nicht im Repo)

Die Tripo-Modelle liegen nicht im (öffentlichen) Repo. Für den Namek-Kampf gehören sie nach
`assets/models/dragonball/`:
- `goku5/gokuactionfigure3dmodel_repariert.glb`: Goku SSJ, Standpose, Mixamo-Rig, Textur. Die gelieferte GLB war
  ein unvollständiger Download (72 %); Mesh, Farb- und Normal-Map waren vollständig, die Bind-Matrizen wurden aus
  dem Skelett neu berechnet und die abgeschnittene Metallic-Map entfernt.
- `freezer4/friezafinalform3dmodel.glb`: Freezer Endform, Standpose, Mixamo-Rig, Textur
- ältere Fassungen (Fallback): `goku2/…fbx` (SSJ, Sprungpose), `freezer/…fbx` (erste Fassung)

Für die Crew auf der Thousand Sunny nach `assets/models/onepiece/<figur>/` (`tripo_crew.SPEC`):
`luffy/luffycharacter3dmodel.glb`, `zoro/zorofigure3dmodel_rig_repariert.glb` (unvollständiger Download, mit
`repair_glb.py` repariert), `nami/animegirlfigure3dmodel.glb`, `usopp/piratecharacter3dmodel.glb`,
`sanji/sanjifigure3dmodel.glb`, `chopper/choppertoy3dmodel.glb`, `robin/femalecharacter3dmodel.glb`,
`franky/muscularactionfigure3dmodel.glb` (ohne Rig), `brook/skeletonclown3dmodel.glb`,
`jinbe/fantasyogre3dmodel.glb`.

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
One Piece simuliert beim Aufbau die Flaggen (Cloth) und berechnet die Bug-Gischt vor; beides landet als
PC2-Punktcache in `<out>_cache` (oder `NIDO_CACHE`), damit jeder Frame einzeln renderbar bleibt. Die
Sounddesign-Marker (WHOOSH, BEAT_DROP, AMBIENCE) stehen zusätzlich in `<out>/sound_markers.csv`.
