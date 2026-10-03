# Parameter-Katalog: Blender-Städte und FPV-Welten

Alle Stellschrauben einer Stadt in Kinoqualität und der übrigen 3D-Welten, **ohne Werte**. Die Werte legst du je
Video fest und trägst sie nach dem Final mit denselben IDs ins Werte-Archiv ein
(`drohnen_3d_welten/werte/<NN_name>/werte.json`, Vorlage: `werte_vorlage.json`). Gleiche IDs über alle Videos
erlauben später die Auswertung, z. B. welche Kameratempi, Detaildichten, Sample-Zahlen oder Grades gut ankamen.

- **Teil A** enthält die Stadt- und Kino-Parameter (Abschnitte S1 bis S15).
- **Teil B** enthält die gemeinsamen Parameter aller 3D-Welten (Abschnitte 1 bis 15).
- Derselbe Katalog (und dieselbe Werte-Vorlage) liegt in den Skills `blender-stadt-vfx` und
  `anime-fpv-kampfvideo`. Alle Videos nutzen dieselben IDs, deshalb sind Stadt-, Anime- und Kampfvideos direkt
  vergleichbar.
- Jede ID gibt es nur einmal. Wo Teil A einen Bereich aus Teil B ergänzt, steht ein Verweis statt einer Kopie.

**Code-Ort:**
- Repo `SnaQOG/Cursor`, Ordner `drohnen_3d_welten/`.
- Referenz-Umsetzung einer Stadt ist `world_naruto.py` (Konoha) mit `konoha.py` (Gebäude, Materialien,
  Felsen) und `konoha_life.py` (Straßenleben). Für Teil B ist es `world_dragonball.py` (Namek).
- Blender-Eigenschaften stehen als Pfad da (z. B. `scene.cycles.*`), Einstellungen der Skill-Skripte als
  `CONFIG["…"]` (`scripts/setup_cinema_render.py`) bzw. als Konstante in `scripts/preflight_check.py`.

**Nicht zutreffend:** Gibt es einen Parameter im Video nicht (z. B. kein Wasser), trägst du im Archiv „entfällt“
ein. Die ID wird nicht gelöscht.

## Inhalt
Teil A: Stadt und Kino
- S1. Briefing und Planung
- S2. Stadtlayout und Blockout
- S3. Gebäude und Aufbau
- S4. Stadt-Materialien
- S5. Stadt-Licht
- S6. Atmosphäre
- S7. Leben und Bewegung
- S8. Vegetation in der Stadt
- S9. Wahrzeichen-Relief (Felswand, Monument)
- S10. Kamera wie im Film
- S11. Render-Setup in Kinoqualität
- S12. Compositing (Kinolook)
- S13. Video-zu-Video-KI
- S14. QC, Lieferung, Gates
- S15. Kampf in der Stadt

Teil B: gemeinsame Parameter aller 3D-Welten
- 1. Format und Ablauf
- 2. Licht und Himmel
- 3. Gelände und Welt-Objekte
- 4. Wasser
- 5. Vegetation, Gras, Kleinkram
- 6. Kamera
- 7. Figuren
- 8. Kampfplan und Timing
- 9. Effekte
- 10. Render
- 11. Nachbearbeitung und Encoding
- 12. Laufzeiten und Ressourcen
- 13. Sound-Marker
- 14. Bewertung
- 15. Schiff und Crew

**Phasen und Gates des Skills:**

| Phase (Gate) | Abschnitte |
|---|---|
| 0 Briefing (A) | S1, 1 |
| 1 Shotlist, Referenzen, Maßstab (A2) | S1, S2, 6 |
| 2 Blockout und Kamera (B) | S2, S10, 6 |
| 3 Assets und Aufbau | S3, S8, S9, 3, 5 |
| 4 Look-Dev (C) | S4, S5, S6, 2 |
| 5 Leben und Bewegung | S7, S15, 7–9, 15 |
| 6 Preview-Animation (D) | S14 (Gates), 12, 14 |
| 7 Render-Setup und Preflight (E) | S11, 10 |
| 8 Compositing und Lieferung | S12, S13, S14, 11–14 |

---

# Teil A: Stadt und Kino

## S1. Briefing und Planung
| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| briefing.stadt | Stadt, Thema, Epoche, Kultur | Text | Docstring der Welt |
| briefing.stil | fotoreal, stilisiert oder Anime | Text | – |
| briefing.tageszeit | Tageszeit (genau benannt) | Text, Uhrzeit | – |
| briefing.wetter | Wetter und Jahreszeit | Text | – |
| briefing.stimmung | Stimmung in einem Satz | Text | – |
| briefing.kameratyp | FPV, Flyover oder Static | Text | – |
| briefing.verwendung | privat, Video-zu-Video-KI oder Veröffentlichung | Text | – |
| briefing.engine | Engine und Device | Text | `CONFIG["device"]` |
| briefing.budget | Renderbudget: max. Minuten pro Frame, Gesamtstunden | min, h | `BUDGET_HOURS` |
| briefing.hardware | VRAM, RAM, CPU-Kerne, freier Speicher für die Sequenz | GB, Anzahl | – |
| briefing.ausgabe | Ausgabeformat und Zielordner | Text, Pfad | `CONFIG["output_root"]` |
| briefing.referenzen | Referenzen für Architektur, Material, Licht, Atmosphäre, Kamera | Liste | Docstring der Welt |
| briefing.palette | Farbpalette und Kontrastidee (aus Referenzen gemessen) | Liste RGB | Material-Farben in `build()` |
| planung.einheiten | Einheitensystem, Maßstab, Skalierung angewendet | Text | `scene.unit_settings` |
| planung.detailbudget | Grenzen nah / mittel / fern und Detailstufe je Zone | m, Text | – |
| planung.collections | Collection-Struktur und Namensschema | Liste | – |
| planung.versionierung | Dateiname, Version, Backup-Ort | Text | `CONFIG["city"]`, `CONFIG["version"]` |

## S2. Stadtlayout und Blockout
| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| stadt.koordinaten | Ursprung, Hauptflugrichtung, Achsen | Text | Welt-Modul |
| stadt.flaeche | Stadtfläche: Größe, Mitte | m | `VillageGround` in `build()` |
| stadt.mauer | Stadtmauer: Mitte, Radius, Segmentwinkel, Höhe, Dicke, Abdeckung, Toröffnung | m, ° | `WALL_C`, `WALL_R`, `build()` |
| stadt.tor | Tor: Ort, halbe Öffnung, Höhe, Oberkante des Querbalkens | m | `konoha.gate`, `BEAM2_TOP` |
| stadt.hauptstrasse | Hauptstraße: Halbbreite, Länge, Mitte, Belag | m | `STREET_HW`, `MainStreet` |
| stadt.zufahrt | Zufahrtsweg: Breite, Länge, Belag | m | `Road` in `build()` |
| stadt.platz | Platz: Mitte, Größe, Belag | m | `Plaza` in `build()` |
| stadt.gelaende | Höhenprofil: Hügelring (Abstand, Breite, Höhe, fbm), Tal der Zufahrt, Plateau, Gipfel | m, Formel | `terrain_height` |
| stadt.gelaende_mesh | Geländenetz: Größe, Auflösung, Ursprung | m, Punkte | `fpv.grid_mesh("Terrain", …)` |
| stadt.wasser | Fluss, Hafen, Kanäle, Uferlinien | m, Liste | Welt-Modul, `ocean.py` |
| stadt.bruecken_treppen | Brücken, Treppen, Terrassen | Liste | Welt-Modul |
| stadt.landmarken | Wahrzeichen: Name, Ort, Größe, Rolle im Flug (Hook, Manöver, Reveal, Schlussbild) | Liste | `RES_POS`, `TREE_POS`, `CLIFF_Y`, `HEADS` |
| stadt.sichtachsen | Sichtachsen und Kompositionsmomente entlang der Route | Liste (s, Motiv) | Kommentare in `CAM_KEYS` |
| stadt.route | grobe Flugroute auf dem Blockout | Liste (x, y, z) | `ROUTE` |
| stadt.seed | Zufalls-Seed des Aufbaus | Zahl | `np.random.default_rng` in `build()` |

## S3. Gebäude und Aufbau
| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| gebaeude.anzahl | Gebäude gesamt (Messwert) | Anzahl | Ausgabe „buildings“ |
| gebaeude.strassenzeile | Häuser an der Hauptstraße: Breite, Tiefe, Stockwerke, Lücke, Rücksprung, y-Bereich | m, Anzahl | `build()` Teil a |
| gebaeude.stile | Bauformen und ihre Gewichte (flach, Tonnendach, rund, Walmdach) | Liste | `rng.choice` in `build()` |
| gebaeude.raster | übrige Stadt: Rasterweite, Versatz, Aussparungen, Ausfallquote, Drehung, Größen, Stockwerke | m, Anteil, ° | `build()` Teil b |
| gebaeude.platzrand | Randbebauung des Platzes: je Gebäude x, y, Breite, Tiefe, Stockwerke, Stil | Liste | `build()` Teil c |
| gebaeude.stockwerkhoehe | Stockwerkhöhe, Sockel, Brüstung | m | `konoha.building` |
| gebaeude.tuerme | Stufentürme: Anteil, Radius, Stufen, Dachfarbe, feste Türme an der Straße | Anteil, m, Anzahl | `konoha.tiered_tower` |
| gebaeude.daecher | Dachformen: Stich des Tonnendachs, Stich und Überstand des Walmdachs, Ziegelmaße | Faktor, m | `barrel_roof`, `hip_roof`, `tile_roof_mesh` |
| gebaeude.dachaufbauten | Wassertanks, Klimageräte, Rohre, Antennen, Schornsteine: Anteil, Größe | Anteil, m | `water_tank_collection`, `building` |
| gebaeude.fassade | Fenster je Stockwerk, Rahmen, Bänke, Läden, Rohre, Kabel, Markisen | Anzahl, Anteil | `facade_details`, `window_collection` |
| gebaeude.banner | Banner: Anteil, Zeichen, Größe, Höhe, Farben | Anteil, m, RGB | `build()` |
| gebaeude.schilder | Ladenschilder: Zeichen, Vorder- und Hintergrundfarbe | Liste | `sign_mats`, `textures.kanji_disc` |
| gebaeude.tueren | Türen: Motiv, Farben | Liste | `konoha.door_material` |
| gebaeude.hero | Hauptgebäude: Ort, Radius, Dachhöhe, Brüstung, Dachbelag | m | `konoha.residence`, `RES_POS`, `DECK_Z`, `C_ROOF` |
| gebaeude.bevel | Kantenfase der Bauteile | m | `konoha.box(bevel)`, `weather(bevel_r)` |
| gebaeude.lod | Detailstufe je Zone, Proxies und Karten in der Ferne | Text | – |
| gebaeude.instancing | Instanzierung und Variation (Höhe, Dach, Farbe, Drehung, Skalierung) | Text, Bereiche | `place_instance`, `nature.scatter_gn` |
| gebaeude.strassendetails | Bordsteine, Gullys, Markierungen, Pflaster | Liste | – |

## S4. Stadt-Materialien
Boden, Felsen und Steilwände der Welt stehen in Abschnitt 3 (`material.boden`, `material.fels`,
`material.steilwand`).

| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| material.putz | Putz: Farben, Fenster an/aus, PBR-Variante | RGB, an/aus | `plaster_material`, `plaster_pbr_material` |
| material.daecher | Dachfarben, Ziegelbreite und -höhe, Streuung, zylindrisch | RGB, m, Faktor | `tile_roof_material`, `roof_material` |
| material.terrakotta | Farben der Ziegel-Meshes | RGB | `terracotta_material` |
| material.fenster | Fensterglas: lit, curtain, per_cell, Innenraum | Faktoren | `window_glass_material` |
| material.verwitterung | je Material: dirt_h, dirt, ground_z, edge, edge_col, bevel_r, streak, var | m, RGB, Faktoren | `konoha.weather` |
| material.dachbelag | Abnutzung des Dachbelags (r_edge) | m | `konoha.deck_wear` |
| material.strasse | Straßenboden (half_w), Pflaster, Feldweg, Sandstraße | m, Faktoren | `street_ground_material`, `street_material`, `dirt_road_material`, `sand_street_material` |
| material.gras | Gras zwischen den Häusern und am Hang: c1, c2, dry, scale | RGB, Faktor | `nature.grass_material` |
| material.metall | Rostmetall für Tanks und Rohre: tint | RGB | `rust_metal_material` |
| material.holz | Balken, Dachdeck, Masten: dark, board, Achse | Faktoren | `sunny.wood_material`, `nature.bark_material` |
| material.lack | lackierte Teile (Läden, Klimageräte, Horn): Farbe, Rauheit | RGB, Faktor | `sunny.paint_material` |
| material.stoff | Markisen, Stände, Wäsche: Farben, Rauheit | RGB, Faktor | `simple_mat` in `build()` |
| material.mauer | Mauer- und Torstein: c1–c3, bump, crack_w | RGB, Faktoren | `nature.rock_material` |
| material.wahrzeichen_fels | Fels des Wahrzeichens: c1–c3, moss, moss_amount, scale, bump, crack_w, strata_scale, lichen, cavity | RGB, Faktoren | `CliffRock`, `FaceRock` |
| material.cc0 | CC0-PBR-Texturen: Dateien, Skalierung, Überblendung | Pfad, Faktor | `konoha.CC0`, `pbr_box` |
| material.emission | Laternen, Fenster, Schilder: Farbe, Stärke | RGB, Faktor | `lantern_materials`, `simple_mat(emission)` |
| material.texeldichte | Texeldichte | px/m | – |
| material.farbraum | Farbtexturen sRGB, Datentexturen Non-Color (geprüft) | Text | Image-Nodes |
| material.displacement | Displacement und Bump nah an der Kamera | m, Faktor | Material-Nodes |
| material.kachelvariation | Mittel gegen sichtbare Kacheln (Noise-Mix, Objekt-Zufall, Attribute) | Text, Faktoren | Material-Nodes |

## S5. Stadt-Licht
Ergänzt Abschnitt 2 (Hauptsonne, Randlicht, Himmel, Wolken).

| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| licht.himmel_typ | Nishita, HDRI oder eigener Verlauf | Text | `fpv.build_world` |
| licht.nishita | aerosol, ozone, air, Sonnenscheibe | Faktoren | `build_world(aerosol, ozone)` |
| licht.hdri | Datei, Drehung, Stärke | Pfad, °, Faktor | World-Nodes |
| licht.nacht_quellen | Laternen, Fenster, Schilder, Fahrzeuge, Feuer: Anzahl, Stärke | Anzahl, W | Welt-Modul |
| licht.farbtemperaturen | Kelvin je Quellenart (Natrium, warm, neutral, kühl) | K | Lichter |
| licht.schatten | Richtung, Länge, Härte (Prüfung gegen Tageszeit) | Text | – |
| licht.mond_sterne | Mond, Sterne | Text | World-Nodes |
| licht.rig | Lichtrig als Collection (Name, Inhalt) | Text | – |

## S6. Atmosphäre
Ergänzt `atmosphaere.dichte`, `wolken.*` (Abschnitt 2) und `post.dunst` (Abschnitt 11).

| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| atmosphaere.volumen | Streuvolumen: Größe, Mitte, Farbe | m, RGB | `fpv.add_atmosphere` |
| atmosphaere.bodennebel | Höhen- und Bodennebel: Höhe, Dichte, Bereich | m, 1/m | – |
| atmosphaere.lichtstrahlen | volumetrische Lichtstrahlen | Faktor | – |
| atmosphaere.wetter | Regen, Schnee, Staub, Asche: Anzahl, Tempo | Anzahl, m/s | – |
| atmosphaere.tiefenschichten | Vorder-, Mittel-, Hintergrund: Abstände | m | – |

## S7. Leben und Bewegung
| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| leben.passanten | Passanten: Abstand entlang der Straße, Anteil je Seite, Abstand zur Straßenmitte, Drehungsstreuung, Aussparungen, Anzahl | m, Anteil, °, Anzahl | `build()`, `konoha_life.place_villagers` |
| leben.passanten_vorlagen | Vorlagen, Posen, Kleidungsfarben, Seed | Liste | `villager_templates`, `VILLAGER_POSES`, `villager_materials` |
| leben.staende | Marktstände: Anzahl, Abstand, Seite, Maße, Dachstoff, Waren, Kisten | Anzahl, m | `build()`, `stall_goods` |
| leben.laternen | Laternenleinen: y-Positionen, x_end, z_end, sag, n je Leine, Seed | m, Anzahl | `konoha_life.lantern_lines` |
| leben.waesche | Wäscheleinen: y-Positionen, Höhe (aus Fassaden), sag, Abstand zum Baum | m | `konoha_life.laundry_lines` |
| leben.leitungen | Strommasten: Abstand, Höhe, Querbalken, Drahtabstand, Durchhang | m | `build()` |
| leben.voegel | je Schwarm: Anzahl, Startbox, Flugvektor, t0, t1, Seed | Anzahl, m, s | `konoha_life.birds` |
| leben.blaetter | treibende Blätter je Feld: Box, Anzahl, Wind, Seed | m, Anzahl, m/s | `konoha_life.leaves_gn` |
| leben.pendeln | Schwingen: Achse, Amplitude, Frequenz, Phase, Rauschen | °, Hz | `konoha_life.sway` |
| leben.verkehr | Fahrzeuge, Bahnen, Boote: Anzahl, Pfade, Tempo, Lichter | Anzahl, m/s | – |
| leben.rauch | Rauch und Dampf: Quellen, Dichte, Tempo | Anzahl, Faktoren | – |
| leben.flaggen | Flaggen, Planen, Wäsche im Wind (Stoff-Cache) | Text | `sunny.flag_cloth` |
| leben.caches | Simulations-Caches: Ordner, Frames | Pfad | – |
| leben.asynchron | Variation, damit nichts synchron läuft (Phase, Tempo) | Bereiche | – |

## S8. Vegetation in der Stadt
Ergänzt Abschnitt 5 (`baum.varianten`, `baum.material`, `baum.abstaende`, `baum.sichtlinie`).

| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| baum.hero | Manöver-Baum: Ort, Höhe, Kronenradius, Cluster, Blätter je Cluster, Steinring | m, Anzahl | `TREE_POS`, `BigTree` |
| baum.nadelbaum | Nadelbäume: Höhe, Krone, Cluster, Form, Transluzenz | m, Anzahl | `KSugi` in `build()` |
| baum.wald | Waldbereiche: Anzahl, x/y-Bereich, Ausschlüsse, Skalierung | Liste | `add_forest` |
| baum.dorf | Bäume im Ort: Versuche, Abstand zu Gebäuden, Hauptgebäude, Straße; Skalierung | Anzahl, m | `build()` |
| baum.buesche | Büsche an der Zufahrt: Anzahl, Bereich, Skalierung | Anzahl, m | `build()` |
| baum.felsbewuchs | Bewuchs auf Felsbändern: Normalen-Schwelle, Mindesthöhe, Anzahl, Skalierung, Aussparung | Faktor, m, Anzahl | `build()` |
| baum.streuung | Streuung per Geometry Nodes: Seed, Drehung, Größe, Punkte gesamt | Zahl, Anzahl | `nature.scatter_gn` |

## S9. Wahrzeichen-Relief (Felswand, Monument)
| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| wahrzeichen.wand | Felswand: x-Bereich, y, z-Bereich, Auflösung, Seed, Panel | m | `CLIFF_Y`, `konoha.cliff_mesh` |
| wahrzeichen.risse | Risse: Anzahl, Bereich, Tiefe, Breite | Anzahl, m | `konoha.add_cracks` |
| wahrzeichen.gesichter | je Gesicht x, Höhenversatz, Haar; Skalierung, Grundhöhe, Kopfvorlage, Voxelgröße | m, Text | `HEADS`, `HEAD_S`, `HEAD_Z`, `hokage_head`, `fuse_parts` |
| wahrzeichen.kinnlinie | Kinnlinie, Panel-Aussparung | Formel | `chin` in `cliff_mesh` |
| wahrzeichen.details | Treppen, Geländer, Hütten, Kuppeln an der Wand | Liste | `cliff_details` |

## S10. Kamera wie im Film
Ergänzt Abschnitt 6 (`kamera.brennweite`, `kamera.sensor`, `kamera.wegpunkte`, `kamera.tempo` …).

| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| kamera.fov | Sichtfeld | ° | `fpv.make_camera(fov_deg)` |
| kamera.blende | Blende, Fokusdistanz, Schärfentiefe an/aus | f, m | `cam.data.dof` |
| kamera.clipping | Clip Start, Clip End | m | `cam.data.clip_start`, `clip_end` |
| kamera.overscan | Overscan für Stabilisierung und Verzerrung im Comp | % | – |
| kamera.manoever | je Manöver: Zeit, festes Objekt, Art (Tor, Kabel, Baum, Steigflug, Brüstung) | Liste | Kommentare in `CAM_KEYS` |
| kamera.mindestabstand | kleinster Abstand zur Geometrie (Messwert) | m | – |
| kamera.hook | Hook-Bild am Anfang: Inhalt, Dauer | Text, s | `ablauf.abschnitte` |
| kamera.schlussbild | Schlussbild: Motiv, Blickwinkel | Text, ° | letzter Eintrag in `CAM_KEYS` |
| kamera.loop | Loop oder fester Anfang und Ende | Text | – |

## S11. Render-Setup in Kinoqualität
Ergänzt Abschnitt 10 (`render.samples`, `render.adaptiv`, `render.denoiser`, `render.bounces`, `render.clamp`,
`render.light_tree`, `render.persistent`, `render.motion_blur`, `render.paesse`, `render.seed` …).

| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| render.engine | Engine und Device | Text | `scene.render.engine`, `CONFIG["device"]` |
| render.min_samples | Mindest-Samples beim adaptiven Sampling | Anzahl | `CONFIG["min_samples"]` |
| render.kaustiken | Kaustiken reflektiv / refraktiv | an/aus | `scene.cycles.caustics_*` |
| render.vector_pass | Vector-Pass (nur ohne Cycles-Motion-Blur) | an/aus | `CONFIG["vector_pass"]` |
| render.cryptomatte | Object, Material, Asset; Tiefe | Text, Anzahl | `use_pass_cryptomatte_*`, `pass_cryptomatte_depth` |
| render.mist | Mist Start, Tiefe, Abfall | m, Text | `CONFIG["mist_start"]`, `CONFIG["mist_depth"]` |
| render.farbmanagement | View Transform, Look, Exposure, Gamma, Display | Text | `CONFIG["view_transform"]`, `CONFIG["look"]` |
| render.ausgabe | Format, Bit-Tiefe, Codec, Bildsequenz, Pfadschema | Text | `CONFIG["exr_codec"]`, `CONFIG["exr_depth"]`, `output_root` |
| render.fortsetzbar | Overwrite aus, Placeholders an, vorhandene Frames überspringen | an/aus | `use_overwrite`, `use_placeholder` |
| render.vram | VRAM- und RAM-Spitze bei schweren Frames (Messwert) | GB | – |
| render.testframes | Testframes in Endqualität: Frame, Sekunden | Liste | – |
| render.hochrechnung | s pro Frame × Frames = Gesamtzeit, gegen Budget | s, h | `SEC_PER_FRAME`, `BUDGET_HOURS` |
| render.preflight | Ergebnis des Preflight-Checks (Fehler, Warnungen) | Liste | `scripts/preflight_check.py` |

## S12. Compositing (Kinolook)
Ergänzt Abschnitt 11 (`post.bloom` für Glare/Fog Glow, `post.optik` für Dispersion, Vignette, Korn, `post.look`,
`post.farbe`, `post.dunst` …).

| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| comp.linsenverzerrung | Linsenverzerrung (Distortion) | Faktor | Lens-Distortion-Node |
| comp.halation | Halation an hellen Kanten | Faktor | Compositor |
| comp.grading | Schwarzwert, Highlights, Lift/Gamma/Gain, Farbtrennung Schatten/Lichter | Faktoren, RGB | Compositor |
| comp.cryptomatte_korrekturen | selektive Korrekturen (Fenster, Himmel, Wasser) | Liste | Compositor |
| comp.shake | Kamerashake im Comp | Faktoren | Compositor |
| comp.letterbox | Letterbox / Scope | Seitenverhältnis | Compositor |
| comp.ki_variante | saubere Fassung ohne Korn, Glare, Verzerrung | Pfad | – |

## S13. Video-zu-Video-KI
Nur, wenn die Verwendung „Video-zu-Video-KI“ ist. Generiert wird erst nach ausdrücklicher Freigabe.

| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| ki.tool | Plattform und Modell | Text | – |
| ki.einstellungen | Einstellungen der Generierung | Text | – |
| ki.kosten | vorab genannte Kosten | Credits | – |
| ki.freigabe | Freigabe: Datum, Zitat | Text | – |
| ki.eingabe | Eingabevideo (saubere Fassung) | Pfad | `comp.ki_variante` |

## S14. QC, Lieferung, Gates
| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| bewertung.gates | je Gate (A, A2, B, C, D, E): Datum, freigegeben ja/nein, Zitat | Liste | – |
| qc.durchsicht | komplett in Originalgeschwindigkeit gesehen | ja/nein | – |
| qc.frames | fehlende oder schwarze Frames, Fireflies | Anzahl, Liste | – |
| qc.flackern | Flackern in Schatten, Lichtern, Wasser, Fenstern | Befund | – |
| qc.textur | Kachelmuster, Texturstreckung, Z-Fighting | Befund | – |
| qc.banding | Banding in Himmel und Nebel | Befund | – |
| qc.zweitdisplay | auf Handy oder zweitem Display geprüft | ja/nein | – |
| qc.anfang_ende | Anfang und Ende sauber (Loop, Fade) | Text | – |
| lieferung.master | Master (EXR-Sequenz, ProRes oder MP4): Pfad, Größe | Text, MB | `videos/` |
| lieferung.varianten | weitere Formate (9:16, 1:1, Webversion) | Liste | `web_version.sh` |
| lieferung.dateiname | Dateiname mit Version und Datum | Text | – |
| lieferung.archiv | archivierte Teile (Projekt, Texturen, Caches, Skripte), Ort | Liste | – |
| lieferung.obsidian | ZIP-Paket für den Vault | Pfad | – |
| lieferung.wiederverwendbar | Assets, Materialien, Node-Gruppen, Lichtrig, Kameraskript für die nächste Stadt | Liste | – |
| lieferung.notiz | was beim nächsten Mal früher entschieden werden muss | Text | – |

## S15. Kampf in der Stadt
Hat die Stadt eine Handlung mit Figuren, gelten zusätzlich Abschnitt 7 bis 9. Diese IDs kommen dazu.

| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| kampf.ort | Kampfort (z. B. Dach): Mitte, Höhe, lokales Koordinatensystem | m | `C_ROOF`, `DECK_Z`, `roof()` |
| fx.impact_klein | kleines Impact-Paket: Farbe, Licht, Ring, Funken, Seed | RGB, W, m, Anzahl | `impact_small` |
| fx.bruch | Bruchstücke an Gebäudeteilen: Objekt, Zeit, Bruchhöhe, Stücke | s, m, Anzahl | `break_fx` |
| fx.techniken | Signatur-Techniken (z. B. Rasengan, Chidori): Farbe, Radius, Licht, Zeiten | RGB, m, W, s | `fight_fx`, `BLUE` |

---

# Teil B: gemeinsame Parameter aller 3D-Welten
Entstanden mit dem Namek-Video. Code-Orte beziehen sich auf die Namek-Umsetzung
(`world_dragonball.py`); in einer Stadt heißen sie gleich oder stehen im Welt-Modul der Stadt
(z. B. `SUN_ELEV`, `RIM`, `CAM_KEYS`, `HITS`, `SHAKES` in `world_naruto.py`, `params/naruto.py`).

## 1. Format und Ablauf
| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| format.dauer | Videolänge | s | `SECONDS` |
| format.fps | Bildrate | fps | `FPS` |
| format.frames | Frames gesamt | Anzahl | FPS × SECONDS |
| format.seitenverhaeltnis | Seitenverhältnis | z. B. 9:16 | Renderauflösung |
| format.ausgabe_aufloesung | Endgröße | B × H px | `post.process(size=…)` |
| ablauf.abschnitte | Zeitleiste: Abschnitt, von–bis, Inhalt, Kamera-Tempo | Liste (s, s, Text, m/s) | Docstring der Welt |
| ablauf.koordinaten | Hauptflugrichtung, Achsen, Einheit | Text | Welt-Modul |

## 2. Licht und Himmel
| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| licht.hauptsonne.hoehe | Höhe der Hauptsonne | ° | `SUN_ELEV` |
| licht.hauptsonne.azimut | Azimut der Hauptsonne | ° | `SUN_AZIM` |
| licht.hauptsonne.staerke | Stärke | W/m² | `SUN_STRENGTH` |
| licht.hauptsonne.kelvin | Farbtemperatur | K | `SUN_KELVIN` |
| licht.hauptsonne.winkel | Winkelgröße (Schattenweichheit) | ° | `add_sun(angle_deg)` |
| licht.zusatzsonnen | weitere Sonnen: Höhe, Azimut, Stärke, Kelvin, Winkel | Liste | `SUNS2`, `build()` |
| licht.sonnenscheiben | sichtbare Scheiben: Radius, Stärke, Farbe | Liste | `extra_suns` in `build()` |
| licht.randlicht | Randlicht nur auf Figuren: Höhe, Azimut, Stärke, Kelvin, Winkel | dict | `RIM`, `fpv.rim_light` |
| licht.toon_richtung | Licht-/Randlichtrichtung für das Toon-Shading | Vektoren | `anime_chars.set_light` |
| himmel.verlauf | Himmelsfarben über der Höhe | Liste (Position, RGB) | `namek_sky`-Ramp |
| himmel.unter_horizont | Farbe unter dem Horizont | RGB | `namek_sky` |
| himmel.sonnenleuchten | Leuchten um Sonnen: Exponenten, Stärken, Farbe | Liste | `namek_sky` |
| himmel.entsaettigung_diffus | Entsättigung des diffusen Himmelslichts | 0–1 | `namek_sky` (`neutral`) |
| himmel.staerke | Himmelsstärke | Faktor | `build_world(sky_strength)` |
| wolken.bedeckung / .ref / .farbe / .skala | Wolkenschicht | Anteil, Faktor, RGB, Faktor | `build_world` |
| atmosphaere.dichte | Streuvolumen (0 = aus) | 1/m | `NIDO_ATMO` |

## 3. Gelände und Welt-Objekte
| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| gelaende.hauptform | Haupt-Landmarke (Tafelberg o. Ä.): Mitte, Radius | m | `MESA_C`, `MESA_R` |
| gelaende.umriss | Umrissfunktion (Sinus-Anteile, fbm, Seeds) | Formel | `mesa_edge` |
| gelaende.plateau | Plateauhöhe + Rauschanteile | m, Formel | `plateau` |
| gelaende.kante | Kantenabfall, Schuttfuß | m | `mesa_height` |
| gelaende.heightfield | Größe, Auflösung | m, Punkte | `fpv.grid_mesh("Mesa", …)` |
| gelaende.steilwand | Ringwand: n_ang, n_z, seed, Felskante (lip), base_z | Anzahl, m | `namek.ring_wall` |
| gelaende.felsnadeln | je Nadel: x, y, r, h, seed, Neigung x/y | Liste | `SPIRES` |
| gelaende.felsnadel_mesh | rock_mesh: detail, taper, lumpy, strata, flute, ledges, bed | Faktoren | `build()` |
| gelaende.ferne_felsen | ferne Nadeln: x, y, r, h | Liste | `build()` |
| gelaende.inseln | Horizont-Inseln: x, y, r, h | Liste | `build()` |
| material.fels | Felsfarben c1–c3, wet_line, algae, moss, moss_amount, strata, bump, crack, lichen, variation, bedding, streak | RGB, Faktoren | `namek_rock` |
| material.steilwand | dito für die Wand | RGB, Faktoren | `namek_cliff` |
| material.boden | Erde, Moosflecken, Roughness, Bump | RGB, Faktoren | `plateau_ground` |
| objekte.haeuser | je Haus: x, y, r; Lehm-, Glas-, Türfarben | Liste, RGB | `houses`, `mats` |
| objekte.dragonballs | Mitte, Sockelradien, Kugelradius, Ring | m | `DB_C`, `namek.dragon_balls` |
| objekte.raumschiff | Ort, Höhe, Radius, Rumpf-/Randfarben, Fenster-Emission | m, RGB | `SHIP_C`, `namek.spaceship` |

## 4. Wasser
| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| wasser.kacheln | Start, Anzahl x/y, Kachelgröße | m, Anzahl | `make_ocean` |
| wasser.wellen | res, wind, wave_scale, chop, Richtung, alignment, foam_coverage | Faktoren, ° | `make_ocean` |
| wasser.material | deep, shallow, foam_amount, view_dark, micro | RGB, Faktoren | `water_material` |
| wasser.brandung | Abstandsfeld: Bereich, Zelle | m | `shore_distance_image` |

## 5. Vegetation, Gras, Kleinkram
| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| baum.varianten | je Variante: Höhe, Kronenradius, Drehung, Blattzahl, Blattgröße, Seed | Liste | `namek.ajisa_variant` |
| baum.material | Blattfarben, Transluzenz, Rinde | RGB, Faktor | `leaf_material`, `bark_material` |
| baum.haine | Haine: x, y, r, n; Skalierungen | Liste | `groves` |
| baum.abstaende | Mindestabstände zu Häusern, Objekten, Kratern, Weg, Kante | m | `avoid`, `build()` |
| baum.sichtlinie | baumfreier Korridor Kamera → Kämpfer: Zeitfenster, Radius | s, m | `build()` |
| findlinge | Anzahl, Radius, Skalierung, Displace, Abstände | Anzahl, m | `build()` |
| sandflecken | Anzahl, Radien, Randabstand | Anzahl, m | `build()` |
| pilze | Gruppen, seitlicher Abstand zur Route, y-Bereich | Anzahl, m | `build()` |
| gras.feld | Bereich, Kantenabstand, Dichte (nah, Grund, Abfall), Band, clump, max Halme, Höhe | m, Anzahl | `namek.grass_field` |
| gras.farben | Basis, Spitze, Büschel-Tönungen | RGB | `grass_blade_material` |
| gras.aussparungen | Radien um Häuser, Objekte, Findlinge, Sand | m | `excl` |
| gras.bewegung | Wind, Böen (Tempo, Stärke), sway | Faktoren, m/s | `namek.grass_motion` |
| gras.druckwellen | je Welle: x, y, t, Tempo, Stärke, Breite | Liste | `shocks` |
| gras.brennen | Krater, in denen Halme verschwinden (Radius-Faktor) | Faktor | `burns` |

## 6. Kamera
| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| kamera.brennweite | Objektiv | mm | `LENS_MM` |
| kamera.sensor | Sensorgröße / Fit | mm | `cam.data.sensor_*` |
| kamera.wegpunkte | (Zeit, Ort) für die Speed-Ramp | Liste (s, x, y, z) | `CAM_KEYS` |
| kamera.tempo | resultierendes Tempo je Zeitpunkt (Messwert) | m/s | `info["v"]` |
| kamera.halte | Kamera-Halte bei Treffern | Liste (s, Frames) | `CAM_HOLDS` |
| kamera.ausrichtung | look_pitch, pitch_follow, bank_gain, max_bank, micro, seed | °, Faktoren | `fpv.fpv_orient` |
| kamera.pitch_overrides | Blick-Neigung je Zeitfenster | Liste (s, s, °) | `pitch_overrides` |
| kamera.blickgewicht | Gewicht Flugrichtung ↔ Blickziel über die Zeit | Liste (s, 0–1) | `keys_w` |
| kamera.blickziele | Ziel je Phase (Paar, Krater, Strahlen, Duell, Ende) mit Zeitfenstern und Gewichten | Liste | `camera_path` |
| kamera.glaettung_ziele | Glättung der Kämpferbahnen für den Blick | s | `_gauss_smooth` |
| kamera.stoesse | Kamerastöße: Zeit, Stärke, Frequenz; seed, pos_amp | Liste | `SHAKES`, `camera_shakes` |

## 7. Figuren
| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| figur.<name>.modell | Datei, Quelle, Reparaturen | Pfad, Text | `tripo_chars.<fn>()` |
| figur.<name>.hoehe | Zielgröße | m | `rig_model(height)` |
| figur.<name>.gelenke | Quelle der Gelenke (Mixamo / vermessen) | Text, Tabelle | `MIXAMO`, `*_JOINTS` |
| figur.<name>.schwanz | Mittellinie, Gliederzahl | Liste, Anzahl | `*_TAIL`, `n_tail` |
| figur.<name>.material | lit, mid, shade, bands, tint_lit, rim, rim_w, sat, value, diffuse_mix, hair_glow | Faktoren, RGB | `toon_tex_material` |
| figur.<name>.kontur | Konturstärke relativ zur Höhe | Faktor | `outline` |
| figur.<name>.gewichte | smooth, power, Saat-Bereiche | Faktoren | `rig_model`, `_geodesic_weights` |
| figur.backen | style, lag_scale, lean_tau, seeds, look_win | Funktion, Faktoren, s | `choreo.bake_fighter` |
| figur.schwanz_peitsche | Peitschenhiebe (Zeit, Winkel), sway | s, ° | `tail_follow` |
| figur.aura | Farbe, Frames an/aus, Höhe, Breite, Licht, opacity, edge_w, strength | RGB, Frames, m, W | `vfx.aura` |
| figur.verschwinden | Frames/Skalierung beim Ausblenden | Frames, Faktor | `stage_fight` |
| figur.mimik | Ausdrucks-Zeitplan (nur Fallback-Figuren) | Liste | `_expr` |

## 8. Kampfplan und Timing
| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| kampf.brusthoehe | Brust über dem Figurenursprung | m | `CH` |
| kampf.stil | je Posenklasse: Dauer, Ausholen, Überschwingen | s, Anteile | `choreo.dbz_style` |
| kampf.bahnprofile | lin / in / out je Abschnitt | Text | `ease` in `_key` |
| kampf.hitstops | Treffer-Halte: Zeit, Frames | Liste | `HITS` |
| kampf.teaser | Fern-Zusammenstöße: Zeit, Mitte, Achse | Liste | `TEASERS` |
| kampf.vorstoss | Startpunkte, Zeitpunkt, Tempo | m, s, m/s | `RG`, `RZ`, `T_RUSH` |
| kampf.zusammenprall | Zeit, Ort | s, m | `T_CLASH`, `C0` |
| kampf.schlaghagel | Wechsel: Zeit, wer, Pose; Ausholvorlauf; Drehung der Kampfachse | Liste, s, ° | `FLURRY`, `_ax` |
| kampf.aktionen | Schwanzhieb, Knie, Doppelfaust, Einschlag, Ausbruch: Zeiten, Orte | s, m | `T_WHIP`, `T_KNEE`, `T_AXE`, `T_SLAM`, `P_CRATER` |
| kampf.zanzoken | Zeitfenster weg/wieder, Frames, Orte | s, Frames, m | `ZAN1`, `ZAN2` |
| kampf.strahlen | Feuerpositionen, Ziele, Ausweichpositionen | m | `A1`, `A2`, `GB1`, `GB2` |
| kampf.trennung | Zielorte, Tempo | m, m/s | `G_RIM`, `Z_SEA` |
| kampf.duell | Aufladen, Start, Klimax | s | `fight_plan`, `T_CLIMAX` |
| kampf.posen | eigene Posen (Gelenkwinkel) | dict | `POSES.update` |
| kampf.schluessel | alle Schlüssel je Figur (t, Pose, Ort, Blick, Luft, lean/roll/rot/ease/spin) | Liste | `fight_plan()` → JSON |

## 9. Effekte
| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| fx.farben | Aura, Kamehameha, Todesstrahl, Ki, Spurfarben | RGB | `GOLD`, `KAME`, `DEATH`, `KI`, `*_TRAIL` |
| fx.kispur | Schweiflänge, center_z, strength, Kernanteil, Radius je Ereignis | s, m, Faktor | `dbz_fx.ki_trail`, `ev["trails"]` |
| fx.zanzoken | Flacker-Folge, Luftring-Radien | Liste, m | `dbz_fx.FLICKER`, `pop_ring` |
| fx.treffer | je Art: Radius, Licht, Funken, Luftring; Dauer; core_k | m, W, Anzahl | `HIT_FX`, `fight_fx` |
| fx.teaser_blitze | r, Dauer, Licht, core_s, glow_s, glow_alpha | m, Frames, W | `fight_fx` |
| fx.druckwellen | Radius, Dauer, Dicke, Glühen, ior | m, Frames | `vfx.shockwave` |
| fx.einschlag | Kraterradius, Abkühlzeit, Burst, Staub, Funken, Bruchstücke, Impuls | m, s, Anzahl | `fight_fx` |
| fx.strahlen | Radius, Licht, Frames an/voll/aus, Einschlag-Burst, Staub, Brocken | m, W, Frames | `vfx.beam`, `fight_fx` |
| fx.duell | Ladungen, Strahlradien, Licht, Wobble, Treffpunkt-Pendel, Kugel, Blitzbögen | m, W, Liste | `fight_fx` |
| fx.explosion | Blitz, Volumen (Feuer, Rauch, Steigen, Dauer, fire), Licht-Kurve, Farbverlauf, Wellen, Funken, Gischt | m, s, W, RGB | `explosion` |
| fx.sicherheit | Burst-Ausblenden, transparente Bounces, ior der Ringe | Text, Anzahl | `vfx.burst`, `build()` |

## 10. Render
| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| render.aufloesung | Render-Auflösung Final / Vorschau / Standbild | B × H px | Pipeline-Argumente |
| render.samples | Samples Final / Vorschau / Standbild | Anzahl | Pipeline-Argumente |
| render.adaptiv | adaptive Schwelle | Faktor | `fpv.setup_render` |
| render.denoiser | Denoiser, Pässe, Prefilter | Text | `fpv.setup_render` |
| render.bounces | max, diffus, glossy, Transmission, Volumen, transparent | Anzahl | `setup_render`, `build()` |
| render.clamp / .blur_glossy / .light_tree / .persistent | Sampling-Optionen | Faktor, an/aus | `setup_render` |
| render.volumen | step_rate, max_steps | Faktor, Anzahl | `build()` |
| render.motion_blur | an/aus, Shutter | Faktor | `setup_render` |
| render.filter | Pixelfilter, Breite | px | `setup_render` |
| render.paesse | Pässe, Mist-Tiefe, Film transparent | Text, m | `setup_render` |
| render.seed | Seed, animiert ja/nein | Zahl | `setup_render` |

## 11. Nachbearbeitung und Encoding
| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| post.dunst | mist_depth, haze_dist, haze_start, haze_color, haze_sky, haze_fade | m, RGB, Liste | `params/<welt>.py` |
| post.look | View Transform / Look | Text | `look` |
| post.farbe | sat, ev | Faktor, EV | `sat`, `ev` |
| post.autobelichtung | ae_strength, ae_tau, ae_ref | Faktor, s | `ae_*` |
| post.bloom | bloom, bloom_thr, fog_glow, bloom_clamp | Faktoren | `bloom*` |
| post.optik | dispersion, vignette, grain | Faktoren | `params` |
| post.pulse | Belichtungsstöße: Zeit, +EV, Dispersion, Dauer | Liste | `pulses` |
| post.flare | Lens Flare: t0, t1, thr, strength, streak, fade_out | s, Faktoren | `flare` |
| encode.skalierung | Filter, Zielgröße | Text, px | `post.process` |
| encode.master | Codec, Profil, Pässe, Bitrate (Soll/Ist), Dateigröße | Text, Mbit/s, MB | `bitrate` |
| encode.web | Bitrate, maxrate, bufsize, Dateigröße | Mbit/s, MB | `web_version.sh` |

## 12. Laufzeiten und Ressourcen (Messwerte nach dem Final)
| ID | Bedeutung | Einheit |
|---|---|---|
| zeit.szene_bauen | Aufbau der Szene | min |
| zeit.frame_mittel / .frame_max | Renderzeit pro Frame | s |
| zeit.render_gesamt | reine Renderzeit | h |
| zeit.wanduhr | Start bis Ende inkl. Ausfälle | h |
| zeit.vorschau | Vorschau-Sequenz(en) | min |
| ressourcen.ram / .blend / .exr | Speicher, Dateigrößen | GB, MB |
| ressourcen.neustarts | Container-Neustarts während des Finals | Anzahl |

## 13. Sound-Marker
| ID | Bedeutung | Format | Code-Ort |
|---|---|---|---|
| sound.marker | Name, Zeit, Frame (AMBIENCE, WHOOSH, IMPACT, ZANZOKEN, BEAT_DROP) | Liste | `sound_markers` → `sound_markers.csv` |

## 14. Bewertung (für die spätere Analyse)
| ID | Bedeutung | Format |
|---|---|---|
| bewertung.freigabe | Datum, freigegeben ja/nein, Zitat des Nutzers | Text |
| bewertung.korrekturen | was nach Vorschauen geändert wurde (Parameter-ID → alt/neu, Grund) | Liste |
| bewertung.reaktion | Reaktion auf das Ergebnis (wörtlich) | Text |
| bewertung.social | später: Aufrufe, Watchtime, Likes (falls gepostet) | Zahlen |

## 15. Schiff und Crew (Videos mit Fahrzeug und Figurengruppe statt Kampf)
Bei Videos ohne Kampf (z. B. One Piece: Thousand Sunny mit Strohhutbande) entfallen `kampf.*` und große Teile von
`fx.*`; dafür gelten diese IDs. Referenz-Umsetzung: `world_onepiece.py`, `sunny.py`, `crew.py`, `tripo_crew.py`.

| ID | Bedeutung | Einheit / Format | Code-Ort |
|---|---|---|---|
| schiff.modell | Schiffsmodell, Bauweise, Quelle (Model Sheets) | Text | `sunny.build` |
| schiff.kurs / .tempo / .start | Kurs, Fahrt, Startpunkt | °, m/s, m | `SHIP_HEADING`, `SHIP_SPEED`, `SHIP_START` |
| schiff.bewegung | Stampfen/Rollen | Faktoren | `sunny.animate` |
| schiff.flaggen | Stoff-Simulation (Cache) | Text | `sunny.flag_cloth` |
| schiff.gischt | Bug-Gischt: Raten, Aufwärtstempo, Tropfenradien | 1/s, m/s, m | `SPRAY`, `ocean.bow_spray` |
| schiff.kielwasser | Bugwelle, Heckwelle, Rumpfschaum | Text | `ocean.ocean_fx_gn` |
| kamera.speed_keys | Tempo über die Zeit (statt Wegpunkt-Zeiten) | Liste (s, m/s) | `SPEED_KEYS` |
| kamera.route_welt / .route_schiff | Bahn weltfest / im Schiffssystem | Liste (x, y, z) | `WORLD_ROUTE`, `SHIP_ROUTE` |
| kamera.uebergang | Überblendung weltfest → schiffsfest | m, s | `BLEND_M`, `t_join` |
| kamera.impact | Kamerastoß beim Überfliegen der Reling | dict | `IMPACT` |
| licht.randlicht_fade | Randlicht aus, bevor die Kamera zurückblickt | s, s | `RIM_FADE` |
| crew.figuren | je Figur: Modell, Höhe, Rig-Quelle, Look | Liste | `tripo_crew.SPEC`, `TOON` |
| crew.aufstellung | je Figur: Ort (Schiffssystem), Yaw, Boden/Sitz | Liste | `crew.place_crew` |
| crew.schauspiel | je Figur: Posenfolge (Zeit, Pose, Dauer, Ausholen, Überschwingen), Atmen, Schwanken, Blick | Liste | `crew.place_crew`, `act` |
| crew.posen_map | figurenspezifische Posen-Ersetzungen | dict | `tripo_crew.POSE_MAP` |
| crew.drehung | Körperdrehung über die Zeit (z. B. zur Kamera) | Liste (s, °) | `act(turn=…)` |
| crew.requisiten | Requisiten an Gelenken | Liste | `tripo_crew.PROPS` |
| crew.texturen | Texturgröße der Figuren | px | `tripo_crew.TEX` |
