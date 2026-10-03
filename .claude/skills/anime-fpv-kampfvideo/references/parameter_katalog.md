# Parameter-Katalog für ein Anime-FPV-Kampfvideo

Alle Stellschrauben eines Videos nach dem Vorbild Namek, **ohne Werte**. Die Werte legst du je Video fest und
trägst sie nach dem Final mit denselben IDs ins Werte-Archiv ein
(`drohnen_3d_welten/werte/<NN_name>/werte.json`, Vorlage: `werte_vorlage.json`). Gleiche IDs über alle Videos
erlauben später die Auswertung, z. B. welche Kameratempi, Schlagdauern oder Effektgrößen gut ankamen.

Code-Ort bezieht sich auf das Repo `SnaQOG/Cursor`, Ordner `drohnen_3d_welten/`. Die Referenz-Umsetzung ist
`world_dragonball.py` (Namek).

## Inhalt
1. Format und Ablauf
2. Licht und Himmel
3. Gelände und Welt-Objekte
4. Wasser
5. Vegetation, Gras, Kleinkram
6. Kamera
7. Figuren
8. Kampfplan und Timing
9. Effekte
10. Render
11. Nachbearbeitung und Encoding
12. Laufzeiten und Ressourcen (Messwerte)
13. Sound-Marker
14. Bewertung (für die spätere Analyse)
15. Schiff und Crew (Videos mit Fahrzeug und Figurengruppe statt Kampf)

---

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
