# Namek-Video – alle Parameter (Stand Final, 2. Oktober 2026)

Exakte Werte des freigegebenen Finals `03_DragonBall_Namek.mp4`. Quelle ist der eingefrorene Code in
`scripts/drohnen_3d_welten/` (Commit `ef59ba2`). Die Zeitpläne für Kampf und Kamera stehen Wert für Wert in
`kampfplan_und_kamera.json`. Was hier nicht steht, steht im Code; jede Zahl dort ist der Wert des Finals.

## Inhalt
1. Format und Ablauf
2. Licht und Himmel
3. Gelände und Welt-Objekte
4. Meer
5. Vegetation, Gras, Kleinkram
6. Kamera
7. Figuren (Tripo-Modelle, Rig, Look)
8. Kampfplan und Timing
9. Effekte
10. Render (Cycles)
11. Nachbearbeitung und Encoding
12. Laufzeiten und Ressourcen
13. Sound-Marker

---

## 1. Format und Ablauf
- 20,0 s, 24 fps, 480 Frames, Hochformat 9:16; Final 1080 × 1920
- Eine einzige durchgehende FPV-Aufnahme, kein Schnitt, kein Outro
- Koordinaten: +Y ist die Hauptflugrichtung (Süd → Nord), Z oben, Meter

| Zeit | Inhalt | Kamera-Tempo |
|---|---|---|
| 0–2,7 s | Hook: hell, 2,5–3,5 m über dem Meer, Slalom links an Felsnadel 1 (y 52), rechts an Felsnadel 2 (y 72) | 26–29 m/s |
| 2,7–5,2 s | offenes Wasser, ferne Lichtspur-Zusammenstöße über der Tafelbergkante (Teaser 2,3 / 4,6 / 7,6 / 8,8 s) | 29–31 m/s |
| 5,2–7,8 s | Hochziehen, Steigflug dicht an der Steilwand (~11 m Abstand), über die Kante: Reveal des Plateaus | 30 → 10 m/s |
| 7,8–10 s | tief (3,2–4,4 m) über blauem Gras, Kuppelhäuser, Dragon Balls links (9,55 s) | 25–32 m/s |
| 10–13,6 s | Nahkampf 12–40 m voraus | 27 → 1,7 m/s |
| 13,6–15,8 s | Trennung, Aufladen, Strahlenduell | 26 → 3 m/s |
| 15,8 s | Klimax: Explosion über dem Meer | 3 m/s |
| 16–20 s | Nachglühen, Rauchsäule, Kamera zieht zurück und steigt (z 46 → 51) | 3,0–3,1 m/s |

## 2. Licht und Himmel
- **Hauptsonne** `Sun1`: Höhe 19°, Azimut 225° (links hinten), 5,0 W/m², 4300 K, Winkel 1,2°
- **Sonne 2:** 36° / 140°, 1,3 W/m², 5600 K, Winkel 2,5° (Aufhellung hinten rechts)
- **Sonne 3:** 7° / 16°, 0,7 W/m², 4700 K, Winkel 1,5° (tief vorn rechts, sichtbar am Horizont, Gegenlicht)
- **Sichtbare Sonnenscheiben** im Himmel (`extra_suns`):
  - Hauptsonne: Radius 0,9°, Stärke 900, Farbe (1,0, 0,95, 0,85)
  - Sonnen 2 und 3: Radius 0,6°, Stärke 350 × W/m², Farbe (1,0, 0,96, 0,9)
- **Randlicht** nur auf den Kämpfern (Light Linking): Höhe 18°, Azimut 10°, Stärke 2,6, 7800 K, Winkel 3°
- **Himmel** (`namek_sky`, eigener Shader, ~30 % entsättigt gegenüber dem Anime):
  - Farbverlauf über sqrt(z): 0,0 (0,80, 0,89, 0,56) · 0,07 (0,50, 0,75, 0,33) · 0,30 (0,19, 0,50, 0,19) ·
    1,0 (0,06, 0,30, 0,12)
  - unterhalb des Horizonts (0,52, 0,64, 0,50)
  - Leuchten um Hauptsonne und Sonne 3: dot^8 × 0,55 + dot^64 × 1,6, Farbe (1,0, 0,93, 0,78)
  - diffuses Himmelslicht zu 60 % entsättigt; nur Kamera- und Glanzstrahlen sehen den vollen Farbton (sonst
    würden blaues Gras und beiger Fels türkis)
- **Welt:** `sky_strength` 0,8, Wolken an, `cloud_cover` 0,22, `cloud_ref` 1,1, Wolkenfarbe (1,0, 1,0, 0,86),
  `cloud_scale` 1,3
- **Atmosphärenvolumen:** aus (`NIDO_ATMO`=0); den Dunst macht die Nachbearbeitung
- **Toon-Licht der Figuren:** `anime_chars.set_light(sun_dir(19, 225), sun_dir(18, 10))`

## 3. Gelände und Welt-Objekte
- **Tafelberg:**
  - Mitte (0, 250), Grundradius `MESA_R` = 74 m
  - Umriss: R · (1 + 0,08 sin(3a+1) + 0,05 sin(7a+2) + 0,06 fbm(cos·3, sin·3, 3 Oktaven, seed 21)), nur große
    Buchten (feine Rinnen würden an der senkrechten Wand zu Orgelpfeifen)
  - Plateau: 39,0 + 2,0·fbm(x/40, y/40, 4, seed 22) + 0,4·fbm(x/6, y/6, 3, seed 23)
  - Kantenabfall 2,5–6,5 m hinter der Kante (Smoothstep); Schuttfuß außen: ((R+18−r)/18)² · 7 − 3
  - Heightfield „Mesa“: (2·74+60)² m, 400 × 400 Punkte; Faces unter z 30 bekommen den Felsen-Slot
  - Steilwand `namek.ring_wall`: n_ang 1500, n_z 150, seed 5, Felskante (lip) (2,0, 4,0, 6,8), base_z −3
- **Felsnadeln** `SPIRES` (x, y, r, h, seed, Neigung x°, y°); Mesh `fpv.rock_mesh`: height h+4, seed seed+50,
  detail 6, taper −0,35 (Pilzkopf), base_z −4, lumpy 0,22, strata 0,6, flute 0,06, ledges 0,03, bed (1,8, 4,0),
  Drehung um z = seed·2,1 rad
  (−15,52,7,46,1,4,−3) (16,72,8,58,2,−3,5) (−48,20,6,30,3,6,2) (42,18,5,24,4,−5,−4) (−42,110,10,52,5,2,6)
  (55,128,9,40,6,−4,3) (−95,170,14,70,7,3,−2) (110,200,12,60,8,−2,4) (−150,60,16,55,9,5,1) (140,40,10,48,10,−6,−2)
  (25,120,4,14,11,8,−5) (−22,150,5,20,12,−7,4)
- **Ferne Felsnadeln** (x, y, r, h): (−160,470,26,95) (120,520,20,70) (−60,640,34,120) (260,700,30,85)
  (−330,820,45,140) (40,980,55,150) (420,1050,50,120) (−620,1150,70,170); detail 4, taper −0,3, seed 200+k, je
  r·0,5 Bäume auf dem Kopf
- **Inseln** am Horizont (x, y, r, h): (−900,1400,260,150), (800,1700,300,170); 180² Gitter
- **Fels-Material** `NamekRock`: c1 (0,15, 0,10, 0,10), c2 (0,58, 0,40, 0,28), c3 (0,44, 0,31, 0,25), wet_line 1,6,
  algae (0,03, 0,06, 0,05), moss (0,02, 0,10, 0,28), moss_amount 0,22, strata_scale 2,2, bump 1,0, crack_w 0,15,
  scale 1,5, lichen 0,12, Moos-Textur `grasslight-big.jpg` × 4, variation 0,6, bedding 0,45, bed_h 2,2, streak 0,5
- **Steilwand** `NamekCliff`: c2 (0,60, 0,42, 0,29), c3 (0,45, 0,32, 0,25), moss (0,03, 0,08, 0,16), moss_amount 0,04,
  lichen 0,1, variation 0,4, bedding 0,35, streak 0,3 (Rest wie Fels)
- **Plateau-Boden** `NamekGround`: Erde (0,08, 0,065, 0,05) … (0,16, 0,13, 0,10), Moosflecken (0,03, 0,08, 0,16)
  ab Rauschen > 0,55, Roughness 0,85, Bump 0,3
- **Kuppelhäuser** (x, y, r): (−20,204,6,5) (−36,224,8) (−22,252,5,5) (−46,252,6) (−28,276,7,5) (26,204,6)
  (30,228,5) (−17,300,6) (−44,290,5) (27,300,6,5)
  - Material Lehm (0,78, 0,79, 0,74), dunkel (0,62, 0,64, 0,58)
  - Glas (0,015, 0,03, 0,03) mit Roughness 0,06; Tür (0,03, 0,05, 0,045)
- **Dragon Balls:** Mitte (1, 232), Sockel r 1,7 → 1,5 (Höhe bis Plateau + 0,4), Kugeln r 0,2 im Ring 0,8
- **Freezers Raumschiff:** (42, 262), z = Plateau + 1,5, R 16
  - Rumpf (0,56, 0,53, 0,48), dunkel (0,30, 0,29, 0,28), Nietenplatten-Textur (Godot TPS)
  - schwarz-gelber Rand mit 48 Streifen
  - Fenster: Emission (1,0, 0,3, 0,2) × 1,2

## 4. Meer
- Kacheln 3 × 5 à 100 m ab (−140, −40); Ocean-Modifier res 13, wind 6,5, wave_scale 0,7, chop 1,1,
  Richtung 60°, alignment 0,3, foam_coverage 0,1
- Material `NamekSea`: deep (0,004, 0,032, 0,030), shallow (0,025, 0,14, 0,11), foam_amount 0,4, view_dark 0,5,
  micro 0,6; Brandung über Abstandsfeld (Bereich −200…200 × −100…420, Zelle 0,5 m), Fresnel über Principled IOR 1,333
- Fernebene bis zum Horizont (z −0,05) mit `far=True`

## 5. Vegetation, Gras, Kleinkram
- **Ajisa-Bäume** (4 Varianten: Höhe, Kronenradius, Drehung):
  - Varianten: (12, 2,8, 0,8), (15, 3,3, 1,0), (9, 2,3, 0,6), (17, 3,6, 1,1)
  - leaves = 900·cr²/4, leaf_size 0,36, seed 70+i
  - Blatt c1 (0,004, 0,045, 0,30), c2 (0,03, 0,12, 0,50), trans 0,3; Rinde (0,42, 0,33, 0,18)
- **Haine** (x, y, r, n): (−56,205,9,8) (50,210,8,6) (−6,270,6,4) (−52,312,10,8) (38,322,9,7) (62,240,7,5)
  (−66,264,8,6) (20,186,6,4) (−28,184,6,5) (−4,334,6,4) (24,250,5,3) (−16,226,4,3) (26,286,6,4) (−24,312,5,4)
  (18,316,4,3)
  - je n·3 Kandidaten, Skalierung 0,7–1,25
  - Felsnadel-Köpfe: 3 + 0,8·r Bäume, Skalierung 0,45–0,9; ferne Felsnadeln 1,2–2,2
- **Abstand der Bäume zu …:**
  - Häusern: r + 3
  - Dragon Balls: 5; Schiff: 22; Krater: 12; Strahlkratern: 8
  - Kameraweg: 9; Tafelbergkante: 9
  - Sichtlinie Kamera → Kämpfer 9,8–13,7 s (jeder 3. Frame, 7 Punkte): 8,5 m, sonst verdecken Kronen die Treffer
- **Findlinge:** 45 Versuche, r 0,6–2,4, Skalierung (0,9–1,4, 0,8–1,2, 0,5–0,8), Displace CLOUDS (size r·0,6,
  strength r·0,35); Abstand Weg 5, Kante 8
- **Sandflecken:** 30, rx 1,5–6, ry rx·(0,45–0,9), Rand 8 + rx zur Kante
- **Pilze:** 50 Gruppen 2,5–14 m seitlich der Route (y 190–318)
- **Blaues Gras** `namek.grass_field`:
  - Bereich y 176–330, nur ≥ 7,5 m hinter der Kante
  - Dichte (nah 360, Grund 170, Abfall 8 m), Band 30 m
  - clump 0,6, max 2.200.000 Halme, Höhe 0,10–0,42 m
  - Halmfarbe: Basis (0,008, 0,035, 0,13) … (0,014, 0,055, 0,19), Spitze (0,035, 0,16, 0,44) … (0,07, 0,25, 0,56),
    Büschel türkis/violett (0,05, 0,22, 0,40) … (0,10, 0,20, 0,52)
  - Ausgespart: Häuser r·0,95, Dragon Balls 1,7, Schiff 17,5, Findlinge, Sandflecken
- **Grasbewegung** `namek.grass_motion`:
  - Wind 0,35, Böen (0,9 m/s, 0,35), sway 0,28
  - Druckwellen (x, y, t, m/s, Stärke, Breite): Zusammenprall (1, 273, 10,0, 45, 0,35, 1,5), Krater
    (3, 278,5, 12,15, 38, 0,9, 2,0), Strahlkrater (…, 30, 0,55, 1,5), Explosion (0,5, 369, 15,8, 75, 1,0, 4,0)
  - Halme in den Kratern verschwinden (Radius × 1,15)

## 6. Kamera
- **Objektiv:** 28 mm, Sensor vertikal 36 mm (Vollformat-Äquivalent, lange Seite)
- **Bahn:** `CAM_KEYS` (Zeit, Ort) mit Speed-Ramp aus Abstand/Zeit, monoton-kubisch (`choreo.keyed_path`); alle
  26 Wegpunkte in der JSON
  - Wegpunkte über dem Plateau: `over(x, y, dz)` = Plateauhöhe + dz
  - 12,90 s: `over(9.5, 266.0, 6.0)`, etwas zurück, damit Freezer, Goku und der Einschlag zusammen im Bild sind
- **Halte:** `CAM_HOLDS` = [(12,15 s, 3 Frames), (15,8 s, 3 Frames)], Zeitverzerrung über `choreo.time_warp`
- **Ausrichtung** `fpv.fpv_orient`:
  - look_pitch −3°, pitch_follow 0,45, bank_gain 1,0, max_bank 32°, micro 1,0, seed 13
  - pitch_overrides: (−1,5–2,2 s: +5°), (5,4–7,5 s: +12°)
- **Blickführung:**
  - Gewicht (Zeit, w): (0, 0) (5, 0) (7,3, 0) (8,4, 0,25) (9,4, 0,5) (10, 0,85) (10,4, 1) (21, 1), PCHIP
  - Blickziele: Kämpferpaar (Mitte nach Winkel; Kämpferbahnen mit Gauß 0,12 s geglättet, kein Reißen bei Haken und
    Teleports)
  - 12,0–13,0 s Krater (Gewicht 0,62)
  - 12,85–13,75 s Freezer, Goku und Einschläge
  - ab 13,6 s Duell
  - ab 16,2 s Goku und Rauchsäule (z + 7, Gewicht 0,45)
- **Kamerastöße** `SHAKES` (Zeit, Stärke, Frequenz-Frames): (10,0, 1,2, 6), Schlaghagel je (t, 0,35, 3),
  (11,11, 0,7, 5) (11,47, 1,1, 6) (11,95, 1,0, 6) (12,15, 2,4, 9) (13,2, 1,0, 6) (13,45, 1,0, 6) (15,05, 0,7, 8)
  (15,8, 3,6, 12) (16,35, 1,3, 10); seed 78, pos_amp 0,03

## 7. Figuren
- **Son Goku SSJ:** `tripo_chars.goku5()`
  - Datei `assets/models/dragonball/goku5/gokuactionfigure3dmodel_repariert.glb` (18,4 MB, repariert: Bind-Matrizen
    neu, Metallic-Map entfernt)
  - Höhe 1,75 m
  - Toon-Material: sat 1,12, value 1,04, shade (0,52, 0,48, 0,64), hair_glow 0,25
  - Kontur 0,004 × Höhe
- **Freezer Endform:** `tripo_chars.freezer4()`
  - Datei `assets/models/dragonball/freezer4/friezafinalform3dmodel.glb` (15,1 MB)
  - Höhe 1,50 m
  - Schwanz `FREEZER3_TAIL` (12 Mittellinienpunkte mit Radius) → 10-gliedrige Kette
  - Toon-Material: sat 1,05, shade (0,58, 0,55, 0,74), rim (0,92, 0,88, 1,0)
  - Kontur 0,004
- **Gelenke:** aus dem Mixamo-Skelett, Zuordnung `tripo_chars.MIXAMO`: Hips, Spine1, Neck, Head, HeadTop_End, Arm,
  ForeArm, Hand, HandMiddle4 (Ende), UpLeg, Leg, Foot, Toe_End
- **Rig** (`rig_model`):
  - Ruhelage mit den Modellproportionen; Armature in der Modellpose; Copy Transforms (WORLD) von den Gelenk-Empties
  - Gewichte geodätisch über die Mesh-Kanten (power 4, Saat < 1,5 Dicken, Innenteil 0,15–0,85, Wurzel −1,5–0,85,
    Kopf/Hände/Füße bis 3–4), top 3, 3 Glättungsdurchgänge
  - UV-Nähte werden für die Rechnung zusammengelegt (seit Oktober; das Final lief noch ohne, Goku und Freezer
    zeigten dabei keine Risse)
- **Toon-Material-Grundwerte:** lit 1,0, mid 0,8, Stufen (−0,1, 0,35), tint_lit (1,0, 0,96, 0,9),
  rim (1,0, 0,97, 0,9), rim_w 0,4, diffuse_mix 0,14 (Rest echte Diffusion, damit Energie-Lichter färben)
- **Backen** `choreo.bake_fighter`:
  - style `dbz_style`, lag_scale 0,35, lean_tau 0,04
  - seed 3 (Goku) / 4 (Freezer); Goku `look_win` (17, 21) schaut am Ende zur Kamera
- **Schwanz:** `anime_chars.tail_follow(Z, n, 24, whips=[(11,11, 60°)], warp=hit_warp)`; sway 9°
- **Freezer** verschwindet im Feuerball: scale 1 → 0,001 in den Frames F(15,8)+1 → +2
- **Goku-Aura** (gold): Frames F(9,4)–F(17,8), Höhe 1,95, Breite 0,95, Licht 250 W, opacity 0,25, edge_w 0,1,
  strength 0,6
- **Mimik:** gibt es nur bei den selbst gebauten Fallback-Figuren (`NIDO_CHARS=anime`); bei den Tripo-Figuren ist das
  Gesicht in der Textur

## 8. Kampfplan und Timing
- **Blocking:** `_key(t, Pose, Brust-Ort, Blickziel, lean, roll, rot, air, ease, spin, style)`; Brust liegt
  `CH` = 1,2 m über dem Figurenursprung
- **Dragon-Ball-Stil** `dbz_style` (Dauer s, Ausholen, Überschwingen):
  - Angriffe (d_punch, d_kick, knee, axe_down, tail_whip): 0,09 / 0 / 0,22
  - Ausholposen *_wind, axe_up: 0,11 / 0 / 0,1
  - Treffer-Reaktionen (recoil, gut_hit, slam, land): 0,07 / 0 / 0,25
  - Abwehr (guard, block): 0,08 / 0 / 0,18
  - Flug (fly, dash): 0,12 / 0 / 0,06
  - Kamehameha/Aufladen: 0,3 / 0,1 / 0,06
  - sonst: 0,2 / 0 / 0,08
- **Bahnprofile** (`ease`): `lin` (Vorstoß ohne Bremsen bis zum Kontakt), `in` (aus dem Stand, schnell an), `out`
  (mit voller Wucht los, auslaufen)
- **Hit-Stops** `HITS` (Zeit, Frames): Zusammenprall 3, Schlaghagel je 1, Schwanzhieb 2, Knie 3, Doppelfaust 3,
  Einschlag 4, Klimax 4
- **Zeitplan:**
  - Teaser-Zusammenstöße bei 2,3 / 4,6 / 7,6 / 8,8 s: aus 12 m aufeinander zu, Rückprall 6–7 m; Mitten und
    Achsen in `TEASERS`
  - Start des großen Vorstoßes: Goku (−9, 282, 53), Freezer (13,5, 284, 52); T_RUSH 9,62, ~45 m/s
  - **10,00 Faust auf Faust** bei C0 (1, 273, 45,5)
  - **10,28–10,98 Schlaghagel**, 6 Wechsel alle 0,14 s:
    - G d_punch_R, Z d_punch_R, G d_kick_R, Z d_kick_L, G d_punch_L, Z d_punch_L
    - Ausholen 0,07 s vorher; Kampfachse dreht 8,5° → 50°, das Paar driftet nach C1 (1,6, 275,2, 45,8)
  - **11,11 Schwanzhieb** (Freezer 360°-Drehung 11,02–11,20)
  - **11,47 Knie in den Magen**
  - **Zanzoken 1:** 11,53 → 11,75, Frames 280 → 283
  - **11,95 Doppelfaust** → **12,15 Einschlag** im Krater P_CRATER (3, 278,5)
  - 12,8 Freezer bricht aus dem Krater
  - **13,15 Todesstrahl 1** aufs Nachbild (Zanzoken 2: 13,15 → 13,27, Frames 317 → 320)
  - **13,40 Todesstrahl 2** knapp unter Goku durch
    - Goku vor/nach dem Ausweichen: GB1 (−2,5, 272,5, 42,3), GB2 (−6, 274,5, 45,2)
    - Freezer beim Feuern: A1 (−1,3, 276, 48,8), A2 (−1, 276,3, 49,2)
    - Hochformat: senkrecht gestaffelt, damit alles ins Bild passt
  - **13,6–14,3 Trennung** mit 70–130 m/s: Freezer übers Meer nach Z_SEA (0,5, 369, 36), Goku an den Nordrand
    G_RIM (−0,5, 323,2, 43,9)
  - **14,45–14,95 Aufladen**, **15,05 Strahlenduell**, **15,80 Durchbruch / Explosion** (T_CLIMAX)
- **Eigene Posen** (Gelenkwinkel im Code, `POSES.update`): axe_up, axe_down, hover, slam, d_wind_R, d_punch_R,
  d_kick_wind_R, d_kick_R, block, dash, knee_R, gut_hit, tail_whip (+ gespiegelte _L)
- **Alle Schlüssel** (52 Goku, 51 Freezer), Treffer, Ki-Spuren und Strahlen: `kampfplan_und_kamera.json`

## 9. Effekte
- **Farben:**
  - GOLD (1,0, 0,62, 0,10) · KAME (0,10, 0,36, 1,0) · DEATH (0,85, 0,25, 1,0) · KI (1,0, 0,80, 0,28)
  - Ki-Spuren: Goku (1,0, 0,78, 0,25), Freezer (0,78, 0,32, 1,0)
- **Ki-Spuren** `dbz_fx.ki_trail`:
  - Schweif der letzten 0,16 s, center_z 0,95, strength 6, Kern r·0,32
  - Radius je Ereignis 0,18–0,45 (Liste `ki_spuren` in der JSON)
- **Zanzoken:**
  - flackerndes Nachbild, Deckkraft je Frame `FLICKER` = 0,95 0,35 0,85 0,2 0,7 0,12 0,5 0,06 0,3 0
  - Luftring beim Verschwinden r 1,4, beim Auftauchen r 1,8; alle Render-Objekte samt Aura werden ausgeblendet
- **Treffer** `HIT_FX` (Radius, Licht W, Funken, Luftring):
  - clash (3,2, 9000, 110, –), flurry (0,9, 1500, 40, 1,6), whip (1,4, 3000, 60, 2,2), knee (1,6, 6000, 110, 4,0),
    axe (1,6, 6000, 110, 3,0)
  - Burst-Dauer 9 bzw. 6 Frames, Kern core_k 1,0 (clash) bzw. 0,5
  - Funken 6–14 m/s, Lebensdauer 0,15–0,45 s
  - Druckwelle beim Zusammenprall: r 9, 10 Frames, thick 0,12, glow 2
- **Teaser-Blitze:** r 11, 12 Frames, Licht 90.000 W, core_s 25, glow_s 6, glow_alpha 0,35
- **Einschlag 12,15 s:**
  - Krater r 3,2 mit glühendem Boden (heat_variant, cool_s 3,2); Strahlkrater r 2,1
  - Burst r 4,5 (14 Frames, 40.000 W), Druckwelle r 16, Staub 1400 Partikel (r 9, rise 2,2, 2,2 s), Funken 200
  - 16 Voronoi-Bruchstücke aus einer Platte r 2,3 × 0,45 m, Rigid Body über 60 Frames, Impuls 38.000
  - 12,8 s Austritt: Staub 500, Luftring r 4
- **Todesstrahlen:**
  - Ladekugel r 0,09, Strahl r 0,18, 4000 W, Frames F(t) → +2 → +7
  - Einschlag-Burst r 4 (16 Frames, 35.000 W), Staub 900, 12 Brocken mit 12 m/s
- **Kamehameha-Duell:**
  - Ladung: Goku r 0,34 (F(14,45)–F(15,1), 600 W), Freezer r 0,22 (500 W)
  - Kamehameha r 0,95, Kern 0,3, 5000 W; Todesstrahl r 0,55, 12.000 W; Wobble 0,12 / 0,15
  - Treffpunkt bei 0,55 L, pendelt (0,25 s: Lc, 0,45: −3 m, 0,6: +2 m, 0,75: −1,5 m) und bricht bei 15,8 s durch
  - Kugel am Treffpunkt r 2,0 (30.000 W), 10 Blitzbögen r 4,5, Burst beim Aufeinandertreffen r 6 (90.000 W)
- **Explosion 15,8 s:**
  - Blitz r 7; prozedurales Volumen (Feuer r 8, Rauch r 9, rise 14 m, 4,4 s, fire 14) statt Mantaflow
  - Punktlicht 900.000 → 600.000 (+4) → 220.000 (+14) → 60.000 (+40) → 20.000 W (20,2 s); Farbe warm (1,0, 0,62, 0,3)
    → (1,0, 0,7, 0,45) → kühl (0,72, 0,8, 1,0)
  - Druckwellen: Luft r 60 (20 Frames), Wasser r 70 (26 Frames)
  - Funken 500 (15–35 m/s), Gischt 2500 Partikel (r 30, rise 9, 3,2 s)
- **Rendersicherheit:**
  - Burst-Meshes außerhalb ihrer Lebenszeit ausgeblendet (sonst stapeln sich transparente Flächen über das Limit)
  - transparent_max_bounces 16
  - Luftringe ohne Brechung (ior 1,0)

## 10. Render (Cycles, CPU)
- **Auflösung und Qualität:** Final 720 × 1280, 32 Samples, adaptiv (Schwelle 0,04)
- **Denoiser:** OpenImageDenoise mit Albedo und Normal, Prefilter ACCURATE; Seed 7, nicht animiert
- **Bounces:** max 4, diffus 2, glossy 2, Transmission 6, Volumen 0, transparent 16; keine Kaustiken
- **Sampling:** Clamp indirekt 6,0, Blur Glossy 1,0, Light Tree an, Persistent Data an
- **Volumen:** volume_step_rate 2,0, volume_max_steps 128
- **Bewegungsunschärfe:** an, Shutter 0,5; Pixelfilter Blackman-Harris 1,5 px
- **Ausgabe:** Film transparent; Pässe Combined, Mist (Tiefe 6000 m), Environment als Multilayer-EXR je Frame
  (`f_####Image.exr`, `…Mist.exr`, `…Env.exr`, ~0,8 MB pro Frame)
- **Wind-Modifier** (`gust`, `flutter`): ohne Deform-Motion-Blur

## 11. Nachbearbeitung und Encoding (`params/dragonball.py`, `post.process`)
```python
POST = dict(mist_depth=6000, haze_dist=1500, haze_start=30, haze_color=(0.66, 0.78, 0.62), haze_sky=True,
            look="AgX - Medium High Contrast", sat=1.0, ev=0.0, ae_strength=0.55,
            bloom=0.25, bloom_thr=1.8, fog_glow=0.6, bloom_clamp=4.0, dispersion=0.006, vignette=0.22, grain=0.025,
            pulses=[(10.0, 0.4, 0.008, 0.06), (11.11, 0.2, 0.004, 0.05), (11.47, 0.3, 0.006, 0.06),
                    (11.95, 0.35, 0.006, 0.06), (12.15, 0.7, 0.012, 0.09), (13.25, 0.4, 0.008, 0.07),
                    (13.5, 0.4, 0.008, 0.07), (15.05, 0.3, 0.006, 0.1), (15.8, 1.8, 0.02, 0.08)],
            flare=dict(t0=15.75, t1=17.2, thr=6.0, strength=0.6, streak=0.5, fade_out=0.8),
            haze_fade=[(15.6, 1.0), (16.0, 0.0), (21.0, 0.0)],
            bitrate="18M", still_full=True)
```
- **Pulse** (Zeit, +EV, Dispersion, Dauer s): Belichtungsstöße auf den Treffern
- **Dunst:** blendet ab 15,6 s aus, damit die Explosion nicht im Dunst verblasst
- **Auto-Belichtung:** τ 0,45 s, Stärke 0,55
- **Encoding:**
  - Hochskalieren mit Lanczos auf 1080 × 1920
  - Master H.264 High, Zwei-Pass 18 Mbit/s (Ergebnis 17,9 Mbit/s, 44,9 MB)
  - Webversion Zwei-Pass 10 Mbit/s, maxrate 11M, bufsize 20M, faststart (24,8 MB): `scripts/pipeline/web_version.sh`

## 12. Laufzeiten und Ressourcen (4 CPU-Kerne, keine GPU)
- **Szene bauen:** ~10–15 min; blend ~250 MB
- **Final:** 136 s pro Frame im Schnitt (max. 314 s); 18,1 h reine Renderzeit für 480 Frames
- **Vorschau** 270 × 480, 6 Samples: Kampfteil (241 Frames) ≈ 36 min, Flugteil (239 Frames) ≈ 2 h (Gras)
- **Standbild** 540 × 960, 10 Samples: 1–3 min
- **RAM:** 4–9 GB; Platz: EXRs ~0,4 GB, Vorschauen mehrere GB (alte löschen)

## 13. Sound-Marker (`sound_markers.csv` neben den Frames)
AMBIENCE 0,0 · WHOOSH 1,92 / 2,67 (Felsnadeln) · WHOOSH 5,7 (Hochziehen) · WHOOSH 7,04 (Kante) ·
WHOOSH 9,62 (Vorstoß) · IMPACT 10,0, 10,28, 10,42, 10,56, 10,70, 10,84, 10,98, 11,11, 11,47 ·
ZANZOKEN 11,62 · IMPACT 11,95, 12,15 · ZANZOKEN 13,15 · IMPACT 13,15, 13,40 · WHOOSH 13,6 ·
BEAT_DROP 15,8 · AMBIENCE 17,0

---

# Ergänzung: Stadt- und Kino-Parameter (Katalog Teil A) und fehlende IDs

Nachgetragen am 2026-10-03, als der Parameter-Katalog um die Stadt- und Kino-Parameter erweitert wurde. „entfällt“ heißt, dass es den Parameter in diesem Video nicht gibt; „nicht erfasst“, dass er nicht gemessen oder festgehalten wurde.

## S1. Briefing und Planung
- `briefing.stadt` (Stadt, Thema, Epoche, Kultur): Planet Namek (Dragon Ball Z), Kampf Son Goku gegen Freezer
- `briefing.stil` (fotoreal, stilisiert oder Anime): Umgebung halbrealistisch (physikalisch), Figuren Toon (Tripo-Modelle mit Kontur)
- `briefing.tageszeit` (Tageszeit (genau benannt)): Dauertag mit drei Sonnen; Hauptsonne tief (19°) links hinten
- `briefing.wetter` (Wetter und Jahreszeit): klar, Wolkenbedeckung 22 %
- `briefing.stimmung` (Stimmung in einem Satz): Kampf im Dragon-Ball-Tempo über blauem Gras, Klimax Explosion über dem Meer
- `briefing.kameratyp` (FPV, Flyover oder Static): FPV, eine durchgehende Aufnahme ohne Schnitt
- `briefing.verwendung` (privat, Video-zu-Video-KI oder Veröffentlichung): nicht ausdrücklich festgelegt (NIDO-Drohnenvideo)
- `briefing.engine` (Engine und Device): Cycles, CPU
- `briefing.budget` (Renderbudget: max. Minuten pro Frame, Gesamtstunden): vorab geschätzt 15–20 h für das Final (720p, 32 Samples)
- `briefing.hardware` (VRAM, RAM, CPU-Kerne, freier Speicher für die Sequenz): {cpu_kerne: 4; gpu: keine; ram_gb: 16; umgebung: Cloud-Container, wird bei Leerlauf neu gestartet}
- `briefing.ausgabe` (Ausgabeformat und Zielordner): {frames: EXR je Pass (Scratchpad); video: H.264 MP4 1080 × 1920 → drohnen_3d_welten/videos/}
- `briefing.referenzen` (Referenzen für Architektur, Material, Licht, Atmosphäre, Kamera): Dominiks Anime-Bilder von Namek (Farben per Pixelmessung), Model Sheets Goku/Freezer, Tripo-Modelle
- `briefing.palette` (Farbpalette und Kontrastidee (aus Referenzen gemessen)): siehe himmel.verlauf, gras.farben, material.fels (aus den Referenzbildern gemessen)
- `planung.einheiten` (Einheitensystem, Maßstab, Skalierung angewendet): {system: METRIC; skalierung: 1; bu: 1 BU = 1 m}
- `planung.detailbudget` (Grenzen nah / mittel / fern und Detailstufe je Zone): Gras nah 360 Halme/m², Grund 170, fern 8 (gras.feld); ferne Felsen und Inseln als einfache Meshes; keine LOD-Stufen
- `planung.collections` (Collection-Struktur und Namensschema): AjisaTrees · AjisaTrees_0 · AjisaTrees_1 · AjisaTrees_2 · AjisaTrees_3 · glTF_not_exported · RigidBodyWorld · RimLightReceivers
- `planung.versionierung` (Dateiname, Version, Backup-Ort): Szene final/namek_final_v2.blend (Scratchpad), Code-Commit ef59ba2

## S2. Stadtlayout und Blockout
- `stadt.koordinaten` (Ursprung, Hauptflugrichtung, Achsen): siehe ablauf.koordinaten
- `stadt.flaeche` (Stadtfläche: Größe, Mitte): entfällt (keine Stadt)
- `stadt.mauer` (Stadtmauer: Mitte, Radius, Segmentwinkel, Höhe, Dicke, Abdeckung, Toröffnung): entfällt (keine Stadt)
- `stadt.tor` (Tor: Ort, halbe Öffnung, Höhe, Oberkante des Querbalkens): entfällt (keine Stadt)
- `stadt.hauptstrasse` (Hauptstraße: Halbbreite, Länge, Mitte, Belag): entfällt (keine Stadt)
- `stadt.zufahrt` (Zufahrtsweg: Breite, Länge, Belag): entfällt (keine Stadt)
- `stadt.platz` (Platz: Mitte, Größe, Belag): entfällt (keine Stadt)
- `stadt.gelaende` (Höhenprofil: Hügelring (Abstand, Breite, Höhe, fbm), Tal der Zufahrt, Plateau, Gipfel): entfällt (keine Stadt)
- `stadt.gelaende_mesh` (Geländenetz: Größe, Auflösung, Ursprung): entfällt (keine Stadt)
- `stadt.wasser` (Fluss, Hafen, Kanäle, Uferlinien): entfällt (keine Stadt)
- `stadt.bruecken_treppen` (Brücken, Treppen, Terrassen): entfällt (keine Stadt)
- `stadt.landmarken` (Wahrzeichen: Name, Ort, Größe, Rolle im Flug (Hook, Manöver, Reveal, Schlussbild)): keine Stadt; Landmarken: Felsnadel 1 (y 52) und 2 (y 72) für den Slalom, Tafelberg-Wand (Steigflug), Kante (Reveal), Dragon Balls, Raumschiff
- `stadt.sichtachsen` (Sichtachsen und Kompositionsmomente entlang der Route): entfällt (keine Stadt)
- `stadt.route` (grobe Flugroute auf dem Blockout): siehe kamera.wegpunkte
- `stadt.seed` (Zufalls-Seed des Aufbaus): entfällt (keine Stadt)

## S3. Gebäude und Aufbau
- `gebaeude.anzahl` (Gebäude gesamt (Messwert)): entfällt (keine Stadt)
- `gebaeude.strassenzeile` (Häuser an der Hauptstraße: Breite, Tiefe, Stockwerke, Lücke, Rücksprung, y-Bereich): entfällt (keine Stadt)
- `gebaeude.stile` (Bauformen und ihre Gewichte (flach, Tonnendach, rund, Walmdach)): entfällt (keine Stadt)
- `gebaeude.raster` (übrige Stadt: Rasterweite, Versatz, Aussparungen, Ausfallquote, Drehung, Größen, Stockwerke): entfällt (keine Stadt)
- `gebaeude.platzrand` (Randbebauung des Platzes: je Gebäude x, y, Breite, Tiefe, Stockwerke, Stil): entfällt (keine Stadt)
- `gebaeude.stockwerkhoehe` (Stockwerkhöhe, Sockel, Brüstung): entfällt (keine Stadt)
- `gebaeude.tuerme` (Stufentürme: Anteil, Radius, Stufen, Dachfarbe, feste Türme an der Straße): entfällt (keine Stadt)
- `gebaeude.daecher` (Dachformen: Stich des Tonnendachs, Stich und Überstand des Walmdachs, Ziegelmaße): entfällt (keine Stadt)
- `gebaeude.dachaufbauten` (Wassertanks, Klimageräte, Rohre, Antennen, Schornsteine: Anteil, Größe): entfällt (keine Stadt)
- `gebaeude.fassade` (Fenster je Stockwerk, Rahmen, Bänke, Läden, Rohre, Kabel, Markisen): entfällt (keine Stadt)
- `gebaeude.banner` (Banner: Anteil, Zeichen, Größe, Höhe, Farben): entfällt (keine Stadt)
- `gebaeude.schilder` (Ladenschilder: Zeichen, Vorder- und Hintergrundfarbe): entfällt (keine Stadt)
- `gebaeude.tueren` (Türen: Motiv, Farben): entfällt (keine Stadt)
- `gebaeude.hero` (Hauptgebäude: Ort, Radius, Dachhöhe, Brüstung, Dachbelag): entfällt (keine Stadt)
- `gebaeude.bevel` (Kantenfase der Bauteile): entfällt (keine Stadt)
- `gebaeude.lod` (Detailstufe je Zone, Proxies und Karten in der Ferne): entfällt (keine Stadt)
- `gebaeude.instancing` (Instanzierung und Variation (Höhe, Dach, Farbe, Drehung, Skalierung)): entfällt (keine Stadt)
- `gebaeude.strassendetails` (Bordsteine, Gullys, Markierungen, Pflaster): entfällt (keine Stadt)

## S4. Stadt-Materialien
- `material.putz` (Putz: Farben, Fenster an/aus, PBR-Variante): entfällt (keine Stadt)
- `material.daecher` (Dachfarben, Ziegelbreite und -höhe, Streuung, zylindrisch): entfällt (keine Stadt)
- `material.terrakotta` (Farben der Ziegel-Meshes): entfällt (keine Stadt)
- `material.fenster` (Fensterglas: lit, curtain, per_cell, Innenraum): entfällt (keine Stadt)
- `material.verwitterung` (je Material: dirt_h, dirt, ground_z, edge, edge_col, bevel_r, streak, var): entfällt (keine Stadt)
- `material.dachbelag` (Abnutzung des Dachbelags (r_edge)): entfällt (keine Stadt)
- `material.strasse` (Straßenboden (half_w), Pflaster, Feldweg, Sandstraße): entfällt (keine Stadt)
- `material.gras` (Gras zwischen den Häusern und am Hang: c1, c2, dry, scale): siehe gras.farben (Halme) und material.boden
- `material.metall` (Rostmetall für Tanks und Rohre: tint): entfällt (keine Stadt)
- `material.holz` (Balken, Dachdeck, Masten: dark, board, Achse): entfällt (keine Stadt)
- `material.lack` (lackierte Teile (Läden, Klimageräte, Horn): Farbe, Rauheit): entfällt (keine Stadt)
- `material.stoff` (Markisen, Stände, Wäsche: Farben, Rauheit): entfällt (keine Stadt)
- `material.mauer` (Mauer- und Torstein: c1–c3, bump, crack_w): entfällt (keine Stadt)
- `material.wahrzeichen_fels` (Fels des Wahrzeichens: c1–c3, moss, moss_amount, scale, bump, crack_w, strata_scale, lichen, cavity): entfällt (keine Stadt)
- `material.cc0` (CC0-PBR-Texturen: Dateien, Skalierung, Überblendung): entfällt (keine Stadt)
- `material.emission` (Laternen, Fenster, Schilder: Farbe, Stärke): Raumschiff-Fenster, siehe objekte.raumschiff
- `material.texeldichte` (Texeldichte): nicht erfasst
- `material.farbraum` (Farbtexturen sRGB, Datentexturen Non-Color (geprüft)): Tripo-Farbtexturen sRGB, Normal-Maps Non-Color; Umgebung prozedural
- `material.displacement` (Displacement und Bump nah an der Kamera): nicht erfasst
- `material.kachelvariation` (Mittel gegen sichtbare Kacheln (Noise-Mix, Objekt-Zufall, Attribute)): prozedurale Materialien mit Rauschen (keine Kacheltexturen)

## S5. Stadt-Licht
- `licht.himmel_typ` (Nishita, HDRI oder eigener Verlauf): eigener Shader (namek_sky)
- `licht.nishita` (aerosol, ozone, air, Sonnenscheibe): entfällt (eigener Himmel)
- `licht.hdri` (Datei, Drehung, Stärke): entfällt (kein HDRI)
- `licht.nacht_quellen` (Laternen, Fenster, Schilder, Fahrzeuge, Feuer: Anzahl, Stärke): entfällt (Tagszene)
- `licht.farbtemperaturen` (Kelvin je Quellenart (Natrium, warm, neutral, kühl)): {sonne1: 4300; sonne2: 5600; sonne3: 4700; randlicht: 7800}
- `licht.schatten` (Richtung, Länge, Härte (Prüfung gegen Tageszeit)): Hauptsonne 19° aus 225° (links hinten): lange Schatten, Gegenlicht durch Sonne 3
- `licht.mond_sterne` (Mond, Sterne): entfällt (Tagszene)
- `licht.rig` (Lichtrig als Collection (Name, Inhalt)): 3 Sonnen + Randlicht (Light Linking auf die Kämpfer) + Effektlichter

## S6. Atmosphäre
- `atmosphaere.volumen` (Streuvolumen: Größe, Mitte, Farbe): entfällt (NIDO_ATMO = 0; Explosion als eigenes Volumen (fx.explosion))
- `atmosphaere.bodennebel` (Höhen- und Bodennebel: Höhe, Dichte, Bereich): entfällt (keiner)
- `atmosphaere.lichtstrahlen` (volumetrische Lichtstrahlen): entfällt (keine)
- `atmosphaere.wetter` (Regen, Schnee, Staub, Asche: Anzahl, Tempo): entfällt (kein Wetter)
- `atmosphaere.tiefenschichten` (Vorder-, Mittel-, Hintergrund: Abstände): {vorn: Meer und Gras 2–30 m; mitte: Tafelberg und Kampf 30–200 m; hinten: Felsnadeln bis 1150 m, Inseln 1400–1700 m}

## S7. Leben und Bewegung
- `leben.passanten` (Passanten: Abstand entlang der Straße, Anteil je Seite, Abstand zur Straßenmitte, Drehungsstreuung, Aussparungen, Anzahl): entfällt (keine Stadt)
- `leben.passanten_vorlagen` (Vorlagen, Posen, Kleidungsfarben, Seed): entfällt (keine Stadt)
- `leben.staende` (Marktstände: Anzahl, Abstand, Seite, Maße, Dachstoff, Waren, Kisten): entfällt (keine Stadt)
- `leben.laternen` (Laternenleinen: y-Positionen, x_end, z_end, sag, n je Leine, Seed): entfällt (keine Stadt)
- `leben.waesche` (Wäscheleinen: y-Positionen, Höhe (aus Fassaden), sag, Abstand zum Baum): entfällt (keine Stadt)
- `leben.leitungen` (Strommasten: Abstand, Höhe, Querbalken, Drahtabstand, Durchhang): entfällt (keine Stadt)
- `leben.voegel` (je Schwarm: Anzahl, Startbox, Flugvektor, t0, t1, Seed): entfällt (keine)
- `leben.blaetter` (treibende Blätter je Feld: Box, Anzahl, Wind, Seed): entfällt (keine)
- `leben.pendeln` (Schwingen: Achse, Amplitude, Frequenz, Phase, Rauschen): Gras im Wind mit Böen (gras.bewegung)
- `leben.verkehr` (Fahrzeuge, Bahnen, Boote: Anzahl, Pfade, Tempo, Lichter): entfällt (keine Stadt)
- `leben.rauch` (Rauch und Dampf: Quellen, Dichte, Tempo): Explosions-Rauchsäule (fx.explosion)
- `leben.flaggen` (Flaggen, Planen, Wäsche im Wind (Stoff-Cache)): entfällt (keine)
- `leben.caches` (Simulations-Caches: Ordner, Frames): Rigid-Body-Bruchstücke als Keyframes gebacken
- `leben.asynchron` (Variation, damit nichts synchron läuft (Phase, Tempo)): Gras: Böen und Druckwellen (gras.druckwellen)

## S8. Vegetation in der Stadt
- `baum.hero` (Manöver-Baum: Ort, Höhe, Kronenradius, Cluster, Blätter je Cluster, Steinring): entfällt (keine Stadt)
- `baum.nadelbaum` (Nadelbäume: Höhe, Krone, Cluster, Form, Transluzenz): entfällt (keine Stadt)
- `baum.wald` (Waldbereiche: Anzahl, x/y-Bereich, Ausschlüsse, Skalierung): siehe baum.haine
- `baum.dorf` (Bäume im Ort: Versuche, Abstand zu Gebäuden, Hauptgebäude, Straße; Skalierung): entfällt (keine Stadt)
- `baum.buesche` (Büsche an der Zufahrt: Anzahl, Bereich, Skalierung): entfällt (keine Stadt)
- `baum.felsbewuchs` (Bewuchs auf Felsbändern: Normalen-Schwelle, Mindesthöhe, Anzahl, Skalierung, Aussparung): entfällt (keine Stadt)
- `baum.streuung` (Streuung per Geometry Nodes: Seed, Drehung, Größe, Punkte gesamt): Ajisa-Bäume in Hainen (baum.haine)

## S9. Wahrzeichen-Relief (Felswand, Monument)
- `wahrzeichen.wand` (Felswand: x-Bereich, y, z-Bereich, Auflösung, Seed, Panel): entfällt (kein Relief; Tafelberg-Wand siehe gelaende.steilwand)
- `wahrzeichen.risse` (Risse: Anzahl, Bereich, Tiefe, Breite): entfällt (kein Relief; Tafelberg-Wand siehe gelaende.steilwand)
- `wahrzeichen.gesichter` (je Gesicht x, Höhenversatz, Haar; Skalierung, Grundhöhe, Kopfvorlage, Voxelgröße): entfällt (kein Relief; Tafelberg-Wand siehe gelaende.steilwand)
- `wahrzeichen.kinnlinie` (Kinnlinie, Panel-Aussparung): entfällt (kein Relief; Tafelberg-Wand siehe gelaende.steilwand)
- `wahrzeichen.details` (Treppen, Geländer, Hütten, Kuppeln an der Wand): entfällt (kein Relief; Tafelberg-Wand siehe gelaende.steilwand)

## S10. Kamera wie im Film
- `kamera.fov` (Sichtfeld): {vertikal_grad: 65,47; brennweite_mm: 28; sensor: 36 mm (Höhe)}
- `kamera.blende` (Blende, Fokusdistanz, Schärfentiefe an/aus): {dof: nein; hinweis: Schärfentiefe aus (großer Schärfebereich wie FPV)}
- `kamera.clipping` (Clip Start, Clip End): (0,05; 30000)
- `kamera.overscan` (Overscan für Stabilisierung und Verzerrung im Comp): entfällt (kein Overscan)
- `kamera.manoever` (je Manöver: Zeit, festes Objekt, Art (Tor, Kabel, Baum, Steigflug, Brüstung)): 1,9 · Felsnadel 1 (y 52) · links vorbei · 2,7 · Felsnadel 2 (y 72) · rechts vorbei · 5,2 · Tafelberg-Wand · Steigflug dicht an der Wand · 7,8 · Kante · drüber: Reveal · 9,55 · Dragon Balls · links vorbei · 10 · Kampf · Abbremsen auf die Kämpfer · 16 · Explosion · zurückziehen und steigen
- `kamera.mindestabstand` (kleinster Abstand zur Geometrie (Messwert)): {flug_meer: {m: 2,27; t: 1,875; objekt: Ocean}; steilwand: {m: 3,87; t: 7; objekt: MesaWall}; plateau: {m: 3,08; t: 9,042; objekt: SandPatches}; kampf: {m: 3,5; t: 10,125; objekt: Mesa}; hinweis: Szene des Finals vermessen, ohne Figuren, Effekte und Grashalme; Meer im Wellenstand von Frame 1}
- `kamera.hook` (Hook-Bild am Anfang: Inhalt, Dauer): 0–2.7 s: Hook tief über dem Meer, Slalom an zwei Felsnadeln
- `kamera.schlussbild` (Schlussbild: Motiv, Blickwinkel): 16.0–20.0 s: Nachglühen, Rauchsäule, Kamera zieht zurück und steigt
- `kamera.loop` (Loop oder fester Anfang und Ende): kein Loop: harter Anfang und ruhiges Schlussbild

## S11. Render-Setup in Kinoqualität
- `render.engine` (Engine und Device): Cycles, CPU (Cloud-Container mit 4 Kernen, keine GPU)
- `render.min_samples` (Mindest-Samples beim adaptiven Sampling): 0 (automatisch)
- `render.kaustiken` (Kaustiken reflektiv / refraktiv): {reflektiv: nein; refraktiv: nein}
- `render.vector_pass` (Vector-Pass (nur ohne Cycles-Motion-Blur)): aus (Motion Blur im Render)
- `render.cryptomatte` (Object, Material, Asset; Tiefe): aus (Object, Material, Asset nicht aktiv)
- `render.mist` (Mist Start, Tiefe, Abfall): {start: 0; tiefe: 6000; abfall: LINEAR}
- `render.farbmanagement` (View Transform, Look, Exposure, Gamma, Display): {view_transform: AgX; look: None; exposure: 0; gamma: 1; display: sRGB; hinweis: der Look kommt in der Post (post.look)}
- `render.ausgabe` (Format, Bit-Tiefe, Codec, Bildsequenz, Pfadschema): {format: OpenEXR je Pass (Image, Mist, Env), 16 Bit Half, Codec DWAA; sequenz: f_####; ort: Scratchpad (flüchtig), danach PNG + MP4}
- `render.fortsetzbar` (Overwrite aus, Placeholders an, vorhandene Frames überspringen): ja: seq_final.py überspringt fertige Frames (fpv.frame_done); Overwrite/Placeholders nicht genutzt
- `render.vram` (VRAM- und RAM-Spitze bei schweren Frames (Messwert)): entfällt (keine GPU; RAM siehe ressourcen.ram)
- `render.testframes` (Testframes in Endqualität: Frame, Sekunden): nicht erfasst
- `render.hochrechnung` (s pro Frame × Frames = Gesamtzeit, gegen Budget): vorab 15–20 h geschätzt (zu Beginn ~150 s/Frame → 18 h), tatsächlich 18,1 h
- `render.preflight` (Ergebnis des Preflight-Checks (Fehler, Warnungen)): entfällt (preflight_check.py gab es zum Zeitpunkt des Finals noch nicht)

## S12. Compositing (Kinolook)
- `comp.linsenverzerrung` (Linsenverzerrung (Distortion)): keine Verzerrung, nur laterale Dispersion (post.optik)
- `comp.halation` (Halation an hellen Kanten): entfällt (nicht verwendet)
- `comp.grading` (Schwarzwert, Highlights, Lift/Gamma/Gain, Farbtrennung Schatten/Lichter): AgX Medium High Contrast, Auto-Belichtung, Pulse und Flare (post.*)
- `comp.cryptomatte_korrekturen` (selektive Korrekturen (Fenster, Himmel, Wasser)): entfällt (keine Cryptomatte)
- `comp.shake` (Kamerashake im Comp): nicht im Comp; Kamerastöße im 3D (kamera.stoesse)
- `comp.letterbox` (Letterbox / Scope): entfällt (Vollbild 9:16)
- `comp.ki_variante` (saubere Fassung ohne Korn, Glare, Verzerrung): entfällt (keine KI-Weiterverarbeitung)

## S13. Video-zu-Video-KI
- `ki.tool` (Plattform und Modell): entfällt (keine KI-Generierung; Higgsfield/Kling/Seedance nicht genutzt)
- `ki.einstellungen` (Einstellungen der Generierung): entfällt (keine KI-Generierung)
- `ki.kosten` (vorab genannte Kosten): entfällt (keine KI-Generierung)
- `ki.freigabe` (Freigabe: Datum, Zitat): entfällt (keine KI-Generierung)
- `ki.eingabe` (Eingabevideo (saubere Fassung)): entfällt (keine KI-Generierung)

## S14. QC, Lieferung, Gates
- `bewertung.gates` (je Gate (A, A2, B, C, D, E): Datum, freigegeben ja/nein, Zitat): {gate: A; datum: 2026-09-28; frei: –; zitat: Cinematic-Briefing; Zeitleiste vorgeschlagen, Freigabe nicht festgehalten} · {gate: D; datum: 2026-09-30 21:49; frei: nein; zitat: „der kampf ist jetzt brutal. wir müssen aber die bewegungen und das aussehen von den 2 verbessen.“} · {gate: D; datum: 2026-09-30 22:24; frei: nein; zitat: „… beachte aber das bei dragon ball diese immer sehr shcnell sind.“} · {gate: D; datum: 2026-10-01 12:54; frei: ja; zitat: „die vorschau passt“} · {gate: E; datum: 2026-10-01 15:05; frei: ja; zitat: „starte das finale …“ / 15:11 „nein, nacheinander passt.“}
- `qc.durchsicht` (komplett in Originalgeschwindigkeit gesehen): Stichprobe aus 10 Zeitpunkten (Flug, Plateau, Kampf, Strahlenduell, Explosion), nicht das ganze Video
- `qc.frames` (fehlende oder schwarze Frames, Fireflies): 480/480 Frames vorhanden
- `qc.flackern` (Flackern in Schatten, Lichtern, Wasser, Fenstern): nicht erfasst
- `qc.textur` (Kachelmuster, Texturstreckung, Z-Fighting): nicht erfasst
- `qc.banding` (Banding in Himmel und Nebel): nicht erfasst
- `qc.zweitdisplay` (auf Handy oder zweitem Display geprüft): nicht erfasst
- `qc.anfang_ende` (Anfang und Ende sauber (Loop, Fade)): kein Loop; Anfang in voller Fahrt, ruhiges Schlussbild
- `lieferung.master` (Master (EXR-Sequenz, ProRes oder MP4): Pfad, Größe): {pfad: drohnen_3d_welten/videos/03_DragonBall_Namek.mp4; mb: 44,9}
- `lieferung.varianten` (weitere Formate (9:16, 1:1, Webversion)): Webversion 10 Mbit/s (< 30 MB) für den Chat
- `lieferung.dateiname` (Dateiname mit Version und Datum): <NN>_<Welt>_<Ort>.mp4, ohne Version und Datum
- `lieferung.archiv` (archivierte Teile (Projekt, Texturen, Caches, Skripte), Ort): Code im Repo (Commit), Werte im Werte-Archiv; .blend, EXR und PNG nur im Scratchpad (nicht dauerhaft archiviert)
- `lieferung.obsidian` (ZIP-Paket für den Vault): ZIP-Paket „Drohnen Videos/3D Welten Blender/<NN Name>/{Test,Final,Werte}“
- `lieferung.wiederverwendbar` (Assets, Materialien, Node-Gruppen, Lichtrig, Kameraskript für die nächste Stadt): choreo.py (dbz_style, Hit-Stops) · dbz_fx.py (Ki-Spuren, Zanzoken) · vfx.py (Strahlen, Explosion als Volumen, Aura) · tripo_chars.py (Tripo-Figuren riggen) · Skill anime-fpv-kampfvideo
- `lieferung.notiz` (was beim nächsten Mal früher entschieden werden muss): Dragon-Ball-Tempo von Anfang an planen; Tripo-Figuren früh einbauen; Sichtkorridor zu den Kämpfern freihalten

## S15. Kampf in der Stadt
- `kampf.ort` (Kampfort (z. B. Dach): Mitte, Höhe, lokales Koordinatensystem): {ort: in der Luft über dem Plateau und dem Nordrand; zusammenprall: (1; 273; 45,5)}
- `fx.impact_klein` (kleines Impact-Paket: Farbe, Licht, Ring, Funken, Seed): siehe fx.treffer
- `fx.bruch` (Bruchstücke an Gebäudeteilen: Objekt, Zeit, Bruchhöhe, Stücke): siehe fx.einschlag (16 Bruchstücke) und fx.strahlen (Brocken)
- `fx.techniken` (Signatur-Techniken (z. B. Rasengan, Chidori): Farbe, Radius, Licht, Zeiten): Kamehameha und Todesstrahl, siehe fx.strahlen und fx.duell

## 7. Figuren
- `figur.<name>.modell` (Datei, Quelle, Reparaturen): siehe figur.goku.modell / figur.freezer.modell
- `figur.<name>.hoehe` (Zielgröße): siehe figur.goku.hoehe / figur.freezer.hoehe
- `figur.<name>.gelenke` (Quelle der Gelenke (Mixamo / vermessen)): siehe figur.goku.gelenke / figur.freezer.gelenke
- `figur.<name>.schwanz` (Mittellinie, Gliederzahl): siehe figur.goku.schwanz / figur.freezer.schwanz
- `figur.<name>.material` (lit, mid, shade, bands, tint_lit, rim, rim_w, sat, value, diffuse_mix, hair_glow): siehe figur.goku.material / figur.freezer.material
- `figur.<name>.kontur` (Konturstärke relativ zur Höhe): siehe figur.goku.kontur / figur.freezer.kontur
- `figur.<name>.gewichte` (smooth, power, Saat-Bereiche): siehe figur.goku.gewichte / figur.freezer.gewichte

## 15. Schiff und Crew (Videos mit Fahrzeug und Figurengruppe statt Kampf)
- `schiff.modell` (Schiffsmodell, Bauweise, Quelle (Model Sheets)): entfällt (kein Schiff, keine Crew)
- `schiff.kurs` (Kurs, Fahrt, Startpunkt): entfällt (kein Schiff, keine Crew)
- `schiff.tempo` (Kurs, Fahrt, Startpunkt): entfällt (kein Schiff, keine Crew)
- `schiff.start` (Kurs, Fahrt, Startpunkt): entfällt (kein Schiff, keine Crew)
- `schiff.bewegung` (Stampfen/Rollen): entfällt (kein Schiff, keine Crew)
- `schiff.flaggen` (Stoff-Simulation (Cache)): entfällt (kein Schiff, keine Crew)
- `schiff.gischt` (Bug-Gischt: Raten, Aufwärtstempo, Tropfenradien): entfällt (kein Schiff, keine Crew)
- `schiff.kielwasser` (Bugwelle, Heckwelle, Rumpfschaum): entfällt (kein Schiff, keine Crew)
- `kamera.speed_keys` (Tempo über die Zeit (statt Wegpunkt-Zeiten)): entfällt (Tempo aus Wegpunkt-Zeiten (kamera.wegpunkte))
- `kamera.route_welt` (Bahn weltfest / im Schiffssystem): siehe kamera.wegpunkte
- `kamera.route_schiff` (Bahn weltfest / im Schiffssystem): entfällt (kein Schiff, keine Crew)
- `kamera.uebergang` (Überblendung weltfest → schiffsfest): entfällt (kein Schiff, keine Crew)
- `kamera.impact` (Kamerastoß beim Überfliegen der Reling): entfällt (Kamerastöße über kamera.stoesse)
- `licht.randlicht_fade` (Randlicht aus, bevor die Kamera zurückblickt): entfällt (Randlicht durchgehend)
- `crew.figuren` (je Figur: Modell, Höhe, Rig-Quelle, Look): entfällt (kein Schiff, keine Crew)
- `crew.aufstellung` (je Figur: Ort (Schiffssystem), Yaw, Boden/Sitz): entfällt (kein Schiff, keine Crew)
- `crew.schauspiel` (je Figur: Posenfolge (Zeit, Pose, Dauer, Ausholen, Überschwingen), Atmen, Schwanken, Blick): entfällt (kein Schiff, keine Crew)
- `crew.posen_map` (figurenspezifische Posen-Ersetzungen): entfällt (kein Schiff, keine Crew)
- `crew.drehung` (Körperdrehung über die Zeit (z. B. zur Kamera)): entfällt (kein Schiff, keine Crew)
- `crew.requisiten` (Requisiten an Gelenken): entfällt (kein Schiff, keine Crew)
- `crew.texturen` (Texturgröße der Figuren): entfällt (kein Schiff, keine Crew)
