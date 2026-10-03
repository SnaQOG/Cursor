# Naruto – Konoha (Naruto gegen Sasuke auf dem Dach der Residenz) – alle Parameter (Stand Final, 2026-09-30)

Exakte Werte des freigegebenen Finals `02_Naruto_Konoha.mp4`. Quelle ist der Code-Stand 761e7ad (Szene gebaut 2026-09-29 06:18, Branch claude/eloquent-archimedes-og1asl), die Final-Szene und das Render-Log. Kampfplan, Kamera und Sound-Marker stehen Wert für Wert in `kampfplan_und_kamera.json`.

Gliederung und IDs wie im Parameter-Katalog (Skill blender-stadt-vfx, Teil A und B). „entfällt“ heißt, dass es den Parameter in diesem Video nicht gibt; „nicht erfasst“, dass er nicht gemessen oder festgehalten wurde.

Teil A: Stadt und Kino (S1–S15), Teil B: gemeinsame Parameter (1–15).

## S1. Briefing und Planung
- `briefing.stadt` (Stadt, Thema, Epoche, Kultur): Konohagakure (Naruto Shippuden) mit Hokage-Felsen; Kampf Naruto gegen Sasuke
- `briefing.stil` (fotoreal, stilisiert oder Anime): halbrealistisch: physikalische Umgebung, Figuren mit Stoff-/Hautmaterialien (kein Toon)
- `briefing.tageszeit` (Tageszeit (genau benannt)): später Nachmittag: tiefe warme Sonne 18° aus Westsüdwest (250°)
- `briefing.wetter` (Wetter und Jahreszeit): klar, Wolkenbedeckung 40 %
- `briefing.stimmung` (Stimmung in einem Satz): warmer Flug durch ein lebendiges Ninja-Dorf, dann ein kurzer, harter Dachkampf mit Rasengan gegen Chidori
- `briefing.kameratyp` (FPV, Flyover oder Static): FPV, eine durchgehende Aufnahme ohne Schnitt
- `briefing.verwendung` (privat, Video-zu-Video-KI oder Veröffentlichung): nicht ausdrücklich festgelegt (NIDO-Drohnenvideo)
- `briefing.engine` (Engine und Device): Cycles, CPU
- `briefing.budget` (Renderbudget: max. Minuten pro Frame, Gesamtstunden): vorab geschätzt 15–17 h für das Final (720p, 32 Samples)
- `briefing.hardware` (VRAM, RAM, CPU-Kerne, freier Speicher für die Sequenz): {cpu_kerne: 4; gpu: keine; ram_gb: 16; umgebung: Cloud-Container, wird bei Leerlauf neu gestartet}
- `briefing.ausgabe` (Ausgabeformat und Zielordner): {frames: EXR je Pass (Scratchpad); video: H.264 MP4 1080 × 1920 → drohnen_3d_welten/videos/}
- `briefing.referenzen` (Referenzen für Architektur, Material, Licht, Atmosphäre, Kamera): Dominiks Anime-/Spiel-Standbilder von Konoha (Pastellfassaden, bunte Dächer, Wassertanks, ockerfarbener Sandstein mit Laufspuren, Treppen, Kuppelbauten), Model Sheets Naruto/Sasuke (2006)
- `briefing.palette` (Farbpalette und Kontrastidee (aus Referenzen gemessen)): {daecher: (0,42; 0,08; 0,04) · (0,55; 0,2; 0,06) · (0,1; 0,28; 0,13) · (0,07; 0,19; 0,46) · (0,25; 0,11; 0,36) · (0,05; 0,28; 0,29); residenz: (0,58; 0,12; 0,045); fels: (0,6; 0,42; 0,22); sonne_k: 4300; rand_k: 7800}
- `planung.einheiten` (Einheitensystem, Maßstab, Skalierung angewendet): {system: METRIC; skalierung: 1; bu: 1 BU = 1 m}
- `planung.detailbudget` (Grenzen nah / mittel / fern und Detailstufe je Zone): keine LOD-Stufen: Straße voll detailliert (Fassaden, Fenster, Stände, Passanten), übriges Dorf als Rasterbebauung, Wald 12 444 GN-Instanzen
- `planung.collections` (Collection-Struktur und Namensschema): AuxScene · KTrees · KTrees_0 · KTrees_1 · KTrees_2 · KTrees_3 · KTrees_4 · KTrees_5 · KWin0 · KWin1 · KWin2 · KWin3 · KWindows · RigidBodyWorld · RigidBodyWorld.001 · RimLightReceivers · Scene · Tank0 · Tank1 · Tank2 · Tanks · Vil0 · Vil1 · Vil2 · Vil3 · Vil4 · Vil5 · Villagers
- `planung.versionierung` (Dateiname, Version, Backup-Ort): Szene pv/n52/scene.blend → final/naruto_final.blend (Scratchpad); Code-Commit 761e7ad

## S2. Stadtlayout und Blockout
- `stadt.koordinaten` (Ursprung, Hauptflugrichtung, Achsen): Ursprung im Tor (y 0), +Y nach Norden zur Residenz und zum Felsen, Z oben
- `stadt.flaeche` (Stadtfläche: Größe, Mitte): {groesse: (420; 420); mitte: (0; 190)}
- `stadt.mauer` (Stadtmauer: Mitte, Radius, Segmentwinkel, Höhe, Dicke, Abdeckung, Toröffnung): {mitte: (0; 190); radius: 190; segmentwinkel: 3,2; hoehe: 12,5; dicke: 3,4; abdeckung: (4,4; 0,8); toroeffnung_grad: 6; offen_zum_felsen_ab_y: 340; bevel: 0,1}
- `stadt.tor` (Tor: Ort, halbe Öffnung, Höhe, Oberkante des Querbalkens): {y: 0; halbe_oeffnung: 10,5; hoehe: 19; pfeiler: 3,2; querbalken_unten_oberkante: 17,2; tueren: あ / ん (A-un)}
- `stadt.hauptstrasse` (Hauptstraße: Halbbreite, Länge, Mitte, Belag): {halbbreite: 12; laenge: 178; mitte_y: 88; belag: street_ground_material}
- `stadt.zufahrt` (Zufahrtsweg: Breite, Länge, Belag): {breite: 12; laenge: 300; mitte_y: -148; belag: Feldweg}
- `stadt.platz` (Platz: Mitte, Größe, Belag): {mitte: (0; 226); groesse: (140; 100); belag: Pflaster (Cobble)}
- `stadt.gelaende` (Höhenprofil: Hügelring (Abstand, Breite, Höhe, fbm), Tal der Zufahrt, Plateau, Gipfel): Hügelring ab r = 198 m über 70 m: (6 + 34·(0,5 + fbm(x/220, y/220, 5, seed 3)))·smoothstep; Tal der Zufahrt bei y < 30; Plateau und Bergspitze siehe gelaende.*
- `stadt.gelaende_mesh` (Geländenetz: Größe, Auflösung, Ursprung): {groesse: (2400; 2400); punkte: (360; 360); ursprung: (0; 400)}
- `stadt.wasser` (Fluss, Hafen, Kanäle, Uferlinien): entfällt (kein Wasser)
- `stadt.bruecken_treppen` (Brücken, Treppen, Terrassen): Zickzack-Treppen am Felsen (6 Läufe, x −238 bis −186, je 25 m Steigung), keine Brücke
- `stadt.landmarken` (Wahrzeichen: Name, Ort, Größe, Rolle im Flug (Hook, Manöver, Reveal, Schlussbild)): {name: Tor; ort: (0; 0); rolle: Hook und Durchflug 1,8 s} · {name: alter Baum; ort: (4; 118); rolle: Manöver links vorbei 7,1 s} · {name: Hokage-Residenz; ort: (-20; 232); radius: 21; rolle: Steigflug und Kampf} · {name: Hokage-Felsen; y: 360; rolle: Teaser im Hook, Schlussbild} · {name: Stufentürme; orte: (-44; 150) · (40; 128); rolle: Straßenrand}
- `stadt.sichtachsen` (Sichtachsen und Kompositionsmomente entlang der Route): 0 · durchs Tor auf Residenz und Gesichter · 3,2 · unter Laternenkabel 1 · 4 · unter der Wäscheleine · 7,1 · links am alten Baum · 9,55 · Steigflug über den Platz · 10,85 · über die Brüstung: Kampf im Gang · 14,45 · beide Kämpfer (6,4 m Abstand) ganz im Bild · 17,1 · tief auf dem Dach: Naruto + Gesichter
- `stadt.route` (grobe Flugroute auf dem Blockout): (0; -60; 4,4) · (0; -30; 4,8) · (0; 0; 5,5) · (0; 35; 7,6) · (-2; 75; 10) · (-7; 112; 11) · (-3; 145; 14) · (-4; 168; 23) · (-10; 190; 33,5) · (-16; 211; 39,3) · (-19; 232; 39,4) · (-19; 251; 39,6) · (-17; 270; 49) · (-14; 290; 62) · (-11; 308; 76) · (-9; 322; 86)
- `stadt.seed` (Zufalls-Seed des Aufbaus): 12

## S3. Gebäude und Aufbau
- `gebaeude.anzahl` (Gebäude gesamt (Messwert)): 229
- `gebaeude.strassenzeile` (Häuser an der Hauptstraße: Breite, Tiefe, Stockwerke, Lücke, Rücksprung, y-Bereich): {breite_entlang: (10; 14); tiefe: (11; 15); stockwerke: (2; 4); luecke: (1; 3,5); abstand_zur_strasse: 1,5; y: (22; 172)}
- `gebaeude.stile` (Bauformen und ihre Gewichte (flach, Tonnendach, rund, Walmdach)): {strasse: {flat: 0,4; barrel: 0,4; round: 0,2}; raster: {flat: 0,33; barrel: 0,33; hip: 0,17; round: 0,17}; stufenturm_im_raster: 0,06}
- `gebaeude.raster` (übrige Stadt: Rasterweite, Versatz, Aussparungen, Ausfallquote, Drehung, Größen, Stockwerke): {weite: 17; versatz: 2,5; x: (-176; 176); y: (10; 310); ausfall: 0,12; drehung_grad: 6; groesse: (8; 13); stockwerke: (1; 4); aussparungen: Straße ±29 m bis y 176, Tor, Platz/Residenz (y 176–272, |x| < 70), außerhalb Mauer −12 m, vor dem Felsen}
- `gebaeude.platzrand` (Randbebauung des Platzes: je Gebäude x, y, Breite, Tiefe, Stockwerke, Stil): 52 · 205 · 22 · 16 · 5 · barrel · 50 · 246 · 18 · 18 · 4 · round · -62 · 198 · 20 · 14 · 4 · hip · -64 · 258 · 16 · 16 · 5 · flat · 30 · 272 · 14 · 12 · 3 · flat
- `gebaeude.stockwerkhoehe` (Stockwerkhöhe, Sockel, Brüstung): {stockwerk: 3,2; streuung_hoehe: (-0,3; 0,6); bruestung: 0,7}
- `gebaeude.tuerme` (Stufentürme: Anteil, Radius, Stufen, Dachfarbe, feste Türme an der Straße): {anteil_raster: 0,06; stufen: (2; 3); erdgeschoss: 6,4; traufkranz: 1,6; verjuengung: 0,8; feste: (-44; 150; 5,5; 3) · (40; 128; 5; 2)}
- `gebaeude.daecher` (Dachformen: Stich des Tonnendachs, Stich und Überstand des Walmdachs, Ziegelmaße): {tonnendach_stich: 0,28·min(B, T); walmdach_stich: 0,35·min(B, T); walmdach_ueberstand: 0,8; ziegel_mesh: (0,3; 0,42)}
- `gebaeude.dachaufbauten` (Wassertanks, Klimageräte, Rohre, Antennen, Schornsteine: Anteil, Größe): {wassertanks: 80 % der Flachdächer, 1 (60 %) oder 2, Skalierung 0,85–1,2; dachaufbau: 30 %: 0,35·B × 0,35·T × 2,4 m; rohre: 1–2 je Gebäude, Ø 0,14–0,26 m; klimageraete: 0–2 (0,9 × 0,55 × 0,65 m)}
- `gebaeude.fassade` (Fenster je Stockwerk, Rahmen, Bänke, Läden, Rohre, Kabel, Markisen): {fensterachsen: max(1, int((Breite − 1,2)/2,6)); fenster_ausfall: 0,12; fenster_z: 0,9 + 3,2·Stockwerk; holzbalken: 0,22; markise: 55 %, 0,8·Breite × 1,8 m in 2,9 m Höhe; balkone: 60 % der ≥ 2-stöckigen, je Stockwerk 65 %; ladenfront: Erdgeschoss zur Straße ohne Fenster}
- `gebaeude.banner` (Banner: Anteil, Zeichen, Größe, Höhe, Farben): {anteil: 0,55; zeichen: 火 · 木 · 忍 · 茶 · 楽 · 薬 · 酒; groesse: (1,4; 3,2); hoehe_mitte: 5,4; farbe: {zeichen: (240; 236; 226); grund: (150; 24; 18)}}
- `gebaeude.schilder` (Ladenschilder: Zeichen, Vorder- und Hintergrundfarbe): {zeichen: 一 · 忍 · 団 · 花 · 書; anteil: 35 % bei Höhe > 6 m; groesse: 0,25 × 1,6 × 2,4–4,0 m; farben: {zeichen: (236; 232; 220); grund: (25; 40; 70) · (90; 30; 18)}}
- `gebaeude.tueren` (Türen: Motiv, Farben): {motiv: あ / ん; farben: {zeichen: (170; 25; 18); grund: (236; 228; 210)}}
- `gebaeude.hero` (Hauptgebäude: Ort, Radius, Dachhöhe, Brüstung, Dachbelag): {name: Hokage-Residenz; ort: (-20; 232); radius: 21; kragen_z: 15; dach_z: 31; dachbelag_z: 31,65; bruestung: 1,1; mittelspitze: 2,6; fensterreihen: (44; 44; 40; 40); emblem: 火 im Ring}
- `gebaeude.bevel` (Kantenfase der Bauteile): {standard: 0,05; mauer: 0,1; tor: 0,08; verwitterung_bevel_r: 0,07}
- `gebaeude.lod` (Detailstufe je Zone, Proxies und Karten in der Ferne): keine LOD-Stufen
- `gebaeude.instancing` (Instanzierung und Variation (Höhe, Dach, Farbe, Drehung, Skalierung)): Wassertanks (3 Varianten) und Fenster (4 Varianten) als Collection-Instanzen, Bäume per Geometry Nodes, Passanten aus 6 Vorlagen
- `gebaeude.strassendetails` (Bordsteine, Gullys, Markierungen, Pflaster): Straßenboden-Shader (Wege, Risse, Pfützen, Platten, dunkle Ränder), keine Bordsteine

## S4. Stadt-Materialien
- `material.putz` (Putz: Farben, Fenster an/aus, PBR-Variante): {plaster: {edge: 0,35; streak: 0,25}; parapet: {edge: 0,3; streak: 0,3; dirt: 0}; strasse_pbr: {edge: 0,4; streak: 0,25; dirt: 0,5; cc0: plaster, scale 0.35}; dachflaeche: (0,22; 0,2; 0,18)}
- `material.daecher` (Dachfarben, Ziegelbreite und -höhe, Streuung, zylindrisch): {farben: (0,42; 0,08; 0,04) · (0,55; 0,2; 0,06) · (0,1; 0,28; 0,13) · (0,07; 0,19; 0,46) · (0,25; 0,11; 0,36) · (0,05; 0,28; 0,29); tile_w: 0,32; tile_h: 0,26; var: 0,18; tuerme: zylindrisch}
- `material.terrakotta` (Farben der Ziegel-Meshes): {rot: (0,4; 0,07; 0,035); orange: (0,5; 0,17; 0,05); blau: (0,08; 0,16; 0,34); gruen: (0,1; 0,24; 0,12); residenz_kragen: (0,62; 0,3; 0,06)}
- `material.fenster` (Fensterglas: lit, curtain, per_cell, Innenraum): {lit: 0,22; curtain: 0,3; per_cell: 0,9}
- `material.verwitterung` (je Material: dirt_h, dirt, ground_z, edge, edge_col, bevel_r, streak, var): {plaster: {edge: 0,35; streak: 0,25}; parapet: {edge: 0,3; streak: 0,3; dirt: 0}; plaster_street: {edge: 0,4; streak: 0,25; dirt: 0,5}; red: {dirt_h: 2,5; dirt: 0,35; streak: 0,35}; red_dark: {dirt: 0; streak: 0,3}; gate: {dirt: 0,4; edge: 0,3; streak: 0,3}; stone: {dirt: 0,3; var: 0,03}; horn: {dirt: 0; streak: 0,35; var: 0}; shutter: {dirt: 0; edge: 0,4}; standard: {dirt_h: 2,2; dirt: 0,45; edge_col: (0,78; 0,74; 0,66); bevel_r: 0,07}}
- `material.dachbelag` (Abnutzung des Dachbelags (r_edge)): {r_edge: 19,3; holz: dark 0,9, Achse X, board 0,2}
- `material.strasse` (Straßenboden (half_w), Pflaster, Feldweg, Sandstraße): {strassenboden_half_w: 12; platz: Pflaster (CC0 paving); zufahrt: Feldweg}
- `material.gras` (Gras zwischen den Häusern und am Hang: c1, c2, dry, scale): {c1: (0,035; 0,06; 0,015); c2: (0,1; 0,12; 0,04); dry: (0,22; 0,19; 0,1); scale: 2}
- `material.metall` (Rostmetall für Tanks und Rohre: tint): {tanks: (0,75; 0,73; 0,68); rohre: (0,6; 0,6; 0,6); cc0: metal, scale 0.5; eisen: {farbe: (0,06; 0,055; 0,05); metal: 0,7}}
- `material.holz` (Balken, Dachdeck, Masten: dark, board, Achse): {balken: {dark: 0,75; board: 5}; dachdeck: {dark: 0,9; board: 0,2; achse: X}; masten: (0,16; 0,11; 0,07); tor: (0,34; 0,1; 0,05)}
- `material.lack` (lackierte Teile (Läden, Klimageräte, Horn): Farbe, Rauheit): {horn: (0,8; 0,78; 0,72); laeden: (0,1; 0,22; 0,15); klima: (0,62; 0,62; 0,6)}
- `material.stoff` (Markisen, Stände, Wäsche: Farben, Rauheit): {markisen: (0,45; 0,06; 0,04) · (0,08; 0,16; 0,4) · (0,7; 0,25; 0,04) · (0,1; 0,26; 0,14); staende: (0,4; 0,05; 0,03) · (0,62; 0,58; 0,5) · (0,06; 0,12; 0,26)}
- `material.mauer` (Mauer- und Torstein: c1–c3, bump, crack_w): {c1: (0,16; 0,15; 0,13); c2: (0,32; 0,3; 0,26); c3: (0,26; 0,24; 0,2); bump: 0,5; crack_w: 0,4; abdeckung: (0,12; 0,1; 0,09)}
- `material.wahrzeichen_fels` (Fels des Wahrzeichens: c1–c3, moss, moss_amount, scale, bump, crack_w, strata_scale, lichen, cavity): {wand: {c1: (0,16; 0,1; 0,05); c2: (0,6; 0,42; 0,22); c3: (0,44; 0,28; 0,13); moss: (0,05; 0,1; 0,02); moss_amount: 0,3; scale: 2; bump: 1; crack_w: 0,3; strata_scale: 1,2; lichen: 0,12}; gesichter: {c1: (0,22; 0,15; 0,08); c2: (0,66; 0,48; 0,27); c3: (0,5; 0,35; 0,18); scale: 1,5; bump: 0,5; crack_w: 0,12; lichen: 0,1; moss: (0,06; 0,1; 0,03); moss_amount: 0,15; cavity: 0,75}}
- `material.cc0` (CC0-PBR-Texturen: Dateien, Skalierung, Überblendung): {texturen: plaster · metal · paving · brick · concrete; projektion: Box, blend 0.25; scale: {putz: 0,35; metall: 0,5}}
- `material.emission` (Laternen, Fenster, Schilder: Farbe, Stärke): {laternen: {farbe: (1; 0,3; 0,1); staerke: 0,12}; fenster: Innenraum-Shader lit 0,22}
- `material.texeldichte` (Texeldichte): nicht erfasst
- `material.farbraum` (Farbtexturen sRGB, Datentexturen Non-Color (geprüft)): CC0-Farbtexturen sRGB, Rauheit/Normal Non-Color
- `material.displacement` (Displacement und Bump nah an der Kamera): kein Displacement, nur Bump/Normal
- `material.kachelvariation` (Mittel gegen sichtbare Kacheln (Noise-Mix, Objekt-Zufall, Attribute)): Box-Projektion mit Überblendung, Noise-Mix, Farbton je Objekt (weather var)

## S5. Stadt-Licht
- `licht.himmel_typ` (Nishita, HDRI oder eigener Verlauf): Nishita (Multiple Scattering)
- `licht.nishita` (aerosol, ozone, air, Sonnenscheibe): {aerosol: 0,9; ozone: 1,6; air: 1; hoehe_m: 50; sun_size: 0,0095; sonnenscheibe: nein}
- `licht.hdri` (Datei, Drehung, Stärke): entfällt (kein HDRI)
- `licht.nacht_quellen` (Laternen, Fenster, Schilder, Fahrzeuge, Feuer: Anzahl, Stärke): entfällt (Tagszene)
- `licht.farbtemperaturen` (Kelvin je Quellenart (Natrium, warm, neutral, kühl)): {sonne: 4300; randlicht: 7800}
- `licht.schatten` (Richtung, Länge, Härte (Prüfung gegen Tageszeit)): lange Schatten quer über die Straße, Streiflicht auf den Gesichtern
- `licht.mond_sterne` (Mond, Sterne): entfällt (Tagszene)
- `licht.rig` (Lichtrig als Collection (Name, Inhalt)): Sonne + Randlicht (Light Linking auf die Kämpfer) + Effektlichter

## S6. Atmosphäre
- `atmosphaere.volumen` (Streuvolumen: Größe, Mitte, Farbe): entfällt (NIDO_ATMO = 0, Dunst in der Post)
- `atmosphaere.bodennebel` (Höhen- und Bodennebel: Höhe, Dichte, Bereich): entfällt (keiner)
- `atmosphaere.lichtstrahlen` (volumetrische Lichtstrahlen): entfällt (keine)
- `atmosphaere.wetter` (Regen, Schnee, Staub, Asche: Anzahl, Tempo): treibende Blätter (leben.blaetter), kein Niederschlag
- `atmosphaere.tiefenschichten` (Vorder-, Mittel-, Hintergrund: Abstände): {vorn: Straße und Fassaden 3–30 m; mitte: Platz und Residenz ~230 m; hinten: Hokage-Felsen 360 m, Wald und Berg bis 950 m}

## S7. Leben und Bewegung
- `leben.passanten` (Passanten: Abstand entlang der Straße, Anteil je Seite, Abstand zur Straßenmitte, Drehungsstreuung, Aussparungen, Anzahl): {abstand_entlang: (3,5; 6); anteil_je_seite: 0,55; abstand_mitte: (5; 7,6) · (11; 12,2); anteil_nah: 0,6; drehung: ±90° zur Straße ± 45°; aussparung: um den alten Baum; anzahl: 43; seed: 11}
- `leben.passanten_vorlagen` (Vorlagen, Posen, Kleidungsfarben, Seed): look_up · 1,72 · point_up · 1,64 · carry · 1,78 · talk · 1,58 · look_up · 1,52 · hands_hips · 1,75
- `leben.staende` (Marktstände: Anzahl, Abstand, Seite, Maße, Dachstoff, Waren, Kisten): {anzahl: 14; abstand: 10,5; start_y: 30; seite: abwechselnd; x: 9,4; dach: (2,8; 3,4; 2,4); tisch: (2,2; 2,8; 0,8); kisten_je_stand: 2}
- `leben.laternen` (Laternenleinen: y-Positionen, x_end, z_end, sag, n je Leine, Seed): {y: (33; 71; 109; 147); x_end: 13,4; z_end: 7,2; sag: 1,1; n: 9; seed: 3; pendeln: {seilachse_grad: (4; 7); seilachse_hz: (0,3; 0,5); quer_grad: (1,5; 3); quer_hz: (0,2; 0,35)}}
- `leben.waesche` (Wäscheleinen: y-Positionen, Höhe (aus Fassaden), sag, Abstand zum Baum): {y: (52); z: 8,32; sag: 0,7; kandidaten_y: (52; 90; 128; 160); bedingung: Fassaden > 7,3 m, ≥ 14 m vom Baum; pendeln_grad: (5; 10); pendeln_hz: (0,35; 0,6)}
- `leben.leitungen` (Strommasten: Abstand, Höhe, Querbalken, Drahtabstand, Durchhang): {mastabstand: 19; y: (14; 176); x: 11,2; masthoehe: 9,5; querbalken: (1,8; 8,6); draehte: 3; drahtabstand: 0,7; drahthoehe: 8,75; durchhang: 0,9; drahtradius: 0,018}
- `leben.voegel` (je Schwarm: Anzahl, Startbox, Flugvektor, t0, t1, Seed): {n: 14; start: (-20; -14) · (64; 80) · (9; 13); flug: (30; 14; 9); t: (3,3; 5,6); seed: 21} · {n: 12; start: (14; 20) · (70; 84) · (9; 13); flug: (-28; 18; 10); t: (3,45; 5,7); seed: 22}
- `leben.blaetter` (treibende Blätter je Feld: Box, Anzahl, Wind, Seed): {name: StreetLeaves; box: (-12; 12) · (18; 176) · (0,3; 12); n: 700; wind: (1; 2,2; -0,3); seed: 4} · {name: TreeLeaves; box: (-6; 14) · (104; 134) · (0,5; 13); n: 260; wind: (0,8; 1,6; -0,5); seed: 6}
- `leben.pendeln` (Schwingen: Achse, Amplitude, Frequenz, Phase, Rauschen): Laternen und Wäsche (Werte dort), Rauschen 0,25–0,4
- `leben.verkehr` (Fahrzeuge, Bahnen, Boote: Anzahl, Pfade, Tempo, Lichter): entfällt (keine Fahrzeuge)
- `leben.rauch` (Rauch und Dampf: Quellen, Dichte, Tempo): entfällt (kein Rauch)
- `leben.flaggen` (Flaggen, Planen, Wäsche im Wind (Stoff-Cache)): Banner statisch, Wäsche pendelt (keine Stoffsimulation)
- `leben.caches` (Simulations-Caches: Ordner, Frames): Rigid-Body-Bruchstücke als Keyframes gebacken (kein Cache-Ordner)
- `leben.asynchron` (Variation, damit nichts synchron läuft (Phase, Tempo)): Amplitude, Frequenz und Phase je Objekt zufällig

## S8. Vegetation in der Stadt
- `baum.hero` (Manöver-Baum: Ort, Höhe, Kronenradius, Cluster, Blätter je Cluster, Steinring): {ort: (4; 118); hoehe: 21; krone: 7; cluster: 44; blaetter: 140; blattgroesse: 0,45; seed: 99; steinring: (3,2; 0,6)}
- `baum.nadelbaum` (Nadelbäume: Höhe, Krone, Cluster, Form, Transluzenz): {varianten: (22; 4; 34) · (16; 3,2; 28); blaetter: 110; form: konisch; blattgroesse: 0,3; farbe: {c1: (0,015; 0,035; 0,012); c2: (0,04; 0,07; 0,025)}; trans: 0,2; seeds: 80 + i}
- `baum.wald` (Waldbereiche: Anzahl, x/y-Bereich, Ausschlüsse, Skalierung): {bereich: Zufahrt; n: 1400; x: (-160; 160); y: (-260; 20)} · {bereich: Ring um die Mauer; n: 6000; x: (-520; 520); y: (-340; 360)} · {bereich: Plateau; n: 5000; x: (-700; 700); y: (360; 950)} · {skalierung: (0,75; 1,5)}
- `baum.dorf` (Bäume im Ort: Versuche, Abstand zu Gebäuden, Hauptgebäude, Straße; Skalierung): {versuche: 700; x: (-170; 170); y: (20; 300); skalierung: (0,6; 1,1)}
- `baum.buesche` (Büsche an der Zufahrt: Anzahl, Bereich, Skalierung): {n: 500; x: ±7,5–40; y: (-240; -8); skalierung: (0,18; 0,4)}
- `baum.felsbewuchs` (Bewuchs auf Felsbändern: Normalen-Schwelle, Mindesthöhe, Anzahl, Skalierung, Aussparung): {normale_z: 0,55; min_hoehe: 8; max: 900; skalierung: (0,25; 0,7); ohne: Gesichter-Panel |x| < 175, z 42–168; gewicht: Fläche·nz²}
- `baum.streuung` (Streuung per Geometry Nodes: Seed, Drehung, Größe, Punkte gesamt): {seed: 5; punkte: 12444; drehung_groesse: zufällig (Geometry Nodes)}

## S9. Wahrzeichen-Relief (Felswand, Monument)
- `wahrzeichen.wand` (Felswand: x-Bereich, y, z-Bereich, Auflösung, Seed, Panel): {x: (-420; 420); y: 360; z: (-2; 176); aufloesung: 1; seed: 8; panel: (-172; 172; 40; 168)}
- `wahrzeichen.risse` (Risse: Anzahl, Bereich, Tiefe, Breite): {n: 10; x: (-378; 378); z_top: 176; tiefe: 2,2; breite: 1,2; seed: 28}
- `wahrzeichen.gesichter` (je Gesicht x, Höhenversatz, Haar; Skalierung, Grundhöhe, Kopfvorlage, Voxelgröße): {liste: -110 · 10 · hashirama · -55 · -13 · tobirama · 0 · 14 · hiruzen · 56 · -13 · minato · 112 · 10 · tsunade; format: [x, Höhenversatz, Haar]; skalierung: 10; grundhoehe: 92; y: 362; vorlage: LeePerrySmith.glb; voxel: 0,22}
- `wahrzeichen.kinnlinie` (Kinnlinie, Panel-Aussparung): Kinn je Kopf = 92 + Versatz − 12 m, dazwischen linear
- `wahrzeichen.details` (Treppen, Geländer, Hütten, Kuppeln an der Wand): {treppen: 6 Läufe, Stufen 0,56 × 1,7 × 0,4 m alle 0,5 m, Geländer; wachhuetten: (-55; 34) · (56; 32) · (-160; 20) · (165; 52); kuppelbauten: (-84; 44; 11) · (-22; 62; 9) · (34; 46; 10) · (98; 70; 12); kuppel_format: [x, Abstand hinter der Kante, Radius], Höhe 8,5 m}

## S10. Kamera wie im Film
- `kamera.fov` (Sichtfeld): {vertikal_grad: 54,43; brennweite_mm: 35; sensor: 36 mm (Höhe)}
- `kamera.blende` (Blende, Fokusdistanz, Schärfentiefe an/aus): {dof: nein; hinweis: Schärfentiefe aus (großer Schärfebereich wie FPV)}
- `kamera.clipping` (Clip Start, Clip End): (0,05; 30000)
- `kamera.overscan` (Overscan für Stabilisierung und Verzerrung im Comp): entfällt (kein Overscan)
- `kamera.manoever` (je Manöver: Zeit, festes Objekt, Art (Tor, Kabel, Baum, Steigflug, Brüstung)): 1,8 · Tor · Durchflug unter dem unteren Querbalken · 3,2 · Laternenkabel 1 · darunter durch · 4 · Wäscheleine · darunter durch · 4,8 · Laternenkabel 2 · darunter durch · 6,4 · Laternenkabel 3 · darunter durch · 7,1 · alter Baum · links vorbei · 9 · Laternenkabel 4 · darüber · 9,55 · Platz/Residenz · Steigflug · 10,85 · Brüstung · drüber, scharf abbremsen · 15 · Klimax · Vorstoß, dann zurückgedrückt · 17,1 · Dach · ganz tief
- `kamera.mindestabstand` (kleinster Abstand zur Geometrie (Messwert)): {flug: {m: 1,36; t: 6,667; objekt: Lantern2.002}; dach: {m: 0,14; t: 17,333; objekt: ResRoofDeck}; hinweis: Szene des Finals vermessen, ohne Figuren und Effekte; 0,14 m ist das gewollte Tiefbild auf dem Dach}
- `kamera.hook` (Hook-Bild am Anfang: Inhalt, Dauer): 0–1,8 s: Blick vom Waldweg durchs Tor auf Dorf, Residenz und Gesichter; Teaser-Blitz 1,3 s
- `kamera.schlussbild` (Schlussbild: Motiv, Blickwinkel): 17–20 s: Naruto in Untersicht (Blick 8° nach oben), Faust zur Kamera, dahinter die fünf Hokage-Gesichter
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
- `render.hochrechnung` (s pro Frame × Frames = Gesamtzeit, gegen Budget): vorab 15–17 h geschätzt, tatsächlich 20,2 h Renderzeit
- `render.preflight` (Ergebnis des Preflight-Checks (Fehler, Warnungen)): entfällt (preflight_check.py gab es zum Zeitpunkt des Finals noch nicht)

## S12. Compositing (Kinolook)
- `comp.linsenverzerrung` (Linsenverzerrung (Distortion)): keine Verzerrung, nur laterale Dispersion (post.optik)
- `comp.halation` (Halation an hellen Kanten): entfällt (nicht verwendet)
- `comp.grading` (Schwarzwert, Highlights, Lift/Gamma/Gain, Farbtrennung Schatten/Lichter): AgX Medium High Contrast, Sättigung 1,12, +0,15 EV, Auto-Belichtung (post.*)
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
- `bewertung.gates` (je Gate (A, A2, B, C, D, E): Datum, freigegeben ja/nein, Zitat): {gate: A; datum: 2026-09-28 14:28; frei: ja; zitat: „ja“ (Zeitleiste Dachkampf, 35 mm, Sonne 18°/250°)} · {gate: C; datum: 2026-09-28; frei: –; zitat: Standbilder je Schritt 3.1–3.6, keine eigene Freigabe festgehalten} · {gate: D; datum: 2026-09-29 06:02; frei: nein; zitat: Vorschau v1: Kampf nur auf dem Dach und länger} · {gate: D; datum: 2026-09-29 07:41; frei: ja; zitat: „naruto sieht so brutal aus“ (Vorschau v2)} · {gate: E; datum: 2026-09-28 19:48; frei: ja; zitat: „720p aber zeig mir vorab das video“}
- `qc.durchsicht` (komplett in Originalgeschwindigkeit gesehen): 8 Stichproben-Frames gegen die freigegebene Vorschau geprüft (nicht das ganze Video)
- `qc.frames` (fehlende oder schwarze Frames, Fireflies): 480/480 Frames vorhanden
- `qc.flackern` (Flackern in Schatten, Lichtern, Wasser, Fenstern): nicht erfasst
- `qc.textur` (Kachelmuster, Texturstreckung, Z-Fighting): nicht erfasst
- `qc.banding` (Banding in Himmel und Nebel): nicht erfasst
- `qc.zweitdisplay` (auf Handy oder zweitem Display geprüft): nicht erfasst
- `qc.anfang_ende` (Anfang und Ende sauber (Loop, Fade)): kein Loop; Anfang in voller Fahrt, ruhiges Schlussbild
- `lieferung.master` (Master (EXR-Sequenz, ProRes oder MP4): Pfad, Größe): {pfad: drohnen_3d_welten/videos/02_Naruto_Konoha.mp4; mb: 45; commit: 4a8cfd2}
- `lieferung.varianten` (weitere Formate (9:16, 1:1, Webversion)): Webversion 10 Mbit/s (< 30 MB) für den Chat
- `lieferung.dateiname` (Dateiname mit Version und Datum): <NN>_<Welt>_<Ort>.mp4, ohne Version und Datum
- `lieferung.archiv` (archivierte Teile (Projekt, Texturen, Caches, Skripte), Ort): Code im Repo (Commit), Werte im Werte-Archiv; .blend, EXR und PNG nur im Scratchpad (nicht dauerhaft archiviert)
- `lieferung.obsidian` (ZIP-Paket für den Vault): ZIP-Paket „Drohnen Videos/3D Welten Blender/<NN Name>/{Test,Final,Werte}“
- `lieferung.wiederverwendbar` (Assets, Materialien, Node-Gruppen, Lichtrig, Kameraskript für die nächste Stadt): konoha.py (Gebäude, Tor, Residenz, Felsen, Verwitterung, Materialien) · konoha_life.py (Laternen, Wäsche, Passanten, Stände, Vögel, Blätter) · choreo.py (Kamera-Wegpunkte, Kampf-Backen, Hit-Stops, Kamerastöße) · vfx.py (Impact-Paket, Voronoi-Bruch, Rigid Body) · ninja.py (Figuren)
- `lieferung.notiz` (was beim nächsten Mal früher entschieden werden muss): Kampfort und -dauer früh festlegen (erst Tor, dann Dach, dann nur Dach und länger); Trefferlichter auf hellen Böden schwächer ansetzen

## S15. Kampf in der Stadt
- `kampf.ort` (Kampfort (z. B. Dach): Mitte, Höhe, lokales Koordinatensystem): {ort: Flachdach der Hokage-Residenz; mitte: (-20; 231); hoehe: 31,65; system: roof(x, y, z) relativ zur Dachmitte}
- `fx.impact_klein` (kleines Impact-Paket: Farbe, Licht, Ring, Funken, Seed): {name: HitAir; t: 10,95; ort_dach: (3,25; 5; 4,4); licht: 5000; ring: 3; farbe: (0,8; 0,88; 1)} · {name: HitKick; t: 11,5; ort_dach: (4,3; 1,8; 1,3); licht: 1000; ring: 2,2} · {name: HitPunch; t: 11,95; ort_dach: (3,8; 1,8; 1,4); licht: 1000; ring: 2,2} · {name: HitStrike; t: 12,35; ort_dach: (4; 1,8; 1,9); licht: 1600; ring: 3; farbe: (0,85; 0,9; 1)} · {name: HitHorn; t: 12,9; ort_dach: (15,4; -6,2; 1,9); licht: 8000; ring: 3,5; funken: 140; farbe: (1; 0,8; 0,5)} · {standard: {funken: 90; tempo: (5; 12); leben: (0,15; 0,45); licht_frames: 0 → max → 50 % (+2) → 0 (+8); ring_dauer: 8}}
- `fx.bruch` (Bruchstücke an Gebäudeteilen: Objekt, Zeit, Bruchhöhe, Stücke): {horn: {objekt: Horn7; t: 12,9 s + 3 Frames; bruchhoehe: 1,6; stuecke: 9; impuls: 40000; dichte: 500; sim_bis_s: 17,5}; dach: {stuecke: 14; platte: (2,6; 2,6; 0,12); t: 15,0 s + 4 Frames; impuls: 34000; dichte: 450; sim_bis_s: 19; max_hoehe: 5,2}; kollision: ResRoofDeck · ResParapet · ResCap; seed: 31}
- `fx.techniken` (Signatur-Techniken (z. B. Rasengan, Chidori): Farbe, Radius, Licht, Zeiten): {rasengan: {farbe: (0,12; 0,42; 1); radius: 0,34; licht: 160; an: 13,55; voll: 14,2; aus: 15,05}; chidori: {farbe: (0,35; 0,6; 1); radius: 0,85; blitze: 10; licht: 260; an: 13,4; aus: 15,05}; klimax: {kugel_r: 0,45; kugel_licht: 600; blitzlicht: (0; 8000) · (1; 14000) · (3; 4000) · (7; 0); boegen: {radius: 1,9; anzahl: 10; frames: 10}; funken: {n: 260; tempo: (8; 20); leben: (0,25; 0,7)}}}

## 1. Format und Ablauf
- `format.dauer` (Videolänge): 20
- `format.fps` (Bildrate): 24
- `format.frames` (Frames gesamt): 480
- `format.seitenverhaeltnis` (Seitenverhältnis): 9:16
- `format.ausgabe_aufloesung` (Endgröße): (1080; 1920)
- `ablauf.abschnitte` (Zeitleiste: Abschnitt, von–bis, Inhalt, Kamera-Tempo): 0 · 1,8 · Hook: vom Waldweg durchs Tor auf Dorf, Residenz und Gesichter; ferner Teaser-Blitz 1,3 s · 17–19 · 1,8 · 7 · Tor-Durchflug, Hauptstraße tief (3,6–5 m) unter Laternenkabeln und Wäsche, Passanten, Stände; Teaser-Blitz 5,3 s · 19–24 · 7 · 9 · links am alten Baum vorbei, über Kabel 4, schnell über den Platz · 17–29 · 9 · 10,85 · Steigflug an der Residenz, über die Brüstung aufs Dach · 29–38 (max. 38,0 bei 9,83 s) · 10,85 · 17 · Dachkampf: Luftzusammenprall 10,95, Tritt/Block 11,5, Schlag/Block 11,95, Sprungtritt 12,35, Konter gegen das Horn 12,9, Aufladen, Rasengan gegen Chidori 15,0 (Klimax) · 18,6 → 0,9–8,5 · 17 · 20 · Schlussbild: Naruto in Untersicht, Faust zur Kamera, dahinter die fünf Hokage-Gesichter · 1,2–2,1
- `ablauf.koordinaten` (Hauptflugrichtung, Achsen, Einheit): +Y Hauptflugrichtung (Tor y 0 → Felsen y 360), Z oben, Meter; Kampf in Dachkoordinaten relativ zu C_ROOF (−20, 231) und DECK_Z 31,65

## 2. Licht und Himmel
- `licht.hauptsonne.hoehe` (Höhe der Hauptsonne): 18
- `licht.hauptsonne.azimut` (Azimut der Hauptsonne): 250
- `licht.hauptsonne.staerke` (Stärke): 5
- `licht.hauptsonne.kelvin` (Farbtemperatur): 4300
- `licht.hauptsonne.winkel` (Winkelgröße (Schattenweichheit)): 0,5
- `licht.zusatzsonnen` (weitere Sonnen: Höhe, Azimut, Stärke, Kelvin, Winkel): ()
- `licht.sonnenscheiben` (sichtbare Scheiben: Radius, Stärke, Farbe): entfällt (Nishita-Himmel, Sonnenscheibe aus)
- `licht.randlicht` (Randlicht nur auf Figuren: Höhe, Azimut, Stärke, Kelvin, Winkel): {hoehe: 22; azimut: 60; staerke: 2,5; kelvin: 7800; winkel: 3; nur: Naruto + Sasuke (Light Linking)}
- `licht.toon_richtung` (Licht-/Randlichtrichtung für das Toon-Shading): entfällt (keine Toon-Figuren)
- `himmel.verlauf` (Himmelsfarben über der Höhe): physikalischer Himmel (Nishita, Multiple Scattering), kein eigener Verlauf
- `himmel.unter_horizont` (Farbe unter dem Horizont): Standard (nicht gesetzt)
- `himmel.sonnenleuchten` (Leuchten um Sonnen: Exponenten, Stärken, Farbe): Standard (nicht gesetzt)
- `himmel.entsaettigung_diffus` (Entsättigung des diffusen Himmelslichts): Standard (nicht gesetzt)
- `himmel.staerke` (Himmelsstärke): 0,1
- `wolken.bedeckung` (Wolkenschicht): 0,4
- `wolken.ref` (Wolkenschicht): 9
- `wolken.farbe` (Wolkenschicht): (1; 0,98; 0,95)
- `wolken.skala` (Wolkenschicht): Standard 1,0 (Höhe 2000 m)
- `atmosphaere.dichte` (Streuvolumen (0 = aus)): 0

## 3. Gelände und Welt-Objekte
- `gelaende.hauptform` (Haupt-Landmarke (Tafelberg o. Ä.): Mitte, Radius): {art: Talkessel mit runder Stadtmauer, Hügelring, Hokage-Felsen im Norden; mitte: (0; 190); radius: 190}
- `gelaende.umriss` (Umrissfunktion (Sinus-Anteile, fbm, Seeds)): entfällt (Kreis (Stadtmauer))
- `gelaende.plateau` (Plateauhöhe + Rauschanteile): nördlich des Felsens: 168 + 22·fbm(x/160, y/160, 5, seed 4), Übergang y 374–390
- `gelaende.kante` (Kantenabfall, Schuttfuß): Hokage-Felsen als eigene Wand (wahrzeichen.wand)
- `gelaende.heightfield` (Größe, Auflösung): {groesse: 2400; punkte: 360; ursprung: (0; 400); offset_z: -0,05}
- `gelaende.steilwand` (Ringwand: n_ang, n_z, seed, Felskante (lip), base_z): siehe wahrzeichen.wand
- `gelaende.felsnadeln` (je Nadel: x, y, r, h, seed, Neigung x/y): entfällt (keine Felsnadeln)
- `gelaende.felsnadel_mesh` (rock_mesh: detail, taper, lumpy, strata, flute, ledges, bed): entfällt (keine Felsnadeln)
- `gelaende.ferne_felsen` (ferne Nadeln: x, y, r, h): Bergspitze links hinter dem Felsen: 250 m hoch bei (−190, 620), σ 150 m (terrain_height)
- `gelaende.inseln` (Horizont-Inseln: x, y, r, h): entfällt (keine Inseln)
- `material.fels` (Felsfarben c1–c3, wet_line, algae, moss, moss_amount, strata, bump, crack, lichen, variation, bedding, streak): {name: GateStone (Torsockel, Baumring); c1: (0,2; 0,19; 0,17); c2: (0,36; 0,34; 0,3); c3: (0,3; 0,28; 0,25); wet: nein; bump: 0,4; crack_w: 0,3; verwitterung: {dirt: 0,3; var: 0,03}}
- `material.steilwand` (dito für die Wand): siehe material.wahrzeichen_fels
- `material.boden` (Erde, Moosflecken, Roughness, Bump): {dorf: konoha.ground_material(); strasse: konoha.street_ground_material(half_w 12); platz: konoha.street_material (Pflaster); zufahrt: konoha.dirt_road_material; huegel: siehe material.gras}
- `objekte.haeuser` (je Haus: x, y, r; Lehm-, Glas-, Türfarben): siehe gebaeude.* (229 Gebäude)
- `objekte.dragonballs` (Mitte, Sockelradien, Kugelradius, Ring): entfällt (nicht in dieser Welt)
- `objekte.raumschiff` (Ort, Höhe, Radius, Rumpf-/Randfarben, Fenster-Emission): entfällt (nicht in dieser Welt)

## 4. Wasser
- `wasser.kacheln` (Start, Anzahl x/y, Kachelgröße): entfällt (kein Wasser)
- `wasser.wellen` (res, wind, wave_scale, chop, Richtung, alignment, foam_coverage): entfällt (kein Wasser)
- `wasser.material` (deep, shallow, foam_amount, view_dark, micro): entfällt (kein Wasser)
- `wasser.brandung` (Abstandsfeld: Bereich, Zelle): entfällt (kein Wasser)

## 5. Vegetation, Gras, Kleinkram
- `baum.varianten` (je Variante: Höhe, Kronenradius, Drehung, Blattzahl, Blattgröße, Seed): {laubbaeume: (14; 5; 26) · (11; 4,2; 22) · (17; 6; 30) · (8; 3,2; 16); format: [Höhe m, Kronenradius m, Blattcluster]; blaetter_je_cluster: 90; blattgroesse: 0,4; seeds: 60 + i; nadelbaeume: siehe baum.nadelbaum}
- `baum.material` (Blattfarben, Transluzenz, Rinde): {KLeaves: {c1: (0,035; 0,1; 0,018); c2: (0,12; 0,22; 0,04)}; KLeaves2: {c1: (0,05; 0,12; 0,02); c2: (0,16; 0,24; 0,05)}; rinde: nature.bark_material (Standard); abwechselnd: gerade Varianten KLeaves}
- `baum.haine` (Haine: x, y, r, n; Skalierungen): siehe baum.wald und baum.dorf
- `baum.abstaende` (Mindestabstände zu Häusern, Objekten, Kratern, Weg, Kante): {dorf_mauer: innerhalb Radius 182 m; dorf_strasse: |x| > 14 m; dorf_gebaeude: außerhalb Grundriss + 2,5 m; dorf_residenz: 28; wald_mauer: Radius > 202 m; wald_torstrasse: |x| < 11 m bei y < 5 frei; wald_felsband: y 335–394 frei}
- `baum.sichtlinie` (baumfreier Korridor Kamera → Kämpfer: Zeitfenster, Radius): entfällt (Kampf auf dem Dach, kein Baum in der Sichtlinie)
- `findlinge` (Anzahl, Radius, Skalierung, Displace, Abstände): entfällt (keine)
- `sandflecken` (Anzahl, Radien, Randabstand): entfällt (keine)
- `pilze` (Gruppen, seitlicher Abstand zur Route, y-Bereich): entfällt (keine)
- `gras.feld` (Bereich, Kantenabstand, Dichte (nah, Grund, Abfall), Band, clump, max Halme, Höhe): entfällt (Gras nur als Material, keine Halme)
- `gras.farben` (Basis, Spitze, Büschel-Tönungen): siehe material.gras
- `gras.aussparungen` (Radien um Häuser, Objekte, Findlinge, Sand): entfällt (keine Halme)
- `gras.bewegung` (Wind, Böen (Tempo, Stärke), sway): entfällt (keine Halme)
- `gras.druckwellen` (je Welle: x, y, t, Tempo, Stärke, Breite): entfällt (keine Halme)
- `gras.brennen` (Krater, in denen Halme verschwinden (Radius-Faktor)): entfällt (keine Halme)

## 6. Kamera
- `kamera.brennweite` (Objektiv): 35
- `kamera.sensor` (Sensorgröße / Fit): {fit: VERTICAL; hoehe_mm: 36}
- `kamera.wegpunkte` ((Zeit, Ort) für die Speed-Ramp): {t: 0; pos: (0; -32; 9)} · {t: 1,8; pos: (0; 0; 7)} · {t: 2,7; pos: (0,5; 20; 5)} · {t: 3,2; pos: (1,5; 31; 3,6)} · {t: 4; pos: (-1,5; 50; 4,2)} · {t: 4,8; pos: (-2; 69; 3,6)} · {t: 5,7; pos: (-1,5; 90; 4)} · {t: 6,4; pos: (-4,5; 104; 3,6)} · {t: 7,1; pos: (-7,5; 116; 4,5)} · {t: 7,8; pos: (-6,5; 128; 5,5)} · {t: 8,4; pos: (-5; 142; 7,5)} · {t: 9; pos: (-5,5; 157; 10,5)} · {t: 9,55; pos: (-10; 174; 16,5)} · {t: 10,1; pos: (-15,5; 192; 24,5)} · {t: 10,55; pos: (-19,5; 205,5; 31)} · {t: 10,85; pos: (-20; 212,2; 35,45)} · {t: 11,3; pos: (-22,5; 218,5; 34,65)} · {t: 12,2; pos: (-19,5; 219; 34,35)} · {t: 13,2; pos: (-16,6; 220; 34,25)} · {t: 14,45; pos: (-15,25; 220,6; 34,05)} · {t: 15; pos: (-15,25; 229; 33,65)} · {t: 15,9; pos: (-13,6; 228; 34,45)} · {t: 17,1; pos: (-13,7; 226,4; 32)} · {t: 20,05; pos: (-13,3; 230,4; 32,4)}
- `kamera.tempo` (resultierendes Tempo je Zeitpunkt (Messwert)): (0; 17,8) · (0,5; 17,1) · (1; 17,4) · (1,5; 18,8) · (2; 21,8) · (2,5; 23) · (3; 22,2) · (3,5; 24,2) · (4; 24) · (4,5; 23,7) · (5; 23,9) · (5,5; 22,9) · (6; 20,7) · (6,5; 18,4) · (7; 17,4) · (7,5; 16,8) · (8; 23,4) · (8,5; 24,3) · (9; 29,2) · (9,5; 35,6) · (10; 37,1) · (10,5; 31,6) · (11; 18,6) · (11,5; 4,2) · (12; 2,7) · (12,5; 3,5) · (13; 2,6) · (13,5; 0,9) · (14; 1) · (14,5; 8,5) · (15; 5,7) · (15,5; 1,9) · (16; 2,6) · (16,5; 2,7) · (17; 2,1) · (17,5; 1,6) · (18; 1,4) · (18,5; 1,3) · (19; 1,2) · (19,5; 1,3) · (20; 1,4)
- `kamera.halte` (Kamera-Halte bei Treffern): (12,35; 2) · (12,9; 2) · (15; 4)
- `kamera.ausrichtung` (look_pitch, pitch_follow, bank_gain, max_bank, micro, seed): {look_pitch: -2; pitch_follow: 0,45; bank_gain: 1; max_bank: 30; micro: 1; seed: 9}
- `kamera.pitch_overrides` (Blick-Neigung je Zeitfenster): entfällt (keine)
- `kamera.blickgewicht` (Gewicht Flugrichtung ↔ Blickziel über die Zeit): (0; 0) · (1,8; 0) · (2,6; 0,25) · (6; 0,25) · (6,5; 0) · (7,8; 0) · (8,6; 0,45) · (10,3; 0,6) · (10,85; 1) · (21; 1)
- `kamera.blickziele` (Ziel je Phase (Paar, Krater, Strahlen, Duell, Ende) mit Zeitfenstern und Gewichten): {phase: Kämpferpaar; ziel: Mitte + 1,0 m} · {phase: Naruto gegen das Horn; von: (12,55; 12,75); bis: (13,05; 13,4); ziel: Naruto + 1,2 m} · {phase: nach dem Zusammenprall; von: (15,12; 15,5); ziel: Naruto + 1,2 m} · {phase: Schlussbild; von: (16,8; 17,8); ziel: Naruto + 1,3 m (Blick 8° nach oben)}
- `kamera.glaettung_ziele` (Glättung der Kämpferbahnen für den Blick): entfällt (Kämpferbahnen ungeglättet (choreo.track))
- `kamera.stoesse` (Kamerastöße: Zeit, Stärke, Frequenz; seed, pos_amp): {liste: (10,95; 1; 6) · (11,5; 0,8; 6) · (11,95; 0,8; 6) · (12,35; 1,2; 7) · (12,9; 1,6; 8) · (15; 3,2; 10); format: [Zeit s, Stärke °, Frames]; seed: 77; pos_amp: 0,03}

## 7. Figuren
- `figur.naruto.modell`: selbst gebaut: ninja.naruto() nach den Shippuden-Model-Sheets (Skin-Modifier-Skelett + Subdivision, Kleidung als Teile, Haar-Stacheln, Uzumaki-Spirale als Decal)
- `figur.sasuke.modell`: selbst gebaut: ninja.sasuke() (hellgraues Hemd mit Uchiha-Wappen, indigo Hüftwickel, lila Seilgürtel, Kusanagi)
- `figur.naruto.hoehe`: 1,66
- `figur.sasuke.hoehe`: 1,66 (Code; Docstring nennt 1,68)
- `figur.naruto.gelenke`: eigenes Gelenk-Rig (figures.Figure)
- `figur.sasuke.gelenke`: eigenes Gelenk-Rig (figures.Figure)
- `figur.naruto.material`: {orange: (0,86; 0,26; 0,035); schwarz: (0,028; 0,028; 0,032); haar: (0,95; 0,7; 0,06); stirnband: (0,02; 0,025; 0,055); platte: (0,62; 0,63; 0,66); bandage: (0,8; 0,78; 0,72); stiefel: (0,045; 0,045; 0,05); haut: skin_material Standard}
- `figur.sasuke.material`: {grau: (0,4; 0,4; 0,46); indigo: (0,045; 0,045; 0,16); hose: (0,035; 0,035; 0,11); beinwickel: (0,16; 0,16; 0,18); seil: (0,2; 0,1; 0,34); haut: (0,74; 0,48; 0,35); haar: (0,012; 0,012; 0,025)}
- `figur.<name>.modell` (Datei, Quelle, Reparaturen): siehe figur.naruto.modell / figur.sasuke.modell
- `figur.<name>.hoehe` (Zielgröße): siehe figur.naruto.hoehe / figur.sasuke.hoehe
- `figur.<name>.gelenke` (Quelle der Gelenke (Mixamo / vermessen)): siehe figur.naruto.gelenke / figur.sasuke.gelenke
- `figur.<name>.schwanz` (Mittellinie, Gliederzahl): entfällt (kein Schwanz)
- `figur.<name>.material` (lit, mid, shade, bands, tint_lit, rim, rim_w, sat, value, diffuse_mix, hair_glow): siehe figur.naruto.material / figur.sasuke.material
- `figur.<name>.kontur` (Konturstärke relativ zur Höhe): entfällt (keine Kontur, halbrealistische Stoff-/Hautmaterialien)
- `figur.<name>.gewichte` (smooth, power, Saat-Bereiche): entfällt (Teile hängen an den Gelenken, keine Gewichtsrechnung)
- `figur.backen` (style, lag_scale, lean_tau, seeds, look_win): {style: choreo.default_style; nachziehen_s: {elbow: 0,07; wrist: 0,12; neck: 0,03; head: 0,06}; seed_naruto: 1; seed_sasuke: 2; look_win_naruto: (17; 21)}
- `figur.schwanz_peitsche` (Peitschenhiebe (Zeit, Winkel), sway): entfällt (kein Schwanz)
- `figur.aura` (Farbe, Frames an/aus, Höhe, Breite, Licht, opacity, edge_w, strength): entfällt (keine Aura)
- `figur.verschwinden` (Frames/Skalierung beim Ausblenden): kein Ausblenden: Sasuke springt 16,15–17,1 s über die Brüstung aus dem Bild
- `figur.mimik` (Ausdrucks-Zeitplan (nur Fallback-Figuren)): nicht erfasst

## 8. Kampfplan und Timing
- `kampf.brusthoehe` (Brust über dem Figurenursprung): {paar: 1; naruto: 1,2; schlussbild: 1,3; hinweis: Blickziel über dem Ursprung}
- `kampf.stil` (je Posenklasse: Dauer, Ausholen, Überschwingen): {angriff: (0,36; 0,2; 0,14); reaktion: (0,12; 0; 0,18); landung: (0,14; 0; 0,14); sprung: (0,2; 0,1; 0,06); lauf: (0,14; 0; 0,05); aufladen: (0,3; 0,1; 0,06); sonst: (0,32; 0,06; 0,07); format: [Dauer s, Ausholen, Überschwingen] (choreo.default_style)}
- `kampf.bahnprofile` (lin / in / out je Abschnitt): Standard-Interpolation (keine ease-Angaben in den Schlüsseln)
- `kampf.hitstops` (Treffer-Halte: Zeit, Frames): (10,95; 3) · (11,5; 3) · (11,95; 3) · (12,35; 3) · (12,9; 3) · (15; 4)
- `kampf.teaser` (Fern-Zusammenstöße: Zeit, Mitte, Achse): {t: 1,3; ort_dach: (3; 5; 3,5)} · {t: 5,3; ort_dach: (3; 5; 3,5)}
- `kampf.vorstoss` (Startpunkte, Zeitpunkt, Tempo): {naruto: {t: (14,65; 15); von_dach: (6,9; 6,5); nach_dach: (5,37; 6,5; 1)}; sasuke: {t: (14,65; 15); von_dach: (2,6; 6,5); nach_dach: (4,13; 6,5; 1)}}
- `kampf.zusammenprall` (Zeit, Ort): {luft: {t: 10,95; ort_dach: (3,25; 5; 3,3)}; klimax: {t: 15; ort_dach: (4,75; 6,5; 1); ort_welt: (-15,25; 237,5; 32,65)}}
- `kampf.schlaghagel` (Wechsel: Zeit, wer, Pose; Ausholvorlauf; Drehung der Kampfachse): {wechsel: 11,5 · S · kick_L · N blockt · 11,95 · N · punch_R · S blockt · 12,35 · N · kick_R (Sprungtritt) · S blockt · 12,65 · S · kick_L (Drehtritt) · trifft; linie_y_dach: 1,8}
- `kampf.aktionen` (Schwanzhieb, Knie, Doppelfaust, Einschlag, Ausbruch: Zeiten, Orte): {konter: 12,67; horn: 12,9; chidori_aufladen: (13,45; 14,5); rasengan_aufladen: (13,6; 14,5); ansturm: 14,65; sasuke_absprung: (16,15; 17,1); naruto_steht_auf: 16,4; hero_pose: 17}
- `kampf.zanzoken` (Zeitfenster weg/wieder, Frames, Orte): entfällt (keine Teleports)
- `kampf.strahlen` (Feuerpositionen, Ziele, Ausweichpositionen): entfällt (keine Strahlen)
- `kampf.trennung` (Zielorte, Tempo): {rueckstoss: 15,35; landung: 15,8; naruto_dach: (8,9; 4,3); sasuke_dach: (-5,5; 8,6)}
- `kampf.duell` (Aufladen, Start, Klimax): {aufladen: (13,45; 14,5); start: 14,65; klimax: 15}
- `kampf.posen` (eigene Posen (Gelenkwinkel)): hero (Faust zur Kamera) + Posenbibliothek figures.POSES (guard, jump, kick_R/L, punch_R/L, recoil, land, run_a/b, crouch_charge_R/L, dash_thrust_R/L, stand)
- `kampf.schluessel` (alle Schlüssel je Figur (t, Pose, Ort, Blick, Luft, lean/roll/rot/ease/spin)): siehe kampfplan_und_kamera.json (Naruto 38, Sasuke 33 Schlüssel)

## 9. Effekte
- `fx.farben` (Aura, Kamehameha, Todesstrahl, Ki, Spurfarben): {rasengan: (0,12; 0,42; 1); chidori: (0,35; 0,6; 1); klimax_kugel: (0,3; 0,55; 1); klimax_blitz: (0,75; 0,85; 1); teaser: (0,55; 0,75; 1); treffer: (1; 0,75; 0,4); horn: (1; 0,8; 0,5)}
- `fx.kispur` (Schweiflänge, center_z, strength, Kernanteil, Radius je Ereignis): entfällt (keine Ki-Spuren)
- `fx.zanzoken` (Flacker-Folge, Luftring-Radien): entfällt (keine Teleports)
- `fx.treffer` (je Art: Radius, Licht, Funken, Luftring; Dauer; core_k): siehe fx.impact_klein
- `fx.teaser_blitze` (r, Dauer, Licht, core_s, glow_s, glow_alpha): {r: 6,5; dauer: 12; licht: 30000; core_s: 25; glow_s: 6; glow_alpha: 0,35; funken: 120; funken_tempo: (6; 14); funken_leben: (0,2; 0,5)}
- `fx.druckwellen` (Radius, Dauer, Dicke, Glühen, ior): {treffer: {r: (2,2; 3,5); dauer: 8; dicke: 0,1; gluehen: 1,5}; klimax: {r: 15; dauer: 12; dicke: 0,2; gluehen: 2,5}; ior: 1,04}
- `fx.einschlag` (Kraterradius, Abkühlzeit, Burst, Staub, Funken, Bruchstücke, Impuls): {horn_staub: {n: 260; r: 3; steigen: 1,2; leben: 1,2}; klimax_staub: {n: 1600; r: 8; steigen: 1,8; leben: 1,6}; brandfleck_r: 1,6; bruchstuecke: siehe fx.bruch}
- `fx.strahlen` (Radius, Licht, Frames an/voll/aus, Einschlag-Burst, Staub, Brocken): entfällt (keine Strahlen)
- `fx.duell` (Ladungen, Strahlradien, Licht, Wobble, Treffpunkt-Pendel, Kugel, Blitzbögen): siehe fx.techniken
- `fx.explosion` (Blitz, Volumen (Feuer, Rauch, Steigen, Dauer, fire), Licht-Kurve, Farbverlauf, Wellen, Funken, Gischt): entfällt (keine Explosion)
- `fx.sicherheit` (Burst-Ausblenden, transparente Bounces, ior der Ringe): {transparent_bounces: 8; bruchstuecke_bis_blitz_ausgeblendet: ja}

## 10. Render
- `render.aufloesung` (Render-Auflösung Final / Vorschau / Standbild): {final: (720; 1280); vorschau: (270; 480); standbild: (540; 960); szene_gebaut_mit: (540; 960)}
- `render.samples` (Samples Final / Vorschau / Standbild): {final: 32; vorschau: 8; standbild: 16}
- `render.adaptiv` (adaptive Schwelle): 0,04
- `render.denoiser` (Denoiser, Pässe, Prefilter): OpenImageDenoise, Albedo+Normal, ACCURATE
- `render.bounces` (max, diffus, glossy, Transmission, Volumen, transparent): {max: 4; diffus: 2; glossy: 2; transmission: 6; volumen: 0; transparent: 8}
- `render.clamp` (Sampling-Optionen): 6
- `render.blur_glossy` (Sampling-Optionen): 1
- `render.light_tree` (Sampling-Optionen): ja
- `render.persistent` (Sampling-Optionen): ja
- `render.volumen` (step_rate, max_steps): Standard (step_rate 1,0, max_steps 1024; keine Volumen)
- `render.motion_blur` (an/aus, Shutter): {an: ja; shutter: 0,5}
- `render.filter` (Pixelfilter, Breite): BLACKMAN_HARRIS · 1,5
- `render.paesse` (Pässe, Mist-Tiefe, Film transparent): {paesse: Combined · Mist · Environment; mist_tiefe: 6000; film_transparent: ja}
- `render.seed` (Seed, animiert ja/nein): {seed: 7; animiert: nein}

## 11. Nachbearbeitung und Encoding
- `post.dunst` (mist_depth, haze_dist, haze_start, haze_color, haze_sky, haze_fade): {mist_depth: 6000; haze_dist: 1600; haze_start: 20; haze_color: (0,62; 0,7; 0,82); haze_sky: ja}
- `post.look` (View Transform / Look): AgX - Medium High Contrast
- `post.farbe` (sat, ev): {sat: 1,12; ev: 0,15}
- `post.autobelichtung` (ae_strength, ae_tau, ae_ref): {ae_ref: -2,3; ae_strength: 0,55; ae_tau: 0,45; ev_bereich: (-0,61; 0,47)}
- `post.bloom` (bloom, bloom_thr, fog_glow, bloom_clamp): {bloom: 0,22; bloom_thr: 1,6; fog_glow: 0,6; bloom_clamp: 4}
- `post.optik` (dispersion, vignette, grain): {dispersion: 0,006; vignette: 0,22; grain: 0,025}
- `post.pulse` (Belichtungsstöße: Zeit, +EV, Dispersion, Dauer): (10,95; 0,4; 0,008; 0,06) · (11,5; 0,35; 0,006; 0,06) · (11,95; 0,35; 0,006; 0,06) · (12,35; 0,5; 0,01; 0,07) · (12,9; 0,6; 0,012; 0,08) · (15; 1,8; 0,02; 0,07)
- `post.flare` (Lens Flare: t0, t1, thr, strength, streak, fade_out): entfällt (kein Lens Flare)
- `encode.skalierung` (Filter, Zielgröße): Lanczos · (1080; 1920)
- `encode.master` (Codec, Profil, Pässe, Bitrate (Soll/Ist), Dateigröße): {codec: H.264 High, Zwei-Pass, preset slow; soll_mbit: 18; maxrate: 20.2M; bufsize: 36M; ist_mbit: 18; mb: 45}
- `encode.web` (Bitrate, maxrate, bufsize, Dateigröße): {mbit: 10; maxrate: 11.2M; bufsize: 20M; ist_mbit: 10; mb: 24,9}

## 12. Laufzeiten und Ressourcen (Messwerte nach dem Final)
- `zeit.szene_bauen` (Aufbau der Szene): ca. 1,4 min (06:17:07 → Blend gespeichert 06:18:31)
- `zeit.frame_mittel` (Renderzeit pro Frame): 151,5
- `zeit.frame_max` (Renderzeit pro Frame): 1017
- `zeit.render_gesamt` (reine Renderzeit): 20,2
- `zeit.wanduhr` (Start bis Ende inkl. Ausfälle): 22,5
- `zeit.vorschau` (Vorschau-Sequenz(en)): {v1_480_frames_270x480_8spp_min: 45,3; v2_480_frames_270x480_8spp_min: 61,4}
- `ressourcen.ram` (Speicher, Dateigrößen): 3,7–7,6 GB im System belegt (zusammen mit der Namek-Vorschau), 16 GB gesamt
- `ressourcen.blend` (Speicher, Dateigrößen): 120
- `ressourcen.exr` (Speicher, Dateigrößen): ≈0,78 MB/Frame (376 MB gesamt), dazu 839 MB PNG
- `ressourcen.neustarts` (Container-Neustarts während des Finals): 2 (4 Starts im Log; nach Frame 104 ein Doppelstart: Frames 105–113 liefen zweimal parallel, daher bis 1017 s/Frame)

## 13. Sound-Marker
- `sound.marker` (Name, Zeit, Frame (AMBIENCE, WHOOSH, IMPACT, ZANZOKEN, BEAT_DROP)): {name: AMBIENCE; t: 0; frame: 1} · {name: WHOOSH; t: 1,83; frame: 45} · {name: WHOOSH; t: 3,29; frame: 80} · {name: WHOOSH; t: 4,92; frame: 119} · {name: WHOOSH; t: 6,71; frame: 162} · {name: WHOOSH; t: 8,62; frame: 208} · {name: WHOOSH; t: 10,85; frame: 261} · {name: IMPACT; t: 10,95; frame: 264} · {name: IMPACT; t: 11,5; frame: 277} · {name: IMPACT; t: 11,95; frame: 288} · {name: IMPACT; t: 12,35; frame: 297} · {name: IMPACT; t: 12,9; frame: 311} · {name: BEAT_DROP; t: 15; frame: 361} · {name: AMBIENCE; t: 17; frame: 409}

## 14. Bewertung (für die spätere Analyse)
- `bewertung.freigabe` (Datum, freigegeben ja/nein, Zitat des Nutzers): Vorschau v2 am 2026-09-29 07:41 freigegeben („naruto sieht so brutal aus“), Final-Start 07:42, Final geliefert 2026-09-30 06:14
- `bewertung.korrekturen` (was nach Vorschauen geändert wurde (Parameter-ID → alt/neu, Grund)): 2026-09-27: Konoha v3 nach Dominiks Referenzbildern (Pastellfassaden, bunte Dächer, Wassertanks, ockerfarbener Felsen) · 2026-09-27: „Bei Naruto finde ich den Kameraflug sehr gut.“ Wunsch: Kampf Naruto gegen Sasuke auf dem Dach, nur Rasengan gegen Chidori, Wow-Effekt · 2026-09-28: Cinematic-Briefing → Speed-Ramp, Sonne 10–25°, Randlicht, Verwitterung, Impact-Paket, Straßenleben, Laternen an Seilen, Schlussbild an den Gesichtern (Schritte 3.1–3.6) · 2026-09-28: Kampfort-Vorschlag am Tor abgelehnt („neeee die sollen am ende auf dem dach vom haus des hokage einen kurzen kmpf haben.“) → Dachkampf; Brennweite fest 35 mm · 2026-09-28: Final in 720p hochskaliert („720p aber zeig mir vorab das video“) · 2026-09-29: „Der Kampf … hätte ich gerne nur da, wo er am Ende ist, auf dem Dach … länger gezeigt.“ → Flug 0–10,9 s ohne Kampf, Teaser-Blitze 1,3/5,3 s, Dachkampf 10,95–20 s (vorher ca. 5,5 s) · 2026-09-29: Trefferlichter auf dem Dach von 3500–5000 W auf 1000–1600 W gesenkt (grelle Lichtflecken)
- `bewertung.reaktion` (Reaktion auf das Ergebnis (wörtlich)): „naruto sieht so brutal aus“ (Vorschau v2); zum Flug: „Bei Naruto finde ich den Kameraflug sehr gut.“
- `bewertung.social` (später: Aufrufe, Watchtime, Likes (falls gepostet)): –

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
- `kamera.route_welt` (Bahn weltfest / im Schiffssystem): siehe stadt.route (Blockout) und kamera.wegpunkte
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
