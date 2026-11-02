# Weiterarbeiten an Damm Raider

Übergabe-Notiz mit dem Stand, den Konventionen und den nächsten Schritten, damit eine neue
Sitzung (z. B. Claude Code auf dem eigenen Rechner) direkt weitermachen kann.

## Stand

- Zwei Karten: **Der Damm** (offene Talsperre) und **Weltenbrück** (Fantasy-Stadt mit deutscher
  Architektur, eng und verwinkelt, jedes Haus begehbar). Auswahl im Einsatz-Menü.
- Weltenbrück verbindet vier Viertel-Ideen zu einer eigenen Stadt: Hafen- und Seefahrer-Stimmung,
  Waldsiedlung auf Terrassen mit Riesenbäumen, ein Erfinderturm mit Energiekern und ein düsterer
  Gründerfelsen. Bewusst nur Archetypen – keine Figuren, Logos oder Nachbauten aus bestehenden Werken.
- Die ganze Stadt ist begehbar (Grobausbau): Weiße Stadt, Turmstadt und Hafenviertel liegen hinter offenen Toren,
  Code in `src/maps/weltenbrueck/outer.js`, neue Hausstile `weiss`, `weissturm`, `hochhaus`, `speicher`.
- Fast alle festen Formen (Möbel, Fassadenteile, Requisiten, Bäume, Autos, Waffen, Figuren, Gesichter)
  kommen aus Blender-Skripten, Ausnahme sind die SimReady-Requisiten (siehe unten). Wände, Böden, Treppen und Dächer der Stadthäuser erzeugt das Spiel aus
  dem Stadtplan, weil jedes Haus genau auf sein Grundstück passen muss.
- Detail-Runde: Oberflächen mit Normal-Maps, Fassadendetails, Stuben, Wiesen, Autos, Bäume.
- Grafik-Runde: Nachbearbeitung in `src/post.js` (GTAO auf „hoch“, Bloom, Farblook je Wetter,
  Vignette, Korn), weichere und schärfere Schatten, Himmel mit beleuchteten Wolken, Wind in der
  Vegetation (`World.addWind`), tiefere Tagessonne mit längeren Schatten.
- SimReady-Runde: 19 Requisiten aus dem SimReady-Katalog (CC BY 4.0) in `assets/simready.glb`, per Skript
  aus USD umgewandelt. Stahlfass und Sperrholzkiste ersetzen `p_barrel` und `p_crate`, der Rest ist neue Deko.
  Das Sprengfass ist das Stahlfass in Rot mit gelbem Band und hat echte Maße (0,86 m statt 1,2 m).
- Referenzen bisher nur aus Texten (Bildseiten waren in der Cloud-Umgebung gesperrt): Fachwerk
  (Mannfigur, Andreaskreuz, Balkenköpfe), Schwarzwaldhaus (Walmdach, Holzschindeln), Bauernmöbel
  (Brettstuhl mit Herz, Eckbank, Kachelofen, Kuckucksuhr), Wiesenpflanzen (Mohn, Kornblume, Margerite,
  Löwenzahn, Klee), verlassene europäische Autos der 70er.

## Wichtige Dateien

| Datei | Inhalt |
| --- | --- |
| `src/maps/index.js` | Liste der Karten |
| `src/maps/damm.js` | Damm: Terrain, Wasser, Vegetation, POIs, Aufzüge |
| `src/maps/weltenbrueck/layout.js` | Stadtplan: Terrassen, Straßen, Plätze, Grundstücke (BSP) |
| `src/maps/weltenbrueck/house.js` | Hausgenerator: Wände mit Öffnungen, Treppen, Fassaden, Dächer, Möbel |
| `src/maps/weltenbrueck/build.js` | Sonderbauten: Dom, Rathaus, Stadtmauer, Riesenbäume, Gründerfelsen |
| `src/maps/weltenbrueck/outer.js` | Viertel vor der Mauer: Weißer Turm, Energieturm, Himmelsbrücken, Kai, Galeone, Leuchtturm |
| `src/maps/weltenbrueck/index.js` | Terrain, Materialien, Vegetation, Übersichtskarte |
| `src/textures.js` | prozedurale Texturen mit Normal-Maps (Putz, Backstein, Pflaster …) |
| `src/batcher.js` / `src/instancer.js` | zusammengefasste Geometrie bzw. Instanzen mit Culling |
| `blender/m_city.py` | Stadt-Modelle (Vorderseite **-Y**, `__Paint`-Teile einfärbbar) |
| `blender/m_world.py` | Welt-Modelle (Vorderseite **+Y**, `key`-Farbe wird umgefärbt) |
| `blender/simready_*` | SimReady-Assets: Liste, Download, Umwandlung (Ablauf im README) |
| `docs/simready-credits.md` | Namensnennung für CC BY 4.0, bei neuen Assets ergänzen |

## Bauen und Testen

```bash
cd damm-raider
npm install
node tools/build.mjs                        # dist/damm-raider.html (eine Datei, offline spielbar)
npx --yes http-server -c-1 -p 8790 .        # in einem zweiten Terminal

# Browser-Tests (einmalig: npm i --no-save playwright && npx playwright install chromium)
node tools/tests/stats.mjs                  # Ladezeit, Draw-Calls, Dreiecke
node tools/tests/bewegung.mjs               # Türen, Treppen, Brücken, Spawns
node tools/tests/raid.mjs                   # Raids auf beiden Karten
node tools/tests/welt.mjs                   # Tore, äußere Viertel, Himmelsbrücken, Galeone, Strand
node tools/tests/abschnitt.mjs v04 [grau]   # Prüfansichten des Musterabschnitts, grau = nur Form
node tools/tests/details-stadt.mjs          # Detailbilder -> tools/tests/shots/
node tools/tests/details-damm.mjs
node tools/tests/simready.mjs nachher      # feste Kameras für Vergleichsbilder, Draw-Calls, Dreiecke
node tools/tests/fps.mjs 1920x1080          # Bildrate je Grafikstufe (installiertes Chrome, DSF=1 ohne Retina)
```

Ohne Server und ohne Playwright-Browser geht es auch: `GAME_URL=file://$PWD/dist/damm-raider.html PW_CHANNEL=chrome`
vor den Testaufruf setzen, dann nutzt der Test das installierte Chrome samt echter Grafikeinheit.

Blender-Modelle neu bauen und prüfen (Blender 4.5 als Python-Modul, siehe README):

```bash
.venv-blender/bin/python blender/build.py -- models     # assets/models.glb
.venv-blender/bin/python blender/preview.py -- moebel   # Vorschaubilder -> blender/preview/
```

Richtwerte der Stadt (Software-Rendering in der Cloud): ca. 670 Draw-Calls, 2,3 Mio. Dreiecke,
621 Häuser, 52 Ruinen. Bewegungstests: Treppen 60/60, Türen 59/60 (Haus 154 ist ein Durchlauf
durch die Hintertür, kein Fehler), Brücken 6/6, Spawns 11/11.

Richtwerte aus `simready.mjs` (Apple M1, feste Kamera, vorher → nachher SimReady): Damm 196 → 223 Draw-Calls,
0,74 → 0,94 Mio. Dreiecke; Stadt 872 → 899 Draw-Calls, 2,95 → 3,09 Mio. Dreiecke; Einzeldatei 7,8 → 14,2 MB.

## Bildrate (November 2026, gesamte Stadt)

| Gerät | Stufe | Damm | Stadt |
| --- | --- | --- | --- |
| RTX 4080, 1920 × 1080 | mittel | 103–137 fps | 66–76 fps |
| RTX 4080, 1920 × 1080 | hoch | nicht neu gemessen | 43–53 fps |
| Apple M1, 1440 × 900 Retina | niedrig | 65–93 fps | 33–55 fps |
| Apple M1, 1440 × 900 Retina | mittel | 21–27 fps | 14–20 fps |
| Apple M1, 1440 × 900 Retina | hoch | 5–7 fps | 5–6 fps |

Im Raid mit Gegnern, auf Bodenhöhe. M1 mit `tools/tests/fps.mjs` im normalen Spieltakt ohne Vsync, vor dem Ausblenden
ferner ARC. Die 4080 lief über die Chrome-Erweiterung in einem verdeckten Fenster: Bild für Bild warten, bis die GPU
fertig ist, ohne Bildausgabe. Dabei gab es Ausreißer nach oben (ein Durchlauf auf „hoch“ mit 85–114 fps), die Werte
schwanken also je nach Auslastung des PCs. Die Stadt ist CPU-lastig: rund 10 ms Hauptthread gegen 5 ms GPU auf „mittel“,
weil es noch rund 1.100 Draw-Calls sind. GTAO auf „hoch“ zeichnet die Szene ein weiteres Mal. Die Bildrate im echten
Fenster ist noch nicht geprüft. GTAO 0,85 und die Bloom-Werte in `LOOKS` wurden an Vergleichsbildern geprüft und bleiben.

## Nächste Schritte

**Häuser abschnittsweise verbessern** nach [hausstandard.md](hausstandard.md). Fertig: Weiße Stadt Nord (`lot.pilot`).
Nächster Schritt: übrige Weiße Stadt, danach die anderen Viertel.

**Feinheiten der äußeren Viertel** (Grobausbau steht): Galeone ist ein Kasten mit Masten, Leuchtturm und Energieturm
sind nicht begehbar, Hochhäuser haben schlichte Fassaden ohne eigene Sonderteile, die Himmelsbrücken sind Hängebrücken,
der Gründerwald ist nicht erreichbar, `docs/weltenbrueck-entwurf.png` zeigt noch Etappe 1.

1. **Echte Fotos als Vorlage** (Wikimedia Commons, Unsplash): Fachwerk- und Schwarzwaldhäuser,
   Bauernstuben, Blumenwiesen, verlassene Autos. Danach Modelle, Farben und Texturen angleichen.
2. **Schwächere Geräte entlasten:** Auf dem Apple M1 bleibt „mittel“ weit unter 60 fps (siehe Messwerte).
   Größte Hebel laut Messung: im Damm die Nachbearbeitung (Bloom, Farbkorrektur, MSAA), in der Stadt zusätzlich
   die rund 900 Draw-Calls. Die Stufen sollten dafür nicht schlechter aussehen als nötig.
3. **SimReady:** Natursteine, Felsen und Industrieschilder gibt es im Katalog nicht; das Schild ist ein
   „Wet Floor“-Aufsteller und bleibt ein Notbehelf. Der Sprossenstuhl steht nur in jedem dritten Stein- oder
   Backsteinhaus, die Fußbank nur in jedem vierten Haus, damit die Stadt nicht zu viele Dreiecke bekommt.
4. Offene Ideen: Krüppelwalmdächer, mehr Innenraum-Vielfalt (Herrgottswinkel, Bilder, Geschirrschrank),
   Stadttore öffnen, Autos auch in Weltenbrück, Licht und Schatten feiner abstimmen.

Commits gehen auf den Branch `claude/arc-raiders-browser-game-h181ag`.
