# Weiterarbeiten an Damm Raider

Übergabe-Notiz mit dem Stand, den Konventionen und den nächsten Schritten, damit eine neue
Sitzung (z. B. Claude Code auf dem eigenen Rechner) direkt weitermachen kann.

## Stand

- Zwei Karten: **Der Damm** (offene Talsperre) und **Weltenbrück** (Fantasy-Stadt mit deutscher
  Architektur, eng und verwinkelt, jedes Haus begehbar). Auswahl im Einsatz-Menü.
- Weltenbrück verbindet vier Viertel-Ideen zu einer eigenen Stadt: Hafen- und Seefahrer-Stimmung,
  Waldsiedlung auf Terrassen mit Riesenbäumen, ein Erfinderturm mit Energiekern und ein düsterer
  Gründerfelsen. Bewusst nur Archetypen – keine Figuren, Logos oder Nachbauten aus bestehenden Werken.
- Alle festen Formen (Möbel, Fassadenteile, Requisiten, Bäume, Autos, Waffen, Figuren, Gesichter)
  kommen aus Blender-Skripten. Wände, Böden, Treppen und Dächer der Stadthäuser erzeugt das Spiel aus
  dem Stadtplan, weil jedes Haus genau auf sein Grundstück passen muss.
- Letzte Runde: Oberflächen mit Normal-Maps, Fassadendetails, Stuben, Wiesen, Autos, Bäume.
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
| `src/maps/weltenbrueck/index.js` | Terrain, Materialien, Vegetation, Übersichtskarte |
| `src/textures.js` | prozedurale Texturen mit Normal-Maps (Putz, Backstein, Pflaster …) |
| `src/batcher.js` / `src/instancer.js` | zusammengefasste Geometrie bzw. Instanzen mit Culling |
| `blender/m_city.py` | Stadt-Modelle (Vorderseite **-Y**, `__Paint`-Teile einfärbbar) |
| `blender/m_world.py` | Welt-Modelle (Vorderseite **+Y**, `key`-Farbe wird umgefärbt) |

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
node tools/tests/details-stadt.mjs          # Detailbilder -> tools/tests/shots/
node tools/tests/details-damm.mjs
```

Blender-Modelle neu bauen und prüfen (Blender 4.5 als Python-Modul, siehe README):

```bash
.venv-blender/bin/python blender/build.py -- models     # assets/models.glb
.venv-blender/bin/python blender/preview.py -- moebel   # Vorschaubilder -> blender/preview/
```

Richtwerte der Stadt (Software-Rendering in der Cloud): ca. 670 Draw-Calls, 2,3 Mio. Dreiecke,
621 Häuser, 52 Ruinen. Bewegungstests: Treppen 60/60, Türen 59/60 (Haus 154 ist ein Durchlauf
durch die Hintertür, kein Fehler), Brücken 6/6, Spawns 11/11.

## Nächste Schritte

1. **Echte Fotos als Vorlage** (Wikimedia Commons, Unsplash): Fachwerk- und Schwarzwaldhäuser,
   Bauernstuben, Blumenwiesen, verlassene Autos. Danach Modelle, Farben und Texturen angleichen.
2. **Mit Grafikkarte testen:** echte Bildrate messen (Ziel 60 fps auf „mittel“), Optik prüfen.
3. Offene Ideen: Krüppelwalmdächer, mehr Innenraum-Vielfalt (Herrgottswinkel, Bilder, Geschirrschrank),
   Stadttore öffnen, Autos auch in Weltenbrück, Licht und Schatten feiner abstimmen.

Commits gehen auf den Branch `claude/arc-raiders-browser-game-h181ag`.
