# Damm Raider

Ein Extraction-Shooter für den Browser, inspiriert von **Arc Raiders**. Du steigst aus der unterirdischen
Siedlung Speranza an die Oberfläche, plünderst die Karte **„Der Damm“**, kämpfst gegen ARC-Maschinen und
musst es rechtzeitig zu einem Aufzug schaffen – sonst ist deine Ausrüstung weg.

Fanprojekt ohne Verbindung zu Embark Studios. Alle Modelle, Sounds und Texturen werden zur Laufzeit
erzeugt; es gibt keine externen Assets.

## Spielen

**Einfachste Variante:** `dist/damm-raider.html` im Browser öffnen (Doppelklick). Die Datei enthält
alles – auch three.js – und läuft ohne Server und ohne Internet (nur die Schriften kommen von Google Fonts).

**Entwicklung (ohne Build):**

```bash
cd damm-raider
npm install        # nur nötig für den Build
npm run dev        # startet http://localhost:8080
```

`index.html` lädt die ES-Module aus `src/` direkt und three.js über eine Import-Map vom CDN.

**Build der Einzeldatei:**

```bash
npm run build      # schreibt dist/damm-raider.html
```

Benötigt wird ein Desktop-Browser mit WebGL, Tastatur und Maus.

## Steuerung

| Taste | Aktion |
| --- | --- |
| W A S D / Maus | Bewegen / Umsehen |
| Linke / rechte Maustaste | Schießen / Zielen |
| Shift | Sprinten |
| Leertaste | Springen |
| C | Ducken (leiser, schwerer zu entdecken) |
| Q | Ausweichrolle |
| V | Nahkampf – schüttelt auch Zecken ab |
| R | Nachladen |
| 1 / 2 / Mausrad | Waffe wechseln |
| E (halten) | Behälter durchsuchen, Aufzug rufen |
| 3 / 4 / 5 | Verband / Sanitätskit / Schildlader |
| G | Splittergranate |
| Tab | Inventar (inkl. sicherer Tasche) |
| M | Karte (Klick setzt eine Markierung) |
| Esc | Pause |

## Spielablauf

1. **Speranza (Hub):** Ausrüstung zusammenstellen, im Lager verwalten, bei Händlerin Celeste kaufen und
   verkaufen, Rucksack und Lager ausbauen.
2. **Raid:** 20 Minuten auf dem Damm. Zufällige Bedingung: klarer Tag, Dämmerung oder Elektrosturm
   (schlechte Sicht, mehr ARC, bessere Beute).
3. **Extraktion:** An einem der vier Aufzüge die Konsole bedienen, 25 Sekunden durchhalten (der Lärm lockt
   ARC an), dann einsteigen und 5 Sekunden in der Kabine bleiben.
4. **Tod oder Zeitablauf:** Alles, was du dabei hattest, ist verloren – nur die sichere Tasche kommt ins
   Lager. Ohne Waffe gibt es eine zufällige Gratis-Ausrüstung.

Der Spielstand wird im `localStorage` des Browsers gespeichert.

## Die Karte: Der Damm

- **Talsperre 7** mit begehbarer Dammkrone, Portalkran, Torhäusern und Überlauf mit Wasserfall
- **Stausee** im Norden (man kann schwimmen, aber nicht schießen)
- **Kraftwerk** am Fuß der Mauer mit Turbinen, Druckrohren und Galerie
- **Kontrollturm** mit Brücke auf die Dammkrone
- **Forschungsanlage**, **Wasseraufbereitung**, **Umspannwerk**, **Blasse Wohnblöcke**
- **Giftsumpf** mit Stelzenhütten – das Wasser verätzt
- **Altes Schlachtfeld** mit dem Wrack einer riesigen ARC-Königin
- **Hydrokuppeln** und **Pumpstation** auf dem Plateau am See
- Vier Aufzüge: West, Süd, Ost und auf dem Damm

## ARC-Gegner

| ARC | Verhalten | Schwachstelle |
| --- | --- | --- |
| Zecke | lauert, springt dich an und hängt sich fest | Auge |
| Knaller | rollt heran und explodiert (auch bei Beschuss – Kettenreaktionen!) | – |
| Wespe | schnelle Drohne, feuert Salven | Rotoren (2 zerstört = Absturz) |
| Hornisse | gepanzerte Front, schwere Energiebolzen | hintere Antriebe |
| Späher | ruft Verstärkung, wenn er dich zu lange sieht | Scanner |
| Wächter | Geschütz auf Dächern, Laser-Zielerfassung, dann ein harter Schuss | Linse |
| Raketier | schwerer Flieger mit Raketensalven | Düsen |
| Springer | großer Vierbeiner, springt dich an, Flächenschaden bei der Landung | Kernauge |
| Bastion | Kampfläufer mit Miniguns und Stampfattacke | Rückenkern |

Zerstörte ARC hinterlassen Wracks mit wertvollen Teilen.

## Projektstruktur

```
damm-raider/
├── index.html          HTML-Gerüst (Hub, HUD, Fenster)
├── css/style.css       Oberfläche
├── src/
│   ├── main.js         Einstieg
│   ├── game.js         Spielablauf, Treffer, Explosionen, Extraktion
│   ├── world.js        Szene, Licht, Wetter, Wasser, Vegetation
│   ├── terrain.js      Höhenfeld (Tal, Plateau, Stausee, Fluss, Sumpf)
│   ├── map.js          Aufbau der Dammkarte
│   ├── structures.js   Gebäude-Baukasten
│   ├── batcher.js      Zusammenfassen statischer Geometrie
│   ├── collision.js    Kollision und Raycasts
│   ├── player.js       Spielfigur, Kamera, Waffen, Heilung
│   ├── enemies.js      ARC-KI
│   ├── arcmodels.js    ARC-Modelle
│   ├── projectiles.js  Geschosse, Raketen, Granaten
│   ├── loot.js         Behälter und Beutetabellen
│   ├── props.js        Aufzüge, explosive Fässer
│   ├── items.js        Gegenstände und Waffenwerte
│   ├── hud.js / ui.js  HUD und Menüs
│   └── audio.js        synthetisierte Sounds
└── tools/build.mjs     Einzeldatei-Build (esbuild)
```
