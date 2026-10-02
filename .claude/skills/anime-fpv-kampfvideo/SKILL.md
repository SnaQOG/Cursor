---
name: anime-fpv-kampfvideo
description: Baut ein 20-s-FPV-Video in Blender (bpy, Cycles) durch eine Anime-Welt mit Kampf, nach dem Vorbild des Dragon-Ball-Namek-Videos (Goku gegen Freezer). Enthält den Ablauf, den vollständigen Parameter-Katalog (alle Stellschrauben, ohne Werte), Pipeline-Skripte für Vorschau und fortsetzbaren Final-Render sowie das Werte-Archiv für die spätere Analyse. Immer nutzen, wenn Dominik ein weiteres Dragon-Ball-, Anime- oder Kampf-FPV-Video, eine neue 3D-Welt mit Figuren und Effekten, eine Fortsetzung oder Variante von Namek, ein Strahlenduell, Tripo-Figuren in einer Blender-Szene oder „wie beim Namek-Video“ erwähnt, auch wenn das Wort Skill nicht fällt.
---

# Anime-FPV-Kampfvideo (Vorlage: Namek)

Ein Video dieser Art ist eine einzige durchgehende FPV-Aufnahme. Erst kommt ein schneller Flug durch eine Anime-Welt
mit einem Reveal, dann ein Kampf im Originaltempo der Serie mit Treffern, Teleports, Strahlen und einer Explosion
als Klimax. Dieser Skill hält fest, **welche Parameter** ein solches Video bestimmen und wie man sie festlegt. Die
**Werte** gehören nicht hierher, sondern zum jeweiligen Video:

- Jedes Video bekommt seine Werte neu (aus Briefing, Vorlagenbildern und Vorschau-Feedback).
- Nach dem Final werden die Werte im **Werte-Archiv** gespeichert, mit denselben Parameter-IDs wie im Katalog. So
  lassen sich später mehrere Videos vergleichen, z. B. Kameratempo gegen Wirkung oder Schlagdauer gegen Reaktion.

## Wo liegt was
- **Code:** GitHub `SnaQOG/Cursor`, Ordner `drohnen_3d_welten/` (Branch `claude/eloquent-archimedes-og1asl`).
  - Referenz-Umsetzung: `world_dragonball.py` (Namek, Final-Stand Commit `ef59ba2`).
  - Bausteine: `fpv.py` (Render, Kamera, Gelände), `choreo.py` (Timing, Backen, Hit-Stops), `dbz_fx.py`
    (Ki-Spuren, Zanzoken), `vfx.py` (Treffer, Strahlen, Aura, Explosion, Partikel), `tripo_chars.py`
    (Tripo-Figuren riggen), `namek.py` (Namek-Objekte, Gras), `nature.py`, `ocean.py`, `post.py`
    (Grade und Encoding), `params/<welt>.py` (Nachbearbeitung).
- **Parameter-Katalog:** `references/parameter_katalog.md`, alle IDs mit Bedeutung, Einheit und Code-Ort. Lies ihn
  beim Planen eines neuen Videos vollständig.
- **Werte-Vorlage:** `references/werte_vorlage.json`, alle IDs mit leeren Werten.
- **Werte-Archiv:** `drohnen_3d_welten/werte/<NN_name>/` im Repo, mit `werte.json` (IDs → Werte),
  `kampfplan_und_kamera.json` (alle Schlüssel) und `werte.md` (lesbare Fassung). Erster Eintrag: `03_namek`.
  Die Werte bleiben für die Auswertung dort und werden nicht in diesen Skill übernommen.
- **Pipeline:** `scripts/pipeline/`
  - `build_scene.py`, `preview.py`, `render_final.py`: Szene bauen, Vorschau, fortsetzbarer Final-Render
  - `run_final.sh`, `watchdog.sh`: Final-Kette und Überwachung bei Container-Neustarts
  - `web_version.sh`: Webversion unter 30 MB
  - `werte_dump.py`: Kampfplan und Kamera als JSON ins Werte-Archiv
- **Modelle:** Tripo-GLBs liegen nicht im öffentlichen Repo. Sie gehören nach
  `drohnen_3d_welten/assets/models/<franchise>/<figur>/`; Dominik hat sie im Obsidian-Vault unter
  „3D Modelle“. Texturen lädt `fetch_assets.sh`.

## Ablauf
Die Phasen folgen den Abschnitten des Katalogs. In jeder Phase legst du die Werte der genannten IDs fest und
notierst Begründungen, denn die braucht die spätere Analyse.

1. **Briefing und Zeitleiste** (`format.*`, `ablauf.*`): Welt, Figuren und Wow-Moment klären; vorhandene Vorlagen
   (Anime-Bilder, Model Sheets) ansehen und Farben per Pixelmessung übernehmen. Schlag eine Zeitleiste vor
   (Flugteil mit Hook, Reveal hinter einer Verdeckung, Kampfteil mit Klimax) und lass sie bestätigen, bevor du
   baust.
2. **Welt** (`gelaende.*`, `objekte.*`, `wasser.*`, Abschnitt 5): Landmarken so setzen, dass jedes Kamera-Manöver
   an einem festen Objekt passiert (Slalom an Felsnadeln, Steigflug an einer Wand, Reveal über eine Kante).
   Vegetation nie in die Flugbahn und nie in die Sichtlinie zum Kampf.
3. **Licht und Himmel** (`licht.*`, `himmel.*`, `wolken.*`): Hauptsonne seitlich-hinten für Modellierung,
   Gegenlicht für die Tiefe. Diffuses Himmelslicht entsättigen, wenn der Himmel stark gefärbt ist, sonst färbt
   er alle Materialien ein.
4. **Kamera** (`kamera.*`): Wegpunkte mit Uhrzeit, daraus ergibt sich das Tempo. Brennweite fest, Hochformat.
   Im Kampf folgt der Blick den geglätteten Kämpferbahnen, Phasenziele werden weich überblendet.
   Kamerastöße und Halte gehören auf die Treffer.
5. **Figuren** (`figur.*`):
   - Tripo-GLB mit Mixamo-Auto-Rig und eingebetteter Textur; unvollständige Safari-Downloads mit `repair_glb.py`
     retten.
   - `tripo_chars.rig_model(...)` liefert die Figur auf dem Standard-Rig, alle Posen der Posenbibliothek gelten.
   - Vor dem Einbau einen Lineup- und Posentest rendern und ansehen: Risse, mitgezogene Gegenstände an den
     Händen, Ärmel.
6. **Kampfplan** (`kampf.*`):
   - Blocking mit `_key(t, Pose, Brust-Ort, Blickziel, …)`; Dragon-Ball-Tempo über `choreo.dbz_style`.
   - Angriffe extrem kurz, Ausholen über eigene *_wind-Posen, Hit-Stops auf Treffern, Vorstöße mit `ease="lin"`
     bis zum Kontakt, Rückstöße mit `"out"`.
   - Prüfe jede Aktion im Bild (NDC-Projektion je Frame, Kontaktblätter). Im Hochformat Aktionen senkrecht staffeln.
7. **Effekte** (`fx.*`): Ki-Spuren hinter schnellen Bewegungen, Zanzoken (verschwinden, Nachbild, Luftring),
   Treffer-Bursts mit Funken, Einschlag mit Krater und Bruchstücken, Strahlen, Duell, Explosion als prozedurales
   Volumen. Effekt-Meshes nach ihrer Lebenszeit ausblenden.
8. **Vorschau**: Standbilder an Schlüsselmomenten, dann die komplette Sequenz in kleiner Auflösung. Dominik sieht
   immer das **ganze** Video, bevor das Final startet. Seine Rückmeldungen gehen unter `bewertung.korrekturen`
   ins Archiv.
9. **Final** (`render.*`, `post.*`, `encode.*`):
   - `run_final.sh` mit Umgebungsvariablen starten (fortsetzbar).
   - `watchdog.sh` als Hintergrund-Task mit dem maximalen Timeout dazu.
   - Check-ins (send_later) im Abstand kürzer als die Laufzeit des Watchdogs. Bei jedem Check-in den Render
     prüfen und den Watchdog neu starten.
10. **Lieferung:**
    - Webversion unter 30 MB schicken (Bitrate aus Dauer × Bitrate / 8 < 30 MB).
    - Master nach `drohnen_3d_welten/videos/` (Commit, Push).
    - Obsidian-Update als ZIP, Struktur `Drohnen Videos/3D Welten Blender/<NN Name>/{Test,Final}`.
11. **Werte-Archiv:**
    - `werte_vorlage.json` nach `drohnen_3d_welten/werte/<NN_name>/werte.json` kopieren und alle IDs ausfüllen,
      auch Messwerte (Renderzeiten, Neustarts) und Bewertung.
    - `werte_dump.py` schreibt Kampfplan, Kamera und Ereignisse exakt aus dem Code dazu.
    - Committen und als ZIP an Dominik geben.

Skripte aufrufen (Beispiele, Werte je Video):
```bash
python scripts/pipeline/preview.py <code> world_xyz <code>/params/xyz.py <out> stills 1.5,10.3,15.8 <BxH> <spp>
WORK=… CODE=… MODUL=world_xyz PARAMS=… RES=<BxH> SPP=<n> SIZE=<BxH> NAME=<Video> scripts/pipeline/run_final.sh
python scripts/pipeline/werte_dump.py <code> world_xyz drohnen_3d_welten/werte/<NN_name>/kampfplan_und_kamera.json
```

## Hausregeln (Zusammenarbeit mit Dominik)
- Auf Deutsch antworten, genaue Werte nennen; was nicht geht, offen sagen statt faken.
- **Nichts generieren** (Higgsfield, Kling, Seedance o. Ä.) ohne ausdrückliche Erlaubnis; Seedance ist zu teuer.
- Erst die komplette Vorschau zeigen, dann das Final starten.
- Modelle nie ins öffentliche Repo committen.
- Der Obsidian-Vault „dd brain“ ist aus der Cloud nicht erreichbar, deshalb ZIP-Pakete liefern; jede Datei unter
  30 MB.
- Lange Renders nacheinander statt parallel (4 CPU-Kerne). Parallel verzögert das erste Video und beschleunigt das
  zweite nicht.

## Erkenntnisse aus Namek (gelten weiter)
- **Tripo-GLBs** sind an den UV-Nähten in über 1000 Inseln zerlegt. Vor der Gewichtsrechnung Nähte zusammenlegen
  (`rig_model` macht das), sonst reißt Kleidung.
- **Gegenstände an der Hand** werden beim Armheben mitgezogen: diese Arme ruhig lassen oder Posen je Figur anpassen.
- **Transparente Effekt-Meshes**, die nach ihrer Lebenszeit in der Szene bleiben, stapeln sich über das Limit der
  transparenten Bounces und rendern schwarz. Ausblenden.
- **Luftringe** mit Brechung erzeugen dunkle Ringe: Brechungsindex neutral setzen.
- **Volumen** (Explosion) schreiben unendliche Tiefe in den Mist-Pass: Dunst an Volumenpixeln aus der Post nehmen
  bzw. ausblenden.
- **Mantaflow** bricht in diesem bpy-Build ab: Explosion als prozedurales Volumen.
- **Baumkronen** zwischen Kamera und Kämpfern verdecken Treffer: Sichtkorridor freihalten (mit Kronenradius).
- **Nachbild** (Zanzoken) als Kopie mit UVs (`preserve_all_data_layers`), sonst ist es schwarz.
- **Container** werden bei Leerlauf und auch zwischendurch neu gestartet: Renders fortsetzbar halten, Watchdog,
  Check-ins.
- **Prozesssuche:** `pgrep -f` mit einem Muster, das in der eigenen Befehlszeile steht, findet sich selbst. Mit
  `[x]yz`-Trick suchen oder per PID arbeiten. PID-Dateien nach Neustarts prüfen.
- Unvollständige Safari-Downloads (`.glb.download`) sind reparierbar (`repair_glb.py`).

## Auswerten (später)
Alle `drohnen_3d_welten/werte/*/werte.json` laden und je ID vergleichen, z. B. `kamera.tempo` gegen
`bewertung.reaktion`. Numerische IDs eignen sich für Tabellen, Listen-IDs (Zeitpläne) für Zeitachsen-Vergleiche.
Neue Parameter zuerst in den Katalog und die Vorlage aufnehmen, dann in den Videos füllen. So bleiben die IDs über
alle Videos gleich.
