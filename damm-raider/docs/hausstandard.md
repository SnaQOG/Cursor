# Hausstandard

Wie Häuser in Damm Raider verbessert werden. Der Ablauf stammt aus dem Wano-Projekt (Musterhaus v18,
Straßenstück v04 in Google Drive) und ist für ein Spiel im Browser angepasst: Dort wurde offline in Blender
gerendert, hier muss alles im Spieltakt laufen und jedes Haus begehbar bleiben.

## Ablauf je Abschnitt

1. **Ausgangsstand festhalten:** `node tools/tests/abschnitt.mjs v00` und dasselbe mit `grau`.
2. **Form in Grau prüfen:** Straße auf Augenhöhe, von oben, Fassade, Hof. Ohne Texturen fällt auf, was wirklich
   fehlt. Erst wenn das stimmt, kommt Material.
3. **Fehlerliste schreiben** und je Version abarbeiten. Was nicht besteht, wird verworfen und nicht übertüncht.
4. **Material in echter Größe:** Ziegel, Pflaster und Putz müssen zur Körpergröße passen.
5. **Spielprüfung:** `bewegung.mjs`, `welt.mjs`, `raid.mjs`, danach `fps.mjs`. Türen, Treppen und Spawns dürfen nicht leiden.
6. **Erst dann ausrollen:** auf den nächsten Abschnitt und nur mit Varianten, nicht als Kopie.

Ein Ergebnis wird nie als fotorealistisch bezeichnet, solange es sichtbar ein 3D-Modell ist.

## Was ein Haus braucht

- **Tiefe statt Fläche:** Fenster sitzen in der Laibung, dazu Faschen, Fensterbank, Gurtgesims, Traufgesims, Sockel.
- **Dach mit Dicke:** Überstand, Sparrenköpfe, Rinne, Fallrohr, Ortgang, First, Gauben.
- **Verwandte Typen statt einer Schablone:** Bürgerhaus (Eckquader, Verdachungen), Giebelhaus (First quer, Lisenen),
  Erkerhaus. Der Typ folgt aus `lot.id % 3`.
- **Alter je Haus:** `lot.age` bestimmt Schmutz am Sockel, Schatten unter der Traufe und Gesamthelligkeit.
- **Bodenanschluss:** Steinstreifen am Sockel, Pflanzen an geschützten Stellen, Eingänge bleiben frei. Unkraut wächst
  an Mauern und nicht mitten auf der Straße.
- **Keine leeren Löcher:** Jedes Fenster hat einen Rahmen, auch im Giebel.

## Protokoll Weiße Stadt Nord

| Version | Ergebnis | Fehler, die zur nächsten Version führten |
| --- | --- | --- |
| v00 | Ausgangsstand | glatte Fassaden, dünnes Dach, blaue Schatten, Putz mit Kreisen und Rechtecken, Riesenpflaster, Gras auf der Straße, Fenster ohne Rahmen, nur ein Haustyp |
| v01 | Form und Licht besteht | Putz beige statt weiß, Dach zu orange, Ziegel zu klein, Cartoon-Büsche, Pflaster zu dunkel, alter Putz hinter offenen Türen |
| v02 | Material besteht | schwarze Giebelfenster, Putz zu fleckig, Erker nicht geprüft |
| v03 | Giebelfenster und Putz bestanden | Erker ist flacher Kasten mit schwarzem Loch, Efeu wirkt wie schwarze Silhouette |
| v04 | Erker mit echten Öffnungen, Efeu vorerst entfernt, im Spiel geprüft | offen, siehe unten |

Offen im Abschnitt, jeweils außerhalb der Häuser oder als nächste Runde:

- Low-Poly-Felsen am Gründerfelsen und Cartoon-Bäume brechen den Stil. Sie sind der nächste große Hebel.
- Fensterläden sind flach und kräftig gefärbt, Türen noch schlicht.
- Innenräume sind unverändert dunkel.
- Bäume, Häuser und Fenster wiederholen sich weiterhin erkennbar. Kein fertiger Fotorealismus.
- Bildrate im Abschnitt nach den neuen Texturen noch nicht auf der 4080 nachgemessen.
