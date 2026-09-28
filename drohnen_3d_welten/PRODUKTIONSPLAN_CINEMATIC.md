# Produktionsplan „Cinematic Quality“ – One Piece · Naruto · Namek

Stand: Blender **5.0.1 / Cycles**. Alle drei Welten werden per Python in `drohnen_3d_welten/` gebaut
(`world_*.py`, Module `ocean.py`, `nature.py`, `sunny.py`, `konoha.py`, `namek.py`, `vfx.py`, `post.py`).
Jeder Punkt unten ist als Blender-Setup beschrieben (Modifier-Stack, Node-Tree, Parameter) und nennt,
wo er in unserer Pipeline landet.

Notation für Node-Trees:
`[Node] Parameter → [Node].Eingang` · Farben als **lineare RGB-Werte** (so, wie sie in den Node-Feldern
stehen), in Klammern der ungefähre sRGB-Hex-Wert des Farbwählers.

---

## 0. Befund, Grundsatzentscheidungen, Budget

### 0.1 Ist-Stand gegen Ziel

| Bereich | Heute in der Pipeline | Was fehlt für Kino-Look |
|---|---|---|
| Ozean One Piece | FFT-Ocean-Modifier (animiert), GN-Kielwasser (Kelvin 19,47°), Rumpf- und Brandungsschaum (`ocean.py`) | Wasser ist ein **undurchsichtiger** Principled-Shader ohne Transmission/Tiefe, kein Meeresboden, keine Bugwellen-**Geometrie**, keine Caustics |
| Felsen | prozedurale Objekt-Koordinaten (keine UVs), Nässe-Linie über Welt-Z (`nature.rock_material`), Mesh aus `fpv.rock_mesh` | Die „gestreckten“ Streifen kommen aus den **senkrechten Karstrinnen** im Mesh-Generator (hohe Frequenz um den Umfang, niedrige in der Höhe) – kein UV-Problem. Es fehlen Bildtexturen, echte Displacement-Details, Brandungskehle, animierte Nässe |
| Sunny | Holzmaserung + Kantenabrieb über Pointiness, Lack mit Coat, Bevel-Stack (`sunny.py`) | Segel sind **statische** Meshes, Flaggen nur Wave-Modifier; keine Kratzer-/Planken-Variation pro Brett |
| Hokage-Felsen | **Scan-Kopf** (Lee Perry-Smith) + modelliertes Haar, per Voxel-Remesh verschmolzen | Genau das ist der „Museumskopf“: realistische Anatomie statt Anime-Formensprache |
| Konoha | Häuser als Boxen ohne Bevel, Straße als flache Fläche mit Bildtextur, Stromleitungen (Kurven) | Keine Mikro-Displacement-Straße, keine gefasten Kanten, Fenster ohne Tiefe, kein Clutter |
| Namek-Himmel | eigener Verlaufs-Shader + Light-Path-Trick (diffuses Licht entsättigt) | Verlauf endet in Grün statt Türkis, 2D-Wolken in der World |
| Namek-Felsen | gleiche Karstrinnen wie One Piece | Namek braucht **waagerechte** Sedimentbänke statt senkrechter Rinnen |
| Namek-Gras | 1,5 Mio. Halm-Dreiecke als Mesh | Kein Clumping/Wind; Hair Curves sind speicherschonender |
| Wasser Namek | wie One Piece, grün getönt, undurchsichtig | Tiefe/Absorption, Meeresboden |
| VFX | Emissions-Hüllen, Punktlichter, Druckwellenringe (`vfx.py`) | Funken mit Motion Blur, Hitzeflimmern, Linienlicht, Light Linking |
| Post | eigener numpy-Post: Dunst aus Mist-Pass, Auto-Belichtung, AgX, Bloom, Grain, Vignette (`post.py`) | DoF (bewusst nicht – siehe 0.2), Heat-Displace, Grain-Plate |

### 0.2 Korrekturen an der Aufgabenstellung (Blender 5.0 und Physik)

1. **Musgrave Texture gibt es seit Blender 4.1 nicht mehr.** Alle Musgrave-Varianten stecken im
   *Noise Texture* → `Type`: fBM, Multifractal, Ridged Multifractal, Hybrid Multifractal, Hetero Terrain
   (mit den Eingängen Offset/Gain). Alle Rezepte unten verwenden das.
2. **„Nishita“ heißt in 5.0 `Single Scattering`**, neu ist **`Multiple Scattering`** (physikalisch
   genauer, wir verwenden es schon). Parameter: Air, Aerosol (früher Dust), Ozone, Altitude, Sun Disc.
3. **Adaptive Subdivision ist in 5.0 nicht mehr experimentell** und sitzt direkt im
   Subdivision-Surface-Modifier (`Adaptive Subdivision` ✓, Space `Pixel`, Pixel Size). Globale Werte:
   Render → Subdivision: Dicing Rate, Max Subdivisions, Offscreen Scale.
4. **Kein eigener Fresnel-Node fürs Wasser**: der Principled BSDF berechnet Fresnel aus der IOR.
   Ein zusätzlicher Fresnel-Mix würde die Reflexion doppelt zählen. Fresnel/Layer Weight nur für
   Masken (Schaum, Rand-Glühen).
5. **Subsurface Scattering an der Wasseroberfläche ist physikalisch falsch.** Das „milchige“
   Leuchten flachen Wassers ist Volumenstreuung. Richtig: Volume Absorption (+ sparsam Volume Scatter)
   oder der günstige Ray-Length-Trick (1.3).
6. **EEVEE und Cycles lassen sich nicht in einem Bild mischen.** Damit der Ki-Strahl Felsen und Wasser
   physikalisch aufhellt, muss er in derselben Engine wie die Welt rendern → **Cycles**.
7. **Schärfentiefe vs. FPV:** Unsere Hausregeln verlangen den Look einer stabilisierten Action-Cam
   (kleiner Sensor, große Schärfentiefe, kein Kino-Bokeh). „Cinematic“ heißt hier: Kino-Qualität bei
   Assets, Shading und Licht – **nicht** Kino-Optik. DoF daher höchstens f/16-äquivalent (Abschnitt 4).
8. **Volumen kosten in unserer Szene extrem viel:** Heterogene Volumen (Rauschen in der Dichte) haben
   die Namek-Frames bei vielen Punktlichtern **4,5× langsamer** gemacht (gemessen: 20 s → 90 s pro
   Vorschau-Frame), weil jeder Schattenstrahl durch das Volumen muss. Regel: homogene Absorption ist
   billig, Streuung und Rausch-Dichte nur gezielt und mit eingeschränkter Ray-Visibility (3.2).
9. **Dynamic Paint und ein fahrendes Schiff:** funktioniert, aber der Canvas muss sehr fein aufgelöst
   sein (≤ 10 cm) und wird dadurch groß; für einen 20-s-Flug mit fahrender Sunny ist das analytische
   GN-Kielwasser (haben wir) kontrollierbarer. Beides steht in 1.1.

### 0.3 Render-Budget

Heute: 720×1280, 10–12 Samples + OIDN, 4 CPU-Kerne → **20–45 s pro Frame** (480 Frames ≈ 3–6 h pro Video).
Die Maßnahmen unten kosten grob zusätzlich:

| Maßnahme | Faktor Renderzeit | Empfehlung auf 4 CPU-Kernen |
|---|---|---|
| Wasser mit Transmission + homogener Absorption + Meeresboden | ×1,3–1,6 | ja |
| Volume Scatter im Wasser (niedrige Dichte) | ×1,5–2 | nur Namek-Nahbereich |
| MNEE-Caustics | ×1,3 | optional, sonst Fake-Caustics |
| Adaptive Displacement (Fels, Straße) | ×1,2–1,5 + RAM | ja |
| Cloth-Segel/-Flaggen | Bake einmalig, Render ≈ ×1 | ja |
| Volumen-Wolken (voll) | ×3–5 | **nein** → Sky-Plate oder eingeschränkte Ray-Visibility (3.2) |
| Hair Curves statt Halm-Mesh | ≈ ×1, weniger RAM | ja |
| 1080×1920 nativ + 64 Samples | ×10–15 | nur mit GPU (Mac/Metal, siehe README) |

---

## 1. PROJEKT 1 – One Piece: Thousand Sunny & Karstfelsen-Ozean

### 1.1 Ozean & Interaktion mit dem Rumpf

**Grundfläche `Sea_Near`** (heute `ocean.make_ocean`, 4×5 Kacheln à 100 m):

| # | Modifier | Einstellungen |
|---|---|---|
| 1 | **Dynamic Paint** (Canvas) – nur Variante A | siehe unten |
| 2 | **Ocean** | Geometry *Displace*, Resolution Viewport 8 / **Render 14–16**, Spatial Size 100 m, Depth 200 m, Wave Scale 0.9, Choppiness 1.3, Wind Velocity 9.5 m/s, Alignment 0.4, Direction −30°, Damping 0.5, **Generate Foam** ✓, Foam Coverage 0.25, Foam Data Layer `foam`, Time = `frame/24` (Driver), Repeat X/Y passend zur Kachel |
| 3 | **Geometry Nodes „Wake_GN“** – Variante B (unser Weg) | siehe unten |
| 4 | **Solidify** (nur für Volumen-Wasser, 1.2) | Thickness 40 m, Offset −1, Fill Rim ✓, Even Thickness ✓ → geschlossener Wasserkörper |

Reihenfolge wichtig: Dynamic Paint **vor** Ocean, damit die Wake-Höhen auf die FFT-Wellen addiert
werden und der Solver auf einer ruhigen, flachen Fläche rechnet (auf verschobener Geometrie wird er
instabil).

**Variante A – Dynamic Paint (Bugwelle + Heckwasser als Wellensimulation)**

1. Eigene Canvas-Fläche `Sea_Wake` statt der ganzen Ozeankachel: Grid 90 × 40 m entlang des Kurses der
   Sunny (sie fährt 60 m in 20 s), **Kantenlänge 0,08–0,10 m** (≈ 450k Vertices). Unterteilung direkt im
   Grid, kein Subdivision-Modifier (Dynamic Paint rechnet auf den Vertices *vor* nachfolgenden
   Modifiern).
2. Modifier-Stack `Sea_Wake`: `[Dynamic Paint Canvas] → [Ocean] (gleiche Werte wie Sea_Near, damit die
   Ränder passen) → [Solidify]`. Ränder mit `Sea_Near` überblenden (Vertex-Gruppe „edge_fade“ als
   Faktor für den Wellen-Output).
3. Canvas → Surface 1 **„Waves“**: Surface Type *Waves*, Format *Vertex*, Frames 1–528 (48 Frames
   Vorlauf), Substeps 2; Wave: **Open Borders** ✓, Timescale 1.0, Speed 0.8, Damping 0.04,
   Spring 0.20, Smoothness 1.0.
4. Canvas → Surface 2 **„Foam“**: Surface Type *Paint*, Output → Vertex Color `dp_foam`;
   Dissolve ✓ 90 Frames (Slow ✓), Spread ✓ Speed 0.6, Color Spread 1.0.
5. **Brush-Proxy** `Hull_Proxy`: vereinfachter Rumpf (Convex Hull der Unterwasser-Hülle, ~2k Faces,
   Ray Visibility aus), an `ThousandSunny_body` geparentet. Dynamic Paint Brush:
   - für „Waves“: Paint Source *Mesh Volume + Proximity*, Distance 0.4 m, Wave Type **Obstacle**,
     Factor 1.0, Clamp Waves 0.6 m.
   - für „Foam“: Paint Source *Proximity*, Distance 1.2 m, Falloff *Smooth*, Paint Color Weiß, Alpha 1.
6. Bake: Cache → Bake (Frames 1–528), Render ab Frame 49.

**Variante B – Geometry Nodes „Wake_GN“ (empfohlen, erweitert unser `ocean.ocean_fx_gn`)**

Physik: Tiefwasser-Kelvinwinkel 19,47°; Querwellenlänge λ = 2π·v²/g → bei v = 3 m/s **5,76 m**
(Wellenzahl k = g/v² = 1,09 rad/m).

```
[Object Info] (Sunny_body, Relative) .Transform → [Invert Matrix]
[Position] → [Transform Point] (Matrix = invertiert)           → p (Schiffskoordinaten: x Bug, y Backbord)
s = −(p.x − X_STERN)                                             Abstand hinter dem Heck (m)
Kelvin-Arme:  a = | |p.y| − tan(19.47°)·s |  → [Math Exponent] exp(−a²/1.2²)          → arm
Querwellen:   q = sin(1.09·s + t·ω)·exp(−(|p.y|/(0.35·s+2))²),  ω = √(g·k)=3.27 rad/s   → transverse
Dämpfung:     d = 1/√(1 + s/8)  ·  [Map Range] s ≥ 0 → 1, sonst 0
Heckwelle z_w = (0.28·arm·sin(1.6·s − ω·t) + 0.18·transverse) · d
Bugwelle:     r = Länge(p.xy − (X_BUG,0)) − Rumpfbreite(p.x) → Stauwulst z_b = 0.45·exp(−(r/0.9)²)·[p.x > X_BUG−6]
[Set Position] Offset z = z_w + z_b
[Store Named Attribute] "wake" (Float, Point) = clamp(1.6·z_b + 0.9·arm·d + Rumpfschaum, 0, 1)
```

Bugwellen-Gischt als Partikel: `[Distribute Points on Faces]` auf dem Bugwulst (Dichte ∝ z_b, max
400/m²) → Simulation Zone (Geschwindigkeit nach außen/oben 2–4 m/s, Schwerkraft, Lebensdauer 0,6 s,
Attribut `velocity` für Motion Blur) → `[Instance on Points]` Tröpfchen r 1–3 cm.

### 1.2 Wasser-Shader (Cycles)

```
─ Zeit & Koordinaten ─────────────────────────────────────────────
[Geometry].Position ─┐
[Value "TIME"] ──────┴→ [Vector Math: Add] (Drift = TIME·(0.8, −0.5, 0.35)) → P_t

─ Schaum-Maske ──────────────────────────────────────────────────
[Attribute "foam"] (Ocean) ─┐
[Attribute "wake"] (GN) ────┼→ [Math: Maximum] ×2 → F_raw
[Color Attribute "dp_foam"] ┘ (nur Variante A)
[Noise Texture] fBM, Vector P_t, Scale 1.8, Detail 6, Roughness 0.62 → n
[Math: Subtract] F_raw − 0.45·(1 − n) → [Map Range] Smoother Step, From 0.05–0.35 → 0–1 → F

─ Wasser-BSDF ──────────────────────────────────────────────────
[Noise Texture] fBM, P_t·12, Detail 3 → [Bump] Strength 0.08, Distance 0.02 → N_ripple
[Principled BSDF]
    Base Color (1, 1, 1)                  (Farbe kommt aus dem Volumen)
    Roughness 0.025 (+ [Noise]·0.02)
    IOR 1.333, Specular IOR Level 0.5
    Transmission Weight 1.0
    Normal ← N_ripple
─ Schaum-BSDF ──────────────────────────────────────────────────
[Principled BSDF] Base (0.80, 0.85, 0.87) (#E7EEF0), Roughness 0.55,
    Subsurface Weight 0.25, Subsurface Radius (0.2, 0.2, 0.2), Transmission 0
[Mix Shader] Fac = F, 1 = Wasser, 2 = Schaum → Material Output.Surface

─ Volumen (Grand-Line-Tiefe) ───────────────────────────────────
[Volume Absorption] Color (0.16, 0.58, 0.66) (#70C9D4), Density 0.15
[Volume Scatter]    Color (0.20, 0.62, 0.72), Density 0.004, Anisotropy 0.35   (optional)
[Add Shader] → Material Output.Volume
```

Warum diese Absorption: Cycles rechnet σ = Density·(1 − Color). Mit den Werten oben bleibt nach
10 m Weg (0.28, 0.53, 0.60) Licht übrig (türkis), nach 30 m (0.02, 0.15, 0.22) (tiefes Blau).
Genau der Grand-Line-Verlauf von hell über Sandgrund zu dunkel über Tiefe.

Render → Light Paths: Transmission 8, Volume 1 (0, wenn nur Absorption), Transparent 16,
Caustics: Refractive ✓, Reflective ✗, **Filter Glossy 1.0**. Voraussetzung: Solidify-Modifier
(geschlossener Körper) und ein **Meeresboden** (1.3); ohne Boden macht Absorption das Wasser nur dunkel.

### 1.3 Meeresboden & Caustics

- `Seabed`-Grid 400 × 500 m: Höhe aus dem vorhandenen Küsten-Abstandsfeld (`ocean.shore_distance_image`):
  −1,5 m an den Felsen, −25 m in 60 m Entfernung (Smoothstep). Material: heller Korallensand
  (0.55, 0.50, 0.40) mit Rippeln (Wave Bands, Scale 1.2 m, Distortion 2) und Seegras-Flecken nahe der Felsen.
- **Günstige Tiefenfarbe ohne Volumen (Ray-Length-Trick, für 4 CPU-Kerne empfohlen):** Wasser-Material
  ohne Volume, dafür im Boden-Material:
  ```
  [Light Path].Ray Length → [Vector Math: Scale] σ = (0.126, 0.063, 0.051) → [Vector Math: Multiply] (−1,−1,−1)
  → [Separate XYZ] → 3× [Math: Exponent] → [Combine Color] = T
  Farbe = Sand·T + Wasserfarbe (0.01, 0.10, 0.16)·(1 − T)       ([Mix Color: Mix], Fac = 1 − T)
  ```
  Ray Length ist die Länge des letzten Strahlsegments, also der Weg durchs Wasser bis zum Boden.
- **MNEE-Caustics** (echt, Sonne durch Welle auf den Boden): Wasserobjekt → Visibility → Caustics:
  **Cast Shadow Caustics** ✓; Boden und Felsfüße: **Receive Shadow Caustics** ✓; Sonnenlicht:
  Shadow Caustics ✓. Einschränkungen: nur eine brechende Grenzfläche, glatte Normalen, nur echte
  Lichter (nicht die World).
- **Fake-Caustics** (billig): im Boden-Material
  ```
  [Voronoi Texture] 4D, Distance to Edge, Scale 1.6, W = frame/24·0.6, Randomness 1
  → [Map Range] 0–0.08 → 1–0 → [Math: Power] 2.5 → [Math: Multiply] 0.8·exp(−0.08·Tiefe)
  → [Mix Color: Add] auf die Boden-Base-Color
  ```

### 1.4 Klippen & Felsformationen

**Geometrie** (`fpv.rock_mesh`): Die senkrechten Karstrinnen sind für Kalkstein-Nadeln richtig (Ha-Long-
Typ), aber zu gleichmäßig. Deshalb:
- Rinnen-Amplitude halbieren, Frequenz um den Umfang pro Felsen variieren (±40 %).
- **Waagerechte Schichtfugen** alle 1,5–3 m (zufällige Dicke), 0,2–0.5 m tief.
- **Brandungskehle** (typisch für Meeresfelsen): radial −1,2 bis −2 m zwischen z −0,3 und +1,3 m. In GN:
  `[Position].Z → [Float Curve] → [Set Position] Offset = −Normal_xy · Kehle(z)`.
- Überhänge an der Oberkante, Schuttkegel am Fuß (Distribute Points on Faces auf der Wasserlinie →
  Instanzen von 8 Blockvarianten).
- Stack: `[GN Rock-Shape] → [Subdivision] Adaptive ✓, Pixel Size 1.0, Levels Viewport 1`.

**Shader ohne gestreckte UVs** (Box-Projektion + Prozedural):

```
[Texture Coordinate].Object → [Mapping] Scale 0.25 (= 4-m-Kachel)
  → [Image Texture] rock_albedo, Projection BOX, Blend 0.25                  → Alb
  → [Image Texture] rock_rough (Non-Color), BOX, Blend 0.25                  → Rgh
  → [Image Texture] rock_height (Non-Color), BOX, Blend 0.25                 → Hgt
  (Normal-Maps nicht mit Box-Projektion mischen – falsche Tangenten; Details über Height → Bump)
Makro:  [Noise] Ridged Multifractal, Scale 0.08, Detail 7, Roughness 0.55, Lacunarity 2.3, Offset 1.0, Gain 2.0 → R
Schicht: [Wave] Bands, Direction Z, Scale 0.9, Distortion 3.5, Detail 3, Detail Scale 1.5,
         Vector = Object + [Noise fBM Scale 0.2].Color·4   → S
Risse:   [Voronoi] Distance to Edge, Scale 2.5 → [Map Range] 0–0.03 → 1–0 → C
Farbe:   [Color Ramp] (S): 0.00 (0.07,0.065,0.058) · 0.40 (0.30,0.27,0.23) · 0.62 (0.24,0.21,0.17) · 1.00 (0.12,0.11,0.09)
         → [Mix Color: Overlay] Fac 0.6 mit Alb → [Mix Color: Multiply] Fac 0.35·R → [Mix] Risse dunkel Fac 0.8·C
Höhe:    0.55·R + 0.30·S − 0.20·C + 0.15·Hgt → [Displacement] Midlevel 0.5, Scale 0.6 m → Output.Displacement
Material → Settings → Displacement: „Displacement and Bump“
```

**Nasse Algen-/Gischtzone an der Wasserlinie** – zwei Wege:

*a) Vertex Weight Proximity* (folgt den Wellen, weil der Modifier das ausgewertete Ozean-Mesh liest):

| # | Modifier | Einstellungen |
|---|---|---|
| 1 | Subdivision (Simple, 1) | genug Vertices für eine weiche Maske |
| 2 | Vertex Weight Edit | Group `wet` (leer anlegen), Default Weight 1.0, Group Add ✓, Add Threshold 0 → alle Vertices mit 1.0 in der Gruppe |
| 3 | **Vertex Weight Proximity** | Group `wet`, Target `Sea_Near`, Proximity Mode *Geometry* → Face ✓, Lowest 0 m, Highest 2.5 m, Falloff *Smooth*, Normalize ✓ |
| 4 | Subdivision | Adaptive ✓ (Render) |

*b) GN-Raycast* (präziser, auch Spritzzone):
```
[Raycast] Target = [Object Info](Sea_Near).Geometry, Source Position = [Position],
          Ray Direction (0,0,−1), Ray Length 6 m
Is Hit = falsch (Punkt liegt unter Wasser) → wet = 1
sonst  wet = [Map Range] Hit Distance 0–2.5 m → 1–0
Spritzzone Luvseite: Highest 4 m, wenn dot(Normal, Windrichtung) > 0.3
[Store Named Attribute] "wet" (Float, Point)
```

Shader-Anteil (Maske W aus Attribut `wet`, `[Map Range] Smoother Step`):
```
Base Color ×(1 − 0.55·W)                      nass = dunkler
Roughness  mix(R, 0.12, W)
Coat Weight 0.7·W, Coat Roughness 0.04, Coat IOR 1.33      Wasserfilm
Algenband  z ∈ [−0.2, 0.6] m × [Noise] → Base (0.05, 0.07, 0.03), Roughness 0.35
Salz/Seepocken z ∈ [0.4, 1.6] m: [Voronoi F1] Scale 60 → [Map Range] eng → Base (0.55, 0.53, 0.48), Bump 0.2
```

### 1.5 Thousand Sunny: Shading

**Holzplanken** (erweitert `sunny.wood_material` / `_wear`):
```
Planken-UV (ein UV-Island pro Planke), Maserung längs U
[Geometry].Random Per Island → [Map Range] → Farbvariation ±6 % (Hue/Sat/Val) und
   Maserungs-Versatz: UV + RandomPerIsland·(13.7, 7.1)
Maserung: [Mapping] Scale (40, 1.2, 1) → [Noise] fBM Scale 3, Detail 6, Roughness 0.6, Distortion 0.8
          + [Wave] Rings, Scale 2.5, Distortion 6 (Jahresringe an den Stirnseiten)
Kanten (Curvature ohne dichte Geometrie):
   [Bevel] Radius 0.02 m, Samples 8 → [Vector Math: Dot] mit [Geometry].True Normal
   → [Map Range] 0.97–1.0 → 1–0 = Kantenmaske E   (Pointiness nur bei dichten Meshes zuverlässig)
Hohlräume: [Ambient Occlusion] Distance 0.15 m, Only Local ✓ → [Map Range] 0.5–1 → 1–0 = Schmutz D
Kratzer:   [Mapping] Scale (80, 3, 3) → [Noise] Scale 6, Detail 2 → [Map Range] 0.62–0.64 → 0–1
           × [Noise Scale 0.5] (nur stellenweise) = K
Base:      Holz → Kanten +12 % heller (E·Noise) → Kratzer +8 % → Schmutz ×0.6 (D)
Roughness: 0.55 ± 0.1·Noise; Kanten 0.35 (abgegriffen); Schmutz 0.8; Kratzer +0.2
Normal:    [Bump] Maserung 0.15/0.002 → [Bump] Plankenfugen (Wave Bands quer, 0.6/0.004)
           → [Bump] Kratzer 0.05/0.001 (Invert) → Principled.Normal
Nägel:     [Voronoi F1] an den Plankenenden → kleine dunkle Kreise + Bump
```

**Roter Lack der Bordwand**: Principled Base (0.36, 0.018, 0.014), Roughness 0.35, Coat 0.2;
Abplatzungen = E × [Noise]-Schwelle → darunter Holz/Grundierung; Rostfahnen unter Speigatten:
`[Mapping] Scale (8, 8, 0.4)` + Höhenmaske.

**Segel & Flaggen – Cloth-Simulation** (ersetzt die statischen `sail_mesh`-Formen und den Wave-Modifier):

Geometrie: Fock 48 × 40 Quads (flach, Rest-Form), Großsegel 40 × 24, Flagge 30 × 20.

| # | Modifier (Segel) | Einstellungen |
|---|---|---|
| 1 | **Cloth** | siehe Tabelle |
| 2 | Subdivision | Viewport 1, Render 2 |
| 3 | Solidify | 0.004 m, Even Thickness |

| Cloth-Parameter | Segel (Canvas) | Flagge |
|---|---|---|
| Quality Steps | 10 | 12 |
| **Vertex Mass** | (0.4 kg/m² × Fläche) / Vertexzahl → Fock ≈ 0.039 kg | (0.15 kg/m² × 8.6 m²) / 600 ≈ 0.002 kg |
| Air Viscosity | 1.0 | 2.0 |
| Stiffness Tension / Compression / Shear / Bending | 80 / 80 / 40 / 2 | 15 / 15 / 5 / 0.1 |
| Damping Tension / Compression / Shear / Bending | 25 / 25 / 25 / 0.5 | 5 / 5 / 5 / 0.1 |
| Pin Group | `pin`: Rah-Kante + Schothörner = 1.0 | Liek am Mast = 1.0 |
| Collisions | Object ✓ (Mast/Wanten mit Collision-Modifier), Distance 0.015; Self ✗ | ✗ |
| Field Weights | Wind 1.0, Turbulence 1.0 | Wind 1.0, Turbulence 1.0 |
| Cache | Frames 1–528, Render ab 49 (2 s Vorlauf, damit das Segel beim ersten Bild schon gefüllt ist) | gleich |

Kraftfelder (beide an `ThousandSunny` geparentet, damit der Wind relativ zum Schiff konstant bleibt):
- **Wind**: Shape Plane, Richtung Heck → Bug (+X Schiff), Strength zuerst 300, dann kalibrieren,
  bis der Bauch ≈ 8–12 % der Sehne tief ist (Fock: 1,5–2 m); Flow 0, **Noise 1.5** (Böen), Seed 7.
  Hinweis: Air Viscosity bremst das Segel auch gegen die Fahrt (3 m/s) → Wind etwas stärker setzen.
- **Turbulence**: Strength 15 (Segel) / 5 (Flagge), Size 1.5 m / 0.5 m, Flow 0.3 → Flattern an den Lieken.
- Schoten folgen den Schothörnern: Empty per *Vertex Parent* an den Schothorn-Vertex → Hook-Modifier
  am Tau-Ende auf dieses Empty.

### 1.6 World & Atmosphäre

```
[Sky Texture] Multiple Scattering: Sun Elevation 32°, Sun Rotation 200°, Altitude 50 m,
              Air 1.0, Aerosol 1.2, Ozone 2.0, Sun Disc ✗
→ [Background] Strength 0.08 (unsere Belichtung; zusammen mit Sonne 5.2 und AgX)
Sonne: Sun Light, Angle 0.53°, Strength 5.2, Farbe (1.0, 0.90, 0.78), Richtung = Sky-Sonne
```
Sonnenscheibe im Sky aus, dafür echte Sun Light: sauberere Schatten und besseres MIS-Sampling.
Mist: World → Mist Pass Start 25 m, Depth 2500 m, Falloff *Quadratic*; View Layer → Passes → Mist ✓.
Compositor-Anteil in Abschnitt 4 (bei uns heute in `post.py`: `haze_dist`, `haze_color`).
Horizont-Dunstband: im World-Shader `mix(Himmel, (0.72, 0.80, 0.88), MapRange(|z| 0–0.05 → 1–0))`.

---

## 2. PROJEKT 2 – Naruto: Konohagakure & Hokage-Monument

### 2.1 Hokage-Monument: vom Scan-Kopf zur gemeißelten Anime-Skulptur

Ursache des „Museumskopf“-Eindrucks: Die Gesichter basieren auf einem fotogrammetrischen Menschenkopf.
Anime-Formensprache heißt: flache, klar gesetzte Flächen, großer Schädel, Augen als geschlossene
gebogene Kerben, kleine keilförmige Nase, Mund als Linie, schmales Kinn. Und die Köpfe sind **Relief** in
der Wand, keine frei stehenden Büsten: Tiefe ≈ 0,45 × Gesichtsbreite.

Workflow je Kopf (Kopfhöhe ~16–18 m):
1. **Blockout**: UV-Sphere + Box-Proxys für Schädel/Kiefer/Haarmasse nach Model Sheet. Frontal- und
   Seitenansicht der Referenz als Image Empties.
2. **Voxel Remesh** 0,20 m (Preserve Volume ✓, Fix Poles ✓).
3. **Sculpt, Formen setzen** – Clay Strips (Radius 1,2 m, Strength 0.6) für Wangen/Stirn/Haarmassen;
   Scrape/Flatten (Plane Offset −0.1) für die Meißelflächen; Trim Box / Line Project für harte Schnitte
   (Stirnband, Haarsträhnen-Kanten).
4. **Dyntopo**: Detailing *Constant Detail*, Resolution 20 (= 5 cm), Refine *Subdivide & Collapse*.
   Crease (Strength 0.6, Pinch 0.5, Radius 0,25 m) für Lidlinien, Mund und Haarfugen; Draw Sharp für
   Meißelspuren.
5. **Verwitterung**: Crease mit Riss-Alphas entlang der Schichtfugen; Auto-Masking *Cavity* +
   Mesh Filter *Sharpen* (0.3) und *Random* (0.05) an Kanten = Erosion. Nasenspitze, Lippen, Haarspitzen
   bewusst abgerundet und bestoßen.
6. **Verankerung in der Klippe**: Kopf + umgebendes Klippenstück joinen → **Boolean Union (Exact)** →
   **Voxel Remesh 0,12 m** → Smooth an der Naht → Clay Strips baut Gesteinsmasse um Hinterkopf und Haar,
   damit die Haare in die Schichtbänke übergehen. Schichtfugen der Wand laufen durch die Köpfe durch.
7. **Render-Mesh**: Decimate (Collapse 0.25) auf ~0,5–1 Mio. Tris je Kopf; Mikrodetail kommt aus dem
   Shader (2.2) mit adaptiver Unterteilung. Alternative: Displacement von der Sculpt-Version auf ein
   ~200k-Mesh backen.
8. **Gleiches Material wie die Wand, in Weltkoordinaten** (`[Geometry].Position`), sonst entstehen Nähte.
   Dunkle Regenfahnen vom Kinn abwärts: `[Mapping] Scale (6, 6, 0.3)` × Höhenmaske unter jedem Kinn;
   Moos/Gras in Fugen über `[Geometry].Normal.z > 0.6 × AO`.

### 2.2 Prozeduraler Sandstein-/Granit-Shader mit Mikrorissen

```
[Geometry].Position → P (Welt, nahtlos über Köpfe und Wand)
A Makro-Erosion:  [Noise] Ridged Multifractal, Scale 0.06, Detail 8, Roughness 0.5, Lacunarity 2.2,
                  Offset 1.0, Gain 2.0 → R
B Sedimentbänke:  [Wave] Bands, Direction Z, Scale 0.4 (≈ 2,5-m-Bänke), Distortion 5, Detail 3,
                  Detail Scale 1.2, Vector = P + [Noise fBM Scale 0.2].Color·2 → S
C Mikrorisse:     [Voronoi] 3D, Distance to Edge, Scale 3.0, Randomness 1 → [Map Range] 0–0.015 → 1–0 → C1
                  [Voronoi] Distance to Edge, Scale 11 → [Map Range] 0–0.008 → 1–0 → C2
                  Risse = max(C1, 0.5·C2) × [Map Range]([Noise Scale 0.5], 0.45–0.6 → 0–1)
D Korn:           Sandstein: [Noise] Scale 400, Detail 2 → Bump 0.1/0.0005
                  Granit:    [Voronoi] F1 Smooth, Scale 120, Smoothness 1 → .Color → ±8 % Farbe
Farbe:  [Color Ramp] (S): 0.00 (0.28, 0.17, 0.08) · 0.35 (0.58, 0.40, 0.22) · 0.55 (0.69, 0.51, 0.31)
                          · 0.80 (0.39, 0.24, 0.11) · 1.00 (0.20, 0.12, 0.06)
        → [Mix Color: Multiply] Fac 0.35·R → [Mix] Riss-Farbe (0.04, 0.03, 0.02), Fac 0.8·Risse → Base Color
Roughness: 0.82 + 0.10·R
Höhe:   0.5·R + 0.25·S − 0.35·Risse + 0.05·Korn → [Displacement] Midlevel 0.5, Scale 0.4 m → Output
Objekt: Subdivision Adaptive ✓ Pixel 1.0 · Szene: Dicing Rate Render 1.0, Max Subdivisions 10, Offscreen Scale 4
```

### 2.3 Straße: Mikro-Displacement statt flacher Fläche

Straßen-Mesh als Band mit UV (U quer in m, V längs in m) → `Subdivision (Adaptive, Pixel 1.0)`.
```
Pflaster:   [Voronoi] F1 auf UV·3.5, Distance to Edge → [Map Range] 0–0.08 → 0–1
            → [Float Curve] (Kuppelprofil) × 0.035 m ; Steinhöhe ± [Voronoi].Color.R × 0.01 m → H_cob
Sand:       Maske M_s = [Map Range]([Noise Scale 0.3, Detail 4], 0.45–0.6 → 0–1) + Radrillen + Wandnähe
            H_sand = 0.02 + [Noise Scale 8]·0.01 ;  H = mix(H_cob, max(H_cob, H_sand), M_s)
Radrillen:  u0 = Straßenmitte ± 0.8 m: rut = exp(−((U − u0)/0.18)²) → H −= 0.03·rut ; Farbe dunkler,
            Roughness −0.1 (verdichtet), seitliche Wülste +0.01 m
Wandnähe:   Attribut "wall_dist" (GN Geometry Proximity zu den Häusern) → Sandverwehungen, Staub
[Displacement] Height = H, Midlevel 0, Scale 1.0 → Output.Displacement
```

### 2.4 Häuser: Kanten, Normalen, Fenster mit Tiefe

| # | Modifier (jedes Gebäude) | Einstellungen |
|---|---|---|
| 1 | Bevel | Width Type Offset, **Amount 0.025 m** (Wände) / 0.012 m (Zierleisten), **Segments 2**, **Limit Method Angle 30°**, Profile 0.5, Clamp Overlap ✓, **Harden Normals** ✓, Miter Outer *Arc* |
| 2 | Weighted Normal | Mode *Face Area*, Weight 50, Threshold 0.01, **Keep Sharp** ✓ |

(In 5.0 braucht Weighted Normal kein Auto Smooth mehr; bei Schattierungsfehlern zusätzlich den
Node-Modifier *Smooth by Angle* 30°.) Für die Sunny gibt es das schon (`sunny.bevel_stack`), in `konoha.py`
fehlt es noch.

Putz: Sockelschmutz (z 0–0,8 m über Grund dunkler), Regenfahnen unter Fensterbänken
(`[Mapping] Scale (6,6,0.3)` × Maske unter jeder Bank), Kantenabrieb mit der Bevel-Dot-Maske aus 1.5.

**Fenster & Türen:**
- Geometrie: Inset 0.08 m → Extrude −0.15 m (Laibung), Rahmen (Solidify 0.06 m der Rahmenschleife),
  Fensterbank 0.05 × 0.12 m, Glas 0.10 m zurückgesetzt.
- Glas: `[Mix Shader] Fac = Light Path.Is Shadow Ray → (Glass BSDF IOR 1.5, Roughness 0.02 | Transparent)`
  (Licht fällt in den Raum, ohne dass Schattenstrahlen durchs Glas rauschen).
- **Innenraum-Kubus** (empfohlen, physikalisch korrekte Parallaxe, bei FPV-Tempo am günstigsten):
  Box 3 × 3 × 2,6 m hinter jedem Fenster, das näher als ~40 m an der Flugbahn liegt; Normalen nach innen;
  Tapete/Putz mit `Object Info.Random` → Color Ramp (6 Farbtöne), Holzboden, dunkle Möbel-Silhouette
  (1–2 Boxen). Vorhang 0,2 m hinter dem Glas: `Mix(Principled, Translucent, 0.4)`, Farbe per Random,
  halb offen über Noise-Alpha. Ferne Fenster: nur dunkles, spiegelndes Glas, zufällig 20 % mit warmem
  Innenlicht.
- Parallax ohne Geometrie (Interior Mapping) geht auch ohne OSL als Node-Gruppe: Strahl in
  Tangentenraum → je Achse `t = (ceil/floor(p) − p)/d` → Minimum → Trefferpunkt → Wand-/Boden-/
  Deckentextur. OSL-Varianten laufen in Cycles nur auf CPU/OptiX.

### 2.5 Hängende Stromkabel (Geometry Nodes, Durchhang als Kettenlinie)

```
Eingabe: Mesh „CableSpans“ – je Spannfeld ein Edge zwischen zwei Aufhängepunkten (Mastspitze/Hausanker)
[Mesh to Curve] → [Resample Curve] Count 32
t = [Spline Parameter].Factor ; L = [Spline Length].Length
sag = L·0.035·(1 + 0.3·[Random Value] Float −1…1, ausgewertet auf Domain *Spline* ([Evaluate on Domain]))
Kettenlinie (Parabel-Näherung, < 1 % Fehler bei sag/L < 10 %):  dz = −4·sag·t·(1 − t)
Wind:  seitlich = Normalize(Cross(Tangente, Z)) ;
       dy = sin(2π·0.35·[Scene Time].Seconds + Phase_spline)·0.06·4t(1 − t)
[Set Position] Offset = seitlich·dy + (0, 0, dz)
[Curve to Mesh] Profile [Curve Circle] Resolution 6, Radius 0.012 m, Fill Caps ✓
[Set Material] „Cable“ (Base 0.02, Roughness 0.35, Coat 0.3)
Extras: [Endpoint Selection] → [Instance on Points] Isolatoren;
        [Curve to Points] Count 6 → [Instance on Points] Laternen, 0,3 m nach unten versetzt, gleiche Wind-Phase
```
Exakt wäre z(x) = a·cosh((x − xm)/a) − a·cosh(L/2a); für Stromkabel reicht die Parabel.

### 2.6 Clutter in Straßenwinkeln (GN, Masken statt Handplatzierung)

```
[Group Input] (Straße + Plätze)
wall   = [Geometry Proximity] Target = [Collection Info] „Buildings“ (Realize Instances) .Distance
         → [Map Range] 0–1.8 m → 1–0 (Smoother Step)
corner = Anzahl [Raycast]-Treffer in 4 horizontalen Richtungen (Länge 2 m) ≥ 2 → +0.5
cover  = [Raycast] nach oben (Länge 8 m).Is Hit → unter Dachüberständen weniger Laub, mehr Staub
dens   = 25/m² × (0.15 + 0.85·clamp(wall + corner)) × [Noise Scale 0.4]
[Distribute Points on Faces] Poisson Disk, Distance Min 0.03 m, Density Max 60, Density Factor = dens
[Instance on Points] Instance = [Collection Info] „Clutter“ (Separate Children ✓, Pick Instance ✓,
     Instance Index = [Random Value] Int 0…N−1)
     Rotation = [Align Rotation to Vector] (Normal, Z) → [Rotate Rotation] Z zufällig
     Scale = [Random Value] 0.6–1.4
Collection „Clutter“: 6 Kiesel (2–6 cm, displaced Icospheres), 4 Laubkarten (8 cm, Alpha + Translucent),
     2 Papierfetzen, 3 Zweige, Sandverwehungs-Decals
Shader-Kontaktschmutz nur am Boden: [Ambient Occlusion] Distance 1.2 m, Samples 16 → [Map Range] 0.4–1 → 0.6–1 → × Base Color
```

### 2.7 Licht & Dunst

- **Sun Light: Angle 2.5°** (2,0–3,5° → weiche, leicht dunstige Schattenkanten; die echte Sonne hat
  0,53°), Strength 5.0–5.2, Farbe (1.0, 0.95, 0.87).
- **Sky-HDRI** (z. B. ein Poly-Haven-Himmel „partly cloudy“, Sonne im HDRI entfernt/geclampt, damit es
  keine doppelte Sonne gibt), Rotation passend zur Sun Light, Strength 0.8–1.0 – oder Sky Texture
  Multiple Scattering mit Sun Disc aus.
- Light Paths: Max Bounces 8, Diffuse 4 (die Straßenschluchten leben vom Licht, das von den
  Pastellwänden zurückfällt), Clamp Indirect 10.
- **Z-Depth-Dunst**: Mist Start 12 m, Depth 600 m, Falloff Quadratic; im Compositor mit (0.68, 0.61, 0.47)
  (#D7CCB6) bei max. 18 % mischen.
- Optional Lichtstrahlen zwischen Häusern: Volume-Scatter-Box über dem Dorf, Density 0.001,
  Anisotropy 0.65, Ray Visibility nur Camera + Transmission. Teuer (0.2), nur für Hero-Shots mit GPU.

---

## 3. PROJEKT 3 – Dragon Ball: Planet Namek

### 3.1 Himmel: atmosphärischer Verlauf statt grüner Wand

**Variante A – physikalischer Himmel mit Hue-Rotation (empfohlen):** Die Erd-Atmosphäre streut blau;
dreht man den Farbton, bleiben Helligkeitsverteilung, Horizontaufhellung und Sonnenhof physikalisch
korrekt – nur der Planet wird grün-türkis.
```
[Sky Texture] Multiple Scattering: Sun Elevation 24°, Sun Rotation 300°, Air 1.2, Aerosol 1.0,
              Ozone 0.3 (weniger Zenit-Sättigung), Sun Disc ✗
→ [Hue/Saturation/Value] Hue 0.37 (Blau h≈0.60 → Zenit-Türkis h≈0.47; Horizont bleibt hell-smaragd),
                          Saturation 1.25, Value 1.0
→ [Mix Color: Mix] Fac 0.3 mit Variante B (Anime-Farbtreue)
→ Light-Path-Split (haben wir schon): Is Camera Ray OR Is Glossy Ray → kräftiger Himmel,
   sonst entsättigt (sonst färbt der grüne Himmel das Gras türkis)
→ [Background]
```

**Variante B – eigener Rayleigh-Verlauf** (ersetzt die Rampe in `world_dragonball.namek_sky`):
```
[Texture Coordinate].Generated (in der World = Blickrichtung) → [Separate XYZ].Z → [Math: Maximum] 0 → μ
Optische Tiefe τ = 0.35 / (μ + 0.12)                       ([Math: Add], [Math: Divide])
[Color Ramp] auf μ^0.45:
   0.00 (0.62, 0.92, 0.72)  hell-smaragd (#CFF4DC) – Horizont
   0.10 (0.28, 0.80, 0.58)                         (#90E7C7)
   0.35 (0.06, 0.52, 0.46)                         (#46BDB4)
   1.00 (0.012, 0.22, 0.26) tiefes Türkis          (#1C818B) – Zenit
Sonnenhof (Mie-Vorwärtsstreuung): d = max(dot(Generated, SunDir), 0)
   → 0.5·d⁸ + 1.5·d⁶⁴ → [Mix Color: Add] mit (1.0, 0.97, 0.85)
Horizont-Dunstband: [Map Range] μ 0–0.06 → 1–0 → Mix mit (0.75, 0.95, 0.80)
Zwei Nebensonnen: gleicher Hof mit Stärke 0.35 / 0.25 in ihren Richtungen
```

### 3.2 Anime-Kumuluswolken (volumetrisch, bezahlbar)

Modell: 6–10 Wolkenkörper aus Metaballs (Ball, r 20–60 m, Threshold 0.6) → Convert to Mesh →
Voxel Remesh 1.5 m → Displace (Clouds, Size 12 m, Strength 6 m) → flache Unterseite (GN: z < z_base
abschneiden).
```
[Texture Coordinate].Object → [Noise] fBM, Scale 0.04, Detail 6, Roughness 0.55 → n
Density = clamp((n − 0.42)·12, 0, 1)            (steiler Remap = scharfe Anime-Oberkanten)
[Principled Volume] Color (0.95, 1.0, 0.92), Density ← Density, Anisotropy 0.45,
                    Absorption Color (0.9, 1.0, 0.9) → Output.Volume (kein Surface-Shader)
Render: Volume Step Rate 1.0, Max Steps 1024, Volume Bounces 1
```
Wegen der gemessenen Kosten (0.2 Punkt 8) eine der beiden Varianten:
- **Ray Visibility einschränken**: Wolkenobjekte → Visibility → Ray Visibility nur *Camera* (Shadow ✗,
  Diffuse ✗, Glossy ✗). Dann gehen Schattenstrahlen der Szene nicht mehr durch die Volumen.
  Wolkenschatten auf dem Land liefert ein billiges Gobo: große Ebene hoch oben mit
  Noise-Alpha, nur für Schattenstrahlen sichtbar.
- **Sky-Plate**: Wolken einmal mit Panorama-Kamera (Equirectangular, 8K, 256 Samples) vom Zentrum der
  Flugbahn rendern und als Environment Texture über den Himmel legen. Bei > 1,5 km Entfernung ist die
  Parallaxe über 360 m Flug gering.

### 3.3 Felsplateaus: waagerechte Sediment-Schichten statt Zylinder

GN „Sediment“ auf dem Tafelberg bzw. den Felsnadeln (vorher Remesh 0,3 m oder Subdivision):
```
[Position] → [Separate XYZ].Z → z ; z' = z + 0.8·[Noise](z·0.1)        (ungleich dicke Bänke)
layer = [Math: Floor](z'/2.2 m) ; fr = [Math: Fraction](z'/2.2 m)
Profil (harte Bank springt vor, weiche Schicht liegt zurück):
  [Float Curve] auf fr: 0.00→0.0 · 0.15→1.0 · 0.30→0.85 · 0.75→0.1 · 1.00→0.0
Versatz je Bank: [Random Value] Float, ID = layer, Min −0.8, Max 0.8 m, Seed 5
Silhouette:      [Noise] 3D Scale 0.08, Detail 5 → (−0.5…0.5)·1.5 m
radial = Normalize(Position − Mitte, z = 0)
[Set Position] Offset = radial·(0.9·Profil + Versatz + Silhouette) ; oberste Bank +1.5 m Überhang
[Store Named Attribute] "layer" (Int, Point), "fr" (Float, Point)
→ [Subdivision] Adaptive ✓ + Shader-Displacement für Feindetail
```
Shader: `[Attribute "layer"] → [White Noise] (W = layer) → [Color Ramp]` mit Ocker (0.55, 0.36, 0.22),
Beige (0.72, 0.58, 0.42), Rostrot (0.45, 0.22, 0.14), Hellsand (0.80, 0.70, 0.55); harte Bänke (fr < 0.2)
heller und rauer, Verwitterungsfahnen unter den Kanten. Die senkrechten Karstrinnen aus
`fpv.rock_mesh` für Namek abschalten (`strata` statt Rinnen) – sie sind die Ursache der Längsstreifen.

### 3.4 Ajisa-Bäume (GN)

```
Stamm:  [Curve Line] (0,0,0)→(0,0,H) → [Resample Curve] 40
        → [Set Position] Offset = (sin, cos)(t·Windungen·2π)·A·(1 − t) + [Noise]·0.2
        → [Set Curve Radius] [Float Curve](t): 0.35 → 0.12 → [Curve to Mesh] [Curve Circle] Res 10 → Rinde
Krone:  [Ico Sphere] r = 2.5–3.6 m, Subdiv 4 → [Distribute Points on Faces] Poisson, Distance Min 0.25 m
        → [Instance on Points] Collection „LeafClusters“ (3–5 gekreuzte Blattkarten, Pick Instance ✓)
        → [Align Rotation to Vector] (Normal, Z) → [Rotate Rotation] Z zufällig → Scale 0.8–1.2
        + dunkle innere Kugel (0,85·r) mit gleichem Material, damit die Krone nicht durchscheint
Blatt:  [Mix Shader] Fac 0.35:
          [Principled BSDF] Base (0.02, 0.10, 0.55), Roughness 0.45, Sheen 0.2
          [Translucent BSDF] Color (0.10, 0.35, 1.0)        → Gegenlicht leuchtet blau durch
        Variation: [Object Info].Random (pro Instanz) → [Hue/Saturation/Value] Hue 0.5 ± 0.03, Value 0.9–1.1
```

### 3.5 Blauer Grasboden als Hair Curves

Add → Curves → Empty Hair auf dem Plateau; Node-Gruppen aus den Essentials-Assets (*Hair*):

| # | Node-Gruppe | Einstellungen |
|---|---|---|
| 1 | Generate Hair Curves | Surface = Tafelberg, Density 1200/m² × Maske `grass_mask` (1 an der Flugbahn → 0.15 in 40 m, 0 auf Sand/Häusern), Length 0.18 m, Length Variation 0.5, Control Points 5 |
| 2 | Clump Hair Curves | Clump Distance 0.06 m, Shape 0.3, Factor 0.5, Tip Spread 0.02 |
| 3 | Frizz Hair Curves | Factor 0.4 |
| 4 | Noise Hair Curves | Factor 0.35, Scale 4 |
| 5 | eigene GN „WindBend“ | Offset = t²·0.05 m·sin(2π·0.6·Time + dot(Position, k)) in Windrichtung |
| 6 | Set Hair Curve Profile | Radius Wurzel 0.002 → Spitze 0.0002 m |

Material auf den Curves:
```
[Curves Info].Intercept → [Color Ramp] 0 (0.01, 0.03, 0.12) Navy · 0.5 (0.03, 0.12, 0.45) · 1 (0.12, 0.35, 0.85)
[Curves Info].Random → [Hue/Saturation/Value] Hue 0.5 ± 0.04, Value 0.85–1.15
[Principled BSDF] Roughness 0.45, Sheen Weight 0.3 ; [Mix Shader] 0.25 mit [Translucent] (0.2, 0.45, 1.0)
Render → Curves: Shape „3D Curves“ (nah) bzw. „Rounded Ribbons“ (günstiger), Subdivisions 2
```
Ersetzt `namek.grass_field` (1,5 Mio. Dreiecke) bei weniger Speicher und mit echtem Clumping.

### 3.6 Grünes Namek-Wasser

Wie 1.2, nur grün:
```
[Volume Absorption] Color (0.30, 0.75, 0.35) (#94E1A0), Density 0.12
   → nach 10 m bleiben (0.43, 0.74, 0.46), nach 30 m (0.08, 0.41, 0.10): smaragd → tiefgrün
[Volume Scatter] Color (0.35, 0.85, 0.45), Density 0.003, Anisotropy 0.3   → das „leuchtende“ Anime-Grün im Flachwasser
Oberfläche: Principled, Transmission 1.0, IOR 1.333, Roughness 0.02, Base Color Weiß
Meeresboden: heller Sand/Fels (−2 m an den Felsnadeln, −25 m im Freiwasser)
```
Ohne Volumen (CPU-Budget): Ray-Length-Trick aus 1.3 mit σ = 0.12·(0.70, 0.25, 0.65).

### 3.7 VFX: Ki-Strahl, Funken, Hitzeflimmern, Licht

**Strahl** (Erweiterung von `vfx.beam`):
```
[Curve Line] Quelle → Kopf (Länge animiert) → [Resample Curve] 64
→ [Set Position] Offset = [Noise 4D](Position·0.5, W = Time)·0.15 m (nicht in den ersten 2 m an den Händen)
→ drei [Curve to Mesh]-Hüllen mit [Set Curve Radius] (Wulst am Kopf über Float Curve auf t):
Kern   r 0.30 m: [Emission] (0.85, 0.95, 1.0) Strength 30 · Mix mit [Transparent] über Facing³ (weicher Rand)
Mantel r 0.90 m: [Emission] (0.12, 0.38, 1.0) Strength 4 × Wirbelstreifen
                 ([Mapping] Scale (0.3, 0.3, 3) längs, [Noise] 4D, W = Time·3)
                 → [Mix Shader] mit Transparent, Fac = (1 − Facing)^1.1·0.65
Aura   r 1.60 m: als Oberflächen-Glühhülle (wie heute) + Glare im Compositor
                 – kein heterogenes Emissions-Volumen (Kosten, 0.2 Punkt 8)
Alle Hüllen: Visibility → Shadow ✗ (Strahlen werfen keine Schatten)
```

**Funken (Simulation Zone, Motion Blur):**
```
[Simulation Input] (Geometry, Delta Time)
  neu:  [Curve to Points] entlang des Strahls (20 pro Frame, zufällige Position)
        + radialer Versatz 0.6 m ; "velocity" = radial·U(6, 14) m/s + Strahlrichtung·4 ; "age" = 0
  alt:  velocity += (0, 0, −9.81)·0.3·dt ; Position += velocity·dt ; age += dt
        [Delete Geometry] age > 0.6 s
  [Join Geometry]
[Simulation Output]
→ [Instance on Points] Ico Sphere r 0.03 m, [Align Rotation to Vector] (velocity, Z), Scale (1, 1, 3)
→ [Set Material] Emission (0.6, 0.8, 1.0) Strength 20
Motion Blur: Das Punkt-Attribut muss exakt „velocity“ (Vector, Point) heißen – Cycles nimmt es für die
Bewegungsunschärfe, wenn sich die Topologie jedes Frame ändert. Render → Motion Blur ✓, Shutter 0.5.
```

**Hitzeflimmern** – zwei Wege:
- *Compositor*: in der Aura-Hülle `[AOV Output] "heat" (Color)` =
  ((Noise(Position·2 + Time).xy − 0.5), 0) × Facing-Maske; View Layer → Shader AOV „heat“ (Color).
  Im Compositor: `[Render Layers].heat → [Separate Color] → XY → [Vector Math: Scale] 12 px → [Displace].Displacement`,
  Image = Render.
- *Glas-Ring im Render* (physikalisch, günstig): Hülle r 2.5 m mit `[Refraction BSDF] IOR 1.01, Roughness 0`,
  Normal ← `[Bump]` aus Noise 4D (Strength 0.4, Distance 0.02), gemischt mit Transparent über (1 − Facing)²;
  Ray Visibility nur Camera + Transmission.

**Dynamisches Licht:**
- Emissive Meshes sind in Cycles Lichtquellen (Light Tree), aber dünne, sehr helle Strahler rauschen im
  diffusen Licht. Deshalb zusätzlich:
  - **Linienlicht** = Area Light *Rectangle*, Size X = Strahllänge, Size Y = 0.5 m, entlang des Strahls
    ausgerichtet, Spread 180°, Leistung ≈ 3 kW pro Meter (Kamehameha, unsere Belichtung); oder
    Punktlichter alle 8 m (so heute in `vfx.beam`).
  - **Light Linking** (Licht → Light Linking → Receiver-Collection „Terrain+Wasser+Figuren“), damit
    Himmel/Wolken nicht mitleuchten.
- Faustregel Leistung: Bestrahlung E = P/(4π·d²). Um eine Felswand in 30 m Abstand mit ~2 W/m²
  (≈ 40 % unserer Sonne) aufzuhellen: P ≈ 22 kW; in 60 m ≈ 90 kW. Die finale Namek-Explosion nutzt
  450 kW in ~60 m.
- Das Wasser zeigt den Strahl automatisch: Spiegelung über Glossy, Unterwasser-Glühen über
  Transmission + Volumen (3.6).

---

## 4. Compositor (alle drei Szenen)

In Blender 5.0: Szene → `Compositing Node Group`. Unsere Renders laufen heute durch `post.py`; die
Tabelle am Ende zeigt die Entsprechung.

```
[Render Layers] Image, Alpha, Mist, Depth, AOV heat, (Env)
1 Dunst       Mist* → [Map Range] 0–1 → 0–0.85 (Smooth Step) → Fac von [Mix Color: Mix]
              A = Image, B = Dunstfarbe: OP (0.50, 0.62, 0.80) · Naruto (0.62, 0.70, 0.82) · Namek (0.66, 0.80, 0.42)
              *bei halbtransparenten Effekten vorher: mist' = (mist − (1 − α))/α   (so in post.py umgesetzt)
2 Hitze       [Displace] Image, Displacement = heat.xy · 12 px, Interpolation Bilinear
3 DoF (FPV)   [Defocus] Bokeh Circular, F-Stop 16, Max Blur 6 px, Use Z-Buffer ✓ (Depth), Scene = Kamera,
              Fokus 6–10 m → nur Objekte < 1–2 m werden weich. (Oder Camera → Depth of Field ✓,
              F-Stop 11–16 im Render – physikalisch besser als Compositor-Defocus.)
4 Glare       [Glare] Type Bloom, Quality High, Threshold 1.5, Smoothness 0.2, Maximum 10,
              Strength 0.35, Saturation 1.0, Size 0.6
              Namek-VFX zusätzlich: [Glare] Type Streaks, Streaks 4, Streaks Angle 45°, Fade 0.88,
              Strength 0.12 – nur auf eine VFX-Maske (AOV/Cryptomatte)
5 Linse       [Lens Distortion] Type Radial, Distortion 0.0 (FPV-Regel: keine Tonne),
              Dispersion ≤ 0.004 (optional, sehr dezent), Fit ✓
6 Vignette    [Ellipse Mask] Position (0.5, 0.5), Size (0.95, 0.95) → [Blur] Gaussian, Size (300, 300) px
              → [Map Range] 0–1 → 0.72–1.0 → [Mix Color: Multiply] mit dem Bild
7 Grain       Blender 5.0 hat keinen Grain-Node → [Image] Grain-Plate (gescanntes 35-mm-Korn, 24-Frame-Loop,
              1080×1920) → [Mix Color: Overlay] Fac 0.08–0.12 (nur auf Luminanz wirkt es natürlicher)
8 Farbe       View Transform AgX, Look „Medium High Contrast“, Exposure je Szene;
              Ausgabe OpenEXR Multilayer (Half) zum Nachgraden → H.264/H.265 über ffmpeg
```

| Compositor-Schritt | `post.py` / `params/*.py` |
|---|---|
| Dunst | `mist_depth`, `haze_dist`, `haze_start`, `haze_color` (Mist-Alpha-Korrektur eingebaut) |
| Belichtung | `ev`, Auto-Belichtung `ae_strength`, `ae_tau`, `ae_ref` |
| Look / Sättigung | `look`, `sat`, `wb` |
| Glare | `bloom` (Stärke), `bloom_thr` (Schwelle) |
| Vignette / Grain | `vignette` (Standard 0.18), `grain` (Standard 0.012) |
| Hitze, DoF | fehlen noch → als AOV-Displace bzw. Kamera-DoF ergänzen |

---

## 5. Render-Einstellungen

| Einstellung | Wert |
|---|---|
| Device | CPU (Cloud) bzw. GPU/Metal (Mac) |
| Samples | 12–16 (CPU, 720p) · 64–128 (GPU, 1080p); Adaptive Sampling Noise Threshold 0.01 |
| Denoise | OpenImageDenoise, Passes Albedo + Normal, Prefilter Accurate |
| Light Tree | ✓ (viele VFX-Lichter) |
| **Path Guiding** | ✓ Surface (+ Volume) – nur CPU, hilft bei Wasser/Caustics und Innenräumen |
| Light Paths | Max 8 · Diffuse 3–4 · Glossy 4 · Transmission 8 · Volume 0–1 · Transparent 16 |
| Caustics | Refractive ✓, Reflective ✗, Filter Glossy 1.0 |
| Subdivision | Dicing Rate Render 1.0 px, Max Subdivisions 10, Offscreen Scale 4 |
| Motion Blur | ✓, Shutter 0.5 (180°) |
| Film | Pixel Filter Blackman-Harris 1.5 px; Transparent ✓ (für unseren Env-/Dunst-Composite) |
| Performance | Persistent Data ✓ |

---

## 6. Reihenfolge nach Wirkung pro Aufwand

| Priorität | One Piece | Naruto | Namek |
|---|---|---|---|
| 1 | Wasser mit Transmission + Absorption + Meeresboden (1.2/1.3) | Neue Anime-Hokage-Köpfe, in die Wand gemeißelt (2.1) | Himmel per Hue-Rotation + Wolken-Plate (3.1/3.2) |
| 2 | Cloth-Segel und -Flaggen (1.5) | Straßen-Displacement + Clutter (2.3/2.6) | Sediment-Plateaus statt Rinnen-Zylinder (3.3) |
| 3 | Felsen: Brandungskehle, Nässezone, Box-Texturen (1.4) | Bevel/Weighted Normal + Innenraum-Kuben (2.4) | Wasser-Absorption + Hair-Curves-Gras (3.5/3.6) |
| 4 | GN-Bugwulst + Gischtpartikel (1.1 B) | Sonne 2,5° + HDRI + Dunst (2.7) | Funken, Hitzeflimmern, Linienlicht (3.7) |

Die Punkte der Stufen 1–2 lassen sich mit dem heutigen CPU-Budget (≈ 1,5–2× Renderzeit) umsetzen;
volle Volumen, 1080p nativ und 64+ Samples brauchen eine GPU.
