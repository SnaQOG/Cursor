# Modellquellen: SimReady-Requisiten

Die Modelle in `assets/simready.glb` stammen aus Hugging-Face-Repos, die im Verzeichnis
[NVIDIA-Omniverse/simready-central](https://github.com/NVIDIA-Omniverse/simready-central) stehen
oder dort einreichen. Alle sind einzeln (Paketdatei `com.nvidia.simready.packaging.json`) unter der
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/deed.de)
lizenziert. Das Spiel nennt die Quellen auch im Hub unter *Optionen*.

**Änderungen für Damm Raider:** Polygone reduziert (bei Scans über die reduzierte LOD-Stufe, beim hohen
Korb über eine neue geschlossene Hülle), Materialien und Normalen auf neue Texturen gebacken, Verdeckung
in die Grundfarbe eingerechnet, Texturen verkleinert, in glTF umgewandelt. Physik und Kollisionsmodelle
der Pakete werden nicht verwendet. `sr_drum_red` ist zusätzlich rot umgefärbt und hat ein gelbes Warnband.

Die Lizenz deckt keine Marken ab. Die Aufschriften „Kurvex“ (Ölfass) und „Spomper“ (Kompressor) ergaben
bei einer Websuche keinen Markentreffer; eine rechtliche Prüfung ist das nicht.

| Spiel-ID | Modell | Quelle (feste Version) | Urheber | Dreiecke |
| --- | --- | --- | --- | --- |
| `sr_drum_steel` | Stahlfass | [drum_steel_a01](https://huggingface.co/datasets/nvidia/simready-assets/tree/653d91e182011f357cf1644013cc94ddf194d349/drum_steel_a01) | NVIDIA SimReady Assets | 100.452 → 5.313 |
| `sr_oil_drum` | Ölfass | [oil_drum](https://huggingface.co/datasets/ForgeCAD/simready-assets/tree/b94ea770666b6c0f82e09a78e9fcb375b223e3b1/oil_drum) | ForgeCAD SimReady Assets | 278.156 → 2.000 |
| `sr_crate` | Sperrholzkiste 1 m | [crate_plywood_b02](https://huggingface.co/datasets/nvidia/simready-assets/tree/653d91e182011f357cf1644013cc94ddf194d349/crate_plywood_b02) | NVIDIA SimReady Assets | 20.320 → 1.500 |
| `sr_crate_long` | Sperrholzkiste 2 m | [crate_plywood_b04](https://huggingface.co/datasets/nvidia/simready-assets/tree/653d91e182011f357cf1644013cc94ddf194d349/crate_plywood_b04) | NVIDIA SimReady Assets | 28.810 → 2.000 |
| `sr_pallet` | Holzpalette | [pallet_block_b02](https://huggingface.co/datasets/nvidia/simready-assets/tree/653d91e182011f357cf1644013cc94ddf194d349/pallet_block_b02) | NVIDIA SimReady Assets | 2.160 → 1.200 |
| `sr_pallet_load` | Beladene Palette | [pallet_composition_a13](https://huggingface.co/datasets/nvidia/simready-assets/tree/653d91e182011f357cf1644013cc94ddf194d349/pallet_composition_a13) | NVIDIA SimReady Assets | 68.432 → 5.662 |
| `sr_ibc` | IBC-Tank | [ibcTank_a02](https://huggingface.co/datasets/nvidia/simready-assets/tree/653d91e182011f357cf1644013cc94ddf194d349/ibcTank_a02) | NVIDIA SimReady Assets | 135.566 → 5.983 |
| `sr_compressor` | Kompressor | [equipment_airCompressor_a01](https://huggingface.co/datasets/nvidia/simready-assets/tree/653d91e182011f357cf1644013cc94ddf194d349/equipment_airCompressor_a01) | NVIDIA SimReady Assets | 537.696 → 7.940 |
| `sr_valve` | Absperrschieber | [gate_valve](https://huggingface.co/datasets/ForgeCAD/simready-assets/tree/0573053fa2c9174ca2a4678a462ec4dfd3035d59/gate_valve) | ForgeCAD SimReady Assets | 166.700 → 8.290 |
| `sr_pipes` | Rohrbündel | [pipe_plastic_gray_b06](https://huggingface.co/datasets/nvidia/simready-assets/tree/653d91e182011f357cf1644013cc94ddf194d349/pipe_plastic_gray_b06) | NVIDIA SimReady Assets | 69.086 → 3.000 |
| `sr_sign` | Warnaufsteller | [sign_warning_a04](https://huggingface.co/datasets/nvidia/simready-assets/tree/653d91e182011f357cf1644013cc94ddf194d349/sign_warning_a04) | NVIDIA SimReady Assets | 34.904 → 2.994 |
| `sr_cones` | Leitkegel und Absperrgitter | [traffic_cones_barriers](https://huggingface.co/datasets/ForgeCAD/simready-assets/tree/11da4ca51b2bd4b44d2ba7a5f46bb37342615e42/traffic_cones_barriers) | ForgeCAD SimReady Assets | 76.584 → 8.000 |
| `sr_stone` | Steinplatte | [stone_tile_KQKG74NK1J](https://huggingface.co/datasets/ALLSIDES/simready-scanned-objects/tree/50a309f9dda21778808cbea1c10f0918c2b9e42a/stone_tile_KQKG74NK1J) | Allsides SimReady Assets | 599.996 → 250 |
| `sr_slat_crate` | Lattenkiste | [D00480](https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library/tree/69a87329036cec0440403351c12230970c4e67d8/D00480) | Palatial Platforms — Palatial NV Asset Library | 110.794 → 1.468 |
| `sr_wood_box` | Holzkiste | [D00953](https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library/tree/69a87329036cec0440403351c12230970c4e67d8/D00953) | Palatial Platforms — Palatial NV Asset Library | 63.603 → 958 |
| `sr_basket_tall` | Hoher Weidenkorb | [basket_KN6XD9DJXF](https://huggingface.co/datasets/ALLSIDES/simready-scanned-objects/tree/9ef058fe70e02669a8afd9d52fd44ce8815b0af1/basket_KN6XD9DJXF) | Allsides SimReady Assets | 605.088 → 1.800 |
| `sr_basket` | Korb mit Leinen | [D00016](https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library/tree/69a87329036cec0440403351c12230970c4e67d8/D00016) | Palatial Platforms — Palatial NV Asset Library | 228.463 → 1.752 |
| `sr_chair` | Holzstuhl | [dining_chair](https://huggingface.co/datasets/ForgeCAD/simready-assets/tree/0573053fa2c9174ca2a4678a462ec4dfd3035d59/dining_chair) | ForgeCAD SimReady Assets | 76.468 → 1.396 |
| `sr_stool` | Fußbank | [footstool_RP2B4ZW65H](https://huggingface.co/datasets/ALLSIDES/simready-scanned-objects/tree/8ea302b2d721a31ee53a44875a34900982cca6d8/footstool_RP2B4ZW65H) | Allsides SimReady Assets | 600.004 → 700 |

Aus `drum_steel_a01` entsteht außerdem `sr_drum_red` (Sprengfass).

Hinweis von Palatial: „NVIDIA and other product names identify compatibility or depicted objects; no
trademark license or endorsement is granted.“ Das Spiel steht in keiner Verbindung zu den genannten Anbietern.
