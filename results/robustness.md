# Robustness

| condition | macro F1 | accuracy | delta vs clean |
|---|---|---|---|
| clean | 0.9608 | 0.9625 | — |
| gaussian @ 0.05 | 0.8874 | 0.8950 | -0.0733 |
| gaussian @ 0.15 | 0.8242 | 0.8325 | -0.1366 |
| gaussian @ 0.30 | 0.7438 | 0.7500 | -0.2169 |
| salt_pepper @ 0.05 | 0.8692 | 0.8700 | -0.0915 |
| salt_pepper @ 0.15 | 0.7973 | 0.8050 | -0.1635 |
| salt_pepper @ 0.30 | 0.7359 | 0.7350 | -0.2249 |
| band_dropout @ 0.05 | 0.9653 | 0.9675 | +0.0046 |
| band_dropout @ 0.15 | 0.9502 | 0.9525 | -0.0105 |
| band_dropout @ 0.30 | 0.9158 | 0.9225 | -0.0449 |

### Per-class F1 (clean vs worst condition: salt_pepper @ 0.30)

| class | clean F1 | worst-case F1 |
|---|---|---|
| AnnualCrop | 0.9487 | 0.7761 |
| Forest | 0.9899 | 0.8182 |
| HerbaceousVegetation | 0.9400 | 0.5794 |
| Highway | 0.9639 | 0.6462 |
| Industrial | 0.9867 | 0.9444 |
| Pasture | 0.9630 | 0.7451 |
| PermanentCrop | 0.8857 | 0.5476 |
| Residential | 1.0000 | 0.7765 |
| River | 0.9492 | 0.7018 |
| SeaLake | 0.9808 | 0.8235 |
