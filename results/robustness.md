# Robustness

| condition | macro F1 | accuracy | delta vs clean |
|---|---|---|---|
| clean | 0.8097 | 0.8225 | — |
| gaussian @ 0.05 | 0.7224 | 0.7275 | -0.0873 |
| gaussian @ 0.15 | 0.6726 | 0.6850 | -0.1371 |
| gaussian @ 0.30 | 0.5923 | 0.6075 | -0.2175 |
| salt_pepper @ 0.05 | 0.6884 | 0.7050 | -0.1213 |
| salt_pepper @ 0.15 | 0.6407 | 0.6575 | -0.1691 |
| salt_pepper @ 0.30 | 0.5807 | 0.5900 | -0.2290 |
| band_dropout @ 0.05 | 0.8068 | 0.8200 | -0.0030 |
| band_dropout @ 0.15 | 0.7893 | 0.8000 | -0.0205 |
| band_dropout @ 0.30 | 0.7365 | 0.7575 | -0.0732 |

### Per-class F1 (clean vs worst condition: salt_pepper @ 0.30)

| class | clean F1 | worst-case F1 |
|---|---|---|
| AnnualCrop | 0.8861 | 0.6197 |
| Forest | 0.9216 | 0.6829 |
| HerbaceousVegetation | 0.7010 | 0.5600 |
| Highway | 0.6176 | 0.2456 |
| Industrial | 0.9610 | 0.9459 |
| Pasture | 0.7368 | 0.5091 |
| PermanentCrop | 0.6575 | 0.5116 |
| Residential | 0.8861 | 0.6061 |
| River | 0.7667 | 0.4524 |
| SeaLake | 0.9630 | 0.6739 |
