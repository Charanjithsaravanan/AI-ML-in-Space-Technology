# Robustness

| condition | macro F1 | accuracy | delta vs clean |
|---|---|---|---|
| clean | 0.8128 | 0.8250 | — |
| gaussian @ 0.05 | 0.7312 | 0.7350 | -0.0816 |
| gaussian @ 0.15 | 0.6342 | 0.6625 | -0.1786 |
| gaussian @ 0.30 | 0.5756 | 0.6025 | -0.2372 |
| salt_pepper @ 0.05 | 0.6661 | 0.6825 | -0.1467 |
| salt_pepper @ 0.15 | 0.5965 | 0.6225 | -0.2163 |
| salt_pepper @ 0.30 | 0.5252 | 0.5600 | -0.2876 |
| band_dropout @ 0.05 | 0.7763 | 0.7925 | -0.0364 |
| band_dropout @ 0.15 | 0.7397 | 0.7575 | -0.0731 |
| band_dropout @ 0.30 | 0.7423 | 0.7650 | -0.0705 |

### Per-class F1 (clean vs worst condition: salt_pepper @ 0.30)

| class | clean F1 | worst-case F1 |
|---|---|---|
| AnnualCrop | 0.8533 | 0.6301 |
| Forest | 0.9126 | 0.7179 |
| HerbaceousVegetation | 0.7253 | 0.4375 |
| Highway | 0.6154 | 0.0833 |
| Industrial | 0.9487 | 0.8916 |
| Pasture | 0.7857 | 0.5373 |
| PermanentCrop | 0.6753 | 0.3492 |
| Residential | 0.8706 | 0.5882 |
| River | 0.7869 | 0.3871 |
| SeaLake | 0.9541 | 0.6292 |
