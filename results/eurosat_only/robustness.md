# Robustness

| condition | macro F1 | accuracy | delta vs clean |
|---|---|---|---|
| clean | 0.7935 | 0.8025 | — |
| gaussian @ 0.05 | 0.7247 | 0.7275 | -0.0688 |
| gaussian @ 0.15 | 0.7187 | 0.7300 | -0.0748 |
| gaussian @ 0.30 | 0.5931 | 0.6150 | -0.2004 |
| salt_pepper @ 0.05 | 0.7205 | 0.7300 | -0.0730 |
| salt_pepper @ 0.15 | 0.6396 | 0.6600 | -0.1539 |
| salt_pepper @ 0.30 | 0.5444 | 0.5625 | -0.2491 |
| band_dropout @ 0.05 | 0.7893 | 0.7950 | -0.0042 |
| band_dropout @ 0.15 | 0.7620 | 0.7700 | -0.0315 |
| band_dropout @ 0.30 | 0.7411 | 0.7550 | -0.0524 |

### Per-class F1 (clean vs worst condition: salt_pepper @ 0.30)

| class | clean F1 | worst-case F1 |
|---|---|---|
| AnnualCrop | 0.8101 | 0.6377 |
| Forest | 0.8846 | 0.6882 |
| HerbaceousVegetation | 0.6667 | 0.4800 |
| Highway | 0.7324 | 0.0800 |
| Industrial | 0.9250 | 0.9589 |
| Pasture | 0.7586 | 0.4789 |
| PermanentCrop | 0.6061 | 0.4478 |
| Residential | 0.8889 | 0.5862 |
| River | 0.7586 | 0.4359 |
| SeaLake | 0.9038 | 0.6506 |
