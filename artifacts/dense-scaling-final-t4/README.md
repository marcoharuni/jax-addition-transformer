# Final dense scaling experiment

Verified results from 120 Colab T4 runs: five model sizes, eight training budgets, and three seeds per point.

> D is full sequence tokens processed from a fixed pool of 200,000 unique addition pairs. Repeated examples count as additional token exposures.

## Mean greedy exact match across three seeds

| Steps | 162K | 644K | 2.16M | 5.12M | 10.00M |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.065% ± 0.017% | 0.103% ± 0.038% | 0.123% ± 0.019% | 0.183% ± 0.060% | 0.132% ± 0.042% |
| 75 | 0.088% ± 0.046% | 0.538% ± 0.300% | 1.160% ± 1.089% | 7.477% ± 8.061% | 0.333% ± 0.185% |
| 100 | 0.135% ± 0.015% | 4.173% ± 2.371% | 26.457% ± 18.133% | 55.663% ± 13.743% | 54.173% ± 38.196% |
| 125 | 0.140% ± 0.026% | 41.963% ± 3.005% | 52.735% ± 37.660% | 83.740% ± 20.244% | 79.458% ± 23.239% |
| 175 | 1.760% ± 0.555% | 81.120% ± 6.495% | 98.267% ± 2.873% | 99.998% ± 0.003% | 99.997% ± 0.006% |
| 250 | 5.552% ± 1.121% | 97.837% ± 2.234% | 99.985% ± 0.026% | 99.998% ± 0.003% | 100.000% ± 0.000% |
| 400 | 88.407% ± 9.366% | 99.977% ± 0.032% | 100.000% ± 0.000% | 100.000% ± 0.000% | 100.000% ± 0.000% |
| 750 | 99.997% ± 0.006% | 100.000% ± 0.000% | 100.000% ± 0.000% | 100.000% ± 0.000% | 100.000% ± 0.000% |

## Exact-match transition

| Model | Parameters | ≥50% EM | ≥95% EM | ≥99% EM |
|---|---:|---:|---:|---:|
| dense_0p16m | 162,176 | 400 | 750 | 750 |
| dense_0p64m | 643,840 | 175 | 250 | 400 |
| dense_2p16m | 2,164,608 | 125 | 175 | 250 |
| dense_5p12m | 5,123,584 | 100 | 175 | 175 |
| dense_10m | 10,000,000 | 100 | 175 | 175 |

## Interpretation

- Larger models cross the algorithmic transition with fewer token exposures.
- Exact match saturates sharply, while validation loss continues to separate models.
- Three seeds quantify run-to-run variation around the transition.
- The scaling-law fit must pass an identifiability check; the analysis does not force exponents.
