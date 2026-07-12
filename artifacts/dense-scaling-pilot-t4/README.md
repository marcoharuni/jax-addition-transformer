# Dense scaling pilot on a Colab T4

This artifact summarizes 16 completed GPU runs from one random seed.
It is a pilot used to locate the task's transition and saturation regions.

## Greedy exact match

| Steps | 162K | 964K | 5.12M | 10.00M |
|---:|---:|---:|---:|---:|
| 50 | 0.045% | 0.170% | 0.115% | 0.110% |
| 125 | 0.125% | 49.240% | 99.825% | 99.940% |
| 250 | 21.690% | 100.000% | 100.000% | 100.000% |
| 750 | 100.000% | 100.000% | 100.000% | 100.000% |

## Validation loss

| Steps | 162K | 964K | 5.12M | 10.00M |
|---:|---:|---:|---:|---:|
| 50 | 1.91193199 | 1.75096915 | 1.71300914 | 1.71686904 |
| 125 | 1.74429147 | 0.60986944 | 0.04960600 | 0.02888434 |
| 250 | 0.92940896 | 0.02840898 | 0.00207388 | 0.00179550 |
| 750 | 0.01177865 | 0.00067921 | 0.00022261 | 0.00019109 |

## Scaling-law identifiability check

- status: `pilot_does_not_identify_stable_chinchilla_exponents`
- additive candidate alpha: 1.739671
- additive candidate beta: 4.108079
- additive log10 R-squared: 0.730071
- additive floor collapsed to zero: True
- multiplicative diagnostic alpha: 0.900052
- multiplicative diagnostic beta: 2.746754

The pilot does not support a defensible compute-optimal exponent. It has one seed, four model sizes, budget-specific learning-rate schedules, a sharp phase transition, and many saturated observations.

## Main observations

- All four models are undertrained at 50 steps.
- Larger models cross the exact-match transition with fewer examples.
- Exact match saturates before validation loss, so loss is the primary fit target.
- The 5.12M and 10M models show diminishing returns after the task is solved.
