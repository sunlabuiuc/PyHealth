# Neural Fine-Gray reproduction and ablation

This directory records the setup and evidence for the Neural Fine-Gray
contribution. The implementation targets Jeanselme et al., *Neural Fine-Gray:
Monotonic Neural Networks for Competing Risks* (CHIL 2023).

## Reproducible commands

The offline smoke experiment requires no clinical data:

```bash
python examples/framingham_competing_risk_neural_fine_gray.py \
  --samples 600 --epochs 20 --seed 7
```

The Framingham route downloads the pinned public source used by the paper's
reference dependency when `framingham.csv` is absent:

```bash
python examples/framingham_competing_risk_neural_fine_gray.py \
  --data-root ./data/framingham --download --epochs 100 --seed 7
```

Both commands compare the paper parameterization
`1 - exp(-t M_r(t, x))` with the SumoNet-style ablation
`1 - exp(-M_r(t, x))`. The latter does not guarantee zero cumulative incidence
at time zero. The script reports cause-1 time-dependent concordance, IPCW Brier
score, and test negative log-likelihood at the 25th, 50th, and 75th percentiles
of uncensored training event times.

## Expected invariants

The synthetic unit tests independently verify that:

- every cause-specific cumulative incidence starts at zero in NFG mode;
- each cumulative incidence is non-decreasing in time;
- cumulative incidence summed across causes never exceeds one;
- automatic derivatives agree with central finite differences; and
- model loss and parameter gradients are finite.

## Recorded synthetic ablation

The documented command was run on November 13, 2026 with Python 3.13.13,
PyTorch 2.15.0, CPU execution, 600 samples, 20 epochs, seed 7, and the paper's
maximum-training-duration time scaling. Full-precision output is stored in
`synthetic_seed7.json`.

| Mode | Test NLL | C-index q25/q50/q75 | Brier q25/q50/q75 |
| --- | ---: | --- | --- |
| NFG `t M_r` | 0.895 | 0.743 / 0.716 / 0.728 | 0.119 / 0.184 / 0.192 |
| No time factor | 1.366 | 0.668 / 0.679 / 0.690 | 0.122 / 0.195 / 0.207 |

On this controlled dataset, the NFG parameterization improved test NLL and all
six horizon-specific discrimination/calibration measurements. This synthetic
experiment is a pipeline check and ablation, not a claim that the paper's
Framingham numbers have been reproduced. A Framingham run should be reported
separately with its split and search budget.

## Recorded Framingham baseline

The full PyHealth dataset → task → model → metrics pipeline was also run with
the 4,434-participant Framingham cohort, a fixed 70/15/15 patient split, seed 7,
20 epochs, batch size 128, and no hyperparameter search. The paper uses 5-fold
cross-validation and a 100-iteration search, so this is a reduced-budget
reproduction rather than a direct estimate of the paper's reported mean and
standard deviation. Full output is stored in `framingham_seed7.json`.

| Mode | Test NLL | C-index q25/q50/q75 | Brier q25/q50/q75 |
| --- | ---: | --- | --- |
| NFG `t M_r` | 2.313 | 0.896 / 0.769 / 0.713 | 0.046 / 0.106 / 0.150 |
| No time factor | 2.404 | 0.891 / 0.770 / 0.715 | 0.045 / 0.106 / 0.149 |

The NFG parameterization had lower held-out NLL, while horizon metrics were
mixed: NFG was better at early concordance and essentially tied or slightly
worse on several later-horizon values. Relative to the paper's CVD targets
(C-index 0.872/0.812/0.782; Brier 0.050/0.095/0.128), this split matched the
early horizon but underperformed at the median and late horizons. Plausible
causes are the reduced training/search budget, a different split, and the lack
of the paper's 5-fold aggregation. This is a partial reproduction, not support
for claiming that the complete H1 target was met.
