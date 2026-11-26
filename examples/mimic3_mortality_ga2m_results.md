# GA2M on 48h MIMIC-III mortality: example/ablation results

Script: `examples/mimic3_mortality_ga2m.py`. Reproduces the mechanism of
Hegselmann et al. (2020) (GA2M, adapted from the Harutyunyan et al. 2019
in-hospital mortality benchmark) against `pyhealth.models.GA2M` and
`pyhealth.tasks.InHospitalMortality48hMIMIC3`: a bin-count ablation
(B in {32, 64, 128, 256}) and the paper's two-phase pairwise-interaction
selection (train all candidate pairs at coarse bins, rank by contribution
variance on the training split, retrain with only the top 34).

## What this is, and is not

This script runs on **synthetic data by default** (`--root` switches it to
a real `MIMIC3Dataset` + `InHospitalMortality48hMIMIC3`, which needs
PhysioNet credentials). The synthetic cohort is generated from one latent
"severity" value per patient that drives both the 17 clinical variables
(with injected missingness and noise) and the mortality label, so the
model has a real, if simplistic, signal to learn.

**These numbers are not a reproduction of the paper's results** (which
require the real MIMIC-III cohort, the official mimic3-benchmarks
preprocessing pipeline, and the InterpretML EBM baseline) — that
reproduction is run by the team locally against real MIMIC-III data,
outside this PyHealth contribution, per the project's data rule (never
read, print, or commit real MIMIC data in this repository). This script
and its output exist to demonstrate that the model, task, bin-count
ablation, and pairwise-interaction-selection pipeline all run correctly
end to end on data anyone can reproduce without data access.

## Last run

Command: `python examples/mimic3_mortality_ga2m.py` (all defaults,
`--seed 0`; 600 synthetic patients, split 420/59/121 train/val/test by
patient). ~22s on CPU.

### Bin-count ablation (1D terms only)

| Bins | ROC-AUC | PR-AUC |
|-----:|--------:|-------:|
|   32 |   0.658 |  0.474 |
|   64 |   0.688 |  0.521 |
|  128 |   0.656 |  0.502 |
|  256 |   0.610 |  0.461 |

On this synthetic cohort, 64 bins gives the best test-set discrimination;
256 bins (the paper's reported default) is the weakest here, consistent
with the clinicians' observation in the paper that very fine binning can
produce erratic, overfit risk curves — the motivation named in the
project's proposal for running this exact ablation. With ~600 synthetic
patients (420 for training) and no clinically meaningful fine structure
in the synthetic generator, this is expected: finer bins have fewer
samples per bin and overfit faster. On the real MIMIC-III cohort (tens of
thousands of patients), finer binning is expected to behave more
favorably, as in the paper.

### Pairwise-interaction selection

All `C(34, 2) = 561` candidate pairs were trained jointly with the 1D
terms at 8 coarse bins, ranked by contribution variance on the training
split, and the top 34 were kept for a final model retrained at 256 bins:

| Configuration | ROC-AUC | PR-AUC |
|---|---:|---:|
| 1D + top-34 pairs @ 256 bins | 0.601 | 0.466 |

The synthetic generator has no real pairwise interaction structure (every
variable's mean is driven independently by the same scalar "severity"),
so the interaction terms are not expected to beat the best 1D-only
ablation result above on this data — the point of this run is to confirm
the selection procedure (train-all -> rank -> retrain) executes correctly
and produces a valid, differently-shaped model, not to demonstrate a lift
from interactions (that comparison is meaningful only on real data with
genuine pairwise structure, e.g. the paper's reported
pH x respiratory-rate interaction).

## Reproducing

```bash
python examples/mimic3_mortality_ga2m.py                 # synthetic (default)
python examples/mimic3_mortality_ga2m.py --root /path/to/mimic-iii/1.4  # real data
```

See `--help` for all ablation knobs (`--bins`, `--coarse-bins`,
`--n-pairs`, `--epochs`, `--lr`, `--seed`).
