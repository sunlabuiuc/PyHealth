# Sequence Transformer Network

This adds the model from [Oh, Wang and Wiens (MLHC 2018)](https://proceedings.mlr.press/v85/oh18a.html)
to PyHealth. It learns a time warp and a value scale/shift for each sequence,
then passes the transformed sequence to a CNN.

Contributors: Matthew Nguyen (`mnguyen358`, `mnguyen358@gatech.edu`) and
Alireza Ghiam (`aghiam3`, `aghiam3@gatech.edu`).

## Setup

These files are additions to PyHealth, not a standalone copy of the library.
Use Python 3.12 or 3.13. The recorded 60-fit experiment used PyHealth commit
`454b394340f6c2e078c9f54e09668ad79ae17199`. The final contribution was integrated
and its targeted tests checked against upstream commit
`178f09c86029ac3ba5cc3022771d1ad1b33a6164`. These are separate provenance records;
the stored experiment results are not presented as a new run on the later commit.

Copy the files into a PyHealth checkout, keeping their paths. The two existing
index files only need the Sequence Transformer entries; keep any newer upstream
entries when merging them.

```bash
python -m pip install -e '.[lint]' pytest
python -m pytest tests/core/test_sequence_transformer.py \
  tests/core/test_sequence_transformer_example.py -q
python examples/mimic3_mortality_sequence_transformer.py \
  --root /path/to/mimic-iii-clinical-database-demo-1.4
```

The example needs the unzipped [MIMIC-III demo](https://physionet.org/content/mimiciii-demo/1.4/).
The unit tests use synthetic data and do not need the demo. Do not add dataset
files or generated patient caches to the PR.

## What the experiment measures

We could not access full MIMIC-III, so this is a demo study of the model.
The example uses PyHealth's YAIB ICU mortality task, which has 25 hourly bins
(0 through 24). The paper uses 48 hours to predict in-hospital mortality.
The task, features, cohort, and data size therefore differ. Its scores should
not be treated as a direct reproduction of the paper's benchmark.

Only dynamic features are passed to the model. The stock YAIB task calculates
static height and weight using the whole stay, so including them would allow
later measurements into this prediction example. YAIB's cohort rules are
otherwise unchanged.

The experiment compares the baseline CNN, temporal transformation, magnitude
transformation, and both transformations. It also checks dropout settings
0, 0.2, and 0.5 for the full model. The first four configurations come from the
paper; the dropout comparison is our sensitivity analysis on the demo.
Patient splits keep all stays from the same patient together, and preprocessing
is fitted on the training split. Average precision is reported as AP.

## Results and checks

The corrected example completed all 60 fits (six configurations, ten patient
splits) on CPU with one PyTorch thread. After YAIB filtering, the cohort has
99 ICU stays from 70 patients, with 21 positive labels. Values below are mean ±
sample standard deviation across those splits.
The individual runs and environment versions are in
`results/mimic3_stn_dynamic_only.json`.

| Configuration | AUROC | Average precision |
| --- | ---: | ---: |
| baseline CNN | 0.493 ± 0.109 | 0.399 ± 0.081 |
| temporal only | 0.495 ± 0.106 | 0.396 ± 0.096 |
| magnitude only | 0.453 ± 0.126 | 0.356 ± 0.126 |
| STN (both) | 0.472 ± 0.140 | 0.368 ± 0.120 |
| STN, dropout 0 | 0.462 ± 0.132 | 0.370 ± 0.103 |
| STN, dropout 0.5 | 0.497 ± 0.126 | 0.380 ± 0.114 |

The full STN did not improve on the baseline in this experiment. Differences
between dropout settings were small compared with the variation across splits.
These overlapping splits are not independent trials, and this small demo does
not establish whether STN improves performance on the original task.

All 27 tests passed, including empty-sequence cases and a synthetic check that
later height/weight measurements do not change this example's model inputs or
predictions. Ruff lint and formatting checks passed. This feature check leaves
YAIB's retrospective cohort rules unchanged.

## Implementation choices

The localizer follows the paper's 1D convolution diagram. The authors' released
code uses 2D convolutions instead. We also use a continuous leaky clamp for the
transformation parameters and place configurable dropout after the classifier's
hidden layer. The released code has different clipping and dropout placement.
These are documented implementation choices, not claims of exact code parity.

The model accepts fixed tensors and single-level jagged numeric inputs. Short
sequences repeat their last value; empty sequences become zeros. With both
transformations disabled, it reduces to the CNN baseline.

## Before submission

Open one team PR with the paper link, contributor details, experiment results,
and the limitations above. The course also requires the team video and each
student's AI-use form. AI assistance with implementation and review should be
described in those forms, including the checks and corrections made afterward.
