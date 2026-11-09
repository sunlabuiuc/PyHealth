# GRAFT on SUPPORT2 and processed SUPPORT

GRAFT learns a score that orders subjects by survival time. A local
Kaplan-Meier (KM) estimate supplies possible event times for censored training
subjects. After ranking training, a separate Cox model converts the scores
into survival probabilities. Fitting this Cox model does not guarantee that
the probabilities are accurate.

Use `fit`, `fit_calibration`, `predict_score`, `predict_survival` and
`evaluate_survival` in that order. The generic Trainer does not provide this
survival fitting workflow.

Original paper: Mohammad Ashhad, Robert Hoehndorf and Ricardo Henao,
[GRAFT: Decoupling Ranking and Calibration for Survival Analysis](https://arxiv.org/abs/2602.07884v2),
2026, supplied preprint under review. Adapted code retains the
[author implementation's MIT notice](https://github.com/anonymous-785-u/GRAFT/tree/49d9d747869fd9410bf329baa57289a6497f9d60).
The correlation study extends those audited experiments. Wider literature
novelty was not assessed.

## Installation and small example

Verified platform: Linux/WSL2, Python 3.12, CPU with two torch threads.
Torchsort must match PyTorch/Python; building it requires a C++ compiler and
Python development tools. From this checkout, for the tested CPU versions:

```bash
python -m pip install torch==2.10.0+cpu torchvision==0.25.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install setuptools wheel ninja numpy==2.2.6
MAX_JOBS=2 python -m pip install --no-build-isolation torchsort==0.1.10
python -m pip install -c examples/support2_survival_graft_constraints.txt '.[graft]'
python examples/support2_survival_graft.py --output /tmp/graft-demo
```

The clean-install check used a new venv without shared site packages, a built
contribution wheel and the CPU torchsort wheel previously compiled from source.
That verifies this combination, not every Python, OS or CUDA build. Inference
from a checkpoint does not import torchsort, lifelines or scikit-survival.

The default creates 96 fabricated SUPPORT2 rows, uses the existing loader and
task, splits 62/14/20 and runs at most eight epochs. It demonstrates train-only
preprocessing/local KM, restored validation-best fitting, calibration, metrics
and exact checkpoint reload. Temporary files are cleaned up without `--output`.
An explicit output directory retains metrics, checkpoint and local reload
inputs. Existing results are never replaced. Nothing is downloaded.

```bash
python examples/support2_survival_graft.py --variant no_stg \
  --noise-multiplier 3 --rho 0.9 --output /tmp/graft-no-stg
python examples/support2_survival_graft.py --variant linear \
  --encoding benchmark14 --noise-multiplier 3 --rho 0 \
  --output /tmp/graft-linear
```

No STG retains the residual MLP without gates or gate penalties. Linear only
has neither. These synthetic outputs verify execution, not benchmark accuracy.

## Raw data and the verified benchmark

`--root /path/to/data` explicitly selects an existing `support2.csv`. Its ID
column must be `sno`; rename an `id` column if that is how the UCI copy was
distributed. The task preserves censoring and missing race as a category,
excludes missing required numerical predictors and uses positive follow-up.
The default onehot22 encoding and optional benchmark14 encoding both use a
fixed 65/15/20 example split. Changing encoding does not reproduce the study CV.
Three times the feature count is 66 added columns for onehot22 and 42 for
benchmark14.

The study uses the processed DeepSurv/pycox SUPPORT cohort: 8,873 subjects,
6,036 events, 2,837 censored outcomes and 14 predictors `x0`-`x13`.
`--benchmark-file` selects an existing feather file and verifies its
content/order checksum. This separate, explicit acquisition command downloads
the optional pycox dataset; the example does not download it:

```bash
python -m pip install -c examples/support2_survival_graft_constraints.txt \
  pycox==0.3.0 feather-format==0.4.1
python -c "from pycox import datasets; datasets.support.read_df().to_feather('support.feather')"
python examples/support2_survival_graft.py --benchmark-file support.feather \
  --fold 0 --seed 42 --variant full --output /tmp/support-full-f0-s42
python examples/support2_survival_graft.py --benchmark-file support.feather \
  --fold 0 --seed 42 --variant no_stg --noise-multiplier 3 --rho 0.9 \
  --output /tmp/support-no-stg-rho09-f0-s42
```

Processed mode runs one requested cell with a 1,000-epoch cap and validation
stopping. KFold(3, shuffle=True, random_state=42) fixes the outer partitions.
Inner validation reserves 20%, stratified by event with seed 4242+fold. Source
rows are sorted. Training seeds are 42, 43 and 44. There is no train-plus-
validation refit. The CLI never launches a full matrix.

For explicit repetition of all 90 GRAFT-family study fits:

```python
import itertools
import subprocess
import sys

for variant, fold, seed in itertools.product(
    ["full", "no_stg", "linear"], range(3), [42, 43, 44]
):
    conditions = [(0, 0.0), (3, 0.0)]
    if variant != "linear":
        conditions.extend([(3, 0.5), (3, 0.9)])
    for multiplier, rho in conditions:
        destination = f"runs/{variant}-f{fold}-s{seed}-n{multiplier}-r{rho}"
        subprocess.run(
            [sys.executable, "examples/support2_survival_graft.py",
             "--benchmark-file", "support.feather", "--variant", variant,
             "--fold", str(fold), "--seed", str(seed),
             "--noise-multiplier", str(multiplier), "--rho", str(rho),
             "--output", destination],
            check=True,
        )
```

Clean and rho=0 cells are run once and reused across scientific views. The
scalar CSV also includes three deterministic CoxPH and nine DeepSurv fits.
CoxPH used lifelines penalizer 0.01. DeepSurv used pycox CoxPH/torchtuples
MLPVanilla, hidden (32,32), batch norm, dropout 0.2, no output bias, Adam 0.01
and the same validation C-index/restored-best rule. Both fitted preprocessing
and baseline hazards on inner training. The CLI covers GRAFT architectures.

## Fixed training and evaluation

GRAFT uses hidden width 32, tanh, dropout 0.2, sigma/initial eta 0.5, Adam
0.001, five draws, batch size 64, gate penalty 0.01 and explicit L2 0.0001
with optimizer decay zero. Torchsort L2 strength is 0.1; target ranks use
average ties. Final remainders are retained. Validation raw-score C-index
selects the earliest accepted best state, min_delta 0.0001, patience 10.

The generator draws cohort-indexed epsilon `(8873,42)`, then nine common block
variables from PCG64 SeedSequence([5982026, seed, 1]). Column j is
`sqrt(rho)*U_[j//5]+sqrt(1-rho)*epsilon_j`. Blocks have five columns, with two
in the last block. Base arrays are shared across rho, folds and models. No
predictors or outcomes enter generation. Independence is by construction;
finite sample associations need not vanish. Each condition separately fits
training preprocessing and ungated local neighborhoods.

Local KM excludes self, requires 30 neighbors/ten events, caps at 199 nonself
neighbors and uses standardized Euclidean distance with source-row tie breaks.
Conditional draws renormalize represented finite future event masses.
When there is no represented future event, the fallback is 1.1 times the
censoring time. It affects 37.3%-47.5% of censored training subjects in E3,
so it is a substantial approximation. The event distribution beyond the
observed local support remains unidentified.

Conditional independent censoring is an assumption of the imputation method;
it was not verified from this cohort. Fixed small neighborhoods, renormalized
finite event masses and the fallback do not establish asymptotic consistency.
The rank-correlation objective is a surrogate: improving it need not improve
C-index for every score vector.

C-index uses all original held-out outcomes and raw survival scores. IBS
uses training-estimated reverse-KM weights, 50 training-defined times from
q05 to just below H, and all subjects. H is min(training q90, last censoring
time with G>=0.05): 1477.9/1476.7/1443.9 days by fold. Late outcomes are
administratively censored for IBS only. There is no clipping or dropping.
The unpenalized training-only Cox/Breslow calibrator accounts for centering,
uses linear cumulative-hazard interpolation and flags nonnegative slopes.
Test scores are never flipped. IBS is prediction error on this grid, not a
probability-calibration guarantee.

## Measured results

`support2_survival_graft_results.csv` contains all 102 physical fit rows and
no subject records, outcomes, predictions or checkpoints. All fits completed
with independent metric and exact reload checks. E1 compares models on clean
data; E2 compares architectures with and without independent noise; E3 varies
noise correlation. They contain 27/54/54 result cells, including reused fits.
Means give equal weight to nine fold/seed cells; CoxPH reuses three
deterministic fits. Higher C-index means better ordering of comparable
outcomes; lower integrated Brier score (IBS) means less prediction error on
the specified time grid. IBS measures more than probability calibration.

Clean means, with paper values from supplied Table 1:

| Model | Project C-index | Paper C-index | Project IBS | Paper IBS |
| --- | ---: | ---: | ---: | ---: |
| Full GRAFT | 0.5895 | 0.6143 | 0.2122 | 0.1891 |
| CoxPH | 0.5664 | 0.5686 | 0.2172 | 0.2059 |
| DeepSurv | 0.6097 | 0.6071 | 0.2048 | 0.1956 |

This controlled partial reproduction changes preprocessing boundaries,
fold/seed design, validation/restoration, target ties, remainders and L2 from
the author executable. Its IBS grid and censoring convention also differ;
absolute IBS is not a same-interval replication. No cause of the score gap
was isolated. GRAFT's clean-data advantage over DeepSurv was not reproduced.

| Model | Clean C-index | +42 Gaussian C-index | Clean IBS | +42 Gaussian IBS |
| --- | ---: | ---: | ---: | ---: |
| Full GRAFT | 0.5895 | 0.5754 | 0.2122 | 0.2166 |
| No STG | 0.6122 | 0.5662 | 0.2059 | 0.2220 |
| Linear only | 0.5709 | 0.5626 | 0.2191 | 0.2210 |

Correlated nuisance, with positive contrasts favoring full GRAFT:

| rho | Full C-index | No STG C-index | Delta C | IBS no STG minus full |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.5754 | 0.5662 | +0.0093 | +0.0054 |
| 0.5 | 0.5721 | 0.5697 | +0.0023 | +0.0036 |
| 0.9 | 0.5802 | 0.5703 | +0.0099 | +0.0048 |

At rho=0.5 and 0.9, only seed 43 has a positive fold-averaged C-index contrast;
the other two seed means favor No STG. Full GRAFT wins in 5/9 paired cells at
both levels. Mean benefits are seed dependent and do not increase monotonically.

Every nuisance gate is below the predeclared clipped-eta threshold 0.5.
Mean positive nuisance-gate counts remain 19.67/21.22/14.67 of 42 across rho.
Empty selected nuisance sets yield Jaccard=1, which is not stable feature
recovery or removal of all noise. Original predictors are not known true
signals; gates do not establish causal relevance. Correlation also changes
which subjects become local-KM neighbors, so this comparison does not isolate
the effect of gates. SUPPORT
physiology is largely measured on day 3 with study-entry follow-up. This is
not an admission-time or clinical-use validation.

## Variation across seeds and folds

The following table reports sample standard deviations (ddof=1) for every
model/condition. Each pair is **(SD of three seed means, SD of three fold
means)**: first average over folds within each seed, or over seeds within
each fold, then compute the corresponding SD. These describe sensitivity
within this fixed design. Overlapping CV partitions do not provide
independent replicates for confidence intervals or p-values. CoxPH has zero
seed SD because the same three deterministic fits are reused, not because
three initializations produced identical estimates.

| Condition | Model | C-index SD (seed, fold) | IBS SD (seed, fold) |
| --- | --- | ---: | ---: |
| Clean | Full GRAFT | (0.0051, 0.0194) | (0.0022, 0.0079) |
| Clean | CoxPH | (0.0000, 0.0152) | (0.0000, 0.0095) |
| Clean | DeepSurv | (0.0012, 0.0096) | (0.0009, 0.0089) |
| Clean | No STG | (0.0009, 0.0104) | (0.0002, 0.0086) |
| Clean | Linear only | (0.0014, 0.0125) | (0.0003, 0.0085) |
| +42, rho=0 | Full GRAFT | (0.0052, 0.0139) | (0.0023, 0.0080) |
| +42, rho=0 | No STG | (0.0022, 0.0090) | (0.0016, 0.0113) |
| +42, rho=0 | Linear only | (0.0020, 0.0119) | (0.0003, 0.0089) |
| +42, rho=0.5 | Full GRAFT | (0.0040, 0.0152) | (0.0019, 0.0096) |
| +42, rho=0.5 | No STG | (0.0072, 0.0108) | (0.0019, 0.0092) |
| +42, rho=0.9 | Full GRAFT | (0.0190, 0.0094) | (0.0065, 0.0081) |
| +42, rho=0.9 | No STG | (0.0051, 0.0131) | (0.0006, 0.0085) |

## Tests

```bash
python -m unittest tests.core.test_graft tests.core.test_survival_time_support2 -v
```

The 37 tests use at most five fabricated subjects, download no data and clean
temporary files. They cover schemas, preprocessing, gates/penalties, ranks,
gradients, KM tails/ties, training boundaries, validation restoration,
calibration, curve support and Brier/IBS conventions. The larger synthetic
example and real benchmark fits are separate checks.
