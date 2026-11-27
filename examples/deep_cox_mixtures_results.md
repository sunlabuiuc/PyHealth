
# Deep Cox Mixtures: Reproduction and Ablation Study

## 1. Overview

This study implements Deep Cox Mixtures (DCM) within PyHealth
and evaluates its performance on the SUPPORT2 dataset.

DCM models heterogeneous patient populations using a mixture
of component-specific Cox proportional hazards models. The
implementation uses a shared neural network encoder, a
probabilistic gating network, component-specific risk
functions, and approximate expectation-maximization training.

The objectives are to:

- Evaluate DCM against conventional Cox proportional hazards.
- Compare DCM against a single-component deep Cox model.
- Investigate the effect of the number of mixture components.
- Evaluate discrimination and calibration across demographic groups.
- Assess reproducibility of the original published results.

## 2. Dataset

The SUPPORT2 dataset was loaded using the existing PyHealth
Support2Dataset and SurvivalTimeToEventSupport2 task.

| Dataset characteristic | Value |
|---|---:|
| Original patient population | 9,105 |
| Processed patients | 8,873 |
| Input features | 27 |
| Observed deaths | 6,036 |
| Censored patients | 2,837 |
| Censoring rate | 31.97% |
| White patients | 6,998 |
| Non-White patients | 1,834 |
| Unknown race | 41 |

Patients with unknown race were retained in the overall
evaluation but excluded from demographic subgroup comparisons.

## 3. Experimental Methodology

Five-fold stratified cross-validation was used to evaluate
all model configurations.

Stratification accounted for event status and demographic
group. Feature standardization was fitted exclusively on
the training fold to prevent leakage from test data.

Evaluation horizons were determined from the 25th, 50th,
and 75th percentiles of observed event times in each
training fold.

| Fold | Training patients | Test patients | 25th | 50th | 75th |
|---|---:|---:|---:|---:|---:|
| 1 | 7,098 | 1,775 | 14 | 58 | 253 |
| 2 | 7,098 | 1,775 | 13 | 57 | 251 |
| 3 | 7,098 | 1,775 | 14 | 57 | 248 |
| 4 | 7,099 | 1,774 | 14 | 58 | 253 |
| 5 | 7,099 | 1,774 | 14 | 56 | 247 |

Evaluation horizons are expressed in days.

### 3.1 Model Configurations

| Parameter | Setting |
|---|---|
| Hidden layers | 64, 32 |
| Activation | ReLU |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Training epochs | 100 |
| Gradient clipping | 5.0 |
| Dropout | 0.0 |
| Training strategy | Full-batch approximate EM |
| Baseline estimation | Breslow estimator |
| Interpolation | PCHIP |
| Cross-validation folds | 5 |
| Cross-validation random seed | 42 |
| Neural model random seed | 42 + fold |
| Mixture components | 1, 3, 4, 6 |

### 3.2 Benchmark Models

The following models were evaluated:

- CPH: Conventional Cox proportional hazards regression.
- DeepSurv-equivalent / DCM K=1: Single-component neural
  Cox proportional hazards model.
- DCM K=3: Three mixture components.
- DCM K=4: Four mixture components.
- DCM K=6: Six mixture components.

The single-component DCM configuration serves as a
DeepSurv-equivalent control under the matched neural
architecture and optimization procedure.

A total of 25 model-fold configurations were evaluated.

### 3.3 Evaluation Metrics

Three survival evaluation metrics were used:

- Ctd: Horizon-truncated inverse probability of censoring
  weighted concordance index. Higher values indicate
  better discrimination.
- ECE: Expected Calibration Error using 10 equal-width
  probability bins and IPCW weighting. Lower values
  indicate better calibration.
- Brier score: IPCW-adjusted squared prediction error.
  Lower values indicate better probabilistic accuracy.

Results are reported as the mean plus or minus the
descriptive fold-based standard error.

## 4. Overall Population Results

### 4.1 Concordance Index (Higher Is Better)

| Model | 25th percentile | 50th percentile | 75th percentile |
|---|---|---|---|
| CPH | 0.5541 ± 0.0041 | 0.5684 ± 0.0017 | 0.5906 ± 0.0020 |
| DeepSurv / K=1 | 0.5974 ± 0.0036 | 0.5955 ± 0.0035 | **0.6096 ± 0.0023** |
| DCM K=3 | **0.6352 ± 0.0067** | 0.6074 ± 0.0042 | 0.6094 ± 0.0027 |
| DCM K=4 | 0.6241 ± 0.0095 | **0.6077 ± 0.0065** | 0.6049 ± 0.0031 |
| DCM K=6 | 0.6304 ± 0.0071 | 0.6051 ± 0.0062 | 0.6047 ± 0.0026 |

### 4.2 Expected Calibration Error (Lower Is Better)

| Model | 25th percentile | 50th percentile | 75th percentile |
|---|---|---|---|
| CPH | 0.0313 ± 0.0037 | 0.0435 ± 0.0047 | **0.0169 ± 0.0020** |
| DeepSurv / K=1 | 0.0269 ± 0.0028 | 0.0411 ± 0.0041 | 0.0248 ± 0.0020 |
| DCM K=3 | 0.0222 ± 0.0038 | 0.0248 ± 0.0027 | 0.0199 ± 0.0011 |
| DCM K=4 | **0.0188 ± 0.0033** | **0.0231 ± 0.0044** | 0.0248 ± 0.0038 |
| DCM K=6 | 0.0194 ± 0.0030 | **0.0231 ± 0.0026** | 0.0196 ± 0.0038 |

### 4.3 Brier Score (Lower Is Better)

| Model | 25th percentile | 50th percentile | 75th percentile |
|---|---|---|---|
| CPH | 0.1449 ± 0.0026 | 0.2220 ± 0.0011 | 0.2330 ± 0.0008 |
| DeepSurv / K=1 | 0.1418 ± 0.0027 | 0.2181 ± 0.0014 | 0.2303 ± 0.0009 |
| DCM K=3 | **0.1383 ± 0.0031** | 0.2149 ± 0.0012 | **0.2298 ± 0.0008** |
| DCM K=4 | 0.1393 ± 0.0026 | **0.2148 ± 0.0016** | 0.2304 ± 0.0009 |
| DCM K=6 | 0.1390 ± 0.0029 | 0.2153 ± 0.0013 | 0.2302 ± 0.0010 |

### 4.4 Interpretation

DCM improved discrimination relative to conventional CPH
at all evaluated horizons.

The three-component configuration achieved the highest
short-term concordance, improving from 0.5541 for CPH
to 0.6352 for DCM K=3.

Mixture models also improved short-term and intermediate
calibration. However, CPH retained the lowest calibration
error at the longest evaluation horizon.

Differences between mixture configurations became smaller
at later horizons.

## 5. Mixture Component Ablation Study

The number of mixture components was varied while keeping
the hidden architecture, learning rate, training duration,
and evaluation procedure fixed.

The following values are arithmetic averages of the
reported performance across the three evaluation horizons.

| Model | Mean Ctd | Mean ECE | Mean Brier |
|---|---:|---:|---:|
| CPH | 0.5710 | 0.0306 | 0.2000 |
| DCM K=1 | 0.6008 | 0.0309 | 0.1967 |
| DCM K=3 | **0.6173** | 0.0223 | **0.1943** |
| DCM K=4 | 0.6122 | 0.0222 | 0.1948 |
| DCM K=6 | 0.6134 | **0.0207** | 0.1948 |

### Ablation Findings

Increasing the number of mixture components from one to
three improved average concordance, calibration, and
Brier score.

Increasing the number of components beyond three did not
consistently improve discrimination or Brier score.

DCM K=3 provided the strongest observed balance between
predictive discrimination, probabilistic accuracy, and
model complexity.

DCM K=6 achieved the lowest average calibration error.

These findings suggest diminishing returns from increasing
the number of latent survival subgroups beyond three under
the evaluated experimental configuration.

The horizon-averaged Brier score is a descriptive average,
not an integrated Brier score.

The observed differences are not claims of statistical
significance or independently validated model selection.

## 6. Non-White Population Results

### 6.1 Concordance Index

| Model | 25th percentile | 50th percentile | 75th percentile |
|---|---|---|---|
| CPH | 0.5812 ± 0.0222 | 0.5874 ± 0.0169 | 0.6051 ± 0.0119 |
| DeepSurv / K=1 | 0.6087 ± 0.0199 | 0.6055 ± 0.0133 | 0.6201 ± 0.0087 |
| DCM K=3 | **0.6462 ± 0.0210** | **0.6183 ± 0.0114** | **0.6220 ± 0.0106** |
| DCM K=4 | 0.6241 ± 0.0231 | 0.6116 ± 0.0165 | 0.6164 ± 0.0115 |
| DCM K=6 | 0.6395 ± 0.0221 | 0.6112 ± 0.0181 | 0.6178 ± 0.0122 |

### 6.2 Expected Calibration Error

| Model | 25th percentile | 50th percentile | 75th percentile |
|---|---|---|---|
| CPH | 0.0358 ± 0.0039 | 0.0508 ± 0.0080 | **0.0414 ± 0.0049** |
| DeepSurv / K=1 | 0.0407 ± 0.0040 | 0.0607 ± 0.0106 | 0.0503 ± 0.0060 |
| DCM K=3 | 0.0287 ± 0.0062 | 0.0512 ± 0.0067 | 0.0451 ± 0.0038 |
| DCM K=4 | **0.0272 ± 0.0041** | **0.0392 ± 0.0075** | 0.0433 ± 0.0026 |
| DCM K=6 | **0.0272 ± 0.0056** | 0.0450 ± 0.0061 | 0.0432 ± 0.0070 |

### 6.3 Brier Score

| Model | 25th percentile | 50th percentile | 75th percentile |
|---|---|---|---|
| CPH | 0.1422 ± 0.0068 | 0.2200 ± 0.0030 | 0.2304 ± 0.0031 |
| DeepSurv / K=1 | 0.1392 ± 0.0072 | 0.2165 ± 0.0033 | 0.2281 ± 0.0027 |
| DCM K=3 | **0.1359 ± 0.0066** | **0.2136 ± 0.0020** | **0.2268 ± 0.0031** |
| DCM K=4 | 0.1378 ± 0.0068 | 0.2142 ± 0.0022 | 0.2277 ± 0.0028 |
| DCM K=6 | 0.1366 ± 0.0067 | 0.2147 ± 0.0029 | 0.2270 ± 0.0031 |

### 6.4 Demographic Findings

DCM K=3 achieved the highest observed concordance and
lowest Brier score across all three evaluation horizons
for non-White patients.

DCM K=4 achieved the lowest average calibration error
among the evaluated models for this subgroup.

However, demographic calibration disparities persisted.

For DCM K=3, the calibration differences were:

| Percentile | White ECE | Non-White ECE | Difference |
|---|---:|---:|---:|
| 25th | 0.0241 | 0.0287 | 0.0046 |
| 50th | 0.0266 | 0.0512 | 0.0246 |
| 75th | 0.0266 | 0.0451 | 0.0185 |

The non-White subgroup also exhibited greater variability
in concordance estimates across cross-validation folds.

The smaller subgroup sample size may contribute to this
increased variability.

These results demonstrate that improved overall model
performance does not necessarily eliminate demographic
calibration disparities.

## 7. Comparison With the Original Paper

The following comparison uses the published DCM benchmark
values and the current three-component implementation.

| Metric | Percentile | Published DCM | Current DCM K=3 |
|---|---|---:|---:|
| Ctd | 25th | 0.7425 | 0.6352 |
| Ctd | 50th | 0.7057 | 0.6074 |
| Ctd | 75th | 0.6753 | 0.6094 |
| ECE | 25th | 0.0179 | 0.0222 |
| ECE | 50th | 0.0268 | 0.0248 |
| ECE | 75th | 0.0256 | 0.0199 |
| Brier | 25th | 0.1258 | 0.1383 |
| Brier | 50th | 0.1905 | 0.2149 |
| Brier | 75th | 0.2118 | 0.2298 |

The current implementation reproduced several qualitative
performance trends but did not reproduce the exact
published numerical results.

The most substantial performance gap occurred in
concordance.

Calibration errors were numerically comparable, although
differences between calibration estimators prevent direct
equivalence claims.

## 8. Uncertainty Estimation

The sample standard deviation across five cross-validation
folds was calculated for every model, metric, evaluation
horizon, and demographic subgroup.

Descriptive fold-based standard errors were calculated as:

    SE = standard deviation / sqrt(number of folds)

These estimates summarize variation across cross-validation
folds.

They are not equivalent to patient-level bootstrap standard
errors and do not account for dependence between overlapping
cross-validation training sets.

No formal confidence intervals or statistical significance
claims are derived from these estimates.

## 9. Implementation Validation

The implementation was validated using 20 deterministic
synthetic unit tests covering:

- Model initialization.
- Forward output dimensions.
- Mixture probability normalization.
- Cox partial likelihood.
- Tied event times.
- All-censored Cox loss.
- Breslow baseline estimation.
- Survival curve monotonicity.
- Approximate EM training.
- Posterior probability normalization.
- Gradient computation.
- Survival prediction.
- Single-component behavior.
- Invalid component counts.
- Invalid event indicators.
- Prediction before training.
- Trained model checkpoint persistence.
- Untrained model checkpoint persistence.
- Forward output interface.
- Additional forward keyword arguments.

All 20 tests passed.

The implementation also successfully trained on the complete
processed SUPPORT2 cohort through five-fold cross-validation.

A separate synthetic example demonstrates model training,
survival prediction, visualization, and checkpoint saving
without requiring access to SUPPORT2.

## 10. Reproduction Instructions

Run all commands from the PyHealth repository root.

### 10.1 Dependencies

Install PyHealth and the additional experimental dependencies:

    python -m pip install -e .
    python -m pip install scikit-survival matplotlib pytest

### 10.2 Synthetic Example

    python examples/deep_cox_mixtures_example.py

This generates:

- results/dcm_synthetic/synthetic_results.png
- results/dcm_synthetic/dcm_model.pt

### 10.3 SUPPORT2 Dataset

Place the SUPPORT2 CSV file in a local data directory.

The CSV must contain the patient identifier column, `sno`,
followed by the original SUPPORT2 variables.

If the source file contains an unnamed patient identifier
without a corresponding header, add the missing `sno`
header before loading the dataset.

The dataset is not included in this contribution.

### 10.4 Reproduction and Ablation

Replace DATA_DIRECTORY with the directory containing
support2.csv.

    python examples/support2_survival_deep_cox_mixtures.py --data-root "DATA_DIRECTORY" --epochs 100 --folds 5 --components 1 3 4 6

This generates:

- results/dcm/fold_results.csv
- results/dcm/summary_results.csv

### 10.5 Uncertainty Analysis

    python examples/deep_cox_mixtures_uncertainty.py

This generates:

- results/dcm/uncertainty_results.csv

### 10.6 Unit Tests

    python -m pytest tests/core/test_deep_cox_mixtures.py -v

## 11. Limitations and Reproduction Deviations

The following limitations should be considered when
interpreting the results:

1. The processed SUPPORT2 cohort contains 8,873 patients
   rather than the original population of 9,105 patients.

2. The existing PyHealth SUPPORT2 feature representation
   and preprocessing may differ from the original study.

3. PCHIP interpolation is used for baseline cumulative
   hazards rather than reproducing the original spline
   smoothing procedure exactly.

4. Training uses full-batch approximate EM with fixed
   architecture and optimization settings rather than
   extensive validation-based hyperparameter selection.

5. The single-component DCM configuration serves as a
   DeepSurv-equivalent control rather than an independently
   optimized DeepSurv implementation.

6. The calibration estimator uses 10 equal-width bins
   with IPCW weighting rather than the published
   quantile-based Kaplan-Meier calibration procedure.

7. Exact equivalence of the concordance and Brier score
   implementations with the original evaluation code
   has not been established.

8. Patient-level bootstrap uncertainty estimation was
   not implemented.

9. Standard PyHealth Trainer compatibility is not
   supported. DCM requires its dedicated fit_em() method.

10. Full-batch training preserves Cox risk sets but may
    limit scalability to substantially larger datasets.

11. Survival predictions remain constant beyond the last
    observed component-specific event time.

12. Model configurations were compared descriptively.
    Statistical significance was not established.

## 12. Conclusion

Deep Cox Mixtures was implemented within PyHealth and
evaluated using synthetic and real-world survival data.

The SUPPORT2 experiments demonstrated that mixture-based
survival regression improves short-term discrimination
relative to conventional and single-component deep Cox
models under the evaluated configuration.

The ablation study identified three mixture components as
the strongest observed balance between discrimination,
probabilistic accuracy, and model complexity.

Increasing the number of mixture components beyond three
did not consistently improve performance.

Demographic subgroup analysis demonstrated improved
discrimination for non-White patients, although calibration
disparities persisted.

The study achieved a partial qualitative reproduction
of the original Deep Cox Mixtures findings. Differences
in preprocessing, optimization, interpolation, metric
definitions, and uncertainty estimation prevented exact
numerical reproduction.

## 13. Reference

Nagpal et al. (2021).
Deep Cox Mixtures for Survival Regression.
Proceedings of Machine Learning Research, Volume 149.

Paper:
https://proceedings.mlr.press/v149/nagpal21a.html

Original reference implementation:
https://github.com/chiragnagpal/deep_cox_mixtures
