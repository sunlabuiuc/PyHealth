# Framingham Neural Fine-Gray: setup and recorded results

Script: `examples/framingham_competing_risk_neural_fine_gray.py`.
Paper: Jeanselme et al., *Neural Fine-Gray: Monotonic Neural Networks for
Competing Risks*, CHIL 2023 (https://arxiv.org/abs/2305.06703).

This contribution adds the Framingham cohort (`FraminghamDataset`) and task
(`FraminghamCompetingRiskPrediction`), and a `cdf_mode` option on the existing
`NeuralFineGray` model (#1446). Preprocessing (`CompetingRisksPreprocessor`)
and metrics (`competing_risk_concordance`, `competing_risk_brier`) are reused
from #1446, so the Framingham and PBC pipelines share one model and one
evaluation protocol.

## Commands

```bash
# Offline: synthetic 800-patient cohort written to a temporary directory.
python examples/framingham_competing_risk_neural_fine_gray.py

# Public Framingham cohort (downloads a commit-pinned CSV if absent).
python examples/framingham_competing_risk_neural_fine_gray.py \
  --data-root ./data/framingham --download --epochs 100
```

## Setup

- Patient-level 70/15/15 split, stratified by event cause, seed 7.
- `CompetingRisksPreprocessor` fitted on training patients only (standardized
  features; time divided by the maximum training follow-up).
- `NeuralFineGray`: encoder and heads `(50, 50)`, dropout 0.25, float64, Adam
  with learning rate 1e-3, batch size 128, early stopping after 10 epochs
  without improvement in validation NLL. No hyperparameter search.
- Scoring: cause 1 (cardiovascular disease) CIF at the 25th, 50th and 75th
  percentiles of uncensored training event times (years), with the
  censoring distribution fitted on training labels. Every arm is scored on the
  same uncensored test labels at the same horizons.

## Ablations

| Arm | Model | Training labels |
| --- | --- | --- |
| `nfg_full_followup` | `cdf_mode="nfg"`: `1 - exp(-t m_r(t, x))` (paper) | full follow-up |
| `sumonet_full_followup` | `cdf_mode="sumonet"`: `1 - exp(-m_r(t, x))` | full follow-up |
| `nfg_censored_at_q75` | `cdf_mode="nfg"` | task `horizon_days` = training q75 event time |

The model ablation tests the paper's choice to multiply by `t`, which forces
every CIF to start at zero. The task ablation asks how much the predictions at
or before the q75 horizon depend on follow-up recorded after it.

Test NLL is reported in each arm's own scaled time units. It is comparable
between the two full-follow-up arms (same time scale), but not for the
censored arm, whose time scale is shorter.

## Framingham (4,434 participants; run 2026-11-26, CPU)

Split: 3,102 / 665 / 667 patients. Training events: 1,743 censored, 809 CVD,
550 death. Horizons: 5.68 / 12.42 / 17.97 years. Full output:
`framingham_seed7.json`.

| Arm | Test NLL | C-index q25 / q50 / q75 | Brier q25 / q50 / q75 |
| --- | ---: | --- | --- |
| NFG, full follow-up | 0.709 | 0.906 / 0.810 / 0.784 | 0.038 / 0.089 / 0.127 |
| SumoNet-style, full follow-up | 0.662 | 0.908 / 0.809 / 0.781 | 0.039 / 0.090 / 0.129 |
| NFG, censored at q75 | n/a | 0.905 / 0.806 / 0.780 | 0.038 / 0.090 / 0.128 |

Comparison with the paper (Framingham, CVD, NFG): C-index 0.872 / 0.812 /
0.782 and Brier 0.050 / 0.095 / 0.128. Discrimination at the median and late
horizons and all three Brier scores are within about 0.01 of the paper; the
early-horizon C-index is higher. Two caveats limit how far this can be read:
this is a single split without the paper's 5-fold cross-validation or
100-configuration search, and the concordance here is the horizon IPCW score
from #1446, which treats competing events differently from the paper's
evaluation code. We therefore describe it as a close but partial
reproduction.

Ablation findings:

- **`cdf_mode`:** on Framingham the two parameterizations are practically
  tied: horizon metrics differ by at most 0.003, and the SumoNet-style model
  has a slightly lower test NLL. On this split we do not see the advantage of
  the `t` factor that the paper reports; a single split cannot rule it out.
- **Administrative censoring at 17.97 years:** dropping follow-up beyond the
  q75 horizon changed every metric by at most 0.004. The predictions up to
  that horizon depend little on the late tail.

## Synthetic cohort (800 participants; pipeline check)

Split: 559 / 120 / 121. Training events: 97 censored, 259 cause 1, 203 cause
2. Horizons: 1.56 / 3.46 / 6.74 years. Full output: `synthetic_seed7.json`.

| Arm | Test NLL | C-index q25 / q50 / q75 | Brier q25 / q50 / q75 |
| --- | ---: | --- | --- |
| NFG, full follow-up | 0.399 | 0.675 / 0.698 / 0.712 | 0.137 / 0.159 / 0.195 |
| SumoNet-style, full follow-up | 0.585 | 0.695 / 0.707 / 0.724 | 0.131 / 0.154 / 0.189 |
| NFG, censored at q75 | n/a | 0.663 / 0.705 / 0.718 | 0.133 / 0.152 / 0.187 |

Here NFG has the better likelihood and the SumoNet-style model slightly
better horizon metrics. With 121 test patients these differences are within
split noise; the run exists to show that the full dataset → task →
preprocessor → model → metrics path works without clinical data.

## Data provenance

No patient-level data are committed. `--download` fetches the public
longitudinal CSV from the MIT-licensed DeepSurvivalMachines repository at a
pinned commit (the source used by the paper's reference code). Unit tests use
only fabricated rows.
