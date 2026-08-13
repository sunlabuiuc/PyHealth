---
name: pyhealth-calibrate-predictions
description: Calibrate a trained PyHealth model so its probabilities mean what they say, or wrap it in a conformal set predictor that returns a label set with a coverage guarantee. Covers TemperatureScaling, HistogramBinning, KCal, DirichletCalibration, and the LABEL/SCRIB/FavMac/cluster/covariate set predictors. Use when probabilities are overconfident, when ECE matters, or when the user wants guaranteed coverage rather than a single label.
---

# Calibrate predictions

A model that is 90% accurate is not the same as a model whose "90%" means 90%. Deep networks are
systematically overconfident, and in a clinical setting the probability *is* the deliverable —
it is what a threshold, a triage rule, or a downstream decision consumes.

PyHealth handles this in two ways, and they answer different questions:

- **Calibration** — keep one prediction per sample, fix the probabilities.
  *"When this model says 0.9, is it right 90% of the time?"*
- **Set prediction** — return a *set* of labels with a coverage guarantee.
  *"Give me a set that contains the true label 95% of the time."*

Both are post-hoc: train the model first, then fit the wrapper on a held-out calibration split.
Neither retrains anything.

---

## What you need first

A **trained model** and a **calibration split that the model never saw during training**. Reusing
the training set produces a calibrator that is itself overconfident; reusing the test set
destroys your held-out estimate.

The usual arrangement is a four-way split — train / calibration / validation / test — or
reusing validation for calibration once model selection is finished. Say which one you did.

```python
train_data, val_data, test_data = split_by_patient(samples, [0.6, 0.2, 0.2])
# ... train the model on train_data, select on val_data ...
# then calibrate on val_data, and only then touch test_data
```

---

## Calibration methods

All four subclass `PostHocCalibrator` (`pyhealth/calib/base_classes.py:7`) and share one
interface: construct with the model, call `.calibrate(cal_dataset=...)`, then use the wrapper
wherever you used the model.

| Class | Source | Best for | Notes |
|---|---|---|---|
| `TemperatureScaling` | `calib/calibration/temperature_scale.py:20` | **start here** | one parameter; cannot change the ranking, so accuracy and AUC are unchanged |
| `HistogramBinning` | `calib/calibration/hb.py:124` | non-parametric fix | can change ranking; needs enough calibration data per bin |
| `DirichletCalibration` | `calib/calibration/dircal.py:51` | multiclass, class-dependent miscalibration | more parameters, more calibration data |
| `KCal` | `calib/calibration/kcal/__init__.py:119` | kernel-based, per-instance | heaviest; needs embeddings |

```python
from pyhealth.calib.calibration import TemperatureScaling

cal_model = TemperatureScaling(model)
cal_model.calibrate(cal_dataset=val_data)

from pyhealth.trainer import Trainer
print(Trainer(model=cal_model, metrics=["accuracy", "ECE_adapt"]).evaluate(test_loader))
```

**Report ECE before and after.** `ECE` and `ECE_adapt` are available for binary
(`pyhealth/metrics/binary.py`) and multiclass (`multiclass.py`, plus `cwECEt` for classwise).
A calibration step with no before/after ECE number is an unverified claim.

Since `TemperatureScaling` is monotonic, ROC-AUC will not move. If it does, something is wrong.

---

## Set predictors

All subclass `SetPredictor` (`calib/base_classes.py:25`). Instead of one label they return a
set sized to hit a coverage target — wide where the model is unsure, narrow where it is
confident. That width is the useful signal: it tells a clinician when the model is guessing.

| Class | Source | Guarantee / behavior |
|---|---|---|
| `LABEL` | `predictionset/label.py:26` | the standard conformal baseline; marginal coverage |
| `BaseConformal` | `predictionset/base_conformal/__init__.py:85` | general conformal machinery |
| `ClusterLabel` | `predictionset/cluster/cluster_label.py:28` | class-conditional coverage via clustering |
| `NeighborhoodLabel` | `predictionset/cluster/neighborhood_label.py:21` | locally adaptive |
| `CovariateLabel` | `predictionset/covariate/covariate_label.py:227` | conditions coverage on covariates (e.g. subgroup fairness) |
| `SCRIB` | `predictionset/scrib/__init__.py:174` | risk control with class-specific targets |
| `FavMac` | `predictionset/favmac/__init__.py:90` | multilabel; value–cost tradeoff |

```python
from pyhealth.calib.predictionset import LABEL

set_model = LABEL(model, alpha=0.1)          # target ~90% coverage
set_model.calibrate(cal_dataset=val_data)
```

Evaluate with the prediction-set metrics in `pyhealth/metrics/prediction_set.py` —
`rejection_rate`, `set_size`, `miscoverage_ps`, `miscoverage_overall_ps`, `error_ps` and their
means. Two numbers matter together: **coverage** (did it hit the target?) and **average set
size** (is it useful, or does it hedge by returning everything?). A set predictor with perfect
coverage and an average size of `n_classes` has told you nothing.

Worked examples: `examples/conformal_eeg/` (12 scripts).

---

## Choosing

```
Probabilities are overconfident, want a drop-in fix
  → TemperatureScaling  (then report ECE before/after)

Miscalibration differs by class, multiclass task
  → DirichletCalibration

Want a guaranteed-coverage label set instead of one label
  → LABEL             (start here)
  → ClusterLabel      (need per-class coverage)
  → CovariateLabel    (need coverage within subgroups — fairness)
  → SCRIB             (need explicit per-class risk control)
  → FavMac            (multilabel)
```

---

## Honest reporting

- Calibrating changes the probabilities, **not** the discrimination. Do not present a
  calibration step as a performance improvement — ROC-AUC is usually identical by construction.
- Coverage guarantees are *marginal* unless you used a class- or covariate-conditional method.
  Marginal 90% coverage is compatible with 60% coverage on a minority subgroup. If subgroup
  performance matters, say so and use `CovariateLabel` or `ClusterLabel`.
- Everything here assumes the calibration and test data are exchangeable. Distribution shift —
  another hospital, another year, another scanner — voids the guarantee. Say that out loud when
  someone plans to deploy.
- Report the calibration split size. Conformal coverage is only as tight as that sample.

---

## Adding a new calibrator

Subclass `PostHocCalibrator` or `SetPredictor`, implement `calibrate(cal_dataset)` and
`forward(**kwargs) -> Dict[str, torch.Tensor]`, export it from the package `__init__.py`, and
pair it with metrics in `metrics/calibration.py` or `metrics/prediction_set.py`. Full
checklist in [add-a-component](../add-a-component/SKILL.md).
