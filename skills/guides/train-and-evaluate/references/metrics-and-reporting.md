# Evaluation — metrics, leakage, and honest reporting

The mechanics of running `Trainer.evaluate` are in
[training.md](../SKILL.md); this file is about *which* number to look at and
whether you are allowed to believe it.

---

## Which metrics exist

The mode comes from your task's `output_schema`, and it picks the metrics
function (`get_metrics_fn`, `pyhealth/trainer.py:40`).

### binary — `pyhealth/metrics/binary.py`
Defaults if you pass nothing: `pr_auc`, `roc_auc`, `f1`.

Available: `pr_auc`, `roc_auc`, `accuracy`, `balanced_accuracy`, `f1`,
`precision`, `recall`, `cohen_kappa`, `jaccard`, `ECE`, `ECE_adapt`.

### multiclass — `pyhealth/metrics/multiclass.py`
Defaults: `accuracy`, `f1_macro`, `f1_micro`.

Available: `roc_auc_macro_ovo`, `roc_auc_macro_ovr`, `roc_auc_weighted_ovo`,
`roc_auc_weighted_ovr`, `accuracy`, `balanced_accuracy`, `f1_micro`, `f1_macro`,
`f1_weighted`, `jaccard_{micro,macro,weighted}`, `cohen_kappa`, `brier_top1`,
`ECE`, `ECE_adapt`, `cwECEt`, `cwECEt_adapt`.

### multilabel — `pyhealth/metrics/multilabel.py`
Default: `pr_auc_samples`.

Available: `roc_auc_{micro,macro,weighted,samples}`,
`pr_auc_{micro,macro,weighted,samples}`, `accuracy`,
`f1_{micro,macro,weighted,samples}`, `precision_*`, `recall_*`, `jaccard_*`,
`hamming_loss`, `ddi` (drug–drug interaction, for drug recommendation),
`cwECE`, `cwECE_adapt`.

### regression — `pyhealth/metrics/regression.py`
Defaults: `kl_divergence`, `mse`, `mae`.

---

## Which metric to actually report

| Situation | Primary | Also report | Why |
|---|---|---|---|
| Binary, roughly balanced | `roc_auc` | `pr_auc`, `f1` | AUC is threshold-free |
| Binary, rare positive (<10%) | `pr_auc` | `roc_auc` | ROC-AUC looks flattering when negatives dominate |
| Multiclass, balanced | `accuracy` | `f1_macro` | — |
| Multiclass, imbalanced (e.g. LOS buckets) | `f1_macro` | `accuracy`, `balanced_accuracy` | macro weights rare classes equally |
| Multilabel (drug rec) | `pr_auc_samples` | `f1_samples`, `jaccard_samples`, `ddi` | scored per sample, not per label |
| Regression | `mae` | `mse` | MAE is in the units the clinician thinks in |

**Never report accuracy alone on an imbalanced clinical outcome.** In-hospital
mortality is often 2–5% positive: a model that always predicts "survives" scores
95%+ and is worthless. If the user asks for accuracy on such a task, give it to
them alongside PR-AUC and say why the accuracy number is misleading.

Monitor and report should agree. Set `monitor=` to your primary metric so the
checkpoint you keep is the one that is best on what you actually care about.

---

## Anti-leakage rules

These are the rules that decide whether the number means anything.

**1. Split by patient.** `split_by_patient` is the default for a reason: the same
patient in train and test means the model can memorize the person. Chronic
patients with many admissions are the worst offenders, and they are exactly the
patients EHR datasets are full of.

**2. No same-visit label leakage.** For binary outcomes, take features from visit
`i` and the label from visit `i+1`. Same-visit setups leak whenever the label
influences the features — and in EHR data it nearly always does. Discharge-coded
diagnoses are the canonical trap: they were assigned *at* discharge, so they
encode the outcome.

**3. The test set is touched once.** Select everything — features, model,
hyperparameters — on validation. Then run test, once, and report it. If you
looked at test and then changed something, the test number is no longer a
held-out estimate; say so, or re-split.

**4. Fit preprocessing on train only.** Normalization statistics, vocabularies,
feature selection: all derived from the training split. PyHealth's processors fit
on the data they are given, so a z-normalization you compute yourself must use
train statistics and apply them unchanged to val and test.

**5. `dev=True` numbers are not results.** ~1000 patients. Use it to prove the
code runs; never to compare configurations and never to report.

---

## The final run

Once a configuration is selected on validation:

1. Train the winning config with **at least 3 seeds** — 42, 43, 44 by convention.
2. Evaluate each on the held-out test split.
3. Report **mean ± std** across seeds for every metric. A single seed on a
   clinical dataset is largely noise, and reporting the best of several is
   straightforwardly dishonest.
4. State plainly that the test set was not used during selection.

A results file worth writing:

```json
{
  "task": "mortality_next_visit_mimic4",
  "model": "RNN",
  "config": {"lr": 1e-3, "batch_size": 64, "embedding_dim": 128, "hidden_dim": 128},
  "split": {"by": "patient", "ratios": [0.8, 0.1, 0.1], "seed": 42},
  "seeds": [42, 43, 44],
  "per_seed_test_metrics": [
    {"seed": 42, "roc_auc": 0.812, "pr_auc": 0.341},
    {"seed": 43, "roc_auc": 0.807, "pr_auc": 0.336},
    {"seed": 44, "roc_auc": 0.815, "pr_auc": 0.349}
  ],
  "test_metrics_mean_std": {
    "roc_auc": {"mean": 0.811, "std": 0.004},
    "pr_auc":  {"mean": 0.342, "std": 0.007}
  },
  "n_train": 41230, "n_val": 5150, "n_test": 5160,
  "positive_rate": 0.043,
  "notes": "Test split evaluated once, after selection on validation."
}
```

Always report `n_*` and `positive_rate` next to the metrics. A PR-AUC means
nothing without knowing the base rate it is being compared against.

---

## Deciding whether a change actually helped

An improvement counts only if it exceeds the noise. The working rule: run the
candidate with **≥2 seeds**, compute mean ± std of the validation metric, and
adopt it as the new baseline only if the mean improvement is **greater than one
standard deviation** of the current best. Discard anything smaller as noise, and
say you are discarding it rather than quietly banking it.

This is what keeps an optimization loop from turning into an elaborate random
walk that ends up reporting the luckiest seed.

---

## Things worth saying to the user, unprompted

- The base rate of the outcome, in plain terms ("4.3% of samples are positive —
  so a PR-AUC of 0.34 is roughly 8× the base rate").
- The cohort size after filtering, and what fraction of patients were dropped.
- Any metric that looks *too* good. ROC-AUC above ~0.95 on a next-visit clinical
  outcome usually means leakage, not brilliance. Go find it before celebrating.
- What the model is not: it is not calibrated for decision-making, it has not
  been validated on another site, and a retrospective score is not evidence of
  clinical benefit.
