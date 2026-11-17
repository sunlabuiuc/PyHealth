# Imputation strategies and fairness on MIMIC-III: results

Results of `examples/mimic3_clinical_presence_mortality_logistic_regression.py`,
which reproduces the MIMIC-III experiment of [Jeanselme et al., "Imputation
Strategies Under Clinical Presence: Impact on Algorithmic Fairness", ML4H
2022](https://proceedings.mlr.press/v193/jeanselme22a.html) with the
`ClinicalPresenceMortalityMIMIC3` task and `capacity_fairness_metrics`. Every
number below is an aggregate over test patients.

## Setup

The setup is described in the script's docstring; these runs used real
MIMIC-III v1.4 and the default settings.

- **Cohort.** 36,313 patients (the paper reports 36,296), of whom 2,450
  (6.75%) died within 8 days of admission. Features are the latest value of
  each of 67 labs in the first 24 hours.
- **Split.** 80/10/10 by patient (29,050 / 3,632 / 3,631), repeated over
  split seeds 0-4. The imputer, the scaler and the classifier are all fit on
  the training 80%. A third run repeats the main run on the reference code's
  split (see "Split organisation" below).
- **Strategies.** The MICE strategies make 10 imputation draws, the others
  one. One classifier is trained per draw, and the predicted probabilities
  are averaged over draws.
- **Classifiers.** scikit-learn's `LogisticRegression` as in the reference
  code (`sag`, 1,000 epochs at most, C in 0.01, 0.1, 1, 10 chosen by
  validation accuracy), and PyHealth's `LogisticRegression` (`--pyhealth-lr`:
  PyHealth's `Trainer`, up to 20 epochs, early stopping on validation ROC
  AUC). It is not trained on the `+ Black flag` rows, whose extra input it
  does not take, and the `+ indicators first` rows were run without it.
- **Metrics.** The mean over 100 bootstrap resamples of the 3,631 test
  patients, then over the five seeds. Black, female and publicly insured
  patients are compared with the rest at a care capacity of 5%, 30% and 50%:
  the top-scoring share of patients is prioritised. The tables use 30%, the
  paper's main setting. A gap (Δ) is the marginalised group minus the rest.
- **Compute.** 16 cores and 31 GB RAM. The main run (14 strategies, five
  seeds, both classifiers) takes about 7 h from an empty cache; the
  indicator-order run (six strategies, five seeds, scikit-learn only) took
  4 h 18 min, and the reference-split run 7 h 7 min. Python 3.13.5, NumPy
  2.5.4, pandas 2.3.3, scikit-learn 1.9.1, PyHealth 2.0.2 (2.1.0 for the
  reference-split run, whose cohort and Table 1 are identical).

## Reproduction

Population AUC (the paper's Table 2), five-seed means. One seed's bootstrap
sd is 0.015-0.017.

| Strategy | Paper | scikit-learn | PyHealth |
|---|---|---|---|
| Median | 0.748 | 0.754 | 0.753 |
| MICE | 0.742 | 0.755 | 0.756 |
| Group MICE | 0.738 | 0.758 | 0.759 |
| Median Missing | 0.789 | 0.790 | 0.778 |
| MICE Missing | 0.788 | 0.791 | 0.792 |
| Group MICE Missing | 0.786 | 0.793 | 0.793 |
| Group MICE + Black flag | 0.738 | 0.760 | - |
| Group MICE Missing + Black flag | 0.786 | 0.794 | - |

Black patients at 30% capacity (the paper's Tables 6 and 7): the share
prioritised, and the false negative rate (FNR), the share of deaths not
prioritised.

| Strategy | Prioritised: paper | scikit-learn | PyHealth | FNR: paper | scikit-learn | PyHealth |
|---|---|---|---|---|---|---|
| Median | 0.343 | 0.358 | 0.365 | 0.298 | 0.209 | 0.286 |
| MICE | 0.323 | 0.346 | 0.350 | 0.314 | 0.266 | 0.303 |
| Group MICE | 0.150 | 0.346 | 0.358 | 0.520 | 0.266 | 0.266 |
| Median Missing | 0.317 | 0.351 | 0.343 | 0.224 | 0.216 | 0.321 |
| MICE Missing | 0.318 | 0.348 | 0.341 | 0.251 | 0.238 | 0.226 |
| Group MICE Missing | 0.194 | 0.338 | 0.343 | 0.419 | 0.250 | 0.250 |
| Group MICE + Black flag | 0.150 | 0.211 | - | 0.520 | 0.414 | - |
| Group MICE Missing + Black flag | 0.194 | 0.237 | - | 0.419 | 0.402 | - |

The paper's code also passed the Black flag to the classifier under Group
MICE, so the paper's Group MICE values are repeated on the `+ Black flag`
rows.

- **The cohort matches.** 36,313 patients against 36,296, and the missing
  labs of the paper's Table 1 are identical. Its lab-order means differ from
  what its published code computes; ours agree with the code.
- **Population AUCs are within 0.02 of the paper's,** and 106 of the
  paper's 126 table values are within one bootstrap sd of ours.
- **Missingness indicators raise AUC by 0.035 in every seed,** against
  0.04-0.05 in the paper.
- **The headline Black-patient result does not reproduce.** In the paper,
  Group MICE halves Black patients' prioritisation (0.323 under MICE, 0.150
  under Group MICE). Ours stays at 0.346 under both. The paper's code leaves
  the Black flag among the classifier's inputs; putting it back gives 0.211
  prioritised and an FNR of 0.414, most of the paper's effect.
- **PyHealth's `LogisticRegression` agrees with scikit-learn's** to within
  0.004 AUC, except on the three one-draw strategies with indicators, where
  it is 0.008-0.014 lower (Median Missing: 0.778 against 0.790).

## Hypotheses

The paper makes three claims: strategies with similar population
performance can treat groups differently (H1); group-aware imputation is not
uniformly fairer (H2); and the effect depends on the axis, so no strategy is
best for every group (H3). Each is judged on the five-seed means, with the
paper's strategies and scikit-learn unless stated.

- **H1 holds.** Median, MICE and Group MICE are within 0.004 population AUC
  (0.754, 0.755 and 0.758), against a bootstrap sd of 0.017. Yet Group MICE
  widens the prioritisation gap for women from +0.047 to +0.089 and for the
  publicly insured from +0.147 to +0.179, in every seed; one bootstrap sd is
  0.015-0.016. PyHealth's model shows the same pattern (Sex: +0.048 to
  +0.083).
- **H2 holds, but not on the paper's axis.** Group MICE widens the Sex and
  Insurance gaps above. On the Black axis it changes nothing (0.346
  prioritised and an FNR of 0.266 under both MICE and Group MICE) unless the
  classifier also sees the Black flag. The paper's Black-axis evidence for H2
  therefore comes from the classifier's input, not from the group-aware
  imputation.
- **H3 holds on the means, but the ranking is within noise.** Ranked by
  |Δ FNR| at 30%, the closest-to-zero strategy is Group MICE Missing on
  ethnicity (-0.031), Median Missing on insurance (-0.032) and Median on sex
  (-0.007), so no strategy is first on all three axes. The differences
  between strategies are well below one bootstrap sd (0.06-0.12), and the
  first-ranked strategy changes from seed to seed: only at seed 0 does one
  strategy (Group MICE) rank first on all three axes.

## Ablations and extensions

None of these rows is in the paper. scikit-learn, five-seed means, 30%
capacity.

| Strategy | Population AUC | Black prioritised | Black FNR | Δ prioritisation, Sex | Δ prioritisation, Insurance |
|---|---|---|---|---|---|
| MICE | 0.755 | 0.346 | 0.266 | +0.047 | +0.147 |
| Group MICE | 0.758 | 0.346 | 0.266 | +0.089 | +0.179 |
| MissForest | 0.775 | 0.350 | 0.260 | +0.050 | +0.143 |
| Group MissForest | 0.782 | 0.317 | 0.280 | +0.090 | +0.181 |
| MissForest Missing | 0.798 | 0.349 | 0.228 | +0.042 | +0.132 |
| Group MissForest Missing | 0.804 | 0.326 | 0.281 | +0.071 | +0.157 |
| Corrected MICE | 0.752 | 0.359 | 0.243 | +0.041 | +0.146 |
| Corrected Group MICE | 0.753 | 0.369 | 0.274 | +0.072 | +0.162 |
| Group MICE + Black flag | 0.760 | 0.211 | 0.414 | +0.083 | +0.179 |
| Group MICE Missing + Black flag | 0.794 | 0.237 | 0.402 | +0.054 | +0.146 |

- **MissForest** (`ablation`) is a non-linear imputer. It raises AUC by
  0.020 over MICE, and Group MissForest Missing has the highest AUC of any
  strategy (0.804). It never produces extreme imputations. The group flags
  move its gaps as they move MICE's (Sex +0.050 to +0.090), and they also
  lower Black prioritisation slightly (0.350 to 0.317). It changes none of
  the H1-H3 verdicts.
- **Corrected MICE** (`sensitivity`). On this cohort MICE is numerically
  ill-posed: over the draws, 0.11-0.12% of imputed cells fall beyond the
  training range widened by 3 sds, affecting 3.0-3.5% of patients. Under
  Group MICE this is 6.8% of Black patients on average and 69% in its worst
  draw. The corrected version bounds the imputations and draws proper noise.
  It has no extreme imputations and 0.003-0.005 lower AUC, and it reaches
  the same verdicts (Sex +0.041 to +0.072), so they do not rest on the
  numerical failure.
- **Black flag** (`diagnostic`). The `+ Black flag` rows re-create the
  paper's code, which drops only one of the three group columns after Group
  MICE. They reproduce the paper's sign on the Black axis.

**Indicator order** (`order`). The paper's text adds the missingness
indicators after imputation, while its code adds them before MICE, so they
are also imputation predictors. The `+ indicators first` rows test the code's
order:

| Strategy | Indicators added | Population AUC | Black prioritised | Black FNR | Δ prioritisation, Insurance | Δ FNR, Insurance |
|---|---|---|---|---|---|---|
| MICE Missing | after | 0.791 | 0.348 | 0.238 | +0.131 | -0.051 |
| | first | 0.791 | 0.345 | 0.230 | +0.126 | -0.018 |
| | paper | 0.788 | 0.318 | 0.251 | +0.113 | -0.022 |
| Group MICE Missing | after | 0.793 | 0.338 | 0.250 | +0.151 | -0.086 |
| | first | 0.792 | 0.344 | 0.223 | +0.140 | -0.037 |
| Group MICE Missing + Black flag | after | 0.794 | 0.237 | 0.402 | +0.146 | -0.074 |
| | first | 0.793 | 0.240 | 0.368 | +0.134 | -0.033 |
| | paper (Group MICE Missing) | 0.786 | 0.194 | 0.419 | +0.117 | -0.033 |

- **Insurance gaps narrow; nothing else moves.** Population AUC moves by at
  most 0.001, and no Black or Sex gap moves by 0.01 or more beyond 2
  standard errors across seeds. On every seed, adding the indicators first
  narrows the Insurance FNR gap by 0.032-0.049, which brings it to within
  0.004 of the paper's. The privately insured account for it: their FNR
  falls from 0.347 to 0.301 under Group MICE Missing.
- **The Black-axis difference comes from the flag, not the order.** For
  Group MICE Missing, the Black flag alone covers 70% of the gap between the
  paper's Black prioritisation and ours, and over half of the FNR gap (90%
  on the default split, 56% on the reference split); the order adds nothing
  measurable.
- **Extreme imputations rise 15-22 times** with the indicators first,
  reaching 50-65% of patients, because more labs then have fewer observed
  training rows than predictors.

**Split organisation** (`--reference-split`). The paper's text splits
80/10/10. Its code fits the imputer on an 80% pool, trains the classifier on
64.8% of patients and chooses C on 7.2%, both from the pool, and evaluates on
the other 28%: the pool's remaining 8%, which the imputer was fit on, and the
20% outside it. The third run re-creates this for the main run's 14
strategies. Its pool is the main run's training 80%, so the imputer is fit on
the same patients, and only the classifier's training, validation and test
patients change. Means over the 14 strategies, with one seed's bootstrap sd
in parentheses:

| Split | Test patients | Population AUC | Black prioritised | Black FNR | Δ prioritisation, Sex | Δ prioritisation, Insurance |
|---|---|---|---|---|---|---|
| Default | 3,631 | 0.776 (0.016) | 0.329 (0.029) | 0.273 (0.114) | +0.058 (0.016) | +0.152 (0.015) |
| Reference | 10,168 | 0.773 (0.009) | 0.330 (0.017) | 0.233 (0.066) | +0.057 (0.009) | +0.147 (0.009) |

- **No result moves beyond chance.** Of the 126 cells compared (population
  AUC, Black prioritisation and FNR, and the six gaps, for 14 strategies), 2
  move by 0.01 or more beyond 2 standard errors across seeds. Swapping the
  two runs' values within seeds, chance alone flags a median of 4.5. Every
  verdict above holds on both splits: H1-H3, the AUC gain from the
  indicators, MissForest's gain over MICE, the corrected-MICE gaps and the
  Black flag's effect.
- **The noise shrinks as the larger test set predicts.** Bootstrap sds are
  1.6-1.8 times smaller (1.67 expected), and each seed's test set holds about
  36-47 Black deaths instead of 12-18. The Black FNR is 0.04 lower on
  average, but lower in only three of the five seeds, and within the default
  split's noise.
- **It is not why our numbers differ from the paper's.** The mean deviations
  from the paper are the same on both splits. Without the Black flag, our
  Black FNRs fall further below the paper's (Median: 0.298 in the paper,
  0.209 on the default split, 0.188 on the reference split).

## Differences from the paper

The task follows the authors' reference code, which produced the paper's
numbers, and departs from it in these places. The task's documentation page
gives the full list.

| Topic | Reference code | Here | Effect on MIMIC-III v1.4 |
|---|---|---|---|
| Stay anchor | Hospital admission (the text says ICU admission) | Hospital admission, as the code | - |
| Index admission | Last row of `ADMISSIONS.csv` | Latest admission time | 88 patients (0.24%) get a later admission |
| Label | `DEATHTIME - ADMITTIME` under 8 days | `hospital_expire_flag` and `DISCHTIME - ADMITTIME` under 8 days | 0.2% of the positive labels differ |
| Labs of the admission | `HADM_ID` and the 24-hour window | The window only | 37 patients move into or out of the cohort |
| Split | Imputer fit on an 80% pool; classifier trained on 64.8% and tuned on 7.2%, both from the pool, and evaluated on the other 28%: the pool's remaining 8%, which the imputer was fit on, and the 20% outside it | 80/10/10, with everything fit on the 80%; `--reference-split` gives the code's split | A test split 2.8 times smaller; the code's split changes no result beyond chance |
| Missingness indicators | Added before MICE | Added after, as the text says | See the indicator-order table |
| Group MICE flags | Black flag left among the classifier's inputs | All group flags dropped | See the `+ Black flag` rows |
| Sex flag | Always False | `gender == "F"` | Group MICE also conditions on sex |
| MICE noise | NumPy's unseeded global random state | The same noise model, seeded per draw | Reproducible draws |

The label and lab rules are stand-ins: PyHealth's MIMIC-III config did not
load `DEATHTIME` or `LABEVENTS.HADM_ID` when the task was written. It now
does, so switching to the reference's rules is a possible follow-up. The
cohort difference (36,313 against 36,296) comes from the index admission,
label and lab rules. As in the reference, MIMIC-Extract maps both INR and PT
to Prothrombin time, and the two are charted together, so the duplicate rule
leaves that lab almost empty (44 patients keep a value) and models
effectively see 66 labs.

## Limitations

- **The Black axis is noisy.** At seed 0 the test split has 3,631 patients
  and 16 Black deaths, so one seed's Black FNR has a bootstrap sd of
  0.11-0.14. Judge Black-axis cells on the five-seed means.
- **One small test split per seed.** The reference evaluates on 28% of the
  cohort; we hold out 10%, so subgroup estimates are noisier than the
  paper's. On the reference split the bootstrap sds are 1.7 times smaller,
  and no result moves beyond chance.
- **The MICE classifiers are stopped early, as in the reference.** For each
  MICE strategy, 150 of the 200 scikit-learn fits (10 draws, 4 values of C,
  5 seeds) stop at `sag`'s 1,000 epochs.
  At seed 0, fully converged fits change MICE's test AUC by 0.0005. C is
  chosen by validation accuracy, as in the reference: at 6.3% validation
  prevalence, predicting all negatives already scores 0.9367, so the choice
  rests on very few patients.
- **Reproducibility has been checked on one machine,** where two runs
  matched to the last digit. On other hardware, expect small differences,
  mostly in the MICE strategies.

## How to reproduce

The open MIMIC-III demo (100 patients) needs no credentials and runs the
pipeline end to end in about 3 minutes, plus the download. It checks the code
paths only; with 100 patients most subgroup cells are skipped.

```bash
wget -N -P data/mimiciii-demo \
  https://physionet.org/files/mimiciii-demo/1.4/{PATIENTS,ADMISSIONS,ICUSTAYS,LABEVENTS,D_LABITEMS}.csv
python examples/mimic3_clinical_presence_mortality_logistic_regression.py \
  --root data/mimiciii-demo --out results/demo --strategies paper \
  --n-draws 2 --split-ratios 0.5 0.1 0.4 --n-bootstrap 20 --pyhealth-lr
```

The results above come from MIMIC-III v1.4, which needs credentialed
PhysioNet access, in three runs. Each runs its seeds in one command, because
the report averages over seeds and separate runs cannot be merged:

```bash
# The main run: every strategy outside the order family (about 7 h)
python examples/mimic3_clinical_presence_mortality_logistic_regression.py \
  --root /path/to/mimic-iii/1.4 --out results/full_s0-4 --seeds 0 1 2 3 4 \
  --strategies paper appendix ablation diagnostic sensitivity --pyhealth-lr

# The indicator order, next to its base strategies (4 h 18 min)
python examples/mimic3_clinical_presence_mortality_logistic_regression.py \
  --root /path/to/mimic-iii/1.4 --out results/order_s0-4 --seeds 0 1 2 3 4 \
  --strategies "MICE Missing" "Group MICE Missing" \
  "Group MICE Missing + Black flag" order

# The main run on the reference code's split (7 h 7 min)
python examples/mimic3_clinical_presence_mortality_logistic_regression.py \
  --root /path/to/mimic-iii/1.4 --out results/refsplit_s0-4 --seeds 0 1 2 3 4 \
  --strategies paper appendix ablation diagnostic sensitivity \
  --reference-split --pyhealth-lr
```

`report.md` in `--out` has every table, with the paper's values in brackets,
and the figures. With `--reference-split`, `metrics_test_parts.csv` also has
every metric on the internal and the external test separately.
