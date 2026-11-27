# SHy on MIMIC-III next-visit diagnosis prediction: results

Results of `examples/mimic3_next_visit_diagnosis_prediction_shy.py` for SHy
([Yu et al., "Self-Explaining Hypergraph Neural Networks for Diagnosis
Prediction", CHIL 2025](https://arxiv.org/abs/2502.10689)) and PyHealth's
RETAIN baseline.

## Setup

The setup is described in the script's docstring; these runs used real
MIMIC-III v1.4 (`--root`), default settings (300 epochs, patience 25) and one
T4 GPU. 7,499 patients have at least two admissions (the paper reports
7,493), giving 3,622 label codes. Scores are on the test split, with the
checkpoint chosen on validation Recall@20.

## Reproduction

The 3-seed rows are the mean and standard deviation over seeds 0-2; the
authors' protocol rows are seed 0.

| Configuration | Recall@10 | Recall@20 | nDCG@10 | nDCG@20 |
|---|---|---|---|---|
| Paper RETAIN (Table 1) | 0.2024 | 0.2958 | 0.3136 | 0.3167 |
| Paper SHy (Table 1) | 0.2775 | 0.3831 | 0.4135 | 0.4088 |
| RETAIN (3 seeds) | 0.2598 ± 0.0069 | 0.3506 ± 0.0086 | 0.4196 ± 0.0077 | 0.4008 ± 0.0075 |
| SHy (3 seeds) | 0.2615 ± 0.0035 | 0.3600 ± 0.0043 | 0.4067 ± 0.0061 | 0.3956 ± 0.0048 |
| RETAIN, authors' protocol | 0.2753 | 0.3690 | 0.4277 | 0.4103 |
| SHy, authors' protocol | 0.2688 | 0.3731 | 0.4074 | 0.4000 |

The authors' protocol rows (`--experiments protocol`) follow the authors'
released code, which has no validation set: the checkpoint is picked on the
test split, so those rows are optimistic by construction.

- Under the authors' protocol SHy reaches 0.3731 Recall@20, 0.01 below the
  paper's 0.3831.
- With validation-based selection SHy beats RETAIN on Recall@20 on every
  seed, while RETAIN has the higher nDCG@10.
- PyHealth's RETAIN scores 0.3506 Recall@20 against 0.2958 in the paper, so
  the paper's 9-point margin over RETAIN shrinks to about 1 point.

## Ablations and extensions

None of these rows is in the paper. Seed 0, same setup as above.

| Experiment | Configuration | Recall@10 | Recall@20 | nDCG@10 | nDCG@20 |
|---|---|---|---|---|---|
| `recovery` | SHy add_ratio=0.0 | 0.2685 | 0.3713 | 0.4090 | 0.3993 |
| `recovery` | SHy add_ratio=0.1 | 0.2672 | 0.3658 | 0.4090 | 0.3976 |
| `recovery` | SHy add_ratio=0.2 | 0.2669 | 0.3672 | 0.4072 | 0.3970 |
| `recovery` | SHy add_ratio=0.4 | 0.2690 | 0.3665 | 0.4103 | 0.3988 |
| `recovery` | SHy random links 0.2 | 0.2609 | 0.3686 | 0.4021 | 0.3957 |
| `hierarchy` | SHy hierarchical emb. | 0.2682 | 0.3687 | 0.4077 | 0.3973 |
| `hierarchy` | SHy flat emb. | 0.2648 | 0.3604 | 0.4019 | 0.3906 |
| `omega` | SHy omega=0.04 | 0.2657 | 0.3723 | 0.4068 | 0.3989 |
| `time_gaps` | SHy no time gaps | 0.2621 | 0.3701 | 0.4003 | 0.3946 |
| `time_gaps` | SHy + time gaps | 0.2695 | 0.3726 | 0.4122 | 0.4021 |
| `task` | RETAIN all visits | 0.2652 | 0.3567 | 0.4162 | 0.3971 |
| `task` | RETAIN max_visits=3 | 0.2697 | 0.3592 | 0.4218 | 0.4037 |
| `task` | RETAIN max_visits=1 | 0.2531 | 0.3460 | 0.4033 | 0.3876 |
| `task` | RETAIN 3-digit codes | 0.4042 | 0.5453 | 0.5725 | 0.5706 |

Four seed-0 runs of the same SHy configuration (the baseline run at 0.3630
and the control rows `add_ratio=0.2`, `hierarchical emb.` and `no time
gaps`) scored 0.3630-0.3701 Recall@20, so differences below about 0.007 are
run-to-run noise.

- False-negative recovery has no measurable effect: `add_ratio` from 0 to
  0.4 and random links all fall within noise.
- Hierarchical ICD-9 embeddings beat a flat embedding table by a small
  margin, at the edge of noise (0.3687 vs 0.3604 Recall@20).
- Visit time gaps (extension) give a small nDCG gain.
- `omega` 0.04 (the authors' README value) scores 0.3723 Recall@20, just
  above the 0.3630-0.3701 range of the `omega` 0 runs, so it is at least as
  accurate. The paper compares the two settings on explanation quality only
  (its Table 3, see below).
- RETAIN needs only the last three visits; with the last visit alone it
  loses about 0.011 Recall@20. 3-digit codes are a much easier label space,
  so that row is not comparable with the full-code rows.

Deterministic evaluation masks (`stochastic_eval=False`) keep about 0.1% of
the (visit, code) pairs and score 0.23 Recall@20 against 0.37 with sampled
masks, so sampling stays the default.

## Robustness and explanations

The paper's masking test (Section 4.3, Table 2) and explanation metrics
(Table 3), on seed-0 checkpoints from the runs above. They were measured
after training with an evaluation notebook that is not part of this PR, so
the script does not reproduce this section.

### Masking test

Each (visit, code) pair in a test patient's history is hidden independently
with probability 25% or 75%; the models are not retrained. The table gives
Recall@20 and its relative drop from the unmasked score, as in the paper's
Table 2 (which has no RETAIN row), and the share of hidden pairs that SHy's
recovery step adds back, counted before the phenotype masks are applied.

| Model | No masking | 25% masked | 75% masked | Hidden codes recovered (25% / 75%) |
|---|---|---|---|---|
| Paper SHy | 0.3831 | 0.3604 (-5.9%) | 0.3020 (-21.2%) | 4.8% / 0.7% |
| RETAIN | 0.3567 | 0.3371 (-5.5%) | 0.2467 (-30.8%) | - |
| SHy add_ratio=0.2 | 0.3681 | 0.3471 (-5.7%) | 0.2759 (-25.0%) | 6.4% / 0.7% |
| SHy add_ratio=0.0 | 0.3659 | 0.3509 (-4.1%) | 0.2770 (-24.3%) | 0 / 0 |
| SHy random links 0.2 | 0.3677 | 0.3439 (-6.5%) | 0.2705 (-26.4%) | 0.3% / 0.1% |

The RETAIN row is the `task` control checkpoint. `SHy add_ratio=0.2` is the
paper setting, evaluated on the `hierarchy` control checkpoint; the other two
SHy rows are `recovery` checkpoints. Their unmasked scores differ from the
ablation table by up to about 0.005 because SHy samples its masks at
evaluation.

- At 25% masking SHy and RETAIN lose about the same (5.7% and 5.5%), in
  line with the paper's 5.9% for SHy. At 75% SHy loses 25% against RETAIN's
  31% (paper SHy: 21%).
- The recovery step does add back hidden codes: 6.4% / 0.7% of them (the
  paper reports 4.8% / 0.7% for its phenotypes), about 19 times more than
  random links at 25% masking (6.44% against 0.34% before rounding).
- Recovery is not what makes SHy more robust: without it, or with random
  links, Recall@20 stays within 0.007 of the paper setting at every masking
  level.

### Explanation metrics

| Model | Faithfulness | Complexity | Distinctness |
|---|---|---|---|
| Paper RETAIN (Table 3) | 0.1867 | 17.66 | - |
| Paper SHy (Table 3) | 0.5915 | 10.18 | 0.7749 |
| Paper SHy w/o L_alpha (Table 3) | 0.7381 | 5.09 | 1.0000 |
| SHy omega=0 | 0.1629 | 8.27 | 0.8208 |
| SHy omega=0.04 | 0.2834 | 9.84 | 0.7908 |

Complexity is the number of (visit, code) pairs kept, summed over the K = 5
phenotypes of a patient, and distinctness is the paper's formula (1 means no
overlap between phenotypes). Faithfulness is the per-patient Pearson
correlation between the phenotype weights and the drop in predicted
probability on the true next-visit codes when a phenotype is removed from
the mixture and the remaining weights are renormalised. The paper removes
explanation units "from the input" and gives no further detail, so the
faithfulness values are not directly comparable.

- Explanations are concise and distinct, as in the paper: 8.3 kept pairs
  against the 20.6 history codes of an average test patient, with little
  overlap between phenotypes.
- Faithfulness is 0.16, far below the paper's 0.59, so that claim is not
  reproduced under our reading of the metric.
- `omega` 0 is the paper's "w/o L_alpha" setting. The paper reports that for
  most patients it collapses to one non-empty phenotype (distinctness 1.0,
  complexity 5.1); our `omega` 0 model does not (distinctness 0.82,
  complexity 8.3).
- With `omega` 0.04 all three metrics move toward the paper's values for the
  full model.
