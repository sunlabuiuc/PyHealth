# Phase 9 — Final Experimental Results

## 1. Reproduction of published SUPPORT results

The Phase-4 reproduction uses the exact recovered SODEN patient memberships and the
reconstructed training-only preprocessing. The strongest agreement is in the
distribution-sensitive metrics (IBLL and IBS). Across all nine published
metric/horizon comparisons, the reproduction differs from the paper by no more than
1.21 combined standard errors.

| Horizon P(C>τ) | Metric | Reproduction (mean ± SE) | Paper (mean ± SE) | Difference |
|---|---|---:|---:|---:|
| 10^-8 | Concordance C_td | 0.6236 ± 0.0033 | 0.6280 ± 0.0030 | -0.0044 |
| 10^-8 | IBLL | -0.5594 ± 0.0016 | -0.5590 ± 0.0020 | -0.0004 |
| 10^-8 | IBS | 0.1895 ± 0.0005 | 0.1900 ± 0.0020 | -0.0005 |
| 0.2 | Concordance C_td | 0.6237 ± 0.0033 | 0.6280 ± 0.0030 | -0.0043 |
| 0.2 | IBLL | -0.5754 ± 0.0011 | -0.5750 ± 0.0020 | -0.0004 |
| 0.2 | IBS | 0.1964 ± 0.0005 | 0.1960 ± 0.0010 | 0.0004 |
| 0.4 | Concordance C_td | 0.6227 ± 0.0032 | 0.6280 ± 0.0030 | -0.0053 |
| 0.4 | IBLL | -0.5927 ± 0.0016 | -0.5930 ± 0.0010 | 0.0003 |
| 0.4 | IBS | 0.2039 ± 0.0006 | 0.2040 ± 0.0010 | -0.0001 |

**Interpretation:** the SUPPORT reproduction is sufficiently close to serve as the
control for the ablation. Concordance is modestly lower than the paper, while IBLL
and IBS essentially reproduce the published values.

## 2. Primary softplus-vs-exponential ablation

The final ablation consists of 10 exact SUPPORT splits × 5 paired seeds × 2
transformations = **100 final runs**. Each softplus/exponential pair uses the same
split, preprocessing, architecture, initialization seed, optimizer, minibatch order,
and stopping rule. The only conceptual change is the positive-time transformation.

| Measure | Softplus | Exponential | Exp − Softplus | 95% CI of split-level paired difference | Exp wins |
|---|---:|---:|---:|---:|---:|
| Full test NLL | 0.517746 | 0.502901 | -0.014845 | [-0.018627, -0.011063] | 10/10 |
| Concordance | 0.621659 | 0.624975 | 0.003316 | [0.001374, 0.005258] | 9/10 |
| IBS | 0.189817 | 0.189404 | -0.000413 | [-0.000934, 0.000108] | 7/10 |
| IBLL | -0.560497 | -0.557844 | 0.002653 | [0.000748, 0.004558] | 9/10 |

The split-level five-seed averages show:
- Full test NLL favors exponential on **10/10** splits.
- Concordance favors exponential on **9/10** splits.
- IBLL favors exponential on **9/10** splits.
- IBS favors exponential on **7/10** splits, but its 95% CI crosses zero; this
  effect should be treated as inconclusive.

## 3. Optimization behavior

Exponential reaches its best validation checkpoint at a mean epoch of
**6.56**, versus
**15.04** for softplus. The paired split-level
difference is **-8.48 epochs**.

Mean measured training-loop time is
**0.549 s** for exponential versus
**0.834 s** for softplus.

There were **0 failed runs, 0 nonfinite-loss batches, 0 nonfinite-gradient batches,
and 0 gradient-clipped batches** in the 100 final runs. Thus, on SUPPORT, the
exponential mapping did not produce the numerical-instability penalty that might
have been expected from its rapidly growing transformation.

## 4. Tail behavior

The clearest tradeoff is extrapolation beyond the observed follow-up range.

At twice the maximum training follow-up time, mean predicted survival is:

- Softplus: **0.0075**
- Exponential: **0.1532**

This is about **20.4×**
more survival mass in the exponential model's far tail.

This supports the original paper's qualitative rationale for preferring softplus:
softplus behaves as a stronger tail constraint. However, that constraint did not
translate into better held-out SUPPORT performance in this experiment.

## 5. Robustness across the three paper evaluation horizons

For seed 1, the direction of the ablation effect remains similar at all three
censoring horizons:

| Metric | G=10^-8 delta | G=0.2 delta | G=0.4 delta |
|---|---:|---:|---:|
| Concordance (higher better) | 0.001636 | 0.001272 | 0.001413 |
| IBS (lower better) | -0.000076 | -0.000153 | -0.000295 |
| IBLL (higher better) | 0.001817 | 0.001474 | 0.001459 |

## 6. Recommended final-result statement

> The PyHealth Survival MDN reproduction closely matched the published SUPPORT
> results, particularly for integrated Brier score and integrated binomial
> log-likelihood. In a controlled 100-run ablation, replacing the paper's softplus
> positive-time transformation with an exponential transformation improved held-out
> censored likelihood on all ten SUPPORT splits and produced modest improvements in
> concordance and IBLL, while converging substantially earlier and exhibiting no
> numerical-instability failures. The exponential model, however, assigned much
> more probability to survival times beyond the observed follow-up range. These
> results suggest that the softplus transformation acts as a tail regularizer, but
> it was not empirically superior on SUPPORT under the fixed experimental
> configuration.

## 7. Figures

1. `figure1_reproduction_agreement.png` — reproduction-paper differences normalized
   by combined standard error.
2. `figure2_paired_test_nll_by_split.png` — split-level paired full-NLL differences.
3. `figure3_convergence_best_epoch.png` — best-epoch distribution across 50 runs per
   transformation.
4. `figure4_tail_survival.png` — extrapolated survival probabilities near and beyond
   the observed training range.

## Reporting cautions

- The paper used a 100-trial hyperparameter search; the reproduction used a fixed
  validated configuration. This must be disclosed.
- The ten SODEN splits reuse the same overall SUPPORT cohort, so they should not be
  described as ten independent datasets.
- Formal paired p-values are secondary; the main evidence should be effect size,
  direction across splits, and confidence intervals.
- IBS is not a clear win for exponential because the split-level CI includes zero.
- The result supports exponential on SUPPORT only; it does not establish universal
  superiority over softplus.
