# Survival MDN SUPPORT reproduction and ablation

Option 2 model contribution for [Survival Mixture Density Networks](https://proceedings.mlr.press/v182/han22a.html), Han, Goldstein and Ranganath (MLHC 2022). SUPPORT is the sole clinical dataset.

## Recorded results

The supplied `all_runs.csv` contains 10 exact SUPPORT splits × 5 paired seeds × 2 transformations = **100 runs**. These are the completed Phases 6–8 results from the standalone model core, not newly generated outputs from the final PyHealth example. Phase 9 provides the final tables and figures. The companion bundle preserves the source, data, run histories and reproduction checkpoints supplied by the user.

| Metric | Softplus | Exponential | Exp minus softplus | Split-level 95% interval |
|---|---:|---:|---:|---:|
| Test censored NLL ↓ | 0.517746 | 0.502901 | -0.014845 | [-0.018627, -0.011063] |
| Concordance ↑ | 0.621659 | 0.624975 | 0.003316 | [0.001374, 0.005258] |
| IBS ↓ | 0.189817 | 0.189404 | -0.000413 | [-0.000934, 0.000108] |
| IBLL ↑ | -0.560497 | -0.557844 | 0.002653 | [0.000748, 0.004558] |

Exponential improved NLL in all ten split-level five-seed averages. Mean best epoch was 6.56 for exponential and 15.04 for softplus. The 100 recorded runs had no failed runs or nonfinite-loss/gradient batches. At twice the maximum training follow-up, mean survival was approximately 0.1532 versus 0.0075. Heavier extrapolated tails cannot be interpreted as better calibration beyond observed follow-up.

IBS is inconclusive. The intervals are descriptive paired t intervals over overlapping random splits, not independent patient cohorts. Five seeds are averaged within each split. Do not treat 50 pairs as 50 independent datasets or claim universal superiority. `final_reproduction_table.csv` compares the baseline against published SUPPORT means at all three horizons; reproduction used a fixed validated configuration, not the paper's full hyperparameter search.

## Ablation contribution

Section 3.2 of the paper already identifies exponential as an alternative and motivates softplus through tail behavior. The project's extension is a controlled quantitative SUPPORT comparison of transformations, including predictive metrics, optimization and extrapolation. It does not claim to invent exponential mapping. The paper's appendix compares mixture base distributions; this is a different experiment. See [original paper](https://proceedings.mlr.press/v182/han22a/han22a.pdf). Final grading of novelty remains with the course staff.

## Experimental setup

- Cohort: 8,873 patients after excluding missing WBC/creatinine; 14 source covariates encoded into 27 features.
- Exact recovered SODEN memberships; standardization fitted on training patients only.
- Time: `d.time / 365.25 + 0.001`; event: `death` (1 event, 0 censored).
- Three Linear–BatchNorm–PReLU blocks, width 32, 10 Gaussian components, softplus scale parameterization.
- Initial weights equal, means linearly spaced from -3 to 3, scale logits 0.5. Paired initialization and identical minibatch random streams.
- RMSprop, learning rate 0.001, weight decay 1e-6, batch size 512, incomplete training batch dropped, gradient norm limit 100.
- At most 100 epochs, patience 10. Selection uses the historical unweighted mean of validation batch means (batch size 1024); reported test NLL is patient-weighted.
- Both transforms include the inverse Jacobian. NLL is primary; concordance, IBS, IBLL, convergence and tail behavior are secondary.
- Historical evaluator conventions are deliberately retained: test-cohort censoring KM weights, linear interpolation, 1,000-point uniform integration grid and strict concordance comparisons. It is an example-local benchmark evaluator, not a general survival-metrics API.
- All horizons were recorded for seed 1; other seeds have primary-horizon metrics only. Missing secondary-horizon cells are intentional.

## Run commands

Install the current PyHealth checkout with `python -m pip install -e .`. From its root:

```bash
python examples/support_survival_survival_mdn.py --epochs 2 --vary-hidden --output /tmp/mdn_synthetic
```

Synthetic data demonstrate execution only. Two transformation arms are run by default; `--vary-hidden` adds softplus width 64. This width comparison is separate from the historical 100-run experiment.

Use PyHealth's existing dataset with the supplied raw CSV:

```bash
python examples/support_survival_survival_mdn.py --source support2 --data /path/to/bundle/data/support2_with_patient_id.csv --epochs 8 --vary-hidden --output /tmp/mdn_support2
```

This reads records through `Support2Dataset`, then encodes and splits them. Without `--membership` it uses a reproducible random 70/15/15 split, so results are not expected to match the historical tables. Add the exact membership file to use recovered splits. The input must have explicit `sno` or `patient_id`; row identities are never guessed. Raw-mode outputs include an encoded CSV for inspection; keep patient-level outputs outside the PR.

Full paired experiment with the native PyHealth model:

```bash
python examples/support_survival_survival_mdn.py --source benchmark --data /path/to/bundle/data/support2_soden_pyhealth_samples_unstandardized.csv --membership /path/to/bundle/data/support2_soden_exact_split_matrix.csv --splits 1 2 3 4 5 6 7 8 9 10 --seeds 1 2 3 4 5 --epochs 100 --metrics all --output /tmp/mdn_benchmark
```

This evaluates all horizons for all seeds, extending evaluation beyond the historical seed-1-only secondary horizons. Training otherwise follows the same recipe. Use `--splits 1 --seeds 1 --epochs 2 --metrics nll` for a quick smoke run. Choose a fresh output directory; previous results are not overwritten. Numerical differences across dependency versions and hardware can occur.

For a completed width-32 100-run experiment, run the companion `reproduction/summarize_phase6.py --results /tmp/mdn_benchmark` to regenerate paired statistics. Do not include `--vary-hidden` in that 100-run command.

## Manual acceptance checks

1. Synthetic execution: finite NLL for each configuration, output CSV and configuration JSON.
2. Existing SUPPORT2: raw records parsed through PyHealth, expected 8,873 retained patients for the supplied CSV, finite arm results and training-only scaling.
3. Benchmark: unique split/seed/transform keys, all 100 runs present for the full command, no failures, paired summaries recomputed.
4. Interpretation: report the inconclusive IBS result and heavier exponential tails along with the improved NLL.
