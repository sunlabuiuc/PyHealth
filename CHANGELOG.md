# Changelog

All notable changes to PyHealth are documented here. Versions follow the
`major.minor.patch` scheme; see the
[releases page](https://github.com/sunlabuiuc/PyHealth/releases) for the
corresponding tags and wheels.

## 2.0.2 — 2026-09-02

The first release since 2.0.1 (2026-03-30). It adds four datasets, a synthetic-EHR
generation and evaluation stack, several new models and interpretability methods, and
a large batch of correctness fixes across models, tasks, metrics, and calibration.
No public API was removed or renamed — 2.0.1 code continues to work unchanged.

### New datasets

- **MEDS** — `MEDSDataset` plus a typed Parquet scan path on `BaseDataset`. `.parquet`
  / `.pq` files, globs, and directories now route through a typed `_scan_parquet`
  scanner, with a datetime fast path that skips the string round-trip. Includes the
  `in_hospital_mortality_meds` task. ([#1179](https://github.com/sunlabuiuc/PyHealth/pull/1179))
- **FHIR** — a full FHIR pipeline under `pyhealth/datasets/fhir/`, including a
  MIMIC-IV-on-FHIR dataset. ([#1155](https://github.com/sunlabuiuc/PyHealth/pull/1155))
- **EEGBCI** — dataset, helper functions, and tasks. ([#1177](https://github.com/sunlabuiuc/PyHealth/pull/1177))
- **PhysioNet De-Identification** — dataset, NER task (`deid_ner`), and the
  `TransformerDeID` model. ([#981](https://github.com/sunlabuiuc/PyHealth/pull/981))

### New models

- **MedFuse** — multi-modal fusion of EHR time series with chest X-rays. ([#1003](https://github.com/sunlabuiuc/PyHealth/pull/1003))
- **Synthetic EHR generators** — `pyhealth/models/generators/`: HALO, MedGAN, CorGAN,
  PromptEHR, and a GPT-2 generator, with the `generate_ehr` task. ([#1148](https://github.com/sunlabuiuc/PyHealth/pull/1148))
- **CaliForest** — calibrated random forest; requires an explicit `fit` before
  inference. ([#999](https://github.com/sunlabuiuc/PyHealth/pull/999))
- **GRASP** — migrated from the 1.x API to 2.0, with `static_key` support for
  demographic features. ([#905](https://github.com/sunlabuiuc/PyHealth/pull/905))

### New tasks and metrics

- **`DrugRecommendationOMOP`** — a class-based OMOP drug recommendation task. ([#1203](https://github.com/sunlabuiuc/PyHealth/pull/1203))
- **Generative evaluation metrics** — `pyhealth/metrics/generative/` scores synthetic
  EHR data on privacy (NNAAR, membership inference, discriminator privacy), utility,
  and statistical fidelity. ([#1148](https://github.com/sunlabuiuc/PyHealth/pull/1148))
- **Attention rollout** — interpretability method of Abnar & Zuidema (2020). ([#1158](https://github.com/sunlabuiuc/PyHealth/pull/1158))
- **Conformal prediction** — real Adaptive Prediction Sets (Romano, Sesia & Candès
  2020) in the new `pyhealth/calib/predictionset/scores.py`, with a dynamic
  `score_type` on conformal methods ([#1189](https://github.com/sunlabuiuc/PyHealth/pull/1189)),
  plus additional conformal methods and example scripts ([#942](https://github.com/sunlabuiuc/PyHealth/pull/942)).

### Restored from 1.x

- **`code_mapping`** — `SequenceProcessor` accepts an optional `code_mapping` that
  collapses granular codes into grouped vocabularies (ICD9CM→CCSCM, ICD9PROC→CCSPROC,
  NDC→ATC) before building the embedding table, and `BaseTask.__init__` accepts it
  directly so schemas no longer have to be patched by hand. Closes the functional gap
  left by the 1.x→2.0 rewrite. ([#905](https://github.com/sunlabuiuc/PyHealth/pull/905), ref [#535](https://github.com/sunlabuiuc/PyHealth/issues/535))

### Fixes

**Data leakage and label correctness**

- StageNet MIMIC-IV mortality/LOS tasks leaked post-outcome information: diagnosis and
  procedure codes are timestamped at `dischtime`, so for the admission being predicted
  they are only known at or after the outcome, and labs were pulled through
  discharge/death. Those codes are now excluded for that admission and its labs capped
  to the first 48 hours. ([#1205](https://github.com/sunlabuiuc/PyHealth/pull/1205))
- `drug_recommendation_omop_fn` never excluded the current visit's own drugs from
  `drugs_all`, making the last history entry identical to the prediction target. ([#1203](https://github.com/sunlabuiuc/PyHealth/pull/1203))
- Drug tasks extracted `event.drug` (drug *names*, e.g. "Aspirin"), which produce zero
  matches in the NDC→ATC CrossMap; they now extract `event.ndc`. ([#905](https://github.com/sunlabuiuc/PyHealth/pull/905))
- Drug recommendation NDC/ATC3 code handling and padding behaviour. ([#1138](https://github.com/sunlabuiuc/PyHealth/pull/1138))

**Models**

- `CNN` crashed on 1-D tensor and multi-hot inputs: `forward` hardcoded a 3-D
  expectation for `spatial_dim=1`, but `MultiHotProcessor` and 1-D `TensorProcessor`
  inputs embed to `[batch, embedding_dim]` with no sequence axis. Now treated as a
  length-1 sequence. ([#1208](https://github.com/sunlabuiuc/PyHealth/pull/1208))
- `TCN` crashed on tuple-schema features by passing raw kwargs (including
  `StageNetProcessor`'s `(time, value)` tuples) to the embedding model; it now unwraps
  the `value` tensor first, like its sibling sequence models. ([#1212](https://github.com/sunlabuiuc/PyHealth/pull/1212))
- `BIOT` hardcoded `nn.Embedding(n_channels, 256)` for channel tokens, crashing for
  any `emb_size != 256`. ([#1213](https://github.com/sunlabuiuc/PyHealth/pull/1213))
- `MoleRec`'s no-SMILES fallback predictor was created lazily inside `forward`, so an
  optimizer built from `model.parameters()` beforehand never saw its parameters and it
  never trained. It is now created in `__init__`. ([#1214](https://github.com/sunlabuiuc/PyHealth/pull/1214))
- `SdohClassifier` was an `nn.Module` decorated with `@dataclass`, whose generated
  `__init__` never called `nn.Module.__init__`, leaving the module without
  `_parameters`/`_modules` and unusable in torch. ([#1209](https://github.com/sunlabuiuc/PyHealth/pull/1209))
- `SinusoidalTimeEmbedding` divided frequency indices by `half - 1`, so `dim=2` gave
  0/0 and every embedding was NaN. Clamped to at least 1. ([#1216](https://github.com/sunlabuiuc/PyHealth/pull/1216))
- Sparsemax in `AdaCare`. ([#1139](https://github.com/sunlabuiuc/PyHealth/pull/1139))
- `RNNLayer` and `ConCare` crashed on zero-length sequences and on `batch_size=1`;
  `GRASP` collapsed its hidden state at `batch_size=1` and raised when
  `cluster_num > batch_size`. ([#905](https://github.com/sunlabuiuc/PyHealth/pull/905))
- MedLink: `collate_fn` built output keys from only the first sample in a batch, so a
  batch mixing samples with and without a mined hard negative (`s_n`) either raised
  `KeyError` or silently produced a misaligned list; keys are now unioned across the
  batch. ([#1222](https://github.com/sunlabuiuc/PyHealth/pull/1222))
- MedLink BM25 hard-negative mining did not preserve all positives. ([#1195](https://github.com/sunlabuiuc/PyHealth/pull/1195))

**Datasets and tasks**

- Patient merging crashed on tables with null `patient_id`. ([#1193](https://github.com/sunlabuiuc/PyHealth/pull/1193))
- `SampleDataset` subset mappings were wrong. ([#1211](https://github.com/sunlabuiuc/PyHealth/pull/1211))
- `PatientLinkageMIMIC3Task`'s `input_schema` named `"integer"`/`"string"`/
  `"datetime"` processors, none of which are registered, so `set_task()` failed
  immediately with `ValueError: Unknown processor`. ([#1204](https://github.com/sunlabuiuc/PyHealth/pull/1204))

**Metrics, calibration, and interpretability**

- `ece_confidence_binary` indexed `prob[:, 0]`/`label[:, 0]`, requiring 2-D arrays,
  but its only caller passes 1-D positive-class probabilities and 1-D 0/1 labels — so
  `ECE` and `ECE_adapt` always raised `IndexError` on binary tasks. ([#1215](https://github.com/sunlabuiuc/PyHealth/pull/1215))
- `disparate_impact` and `statistical_parity_difference` returned `nan` for empty
  subgroups instead of raising: the rate was a numpy 0/0, and since `nan == 0` is
  always `False` the existing zero guard never caught it (and only ever checked the
  unprotected group). ([#1199](https://github.com/sunlabuiuc/PyHealth/pull/1199))
- `fairness_metrics_fn` was commented out of `pyhealth.metrics`; re-enabled and added
  to `__all__`. ([#1200](https://github.com/sunlabuiuc/PyHealth/pull/1200))
- Removal-based interpretability metrics aliased `original_class_probs` to `y_probs`
  and negated negative-class entries in place, flipping the sign on every iteration of
  the percentage loop — a sample's score depended on where its percentage sat in the
  list. ([#1196](https://github.com/sunlabuiuc/PyHealth/pull/1196))
- Interpretability `target_class_idx` handling, argument naming, and sample-class
  filtering. ([#926](https://github.com/sunlabuiuc/PyHealth/pull/926))
- SCRIB: the overall-risk loss squared the chance-ambiguity term, contradicting Eq. 2
  and Algorithm 2 of the paper and the already-correct class-specific loss in the same
  file; fixed in both the Python and Cython paths, along with a `fill_max` inference
  gap. ([#1190](https://github.com/sunlabuiuc/PyHealth/pull/1190))
- Covariate-shift conformal prediction fixes. ([#1180](https://github.com/sunlabuiuc/PyHealth/pull/1180))

**Examples and docs**

- Removed the deprecated `code_mapping`, `dev`, and `refresh_cache` arguments from
  `README.rst`, example scripts, and leaderboard utilities — the 2.0
  `MIMIC3Dataset`/`MIMIC4Dataset` no longer accept them. ([#935](https://github.com/sunlabuiuc/PyHealth/pull/935), fixes [#535](https://github.com/sunlabuiuc/PyHealth/issues/535))
- Fixed a `SyntaxError` in `examples/benchmark_perf/loc/minimal_los.py`, a missing
  `__main__` guard that hung `readmission_mimic3_fairness.py` under multiprocessing,
  and a stale `Transformer(...)` call. ([#1200](https://github.com/sunlabuiuc/PyHealth/pull/1200))
- Reinitialized the documentation tutorials lost in the UIUC purge and re-linked the
  Colab notebooks. ([#1143](https://github.com/sunlabuiuc/PyHealth/pull/1143), [#1146](https://github.com/sunlabuiuc/PyHealth/pull/1146))
- Added missing paper citations throughout the codebase. ([#1181](https://github.com/sunlabuiuc/PyHealth/pull/1181))

### Infrastructure

- CI gate enforcing the PR contribution rules for changes under `pyhealth/`
  (`tools/check_pr_rules.py`). ([#1176](https://github.com/sunlabuiuc/PyHealth/pull/1176))
- Unit tests for `RNN` and `MultimodalRNN`. ([#936](https://github.com/sunlabuiuc/PyHealth/pull/936))
- Fixed a pixi warning and the version format for the build backend. ([#917](https://github.com/sunlabuiuc/PyHealth/pull/917))
- `tools/bump_version.py` now keeps `pyhealth.__version__` in sync with
  `pyproject.toml`, rewrites only the `[project]` version line, and no longer hangs
  when bumping from a non-pre-release version.

**Full Changelog**: https://github.com/sunlabuiuc/PyHealth/compare/v2.0.1...v2.0.2
