# Changelog

All notable changes to PyHealth are documented here. Versions follow the
`major.minor.patch` scheme; see the
[releases page](https://github.com/sunlabuiuc/PyHealth/releases) for the
corresponding tags and wheels.

## 2.1.0 — 2026-10-16

A minor release focused on making PyHealth results trustworthy: leakage-free
preprocessing with split-aware `set_task`, reproducible training, calibration metrics
with patient-level confidence intervals, and a large set of cache and data-handling
fixes. It also adds an XGBoost model with exact TreeSHAP, Synthea synthetic-patient
generation, ICD-9 ↔ ICD-10 translation, and a new nine-notebook tutorial series.

Most 2.0.2 code runs unchanged, but read **Upgrading from 2.0.2** below: dependency
floors moved (notably `torch>=2.10`), the EHR-generation tasks were reorganised,
and a few behaviours changed on purpose.

### Upgrading from 2.0.2

- **Dependencies.** `torch>=2.10`, `pyarrow>=23.0.1`, `dask[complete]>=2026.1.0` and
  `litdata>=0.2.64,<0.2.68` are now required; numpy, pandas, pydantic, scikit-learn and
  narwhals use bounded ranges instead of `~=` pins, so installs on Google Colab no
  longer downgrade preloaded packages. `urllib3` is no longer a dependency, and
  `icd-mappings` is new. ([#1254](https://github.com/sunlabuiuc/PyHealth/pull/1254),
  [#1287](https://github.com/sunlabuiuc/PyHealth/pull/1287),
  [#1273](https://github.com/sunlabuiuc/PyHealth/pull/1273),
  [#1257](https://github.com/sunlabuiuc/PyHealth/pull/1257))
- **EHR-generation tasks.** The `EHRGeneration` base class is gone. There are now
  six tasks, one per encoding and dataset:
  `EHRGenerationMIMIC3/4` (per-visit multi-hot, for HALO),
  `EHRSequenceGenerationMIMIC3/4` (per-visit code indices, for GPT-2 and PromptEHR) and
  `EHRCodeSetGenerationMIMIC3/4` (one code set per patient, for MedGAN and CorGAN).
  `EHRGenerationMIMIC3/4` now emit `nested_multihot` instead of `nested_sequence`
  rows; GPT-2 and PromptEHR users should switch to `EHRSequenceGenerationMIMIC3/4`.
  ([#1233](https://github.com/sunlabuiuc/PyHealth/pull/1233))
- **Missing `Event` timestamps are `None`.** `Event` used to substitute
  `datetime.now()` for a missing timestamp (e.g. MIMIC `patients`, eICU tables), which
  sorted before every real event and silently passed time-window filters. Code that
  relied on a non-null timestamp must handle `None`. ([#1247](https://github.com/sunlabuiuc/PyHealth/pull/1247))
- **Caches rebuild once.** The dataset cache key now covers the YAML config and the
  source files' size and mtime, and the processed-sample cache key covers the new
  storage format, so existing caches are rebuilt on first use. Old cache folders are
  left in place and can be deleted. ([#1265](https://github.com/sunlabuiuc/PyHealth/pull/1265),
  [#1277](https://github.com/sunlabuiuc/PyHealth/pull/1277))
- **Shuffle orders differ from 2.0.2.** Dataloaders now draw shuffles from a seeded
  per-loader generator, and patient-level splitters sort patient IDs before shuffling.
  Runs are reproducible from here on, but the same seed selects different batches and
  splits than it did in 2.0.2. ([#1280](https://github.com/sunlabuiuc/PyHealth/pull/1280),
  [#1286](https://github.com/sunlabuiuc/PyHealth/pull/1286))
- **Logging.** The `pyhealth` logger no longer propagates to the root logger, so each
  message prints once in Jupyter and Colab. To route PyHealth messages through your own
  root configuration, remove `pyhealth.handler` and re-enable propagation. ([#1285](https://github.com/sunlabuiuc/PyHealth/pull/1285))
- **Default cache warning.** Building a dataset without `cache_dir` now logs a
  warning naming the user cache folder it writes to; set
  `PYHEALTH_REQUIRE_CACHE_DIR=1` to make it an error. ([#1264](https://github.com/sunlabuiuc/PyHealth/pull/1264))
- The unmaintained 2023 `chat-assistant/` demo was removed (it remains in git
  history). ([#1252](https://github.com/sunlabuiuc/PyHealth/pull/1252))

### Highlights

- **Split-aware preprocessing** — `dataset.set_task(task, split=PatientSplit(...))`
  fits every processor (vocabularies and any learned statistics) on the training
  patients only and returns `(train, val, test)` views of one processed cache.
  `PatientSplit` accepts two or more ratios, and custom splits implement the new
  `Split` interface. Processors that learn statistics warn when fitted on all
  samples. ([#1273](https://github.com/sunlabuiuc/PyHealth/pull/1273))
- **Reproducible training** — `get_dataloader(..., seed=None)` gives each loader its
  own seeded, per-epoch shuffle on both the in-memory and disk backends, so
  `torch.manual_seed` alone makes runs repeatable. `pyhealth.utils.preserve_rng_state()`
  is new; `set_task`, dataset creation and interpretability methods no longer
  touch global RNG state, and importing `pyhealth.models` no longer reseeds numpy
  and torch. `Trainer.train` restores the best epoch even with logging off
  (`keep_best_in_memory`). ([#1280](https://github.com/sunlabuiuc/PyHealth/pull/1280))
- **Calibration and confidence intervals** — `binary_metrics_fn` adds `brier`,
  `oe_ratio`, `calibration_slope` and `calibration_intercept`; the new
  `pyhealth.metrics.bootstrap` module provides `bootstrap_ci` and
  `paired_bootstrap_diff`, with `groups=` for patient-clustered resampling.
  ([#1268](https://github.com/sunlabuiuc/PyHealth/pull/1268))
- **Imbalanced outcomes** — `BaseModel.set_pos_weight(...)` takes a number, a
  per-label list, or `"balanced"` computed from a training dataset, and applies to
  every model that uses the default binary/multilabel loss. ([#1267](https://github.com/sunlabuiuc/PyHealth/pull/1267))

### New models and generators

- **`XGBoostModel`** — gradient-boosted trees for binary, multiclass, multilabel and
  regression tasks, fit with `model.fit(train, val=None)` and evaluated through the
  usual `Trainer`. Supports tensor, timeseries, multi-hot and nested multi-hot
  inputs (and bag-of-codes for code sequences), `scale_pos_weight`, early stopping,
  and exact TreeSHAP through `explain()` and the new
  `pyhealth.interpret.methods.TreeSHAP`. Install with `pip install "pyhealth[xgboost]"`.
  ([#1278](https://github.com/sunlabuiuc/PyHealth/pull/1278))
- **Synthea** — `pyhealth.models.Synthea` runs the Synthea synthetic-patient
  generator (Java 17+ required; the JAR is downloaded and cached), and
  `SyntheaCSVDataset` loads its CSV output. ([#1244](https://github.com/sunlabuiuc/PyHealth/pull/1244))
- **HALO** — visit encoding is now vectorised, removing a per-code Python loop that
  took ~85% of a training step on GPU. ([#1233](https://github.com/sunlabuiuc/PyHealth/pull/1233))

### New processors and medical codes

- **`NestedMultiHotProcessor`** (`"nested_multihot"`) — one multi-hot row per visit,
  sized by the vocabulary rather than by the longest visit (8.6× smaller per patient
  on eICU). ([#1233](https://github.com/sunlabuiuc/PyHealth/pull/1233))
- **ICD-9 ↔ ICD-10 translation and grouper vocabularies** — `CrossMap` gains an
  `icd-mappings` backend for pairs PyHealth hosts no table for (ICD-9 ↔ ICD-10,
  CCSR, chapters and blocks, chronic-condition indicators, and more). Existing
  mappings such as ICD9CM→CCSCM are unchanged. ICD-9 ↔ ICD-10 is a primary-mapping
  approximation of the GEMs, reported through a warning and
  `CrossMap.unmapped_codes`. `CrossMap` also downloads vocabulary tables only when
  they are used. ([#1257](https://github.com/sunlabuiuc/PyHealth/pull/1257))
- `TensorProcessor.size()` now returns the fitted feature width, and embeddings are
  sized from already-processed samples instead of re-running `process()`.
  ([#1262](https://github.com/sunlabuiuc/PyHealth/pull/1262))

### Fixes

**Data leakage and label correctness**

- `MultimodalMortalityPredictionMIMIC4` used chest X-rays from after the prediction
  time; X-rays are now limited to those available by the last included admission's
  discharge. ([#1229](https://github.com/sunlabuiuc/PyHealth/pull/1229))
- `T1DDKAPredictionMIMIC4` counted DKA events recorded before a separate T1DM
  diagnosis; those patients are now excluded. ([#1230](https://github.com/sunlabuiuc/PyHealth/pull/1230))
- With more than 10 workers, `set_task` merged worker indexes in string order
  (0, 1, 10, 11, …), so `patient_to_index` pointed at other patients' samples and
  patient-level splits shared patients (on full MIMIC-IV with 32 workers, ~3.6k
  patients appeared in both train and val). Fixed by requiring `litdata>=0.2.64`.
  **Caches built with more than 10 workers before this release keep the wrong order;
  delete and rebuild them.** ([#1273](https://github.com/sunlabuiuc/PyHealth/pull/1273))
- Supplying pre-fitted processors for some fields left the other fields without a
  processor, so raw values reached the model; missing processors are now fitted.
  ([#1273](https://github.com/sunlabuiuc/PyHealth/pull/1273))

**Datasets, caching and samples**

- Rewriting a source file or editing a dataset's YAML config silently reused the
  stale cached event table. ([#1265](https://github.com/sunlabuiuc/PyHealth/pull/1265))
- Sample fields whose length or shape varies between samples (notes, measurement
  times, `None` mixed with lists) failed, were truncated, or were misread on the
  disk-backed cache. Non-tensor fields are now stored as one opaque value per sample.
  ([#1277](https://github.com/sunlabuiuc/PyHealth/pull/1277))
- Nested sequence and float processors now truncate visits longer than the fitted
  width (with a one-time warning) instead of producing ragged batches.
  ([#1263](https://github.com/sunlabuiuc/PyHealth/pull/1263))
- `Event` objects could not be copied or unpickled (`RecursionError`), so a sample
  containing one was written to the cache but crashed when read back. ([#1247](https://github.com/sunlabuiuc/PyHealth/pull/1247))
- Every dataset build leaked three file descriptors through dask's dashboard,
  eventually failing with "Too many open files" on macOS. ([#1245](https://github.com/sunlabuiuc/PyHealth/pull/1245))
- Downloads are now atomic with a timeout: an interrupted download no longer leaves
  a truncated file that later runs reuse, and a stalled server no longer hangs.
  medcode JSON downloads no longer re-download on every call. ChestX-ray14 extraction
  rejects unsafe archive members. ([#1255](https://github.com/sunlabuiuc/PyHealth/pull/1255))
- `split_by_patient`, `split_by_patient_conformal` and the TUH patient splitters
  picked different patients for the same seed depending on sample order.
  ([#1286](https://github.com/sunlabuiuc/PyHealth/pull/1286))
- `create_sample_dataset(in_memory=False)` could leave helper processes running
  after an error and hang at exit; it now writes in-process. The in-memory dataset
  no longer yields every sample once per dataloader worker. ([#1277](https://github.com/sunlabuiuc/PyHealth/pull/1277),
  [#1280](https://github.com/sunlabuiuc/PyHealth/pull/1280))

**Models**

- `model.mode` was overwritten with the raw schema entry in 16 models, so processor
  class or tuple `output_schema` forms crashed `Trainer.evaluate`, calibration and
  `Agent`. `BaseModel.mode` is now a property that always resolves to a mode string.
  ([#1246](https://github.com/sunlabuiuc/PyHealth/pull/1246), fixes [#914](https://github.com/sunlabuiuc/PyHealth/issues/914))
- GRASP never built its similar-patient graph, falling back to an identity
  adjacency; the kNN cluster graph now works with self-connections. ([#1194](https://github.com/sunlabuiuc/PyHealth/pull/1194))
- `TransformerDeID` ignored its configured feature and label keys. ([#1224](https://github.com/sunlabuiuc/PyHealth/pull/1224))

**Metrics and interpretability**

- Removal-based interpretability metrics (comprehensiveness, sufficiency) crashed
  for every model with integer code-sequence inputs; removed codes are now set to the
  padding id. ([#1284](https://github.com/sunlabuiuc/PyHealth/pull/1284))
- Ranking metrics used a bare `except`. ([#1237](https://github.com/sunlabuiuc/PyHealth/pull/1237))

### Docs and examples

- A nine-notebook tutorial series under `examples/tutorials/series/`, from a first
  model through datasets, tasks, training and evaluation, model choice,
  interpretation, medical codes, clinical text, and contributing. Each notebook runs
  on CPU in Colab. ([#1281](https://github.com/sunlabuiuc/PyHealth/pull/1281))
- New docs sections and runnable examples covering reproducible runs, shuffling,
  split-aware `set_task`, reusing fitted processors, imbalanced outcomes,
  calibration and bootstrap CIs, cache behaviour, identified data, logging setup,
  and custom tensor processors.
- Fixed the GAMENet drug-recommendation example; `minimal_legacy_*.py` scripts are
  labelled as PyHealth 1.1.6-only. ([#1206](https://github.com/sunlabuiuc/PyHealth/pull/1206))

### Infrastructure and security

- `SECURITY.md` describes private vulnerability reporting and the trust model
  (pickle-based caches and checkpoints must come from trusted sources). Workflows
  declare least-privilege token permissions. ([#1253](https://github.com/sunlabuiuc/PyHealth/pull/1253))
- Dependency floors were raised past known CVEs in torch, pyarrow, dask and urllib3;
  pip-audit reports no known vulnerabilities on a fresh install. ([#1254](https://github.com/sunlabuiuc/PyHealth/pull/1254))
- CI reports test coverage. ([#1299](https://github.com/sunlabuiuc/PyHealth/pull/1299))
- Tests find their resources relative to the test file, and a flaky ATC-3 ordering
  assertion is now order-insensitive. ([#1292](https://github.com/sunlabuiuc/PyHealth/pull/1292),
  [#1261](https://github.com/sunlabuiuc/PyHealth/pull/1261))

**Full Changelog**: https://github.com/sunlabuiuc/PyHealth/compare/v2.0.2...v2.1.0

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
