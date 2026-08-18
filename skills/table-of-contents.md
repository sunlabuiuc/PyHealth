# Table of contents

The full manifest. [SKILL.md](SKILL.md) carries the abbreviated routing table; this is what to
read when that table is not enough to choose confidently, and what a human reads to see the
whole offering at once.

Every guide is a directory under `guides/` with its own `SKILL.md` carrying `name` and
`description` frontmatter, so any one of them can be installed standalone:

```sh
python tools/install_skills.py --guide define-a-task --target ../my-project
```

---

## Using PyHealth

### `set-up-the-environment`
**Purpose.** Get PyHealth installed and importable before any modeling work — detect what is
already present, choose the user path (`pip install pyhealth`) or the contributor path (clone
plus editable install), create the environment with permission, verify the heavy imports rather
than just the top-level one, and register this skill in the user's project.
**Prerequisites.** None. Runs before everything, including `scope-a-modeling-request`.
**Produces.** A working environment, a printed version, and `.claude/skills/pyhealth` plus the
`AGENTS.md` pointer block in the user's project.
**Touches.** Nothing in the package — this is environment and install work.
**Size.** ~205 lines.

### `scope-a-modeling-request`
**Purpose.** Turn a vague clinical prediction request into a written, confirmed spec — cohort,
unit of prediction, index time, observation window, prediction gap, label definition, split
strategy, metric. Batches the questions into one round, each with a recommended default, so a
user who does not know the answer can still proceed.
**Prerequisites.** None. This is usually the first guide.
**Produces.** `pipeline_spec.md`, confirmed by the user.
**Touches.** Nothing in the package — this is the conversation before the code.
**Size.** ~160 lines.

### `bring-your-own-data`
**Purpose.** Turn a folder of raw patient files into a PyHealth dataset: inventory the tables
with real code, assign each to `patient_id` / `timestamp` / `attributes`, write the YAML table
config (including `join` blocks for tables with no time of their own), load with `BaseDataset`,
and validate. The hardest step for a newcomer and the one where silent failure is most likely.
**Prerequisites.** Files the agent can read.
**Produces.** A YAML config the user can read, plus a dataset that loads with a plausible
patient and event count.
**Touches.** `pyhealth/datasets/base_dataset.py`, `pyhealth/datasets/configs/`.
**Size.** ~220 lines.

### `use-a-dataset`
**Purpose.** Pick and load a built-in dataset — MIMIC-III/IV, eICU, OMOP, EHRShot, MEDS, and
the signal/imaging sets. Per-dataset tables, which column becomes `event.timestamp` versus a
named attribute, and how to filter events down to one visit. Includes the case rule: `tables=`
is case-insensitive but `event_type=` is not, and getting it wrong returns an empty list with
no error.
**Prerequisites.** Data is one of the supported datasets.
**Produces.** A dataset class choice, table list, and cache setup.
**Touches.** `pyhealth/datasets/`, `pyhealth/datasets/configs/*.yaml`.
**Size.** ~200 lines.

### `define-a-task`
**Purpose.** Write a `BaseTask` — cohort filtering, temporal structure, label computation, and
the schemas that turn one `Patient` into sample dicts. Covers the four temporal patterns
(next-visit, previous-visit history, cumulative nested, same-visit) and which are leakage-safe.
Ships seven runnable example task classes to copy.
**Prerequisites.** A loading dataset.
**Produces.** `task.py` with one module-level task class.
**Touches.** `pyhealth/tasks/`, `pyhealth/data/data.py`.
**Size.** ~300 lines + 7 example scripts.

### `choose-processors`
**Purpose.** Pick the processor for each feature and label. The complete 24-name registry with
class, source file, and expected value type, then detailed treatment of the six most-used —
including the missing-data sentinels and the empty-inner-list rules that crash training at
`pack_padded_sequence`.
**Prerequisites.** Knowing what features the task will emit.
**Produces.** `input_schema` / `output_schema` values that resolve.
**Touches.** `pyhealth/processors/`.
**Size.** ~215 lines.

### `choose-a-model`
**Purpose.** Pick a model compatible with the task's processor types. Six families — sequential,
multimodal, non-sequential, drug recommendation, graph, signal, generative — with a
processor-compatibility column, plus the rule for what to do when a requested model rejects the
schema (prefer the multimodal sibling, not an unrelated architecture).
**Prerequisites.** A task with defined schemas.
**Produces.** A model class and constructor arguments.
**Touches.** `pyhealth/models/`.
**Size.** ~205 lines.

### `train-and-evaluate`
**Purpose.** Split by patient, build dataloaders, run the `Trainer`, evaluate honestly. Monitor
metrics, checkpointing, the caching rules, seeds, the `__main__` guard. Its
`references/metrics-and-reporting.md` covers which metric to report per task mode, the
anti-leakage rules, and what a defensible results file contains.
**Prerequisites.** A `SampleDataset` and a model.
**Produces.** A trained model and metrics you can defend.
**Touches.** `pyhealth/trainer.py`, `datasets/splitter.py`, `datasets/utils.py`, `metrics/`.
**Size.** ~185 lines + ~155 line reference.

### `optimize-a-pipeline`
**Purpose.** Improve a working baseline. Task engineering — feature ablation, ICD truncation,
CCS mapping, history windows, z-normalization — with hyperparameter tuning in
`references/hyperparameter-tuning.md`. Carries the significance rule that stops noise being
banked as progress: ≥2 seeds, adopt only if mean improvement exceeds one standard deviation.
**Prerequisites.** A baseline that trains end to end. **Only when the user asks** — a working,
honest baseline is a complete deliverable.
**Produces.** A tuned configuration with a documented reason for each adopted change.
**Touches.** `pyhealth/tasks/`, `pyhealth/models/`, `pyhealth/trainer.py`.
**Size.** ~205 lines + ~170 line reference.

### `calibrate-predictions`
**Purpose.** Make probabilities mean what they say (`TemperatureScaling`, `HistogramBinning`,
`DirichletCalibration`, `KCal`) or return label sets with a coverage guarantee (`LABEL`,
`SCRIB`, `FavMac`, `ClusterLabel`, `NeighborhoodLabel`, `CovariateLabel`, `BaseConformal`).
Explains what each guarantees and what it does not — in particular that marginal coverage says
nothing about subgroups.
**Prerequisites.** A trained model and a calibration split it never saw.
**Produces.** A calibrated model or set predictor, with before/after ECE or coverage plus
average set size.
**Touches.** `pyhealth/calib/`, `pyhealth/metrics/calibration.py`, `metrics/prediction_set.py`.
**Size.** ~145 lines.

### `interpret-a-model`
**Purpose.** Explain what a trained model keys on, and measure whether the explanation is
faithful. Ten attribution methods plus three ensembles, the `Interpretable` /
`CheferInterpretable` mixin requirement (only `MLP`, `StageNet`, `Transformer`, and
`StageAttentionNet` implement them), and the comprehensiveness/sufficiency metrics scored
against `RandomBaseline`.
**Prerequisites.** A trained model; check the mixin before promising a gradient method.
**Produces.** Per-feature attributions plus faithfulness scores against the random control.
**Touches.** `pyhealth/interpret/`, `pyhealth/metrics/interpretability/`.
**Size.** ~130 lines.

### `map-medical-codes`
**Purpose.** Work with clinical vocabularies. `InnerMap` for lookups and hierarchy walking
across ten vocabularies; `CrossMap` for translation (ICD→CCS, NDC→ATC, NDC→RxNorm). Includes
the MIMIC-IV mixed ICD-9/ICD-10 routing trap and the rule that mapping changes the samples, so
it needs a new `task_name`.
**Prerequisites.** Codes to map. First load downloads.
**Produces.** A code-mapping helper on the task, plus a count of unmappable codes.
**Touches.** `pyhealth/medcode/`.
**Size.** ~155 lines.

---

## Contributing to PyHealth

### `add-a-component`
**Purpose.** Add a dataset, task, model, processor, metric, calibrator, or interpreter to the
package. One guide rather than seven because every extension point shares the same five-step
shape — implement, register, document, exemplify, test — and only the base class and
registration step differ. Per-kind sections carry the base class, the file layout, and the
mistakes specific to that kind (notably: a processor needs both the `@register_processor`
decorator *and* an import in `processors/__init__.py`, or the decorator never runs).
**Prerequisites.** Agreement with the user on the five target files.
**Produces.** The implementation plus its registration.
**Touches.** whichever `pyhealth/` subsystem, plus `docs/` and `examples/`.
**Size.** ~215 lines.

### `ship-a-contribution`
**Purpose.** Get the change through CI. The three enforced rules — docs+examples mandatory,
ruff-clean on changed lines, `>>>` docstring examples on new public APIs — plus the docs stub
and toctree pattern, the synthetic-data test requirement, a pre-PR checklist, and a table of
what each CI failure actually means.
**Prerequisites.** An implementation.
**Produces.** A PR that passes `tools/check_pr_rules.py`.
**Touches.** `docs/`, `examples/`, `tests/core/`.
**Size.** ~175 lines.

---

## Shared

Referenced by many guides; not routed to directly.

| File | Contents |
|---|---|
| [shared/glossary.md](shared/glossary.md) | Clinical ML terms in plain language — cohort, index time, observation window, prediction gap, leakage, class imbalance — plus PyHealth's own vocabulary. Read when a term is unfamiliar to you or to the user. |
| [shared/data-model.md](shared/data-model.md) | `Event`, `Patient`, `get_events` filter semantics, sample dicts, `SampleDataset`, the caching rules, the `__main__` guard. |
| [shared/repo-conventions.md](shared/repo-conventions.md) | The CI gate, `docs/api/` stub pattern, test conventions, code style, branching. |
