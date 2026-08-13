---
name: pyhealth-add-a-component
description: Add a new component to the PyHealth package itself — a dataset, task, model, processor, metric, calibrator, or interpreter. Covers the base class or registry for each, the registration step, and the docs/examples/tests files the CI gate requires. Use when extending PyHealth rather than using it, or when contributing a module upstream.
---

# Add a component to PyHealth

Every extension point in PyHealth has the same five-step shape. Only the base class and the
registration step differ.

```
1. Implement    pyhealth/<subsystem>/<name>.py   — subclass the base class
2. Register     export in <subsystem>/__init__.py (processors also need a decorator)
3. Document     docs/api/<subsystem>/pyhealth.<module>.<Name>.rst + toctree entry
4. Exemplify    examples/<topic>/<name>_example.py
5. Test         tests/core/test_<name>.py, synthetic data, 2–5 patients
```

**Steps 3, 4, and 5 are not optional.** `.github/workflows/pr_contribution_rules.yml` fails any
PR that touches `pyhealth/**/*.py` without also touching `docs/` and `examples/`. Read
[repo-conventions.md](../../shared/repo-conventions.md) before writing code, not after.

**Before implementing anything, agree the file list with the user.** Five files, named. That is
this guide's equivalent of the modeling spec — it takes one message and prevents the whole
"beautiful module, failing CI" outcome.

---

## Pick your kind

| Kind | Base class / registry | Existing | Section |
|---|---|---|---|
| Dataset | `BaseDataset` — `datasets/base_dataset.py:308` | 24 | [↓](#dataset) |
| Task | `BaseTask` — `tasks/base_task.py:7` | 46 | [↓](#task) |
| Model | `BaseModel` — `models/base_model.py:13` | 48 | [↓](#model) |
| Processor | `PROCESSOR_REGISTRY` — `processors/__init__.py:1` | 24 | [↓](#processor) |
| Metric | function convention only | ~46 | [↓](#metric) |
| Calibrator | `PostHocCalibrator` / `SetPredictor` — `calib/base_classes.py:7,25` | 4 + 7 | [↓](#calibrator) |
| Interpreter | `BaseInterpreter` — `interpret/methods/base_interpreter.py:21` | 12 | [↓](#interpreter) |

**Read two existing implementations of your kind first.** These subsystems have strong
conventions that no document fully captures, and the closest sibling to what you are building is
the best specification available.

---

## Dataset

Two files: the class and its YAML table config.

```python
# pyhealth/datasets/new_dataset.py
class NewDataset(BaseDataset):
    def __init__(self, root, tables=None, dataset_name=None, config_path=None, **kwargs):
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "new_dataset.yaml"
        super().__init__(root=root, tables=tables or [...],
                         dataset_name=dataset_name or "new_dataset",
                         config_path=config_path, **kwargs)
```

The YAML carries the real work — one entry per table with `file_path`, `patient_id`,
`timestamp`, `attributes`, and optional `join`. Schema:
`pyhealth/datasets/configs/config.py` (`DatasetConfig`, `TableConfig`, `JoinConfig`); model
yours on `configs/mimic4_ehr.yaml`. Writing that config is covered in detail by
[bring-your-own-data](../bring-your-own-data/SKILL.md) — a dataset contribution is that guide's
output, promoted into the package.

Optional: a `preprocess_<table_name>` method is picked up automatically by
`load_table` (`base_dataset.py:690`) for per-table cleaning. A `default_task` property gives
users a one-liner starting point.

Register: export in `pyhealth/datasets/__init__.py`.

## Task

```python
# pyhealth/tasks/new_task.py
class NewTask(BaseTask):
    task_name: str = "NewTask"
    input_schema:  Dict[str, str] = {"conditions": "sequence"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __call__(self, patient) -> List[Dict[str, Any]]:
        ...
        return samples          # [] to exclude this patient
```

Schema values must be registered processor names —
[choose-processors](../choose-processors/SKILL.md). Class must be at module level (pickling).
`pre_filter` (`tasks/base_task.py`) narrows the patient set before `__call__` runs.

Register: export in `pyhealth/tasks/__init__.py`. For an optional heavy dependency, follow the
lazy `__getattr__` pattern already in that file.

Design guidance — cohort, temporal structure, leakage — is in
[define-a-task](../define-a-task/SKILL.md).

## Model

Convention is **two classes**: a pure `nn.Module` layer and the `BaseModel` wrapper. Look at
`pyhealth/models/rnn.py` for the canonical pair.

```python
class NewModelLayer(nn.Module):        # pure torch, no PyHealth types
    ...

class NewModel(BaseModel):
    def __init__(self, dataset, embedding_dim=128, hidden_dim=128, **kwargs):
        super().__init__(dataset=dataset)
        ...
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        return {"loss": loss, "y_prob": y_prob, "y_true": y_true}
```

`forward` **must** return that dict — `Trainer` and every metrics function depend on those three
keys. The model reads its input dimensions from the `SampleDataset`, so it must handle whichever
processor types its task uses; if it only supports sequences, say so in the docstring and add it
to the compatibility table in [choose-a-model](../choose-a-model/SKILL.md).

To make the model work with attribution methods, also implement `Interpretable`
(`interpret/api.py:5`) — note its two requirements: `forward_from_embedding()` treats label keys
as optional, and activations are `nn.Module` instances, never `F.relu` and friends. See
[interpret-a-model](../interpret-a-model/SKILL.md).

Register: export **both** classes in `pyhealth/models/__init__.py`.
`leaderboard/leaderboard_gen.py` does `from pyhealth.models import *`, so a properly exported
model is benchmarked automatically.

## Processor

The only true registry. Two steps, and forgetting the second is the classic mistake.

```python
# pyhealth/processors/new_processor.py
from . import register_processor
from .base_processor import FeatureProcessor

@register_processor("new_name")            # 1. the registered name — this is what schemas use
class NewProcessor(FeatureProcessor):
    def fit(self, samples, field): ...     # learn vocab / statistics from TRAIN data
    def process(self, value): ...          # one sample value → tensor
    def size(self): ...
```

```python
# pyhealth/processors/__init__.py
from . import new_processor               # 2. import it, or the decorator never runs
```

Without step 2 the class exists and `get_processor("new_name")` raises. Registering a duplicate
name also raises (`processors/__init__.py:4`).

Base classes in `processors/base_processor.py`: `FeatureProcessor` for the standard
one-value-one-tensor case, `TemporalFeatureProcessor` for the dict-output multimodal API (see
`temporal_timeseries_processor.py`), `TokenProcessorInterface` as a mixin when the processor
builds a vocabulary. Implement `is_token()`, `schema()`, `dim()`, `spatial()` as your base
requires.

**Fit on training data only.** A processor that learns statistics from the full dataset leaks.

## Metric

No base class — a naming and signature convention:

```python
def new_metrics_fn(y_true, y_prob, metrics=None) -> Dict[str, float]:
    if metrics is None:
        metrics = ["default_one", "default_two"]
    ...
```

Add your metric name to the relevant `pyhealth/metrics/*.py` function's accepted list and
document it in the docstring — that docstring list is what users read to discover metric names.
A genuinely new *task mode* also needs wiring into `get_metrics_fn` (`pyhealth/trainer.py:40`).

The one exception with a real ABC is interpretability: subclass `RemovalBasedMetric`
(`metrics/interpretability/base.py:21`).

## Calibrator

```python
class NewCalibrator(PostHocCalibrator):        # or SetPredictor
    def __init__(self, model, **kwargs):
        super().__init__(model)
    def calibrate(self, cal_dataset): ...
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]: ...
```

`PostHocCalibrator` for one-prediction-per-sample recalibration, `SetPredictor` for
coverage-guaranteed label sets (`calib/base_classes.py:7,25`). Place under
`calib/calibration/` or `calib/predictionset/` and export from that package's `__init__.py`.
Pair it with metrics in `metrics/calibration.py` or `metrics/prediction_set.py` — a calibrator
with nothing to score it by is untestable. Context:
[calibrate-predictions](../calibrate-predictions/SKILL.md).

## Interpreter

```python
class NewInterpreter(BaseInterpreter):
    def __init__(self, model, **kwargs):
        super().__init__(model)
    def attribute(self, **data) -> Dict[str, torch.Tensor]:
        ...        # keys MUST be the task's input_schema feature names
```

That key contract is the whole API — it is what keeps attributions portable across EHR, imaging,
and custom tasks (`interpret/methods/base_interpreter.py:21`). If your method needs embedding
access, document which mixin the target model must implement. Export from
`interpret/methods/__init__.py`, and score it with the comprehensiveness/sufficiency metrics
against `RandomBaseline`.

---

## Then ship it

Steps 3–5 and the CI gate: [ship-a-contribution](../ship-a-contribution/SKILL.md).
