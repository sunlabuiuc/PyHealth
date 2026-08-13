---
name: pyhealth-train-and-evaluate
description: Split by patient, build dataloaders, run the Trainer, and evaluate honestly — monitor metrics, caching, seeds, the __main__ guard, and the anti-leakage rules. Use when turning a SampleDataset into a trained, scored model.
---

# Train and evaluate

Turn a `SampleDataset` into loaders and run the
`Trainer`. For choosing a model, see [models.md](../choose-a-model/SKILL.md). For what the metrics
mean, see [evaluation.md](references/metrics-and-reporting.md).

---

## Split

```python
from pyhealth.datasets import split_by_patient, get_dataloader

train_ds, val_ds, test_ds = split_by_patient(samples, [0.8, 0.1, 0.1], seed=42)
```

`split_by_patient` (`pyhealth/datasets/splitter.py:142`) guarantees that no
patient's samples land in more than one split. **This is the default and you
should need a specific reason to deviate**, because the alternative is a model
that memorizes individual patients and a test score that means nothing.

Ratios must sum to exactly 1.0 — the function asserts it. Pass `seed` so the
split is reproducible; a different seed is a different split and a different
number.

Other splitters exist in the same module (`split_by_visit`, `split_by_sample`,
and conformal variants). `split_by_visit` and `split_by_sample` both leak
patients across splits. Use them only when the user asks for that explicitly and
understands the consequence.

---

## Dataloaders

```python
train_loader = get_dataloader(train_ds, batch_size=64, shuffle=True)
val_loader   = get_dataloader(val_ds,   batch_size=64, shuffle=False)
test_loader  = get_dataloader(test_ds,  batch_size=64, shuffle=False)
```

`get_dataloader` (`pyhealth/datasets/utils.py:331`) wires up the right collate
function and calls `dataset.set_shuffle(shuffle)` on the underlying streaming
dataset. Shuffle **train only**.

Do not build a raw `torch.utils.data.DataLoader` around a PyHealth
`SampleDataset`. It is a streaming dataset that manages its own ordering, and
`DataLoader(..., shuffle=True)` on it will error or silently misbehave. Always go
through `get_dataloader`.

---

## Train

```python
from pyhealth.trainer import Trainer

trainer = Trainer(
    model=model,
    device="cuda",              # omit to auto-pick cuda if available, else cpu
    output_path="runs/baseline",  # checkpoints + logs; defaults to ./output
    exp_name="rnn_seed42",        # defaults to a timestamp
    metrics=["roc_auc", "pr_auc", "f1"],   # None → the mode's defaults
)

trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=20,
    optimizer_class=torch.optim.Adam,
    optimizer_params={"lr": 1e-3},
    weight_decay=0.0,
    monitor="roc_auc",          # must be a key the metrics fn actually returns
    monitor_criterion="max",
    patience=5,                 # None → no early stopping
    load_best_model_at_last=True,
)
```

Signature: `pyhealth/trainer.py:113`.

**`monitor` must name a metric that is actually computed.** If you pass
`metrics=[...]` to `Trainer`, the monitor key has to be in that list; if you left
`metrics=None`, it has to be one of the mode defaults in
[evaluation.md](references/metrics-and-reporting.md). A typo here raises a `KeyError` mid-training,
after you have already paid for the epochs.

**Do not pass `test_dataloader` while you are still iterating.** `Trainer.train`
accepts one, and using it means you are watching your test score while making
choices — which is exactly how the final number stops being honest. Hold it back
until the very last run.

---

## Evaluate

```python
scores = trainer.evaluate(test_loader)   # dict of metric name → float
print(scores)
```

For raw predictions (to compute something the built-in metrics do not cover, or
to build a calibration plot):

```python
y_true, y_prob, loss = trainer.inference(test_loader)

from pyhealth.metrics import binary_metrics_fn
print(binary_metrics_fn(y_true, y_prob, metrics=["roc_auc", "pr_auc"]))
```

---

## The `__main__` guard

PyHealth uses multiprocessing for dataset caching and task processing. **All
dataset construction and training must run under `if __name__ == "__main__":`**,
and any task class must be defined at module level so it can be pickled.

```python
from pyhealth.tasks import BaseTask

class MyTask(BaseTask):        # ← module level, not inside main
    ...

if __name__ == "__main__":     # ← everything else in here
    dataset = ...
```

A task class defined inside a function or under the guard fails to pickle and the
workers die with an opaque error.

---

## Caching

```python
dataset = MIMIC4EHRDataset(root=..., tables=[...], cache_dir="~/.cache/pyhealth")
```

Always pass `cache_dir` explicitly and use the **same one across runs** — the
first run parses the raw files (slow), every later run loads the cache (fast).
The cache path gets a UUID appended, derived from root, table list, dataset name,
and the `dev` flag, so different configurations do not collide.

Two things the UUID does *not* cover, and which will therefore serve you stale
data:

- **Editing the dataset YAML.** Delete that cache directory before re-running.
- **Editing task logic while keeping `task_name` and both schemas unchanged.**
  The task cache key is `task_name` plus schema. Change the logic, change the
  `task_name` — append `_v2`, `_v3`. Otherwise `set_task` cheerfully returns the
  samples from your previous version.

Changing *hyperparameters* changes neither, so leave `task_name` alone during a
hyperparameter sweep — mutating it there forces a full needless rebuild per run.

---

## Reproducibility

Set every seed you can, and report which ones you used:

```python
import random, numpy as np, torch

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

Pass the same value to `split_by_patient(..., seed=seed)`. For a final reported
result, run at least three seeds (42, 43, 44 by convention) and report
mean ± std — a single seed's number on a clinical dataset is mostly noise.

---

## Speed

- `dev=True` first, always. It caps the dataset at ~1000 patients and turns a
  30-minute mistake into a 30-second one.
- Reduce `epochs` before reducing dataset size when compute is tight — a small
  dataset changes the answer; fewer epochs mostly changes the precision.
- `patience` makes most runs stop well before `epochs`.
- The first run on a new dataset pays the full parse cost. Warn the user about
  the wait rather than letting it look like a hang.
