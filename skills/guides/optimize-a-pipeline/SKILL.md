---
name: pyhealth-optimize-a-pipeline
description: Improve a working PyHealth baseline — task engineering (feature ablation, ICD truncation, CCS mapping, history windows, normalization) and hyperparameter tuning, with a significance rule that stops noise being banked as progress. Use only after a baseline trains end to end, and only when the user asks to optimize.
---

# Optimize a pipeline

Use this to improve model performance by changing *what you feed the model*.
Model architecture and hyperparameters stay **fixed**. All gains come from better
feature representation and temporal structure.

---

## Exploration Strategy

Run variants in batches. After each batch, use val metrics to decide what to try next.
Do not design all variants upfront and run them blindly.

```
Batch 1: baseline + 1 promising variant
         ↓ inspect val metrics
Batch 2: build on what worked; drop dead ends
Batch 3: combine best axes if budget allows
```

**Always include a `baseline` variant** (original `task.py` logic, unchanged) as a
sanity check. If a variant underperforms the baseline, something broke — investigate before
continuing.

---

## Log every win to `strategies.md`

The moment a variant beats the current baseline by a statistically meaningful
margin (mean improvement > 1 std across ≥2 seeds), append a note to
`strategies.md` in the workdir — **before moving to the next batch**:

- which axis/variant changed and the exact change made
- val metric mean ± std, and the delta vs. the previous baseline
- adopt that variant as the new baseline for subsequent batches

`strategies.md` is the durable experiment log. Message history is windowed and
old tool outputs drop off, so an optimization that worked is lost unless you
record it. Do not rely on memory.

---

## Exploration Axes

Work through these independently. Combine only after individual axes show gains.
For code snippets for each axis, read [tasks.md](../define-a-task/SKILL.md). As an example, see below.

| Variant ID | Axis | When to try |
|------------|------|-------------|
| `baseline` | Original task.py logic unchanged | Always — required sanity check |
| `simplified` | Subclass the canonical PyHealth task (no `__call__` override) | When a matching class exists in `pyhealth.tasks`; check `pyhealth/tasks/__init__.py` first |
| `feature_drop` | Remove one or more features from `input_schema` and `__call__` | When the task's `input_schema` has features that may be noisy or redundant |
| `icd_trunc` | Drop ICD decimal suffix (`"428.0"→"428"`) or ATC to 3 chars | Always worth trying — reduces vocab, groups related codes |
| `ccs_norm` | Map ICD→CCS via CrossMap (~15k→~285 groups) | When `icd_trunc` helps; larger reduction, requires download |
| `history_window_N` | Cap cumulative visit history to last N visits (2, 3, 5) | Only for `nested_sequence` tasks |
| `z_norm` | Z-score normalize numeric tensor features on train split | Only when `input_schema` has `"tensor"` features |

Don't hesitate to propose your own variant ID + axis if you have a promising idea not covered here. Just be sure to document it clearly in `strategies.md` if it works.

---

## Rules for Every Variant

- **Keep the class name stable** — anything that does `from task import X` must still work after you overwrite `task.py`
- **`task_name` must be unique per variant** — it is the `set_task` cache key; reusing it silently returns stale samples from the previous variant
- **Always set `cache_dir` to one shared path** — a per-run cache dir forces a full dataset rebuild every time
- **No test set** — `val_loader` only, in every variant. The test split stays sealed until you have picked a winner
- **`dev=False`** — compare variants on the full dataset; `dev=True` is for smoke tests only, and its rankings do not transfer

---

## Full Experiment Template

Every variant is a self-contained script: task class at module level, full
training loop inside `if __name__ == '__main__':`.

**Before filling in:**
- Copy the model class + all hyperparameters (LR, BATCH_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OPTIMIZER_TYPE, MONITOR_METRIC) exactly from your baseline — only the task changes in this stage
- Reuse the same task class name; only `task_name` and `__call__` change per variant
- Fill in `ehr_root`, `cache_dir`, `dev_mode`, and `required_tables` for your own setup

```python
import json, os, sys
import torch

RUN_DIR = "runs/variant_id"   # ← unique per variant; checkpoints + logs land here
os.makedirs(RUN_DIR, exist_ok=True)

# ── Task class at MODULE LEVEL (required for multiprocessing pickling) ────────
sys.path.insert(0, os.getcwd())
from pyhealth.tasks import BaseTask
from typing import Dict, List, Any

class OriginalTaskClassName(BaseTask):              # ← SAME name as in task.py
    task_name: str = "OriginalTaskName_variant_id"  # ← UNIQUE per variant

    input_schema: Dict[str, str] = {
        # ← same as task.py input_schema, or subset if dropping features
        "conditions": "sequence",
        "procedures": "sequence",
    }
    output_schema: Dict[str, str] = {"label": "binary"}

    def __call__(self, patient) -> List[Dict[str, Any]]:
        samples = []
        visits = patient.visits
        if len(visits) <= 1:
            return []
        for i in range(len(visits) - 1):
            visit = visits[i]
            next_visit = visits[i + 1]
            # ← variant-specific extraction here
            # e.g., ICD truncation, CrossMap normalization, feature ablation
            # For code patterns per axis, see tasks.md
            conditions = [e.code for e in visit.get_events("diagnoses_icd")]
            conditions = conditions if conditions else ["<missing>"]
            label = int(getattr(next_visit, "hospital_expire_flag", 0) or 0)
            samples.append({"conditions": conditions, "label": label})
        return samples


if __name__ == '__main__':
    # ── Hyperparameters (FIXED — copy exact values from the baseline) ────────
    LR             = 1e-3
    BATCH_SIZE     = 32
    EMBEDDING_DIM  = 128
    HIDDEN_DIM     = 128
    OPTIMIZER_TYPE = "Adam"
    EPOCHS         = 20
    PATIENCE       = 5
    MONITOR_METRIC = "roc_auc"   # ← copy from your baseline run
    # RUN_DIR already defined at module level

    # ── Config (fill in directly for the task at hand) ────────────────────────
    ehr_root  = "<DATA_ROOT>"                       # ← where your raw data lives
    cache_dir = "~/.cache/pyhealth"                 # ✅ one shared cache for all variants
    dev_mode  = False                               # True only for quick dev checks

    required_tables = ["admissions", "diagnoses_icd"]  # ← the tables your task actually uses

    # ── Dataset + Task ────────────────────────────────────────────────────────
    from pyhealth.datasets import MIMIC4Dataset  # ← swap for correct dataset class
    base_dataset = MIMIC4Dataset(
        ehr_root=ehr_root,
        ehr_tables=sorted(required_tables),
        cache_dir=cache_dir,
        dev=dev_mode,
    )
    num_workers = min(8, max(1, (os.cpu_count() or 1) - 1))
    sample_dataset = base_dataset.set_task(OriginalTaskClassName(), num_workers=num_workers)

    # ── Splits ────────────────────────────────────────────────────────────────
    from pyhealth.datasets import split_by_patient, get_dataloader
    train_ds, val_ds, _ = split_by_patient(sample_dataset, [0.8, 0.1, 0.1])
    train_loader = get_dataloader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = get_dataloader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    # ⚠️ NO test_loader — never use test set during task exploration

    # ── Model (copied verbatim from the baseline) ────────────────────────────
    from pyhealth.models import RNN              # ← same model as the baseline
    model = RNN(
        dataset=sample_dataset,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from pyhealth.trainer import Trainer
    import torch.optim as optim

    optimizer_class = optim.AdamW if OPTIMIZER_TYPE == "AdamW" else optim.Adam
    weight_decay    = 1e-4 if OPTIMIZER_TYPE == "AdamW" else 0.0

    trainer = Trainer(model=model, device=device, output_path=RUN_DIR)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        # ⚠️ NO test_dataloader
        epochs=EPOCHS,
        patience=PATIENCE,
        optimizer_class=optimizer_class,
        optimizer_params={"lr": LR},
        weight_decay=weight_decay,
        monitor=MONITOR_METRIC,
        monitor_criterion="max",
    )

    # ── Evaluate + emit results ───────────────────────────────────────────────
    val_results = trainer.evaluate(val_loader)
    results = {
        "val_metrics": {
            k: float(v) if hasattr(v, "item") else v
            for k, v in val_results.items()
        }
    }
    print(f"RESULTS: {json.dumps(results)}")
```

---

## Selecting the Winner

Compare all completed variants on **val set only** using the same `MONITOR_METRIC`
as the baseline. Pick the highest val metric. In ties, prefer fewer features.
Keep failed variants in the log with their error — partial
results are useful context for subsequent rounds.
