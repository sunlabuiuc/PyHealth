# Hyperparameter tuning

Use this to improve model performance by tuning training settings.
The task class is **fixed** (best variant from task optimization).
The model architecture is **fixed** — do not switch models between configs.
All gains come from finding better hyperparameters.

---

## Step 0: Get a Minimal Working Setup First

**Every config you actually compare must run with `dev=False`** (the full
dataset). `dev=True` is a smoke test: it confirms the model trains without
crashing, nothing more.

```python
# Smoke test only — never compare configs on these numbers
base_dataset = MIMIC4Dataset(
    ehr_root=ehr_root,
    ehr_tables=sorted(required_tables),
    cache_dir=cache_dir,
    dev=True,
)
```

Never use `dev=True` results to compare configs — sample sizes differ too much
and the ranking will not transfer.

---

## Seeding the Search

Always seed from the baseline you already ran — its exact hyperparameters, its
monitor metric, and its measured seconds-per-epoch. Record them in
`strategies.md` when the baseline finishes so you have them here:

```
{"LR": 0.001, "BATCH_SIZE": 32, "HIDDEN_DIM": 128, "EMBEDDING_DIM": 64,
 "OPTIMIZER_TYPE": "Adam", "MONITOR_METRIC": "roc_auc", "seconds_per_epoch": 45.2}
```

Do not invent starting values. Also re-read your `task.py` for the **current**
task class name — task optimization may have changed it.

---

## Iterative Search Loop

Run configs in small batches. After each batch, use val metrics to focus the
next batch. Do not run all configs blindly.

```
Batch 1: vary LR only around baseline value
         ↓ pick best LR
Batch 2: fix best LR; vary BATCH_SIZE
         ↓ pick best batch size
Batch 3: fix best LR + batch; vary HIDDEN_DIM or EMBEDDING_DIM if budget allows
```

Use `PATIENCE = 3` for early stopping in all jobs — training halts automatically
when val metric doesn't improve for 3 consecutive epochs, so configs that converge
early don't waste time.

---

## Hyperparameter Axes

Each axis below gives a **min–max range** — pick any value within it that your
reasoning supports; you are not limited to round numbers.

| Axis | Baseline seed | Search range (min – max) |
|---|---|---|
| `LR` | from your baseline | 1e-6 – 1e-2 |
| `BATCH_SIZE` | from your baseline | 16 – 256 |
| `EMBEDDING_DIM` | from your baseline | 32 – 256 |
| `HIDDEN_DIM` | from your baseline | 32 – 256 |
| `OPTIMIZER_TYPE` | from your baseline | choice: `"Adam"` or `"AdamW"` |
| `EPOCHS` (max) | — | 10 – 30 (early stopping handles the rest) |

Do not tune `EMBEDDING_DIM` and `HIDDEN_DIM` simultaneously unless budget allows.
Do not stray more than one order of magnitude on LR.

**These axes are a starting point, not an exhaustive list.** Do not blindly
restrict yourself to the table above. If you have a concrete, well-reasoned
hypothesis, you may add other hyperparameters as tuning axes — e.g. dropout,
weight decay, learning-rate scheduler, gradient clipping, or number of RNN
layers. Justify any added axis with a clear rationale (ideally tied to an
observation from a prior batch), and still vary one axis at a time.

Example 5-config grid:
```
config_0: lr=1e-3, bs=32,  emb=128, optim=Adam,  epochs=20  (baseline)
config_1: lr=1e-4, bs=32,  emb=128, optim=Adam,  epochs=30  (lower LR, more budget)
config_2: lr=1e-3, bs=64,  emb=256, optim=Adam,  epochs=20  (larger model)
config_3: lr=5e-4, bs=16,  emb=128, optim=AdamW, epochs=15  (AdamW + small batch)
config_4: lr=1e-2, bs=32,  emb=64,  optim=Adam,  epochs=10  (high LR, short run)
```

---

## Time Budget Awareness

The baseline's measured `seconds_per_epoch` tells you how long one epoch takes
on the full dataset. Multiply by `EPOCHS` before launching to estimate
per-config cost, and tell the user the estimate before you spend their compute.

If budget is tight:
- Reduce `EPOCHS` during search (e.g. 10 instead of 30); use final training for full epochs.
- `PATIENCE = 3` means most configs stop well before `EPOCHS` anyway.

---

## Rules That Apply to Every Config

**Give each config its own output directory**
```python
run_dir = f"runs/cfg{i}"
trainer = Trainer(model=model, device=device, output_path=run_dir)
```
Checkpoints and logs land there; without it configs overwrite each other.

**Always use one shared `cache_dir` across configs**
```python
cache_dir = "~/.cache/pyhealth"   # ✅ built once, reused by every config
# ❌ Never point cache_dir inside the per-run directory — forces a full
#    dataset rebuild for every config
```

**Never mutate `task_name` for cache isolation**
Adding suffixes like `_cfg0` to `task_name` forces a full cache rebuild per run.
Hyperparameters do not change the samples, so every config should share one task
cache via the file lock in `base_dataset.set_task()`. (Task *logic* changes are
different — those do require a new `task_name`; see `tasks.md`.)

**Never swap the model architecture mid-search**
If the chosen model fails, stop and report the failure with the error. A clean
failure is better than silently switching to a different architecture, which
makes every earlier comparison meaningless.

**Processor compatibility check before running**
Verify the model supports all processor types in `task.input_schema`. If there is
a mismatch, switch to a compatible multimodal sibling (e.g., `AdaCare` →
`MultimodalAdaCare`) rather than changing the task schema. See
[models.md](../../choose-a-model/SKILL.md).

**Test set is never used**
Only `train_loader` and `val_loader`. No `test_dataloader` in any hparam job.

---

## Monitor Metrics by Task Type

| Task mode | `monitor=` | `monitor_criterion=` |
|---|---|---|
| Binary (mortality, readmission) | `"roc_auc"` | `"max"` |
| Multiclass (LOS, diagnosis) | `"f1_micro"` | `"max"` |
| Multilabel (drug recommendation) | `"pr_auc_samples"` | `"max"` |

Use the same monitor metric as your baseline — do not change it between configs.
Changing the metric mid-search means you are ranking configs on two different
scales.

---

## Selecting the Winner

Compare all completed configs on **val set only**. Pick the highest val metric.
If two configs are within 0.001, prefer the one with smaller LR (more stable).

Pass the best config directly to final evaluation (final training on full epoch budget
with test set evaluation for the first and only time).
