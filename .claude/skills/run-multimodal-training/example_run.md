# Worked example: `cxr_mlp`

End-to-end trace of one combo — how the YAML entry becomes a tmux command, what the log looks like at each stage, and how to recompute metrics afterward.

## 1. Goal

Train and evaluate the `cxr_mlp` combo: CXR-only mortality classifier with an MLP head, pinned to GPU 0.

## 2. Lookup

The skill resolves `cxr_mlp` by merging two layers from `experiments.yaml` (entry wins on conflicts):

```yaml
# experiments.yaml entry — holds all RAM-sensitive params explicitly:
- name: cxr_mlp
  task: cxr
  model: mlp
  gpu: 0
  batch_size: 16
  embedding_dim: 128
  hidden_dim: 128
  lr: 1.0e-3

# tasks.cxr:
class: CXRMIMIC4
roots: [ehr, cxr]

# defaults (relevant subset):
ehr_root: /shared/rsaas/physionet.org/files/mimiciv/2.2
cxr_root: /shared/rsaas/physionet.org/files/MIMIC-CXR
cxr_variant: sunlab
epochs: 10
seed: 42
```

## 3. Pre-flight

```bash
# Is GPU 0 free?
nvidia-smi --query-gpu=index,memory.free --format=csv
# index, memory.free [MiB]
# 0, 79200 MiB         ← good

# Is the dataset cache warm?
ls ~/.cache/pyhealth/ee6506fd-*/global_event_df.parquet/*.parquet | head
# part.0.parquet
# part.1.parquet
# ... ← good. If empty, expect a ~10-minute Dask rebuild on first run.
```

## 4. Assembled command

Every flag value is traceable to a YAML source:

| Flag | Value | From |
|---|---|---|
| `CUDA_VISIBLE_DEVICES=` | `0` | `experiments.cxr_mlp.gpu` |
| `--ehr-root` | `/shared/rsaas/physionet.org/files/mimiciv/2.2` | `defaults.ehr_root` |
| `--cxr-root` | `/shared/rsaas/physionet.org/files/MIMIC-CXR` | `defaults.cxr_root` (required because `tasks.cxr.roots` includes `cxr`) |
| `--task` | `cxr` | `experiments.cxr_mlp.task` |
| `--model` | `mlp` | `experiments.cxr_mlp.model` |
| `--batch-size` | `16` | `experiments.cxr_mlp.batch_size` |
| `--embedding-dim` | `128` | `experiments.cxr_mlp.embedding_dim` |
| `--hidden-dim` | `128` | `experiments.cxr_mlp.hidden_dim` |
| `--lr` | `1.0e-3` | `experiments.cxr_mlp.lr` |
| `--epochs` | `10` | `defaults.epochs` |
| `--device` | `cuda:0` | always (`CUDA_VISIBLE_DEVICES` remaps physical GPU 0 → `cuda:0`) |

`--note-root` is omitted because `tasks.cxr.roots` doesn't include `note`. `--heads` and `--num-layers` are omitted because MLP ignores them.

```bash
tmux new-session -d -s cxr
tmux send-keys -t cxr "source ~/miniconda3/etc/profile.d/conda.sh && \
  conda activate pyhealth2 && \
  cd /home/joshua86/PyHealth && \
  CUDA_VISIBLE_DEVICES=0 python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \
    --ehr-root /shared/rsaas/physionet.org/files/mimiciv/2.2 \
    --cxr-root /shared/rsaas/physionet.org/files/MIMIC-CXR \
    --task cxr --model mlp \
    --batch-size 16 \
    --embedding-dim 128 --hidden-dim 128 \
    --epochs 10 \
    --device cuda:0 \
    2>&1 | tee logs/cxr.log" Enter
```

## 5. Annotated log

Approximate lines you should see in `logs/cxr.log`. Use these as verify checkpoints.

```
Memory usage Starting MIMIC4Dataset init: 943.9 MB                   ← imports loaded, dataset constructing
Initializing mimic4 dataset from /shared/.../mimiciv/2.2|None|...    ← roots accepted
Using provided cache_dir: /home/joshua86/.cache/pyhealth/ee6506fd-... ← cache subdir resolved
Initializing MIMIC4 CXR variant 'sunlab' with tables: [...]          ← CXR tables loaded
Setting task CXRMIMIC4 for mimic4 base dataset...                    ← task set_task() called
Task sample count: 273512                                             ← ✓ dataset built / loaded; > 0 means OK
Split sizes: train=218809, val=27351, test=27352                     ← splits derived
wandb: 🚀 View run cxr_mlp_seed42 at: https://wandb.ai/.../runs/xyz   ← team-shareable handle; ~10s after training start
Epoch 1: train_loss=0.42 pr_auc=0.58 roc_auc=0.71                    ← training started; metrics moving
...
Epoch 10: train_loss=0.31 pr_auc=0.69 roc_auc=0.81                   ← run finishing
Saved predictions to: output/.../predictions_mlp.csv                  ← CSV written; ready for § 6
```

If the run dies mid-stream, check § Troubleshooting in [`SKILL.md`](./SKILL.md).

## 6. Post-run metrics

```python
import pandas as pd
from pyhealth.metrics import binary_metrics_fn

df = pd.read_csv("output/unified/mlp_seed42/predictions_mlp.csv")
assert len(df) > 0
scores = binary_metrics_fn(
    df["y_true"].to_numpy(),
    df["y_prob"].to_numpy(),
    metrics=["pr_auc", "roc_auc", "f1", "accuracy"],
)
print(scores)
# {'pr_auc': 0.69, 'roc_auc': 0.81, 'f1': 0.42, 'accuracy': 0.93}
```

## 7. Discovery: what's been tried before?

Before launching, the skill queries wandb for prior runs of the same `exp_name` and surfaces the diff (or halts on an exact match). To do the same lookup by hand:

```python
import wandb
api = wandb.Api()
for r in list(api.runs("pyhealth-multimodal", filters={"display_name": "cxr_mlp_seed42"}))[:5]:
    print(r.id, r.state, r.created_at,
          r.summary.get("val_pr_auc"), r.config.get("env/git_sha"))
```

To filter by modality or GPU, use tags:

```python
api.runs("pyhealth-multimodal", filters={"tags": {"$in": ["A100-SXM4-80GB"]}})
```

Full config schema and a snippet that pulls a prior run's config back into `experiments.yaml` live in [`wandb.md`](./wandb.md).

## 8. Common failure modes

Pointers back to [`SKILL.md`](./SKILL.md) § Troubleshooting:

- Hangs at "Caching event dataframe to..." with disk-full traceback → cache directory full
- `OverflowError` in `_filter_by_time_range_fast` → observation-window guard regression
- `FileNotFoundError` / `Directory not empty` during cleanup → Dask race; check the `ignore_errors=True` patch
- CUDA OOM on epoch 1 → lower `batch_size` (and/or `embedding_dim` / `hidden_dim`) on the `cxr_mlp` entry directly, or pick a smaller model
- tmux dies on launch → see `logs/cxr.log` for the traceback; usually a missing required root
