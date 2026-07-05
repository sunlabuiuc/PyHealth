---
name: run-multimodal-training
description: Launch and post-process MIMIC-IV multimodal mortality training runs from /home/joshua86/PyHealth. Use when the user asks anything even loosely related to run / rerun / relaunch a CXR / notes / labs / ICD combo, check on or attach to a training tmux session, recompute or compute metrics from a finished run's predictions CSV, or pick a free GPU for a new experiment.
when-to-use: Use when the user asks anything even loosely related to run / train / queue / rerun / relaunch a CXR / notes / labs / ICD combo, check on or attach to a training tmux session, recompute or compute metrics from a finished run's predictions CSV, or pick a free GPU for a new experiment.
---

# run-multimodal-training

Operates the unified embedding pipeline at [examples/mortality_prediction/unified_embedding_e2e_mimic4.py](../../../examples/mortality_prediction/unified_embedding_e2e_mimic4.py).

## Source of truth

Read [`experiments.yaml`](./experiments.yaml) before answering any "run X" or "what params for X" question. The full flag dictionary is in [`flags.md`](./flags.md). A worked end-to-end example is in [`example_run.md`](./example_run.md). **If the user's request can't be resolved from a YAML entry, ask — don't invent params.**

### Override precedence

When building a launch command, resolve each parameter by walking these layers (later wins):

1. `experiments.yaml.defaults` (e.g., `epochs`, `seed`, `weight_decay`, dataset roots)
2. `experiments.yaml.experiments[].<entry>` (the entry holds all RAM-sensitive params: `batch_size`, `embedding_dim`, `hidden_dim`, `lr`, plus any model-structural keys like `heads` / `num_layers` / `mamba_state_size`). There is **no** per-model defaults layer — RAM footprint varies per (task, model) combo, so every entry sets its own values explicitly.
3. **User-provided value in the current request** — always wins. If the user says "run `cxr_mlp` with 30 epochs and batch size 4", use 30 and 4 even if the YAML says otherwise. Don't ask permission to override.

## Prerequisites

- Conda env: `pyhealth2`
- Project dir: `/home/joshua86/PyHealth`
- Default roots (override in `experiments.yaml.defaults`):
  - `--ehr-root /shared/rsaas/physionet.org/files/mimiciv/2.2`
  - `--note-root /shared/rsaas/physionet.org/files/mimic-note`
  - `--cxr-root /shared/rsaas/physionet.org/files/MIMIC-CXR`
  - `--cxr-variant sunlab`

## Workflow

Each step has a verify check. Loop on verify failures before moving on.

0. **Verify wandb is usable AND check for prior runs of this `exp_name`.**

   0a. Preconditions (skill must halt and ask the user if either fails):
   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh && conda activate pyhealth2
   python -c "import wandb"                              # must exit 0
   [ -n "$WANDB_API_KEY" ] || grep -q "machine api.wandb.ai" ~/.netrc
   ```
   → on failure, print exactly:
   ```
   pip install wandb
   wandb login   # API key from https://wandb.ai/authorize
   ```
   and **halt** — do not launch.

   0b. Prior-run lookup (`exp_name = f"{task}_{model}_seed{seed}"`):
   ```python
   import wandb
   api = wandb.Api()
   for r in list(api.runs("pyhealth-multimodal", filters={"display_name": "<exp_name>"}))[:5]:
       print(r.id, r.state, r.created_at,
             r.summary.get("val_pr_auc"), r.config.get("env/git_sha"))
   ```
   → verify: compare the resolved `hp/*` and `arch/*` keys against the most recent finished prior run.
   - Identical AND finished → ask the user (rerun anyway with new seed? or fetch metrics from the existing run?). Do not auto-launch.
   - Different → print key-by-key diff, ask.
   - Wandb API unreachable → warn, ask before launching without dedup.

1. **Identify the combo in `experiments.yaml`.**
   → verify: combo's `task` value is in the `tasks:` block AND the entry sets at minimum `batch_size`, `embedding_dim`, `hidden_dim`, `lr`. If missing, stop and ask.

2. **Pick a free GPU** (or use the one pinned to the combo).
   → verify: `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv` shows the chosen index is genuinely idle — `utilization.gpu` ≈ 0% AND `memory.used` near baseline (~800 MiB, i.e. no other job resident). High `memory.free` alone is NOT enough: a GPU mid-run can still report lots of free memory. If the combo's pinned GPU is busy, pick another idle index and override `--device`/`CUDA_VISIBLE_DEVICES` accordingly.

3. **Confirm the dataset cache exists.**
   → verify: `ls ~/.cache/pyhealth/ee6506fd-*/global_event_df.parquet/*.parquet | head` returns files.
   → if missing: warn the user that Dask will rebuild (~10 min). Don't launch multiple cold runs in parallel — they race on `<cache>/tmp/`.

4. **Assemble the bash command** from `experiments.yaml` (`defaults` + `tasks.<task>` + the experiment entry's keys) using the flag table in [`flags.md`](./flags.md).
   → verify: every flag in the command has a documented row in `flags.md`.

5. **Launch with tmux send-keys** (never embed the python command directly in `tmux new-session`'s command — the session dies on exit, and you lose the traceback).
   ```bash
   tmux new-session -d -s <name>
   tmux send-keys -t <name> "source ~/miniconda3/etc/profile.d/conda.sh && conda activate pyhealth2 && cd /home/joshua86/PyHealth && CUDA_VISIBLE_DEVICES=<gpu> python examples/mortality_prediction/unified_embedding_e2e_mimic4.py <flags> 2>&1 | tee logs/<name>.log" Enter
   ```
   → verify within 30 s: `tmux ls` shows the session AND `tail -n 5 logs/<name>.log` shows the imports loading without a traceback.

6. **Print** the exact command and configuration (combination, parameters, etc.) you used, AND the wandb URL captured from the log line `wandb: 🚀 View run … at: https://wandb.ai/…/runs/…` (appears within ~10 s of training start). This URL is the team-shareable handle for the run.

7. **Watch** with `tail -f logs/<name>.log` until training starts.
   → verify: a line like `Task sample count: N` with N > 0, followed by per-epoch metric lines.

8. **On completion**, read the predictions CSV and recompute metrics (see § Post-run metrics).
   → verify: PR-AUC printed AND CSV row count > 0.

## Dev mode rule

Default to **full data** (no `--dev`). Pass `--dev` **only** when the user explicitly says "dev", "smoke test", "quick test", or otherwise signals they want a 1000-patient subset. Dev mode creates a separate dataset cache UUID, so the first dev launch triggers a fresh (minute-scale) Dask build — it isn't free.

## Post-run metrics

The runner writes predictions to `<output_dir>/<exp_name>/predictions_<model>.csv` with columns `patient_id, y_true, y_prob, y_pred_threshold_0_5`. Recompute metrics without retraining:

```python
import pandas as pd
from pyhealth.metrics import binary_metrics_fn

df = pd.read_csv("output/unified/.../predictions_mlp.csv")
scores = binary_metrics_fn(
    df["y_true"].to_numpy(),
    df["y_prob"].to_numpy(),
    metrics=["pr_auc", "roc_auc", "f1", "accuracy"],
)
print(scores)
```

Available metrics from [pyhealth/metrics/binary.py](../../../pyhealth/metrics/binary.py): `pr_auc`, `roc_auc`, `accuracy`, `balanced_accuracy`, `f1`, `precision`, `recall`, `cohen_kappa`, `jaccard`, plus calibration metrics (`*_adapt` variants).

## Troubleshooting

- **`P2POutOfDiskError` / `OSError: No space left on device`** — cache disk is full. Default `~/.cache/pyhealth` has ~975 GB free; `/shared/eng/pyhealth` is at 100% and unusable. Check with `df -h <cache_dir>`.
- **`OverflowError: can't convert negative int to unsigned`** in `_filter_by_time_range_fast` — admission starts after the observation window closes. Guarded in all 8 task classes in [pyhealth/tasks/multimodal_mimic4.py](../../../pyhealth/tasks/multimodal_mimic4.py). If it recurs, re-add `if effective_end is not None and admission_time >= effective_end: continue` right after `admission_time = admission.timestamp` in the offending class.
- **`OSError: Directory not empty` / `FileNotFoundError`** during `clean_tmpdir` — Dask cleanup race. Fixed at [pyhealth/datasets/base_dataset.py:422-428](../../../pyhealth/datasets/base_dataset.py#L422-L428) with `shutil.rmtree(tmp_dir, ignore_errors=True)`. If it returns, that patch was reverted.
- **CUDA OOM** — lower `batch_size` directly on the experiment entry (and/or shrink `embedding_dim` / `hidden_dim`), or switch to a smaller model. Don't expect a shared per-model default to bail you out — each (task, model) cell sets its own RAM-sensitive values.
- **tmux session dies immediately on launch** — the Python process crashed before printing anything. Check `logs/<name>.log` for the traceback. Common cause: missing `--cxr-root` or `--note-root` for a task that requires them (cross-check against `experiments.yaml.tasks.<task>.roots`).

## Halting conditions (stop and ask — never guess)

The skill MUST stop and ask the user, not proceed silently, when any of:

- wandb is not installed OR not logged in on this machine (print the two-line setup from step 0a, halt).
- The resolved config matches a finished prior run exactly (ask: rerun anyway with a new seed, or fetch metrics from the existing run?).
- The resolved config differs from the most recent prior run with the same `exp_name` (print the key-by-key diff and ask).
- The `experiments.yaml` entry is missing any of `batch_size`, `embedding_dim`, `hidden_dim`, `lr`.
- The pinned GPU is in use AND no other GPU is idle (ask before evicting / waiting).
- The wandb API is unreachable (warn + ask: launch without dedup, or fix connectivity?).

## Behavioral guardrails

When acting on a launch request, follow Karpathy-style discipline:
- **Think first**: if any required field is missing from the YAML entry, ask the user. Don't fill in plausible numbers.
- **Surgical changes**: launching a run touches only the new tmux session and `logs/<name>.log`. Don't refactor the runner, reorganize the cache, or `pip install` anything. If something adjacent looks broken (e.g., a stale tmux session, a stale cache subdir), mention it; don't fix it unless asked.
- **Verify before reporting success**: every step above ends with a `verify:` line. Don't claim the launch worked because the tmux command returned 0 — confirm the log shows imports loading and training starting.

## Tooling roadmap

Not built yet. Document new ideas here instead of in scattered notes.

- **`scripts/launch_experiments.py`** — read `experiments.yaml`, parse `nvidia-smi` to auto-assign free GPUs, create tmux sessions with auto-named output dirs, validate roots / task names / cache before launching.
- **`scripts/monitor_runs.py`** — poll tmux sessions, parse logs for OOM / NaN / exceptions, Discord webhook on completion or failure, optional auto-requeue with smaller batch.
- **Planning artifact** — each new combo is an `experiments.yaml` entry. The entry's keys (task, model, gpu, batch_size, embedding_dim, hidden_dim, lr, plus any model-structural keys) are the full spec.

## Common ops

```bash
tmux ls
tmux attach -t <name>             # detach: Ctrl-b then d
tmux kill-session -t <name>
tail -f logs/<name>.log
nvidia-smi --query-gpu=index,memory.free --format=csv
df -h ~/.cache/pyhealth           # confirm cache disk has room
```
