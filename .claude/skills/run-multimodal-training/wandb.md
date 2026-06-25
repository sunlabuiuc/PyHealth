# wandb usage

All training runs log to wandb project `pyhealth-multimodal` via `wandb.init(config=...)` in the runner. Mandatory; not user-toggleable. The project name is hardcoded in [`unified_embedding_e2e_mimic4.py`](../../../examples/mortality_prediction/unified_embedding_e2e_mimic4.py) as `WANDB_PROJECT`.

## One-time setup per machine

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pyhealth2
pip install wandb
wandb login   # paste API key from https://wandb.ai/authorize
```

The skill's workflow step 0 verifies both before any launch and halts with this exact two-line message on failure.

## Config schema

Top-level identity keys (always set):

- `task`, `model`, `exp_name`, `modalities` (list), `seed`, `dev_mode`

Grouped keys (collapsible in wandb UI by their `/`-prefix):

| Prefix | Contents |
|---|---|
| `hp/*`     | Effective hyperparameters: `batch_size`, `lr`, `adam_eps`, `weight_decay`, `max_grad_norm`, `epochs`, `dropout`. "Effective" = after model-specific defaults are applied (e.g., `lr=1e-4` for `bottleneck_transformer`). |
| `arch/*`   | Model architecture: `embedding_dim`, `hidden_dim`, `heads`, `num_layers`, `rnn_type`, `rnn_layers`, `bidirectional`, `bottlenecks_n`, `fusion_startidx`, `mamba_state_size`, `mamba_conv_kernel`, `jamba_transformer_layers`, `jamba_mamba_layers`, `param_count`. |
| `fusion/*` | Multimodal / compression knobs: `vision_pool` (`mean` → 1 vec/image, `none` → all patch tokens), `observation_window_hours`. |
| `data/*`   | `cxr_variant`, `train_samples`, `val_samples`, `test_samples`, `cache_dir`. |
| `paths/*`  | `ehr_root`, `note_root`, `cxr_root`, `output_dir`. |
| `env/*`    | `gpu_name`, `gpu_count`, `cuda_visible_devices`, `torch_version`, `cuda_version`, `python_version`, `hostname`, `git_sha`, `git_dirty`. |

Tags (one-click filters in the UI): `[task, model, vision_pool, gpu_short_name]`. Dev runs additionally tagged `dev`.

## Pre-launch lookup (the skill runs this for you)

```python
import wandb
api = wandb.Api()
exp_name = "cxr_mlp_seed42"
for r in list(api.runs("pyhealth-multimodal", filters={"display_name": exp_name}))[:5]:
    print(r.id, r.state, r.created_at,
          r.summary.get("val_pr_auc"), r.config.get("env/git_sha"))
```

The skill compares the resolved `hp/*` and `arch/*` keys against the most recent finished prior run, then halts and asks if they match or differ — never auto-launches a duplicate.

## Pull a prior run's config back into experiments.yaml

```python
import wandb, yaml
run = wandb.Api().run("pyhealth-multimodal/<run_id>")
keep = {"task", "model", "seed",
        "hp/batch_size", "hp/lr",
        "arch/embedding_dim", "arch/hidden_dim",
        "fusion/vision_pool"}
print(yaml.dump({k.split("/")[-1]: v for k, v in run.config.items() if k in keep}))
```

Paste the result into the matching `experiments.yaml` entry. Also compare `env/git_sha` against current `HEAD` — if it differs, the rerun is not bit-exact even with identical config.

## Live view

- **Per-run URL**: printed by the runner on launch — line begins with `wandb: 🚀 View run`. Appears within ~10 s of training start. This is the team-shareable handle.
- **Team workspace**: `https://wandb.ai/<entity>/pyhealth-multimodal`. Use the URL printed in any of your own runs to identify `<entity>`.
- **What is live during training**: the full config dict (immediate at `wandb.init`), the run URL (immediate), and wandb's auto-collected GPU utilization / memory / power (every 30 s).
- **What appears at the end**: per-epoch metric curves (`train_loss`, `val_pr_auc`, `val_roc_auc`, etc.). These are replayed from `metrics_history` once `trainer.train()` returns. This is the intentional tradeoff for keeping `pyhealth/trainer.py` unmodified.

## Filtering dev runs

Click the `dev` tag in the UI to show only smoke-test runs, or `-dev` (Not) to hide them in the team workspace.

## Artifacts logged at end of run

- `predictions_<exp_name>` (type `predictions`) — the per-sample prediction CSV.
- `metrics_history_<exp_name>` (type `metrics-history`) — the per-epoch JSON, if `metrics_history.json` was written.
