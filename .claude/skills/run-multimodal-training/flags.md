# Flag reference: `unified_embedding_e2e_mimic4.py`

> Defaults for every flag are recorded in [`experiments.yaml`](./experiments.yaml). This file is the line-by-line dictionary. Use it to interpret an unfamiliar flag in a log or assemble a one-off command. For routine launches, read the YAML.

Script: [examples/mortality_prediction/unified_embedding_e2e_mimic4.py](../../../examples/mortality_prediction/unified_embedding_e2e_mimic4.py)

## Reference launch command

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
    --heads 4 --num-layers 2 \
    --epochs 10 \
    --device cuda:0 \
    2>&1 | tee logs/cxr.log" Enter
```

## tmux wrapper

| Flag | Meaning |
|---|---|
| `new-session` | Create a new tmux session. |
| `-d` | Detached — don't attach your terminal to it. |
| `-s <name>` | Session name, e.g. `cxr`. Use this name with `attach`, `kill-session`, `send-keys`. |
| `send-keys -t <name> "…" Enter` | Type the string into session `<name>`, then press Enter. Use this pattern instead of embedding the command in `new-session` — the session stays alive even if the inner command crashes, so the traceback survives. |

## Shell prelude

| Piece | Why |
|---|---|
| `source ~/miniconda3/etc/profile.d/conda.sh` | Make the `conda` command available in a fresh shell. |
| `conda activate pyhealth2` | Switch into the env (polars, torch, pyhealth, etc.). |
| `cd /home/joshua86/PyHealth` | So `examples/...` paths and `pyhealth` imports resolve. |
| `CUDA_VISIBLE_DEVICES=<N>` | Mask which physical GPUs the process sees. The process renames the visible device to `cuda:0`. With `=0`, physical GPU 0 is `cuda:0`; with `=3`, physical GPU 3 is `cuda:0`. **This is why every command uses `--device cuda:0` regardless of which physical GPU it lands on.** |

## Python script flags

### Data paths

| Flag | Effect |
|---|---|
| `--ehr-root <path>` | MIMIC-IV EHR CSV root. **Required.** |
| `--cxr-root <path>` | MIMIC-CXR metadata/images root. Required when the task is a CXR variant. |
| `--note-root <path>` | Clinical notes root. Required only for note-bearing tasks (`clinical_notes_icd_labs`). |
| `--cxr-variant {default, sunlab}` | Defaults to `sunlab` — uses `metadata-pyhealth-sunlab.csv`. |
| `--cache-dir <path>` | Defaults to `~/.cache/pyhealth/`. The cache subdir UUID is derived from the dataset config, so all four CXR-variant tasks share it. Do not point at `/shared/eng/pyhealth/` — that disk is full. |
| `--output-dir <path>` | Where per-run output goes. Defaults to `./output/unified_e2e_cxr`. |

### Task and model

| Flag | Effect |
|---|---|
| `--task {icd_labs, clinical_notes_icd_labs, cxr, icd_cxr, labs_cxr, icd_labs_cxr}` | Which task class to instantiate. Determines the input modalities. |
| `--model {mlp, rnn, transformer, bottleneck_transformer, ehrmamba, jambaehr}` | Backbone architecture. MLP pools over time; transformers and Mamba variants are temporal. |

### Architecture sizing

| Flag | Effect |
|---|---|
| `--embedding-dim <int>` | Per-modality embedding width produced by `UnifiedMultimodalEmbeddingModel`. Default 64. |
| `--hidden-dim <int>` | Backbone hidden dim (e.g., MLP linear width). Default 64. |
| `--heads <int>` | Attention heads. Used by transformer / bottleneck / jamba. Parsed but ignored by MLP and RNN. Default 4. |
| `--num-layers <int>` | Layer count for transformer / bottleneck / ehrmamba / jamba. Ignored by MLP. Default 2. |
| `--dropout <float>` | Default 0.1. |
| `--vision-pool {mean, none}` | CXR patch-token handling in `UnifiedMultimodalEmbeddingModel`. `mean` (default) collapses each image's patch tokens to one vector via global mean pool (1 token/image — the original behavior). `none` keeps all `num_patches` tokens (e.g. 196 for 224px/16px patches), flattening them into the temporal sequence so a downstream temporal model sees spatial detail. `none` greatly inflates sequence length — watch for OOM and lower `--batch-size` if needed. Only affects IMAGE/CXR tasks. |

### Training

| Flag | Effect |
|---|---|
| `--epochs <int>` | Training epochs. Default 1 (almost certainly not what you want — override in `experiments.yaml`). |
| `--batch-size <int>` | Mini-batch size. Default 4. |
| `--device <str>` | Torch device, e.g. `cuda:0` or `cpu`. Combined with `CUDA_VISIBLE_DEVICES` to pin a physical GPU. Default: auto. |
| `--lr <float>` | Defaults to `1e-3` for most models, `1e-4` for `bottleneck_transformer`. |
| `--adam-eps <float>` | Override Adam ε. Only meaningful for `bottleneck_transformer` (set to `1e-6` to stabilize). |
| `--weight-decay <float>` | Default 0.0. |
| `--max-grad-norm <float>` | Gradient clipping. Defaults to none; auto-set to 0.5 for `bottleneck_transformer`. |
| `--seed <int>` | Default 42. |
| `--num-workers <int>` | DataLoader workers (also passed to `set_task`). Default 1. |
| `--observation-window-hours <int>` | Default 24 — features capped at first 24 h after admission. |
| `--dev` | Limit the dataset to 1000 patients for smoke-testing. **Only pass when the user explicitly asks for dev / smoke / quick-test mode.** Creates a separate cache UUID (separate parquet build). |

### Model-specific flags

These do anything only for the matching `--model`:

| Flag | Applies to |
|---|---|
| `--rnn-type {GRU, LSTM}`, `--rnn-layers <int>`, `--bidirectional` | `rnn` |
| `--bottlenecks-n <int>`, `--fusion-startidx <int>` | `bottleneck_transformer` |
| `--mamba-state-size <int>`, `--mamba-conv-kernel <int>` | `ehrmamba`, `jambaehr` |
| `--jamba-transformer-layers <int>`, `--jamba-mamba-layers <int>` | `jambaehr` |

## Output redirection

| Piece | Why |
|---|---|
| `2>&1` | Merge stderr into stdout so tracebacks and progress bars both end up in the same stream. |
| `\| tee logs/<name>.log` | Write to `logs/<name>.log` **and** keep displaying in the tmux pane. `tail -f logs/<name>.log` works from any shell, no tmux attach required. |

## Experiment tracking (mandatory, no flag)

Every run calls `wandb.init(project="pyhealth-multimodal", ...)` unconditionally. There is no `--wandb` flag and no opt-out. Requires `pip install wandb` and `wandb login` once per machine — see [`wandb.md`](./wandb.md) for config schema and lookup snippets. `--dev` runs go to the same project, additionally tagged `dev`.

## Common ops

```bash
tmux ls                          # list sessions
tmux attach -t <name>            # attach (detach: Ctrl-b then d)
tmux kill-session -t <name>
tail -f logs/<name>.log          # watch the log without attaching
nvidia-smi --query-gpu=index,memory.free --format=csv
```
