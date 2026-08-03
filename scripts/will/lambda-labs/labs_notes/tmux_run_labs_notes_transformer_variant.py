project_dir = "PyHealth"
seed = 12
conda_env = "pyhealth2"
ehr_root = "/home/ubuntu/mimiciv-data/ehr"
note_root = "/home/ubuntu/mimiciv-data"
cache_dir = "/home/ubuntu/mimiciv-data/pyhealth_cache_labs_notes"
logs_dir = "/home/ubuntu/logs"
output_dir = "/home/ubuntu/output"
embedding_dim = 128
heads = 4
num_layers = 2
dropout = 0.1
use_amp = True
amp_dtype = "bf16"
epochs = 50
batch_size = 32
lr = 1e-3
weight_decay = 1e-5
patience = 5
pos_weight = 1
num_workers = 4
freeze_encoder = True
include_vitals = False
dev = False
use_old_cache = False
use_wandb = True
wandb_project = "pyhealth-multimodal-labs-notes"
wandb_run_name = None  # defaults to "{model}_seed{seed}" if unset
cuda_visible_devices = "0"
session_name = f"transformer_labs_notes_s{seed}"

# ── Step 0a: Check which CUDA GPU is available ───────────────────────────────
print(f"""
### STEP 0a: Check which CUDA GPU is available (pick an idle index for CUDA_VISIBLE_DEVICES)

nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv
""")

# ── Step 0b: Clean logs and cache ────────────────────────────────────────────
print(f"""
### STEP 0b (optional): Clean logs and cache

rm -rf {logs_dir}/*
{'# cache preserved (use_old_cache=True)' if use_old_cache else f'rm -rf {cache_dir}/*'}
rm -rf {output_dir}/*
""")

# ── Step 1: Start a tmux session ────────────────────────────────────────────
print(f"""
### STEP 1: Start a tmux session (attached)

tmux new-session -s {session_name}
""")

# ── Step 2: Paste this into the session to run training ─────────────────────
print("\n" + "=" * 60 + "\n")

log_dir = logs_dir
log_tag = f"transformer_labs_notes_s{seed}"

flags = [
    f"--ehr-root {ehr_root}",
    f"--note-root {note_root}",
    f"--cache-dir {cache_dir}",
    f"--task notes_labs{' --dev' if dev else ''}",
    "--model transformer",
    f"--embedding-dim {embedding_dim}",
    f"--heads {heads}",
    f"--num-layers {num_layers}",
    f"--dropout {dropout}",
]
if use_amp:
    flags.append("--use-amp")
    flags.append(f"--amp-dtype {amp_dtype}")
flags += [
    f"--epochs {epochs}",
    f"--batch-size {batch_size}",
    f"--lr {lr}",
    f"--weight-decay {weight_decay}",
    f"--patience {patience}",
    f"--pos-weight {pos_weight}",
    f"--num-workers {num_workers}",
]
if freeze_encoder:
    flags.append("--freeze-encoder")
if include_vitals:
    flags.append("--include-vitals")
if use_wandb:
    flags.append("--wandb")
    flags.append(f"--wandb-project {wandb_project}")
    if wandb_run_name:
        flags.append(f"--wandb-run-name {wandb_run_name}")
flags += [
    f"--seed {seed}",
    f"--output-dir {output_dir}",
]
flags_block = " \\\n    ".join(flags)

print(f"""
### STEP 2: Paste this into the tmux session

mkdir -p {log_dir} &&
eval "$(conda shell.bash hook)" &&
conda activate {conda_env} &&
cd {project_dir} &&
export PYTHONPATH={project_dir}:$PYTHONPATH &&
export CUDA_VISIBLE_DEVICES={cuda_visible_devices} &&
python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
    {flags_block} \\
    > >(tee {log_dir}/{log_tag}.out) \\
    2> >(tee {log_dir}/{log_tag}.err >&2)
""")

# ── Step 3: Detach / reattach / monitor ──────────────────────────────────────
print(f"""
### STEP 3: Detach without killing (Ctrl+b d), then reattach later with

tmux attach -t {session_name}

### To check on it later without attaching:

tail -f {log_dir}/{log_tag}.out

### To kill the session when done:

tmux kill-session -t {session_name}
""")

# ── Step 4: Confirm which note sections were used for filtering ─────────────
print(f"""
### STEP 4: Confirm which discharge-note AND radiology-note sections
### NotesLabsMIMIC4 filtered to (e.g. discharge: "chief complaint";
### radiology: "indication", "impression"). Logged once at INFO level by
### pyhealth.tasks.multimodal_mimic4 when the task is constructed.

grep "filtering discharge notes to sections" {log_dir}/{log_tag}.out
""")
