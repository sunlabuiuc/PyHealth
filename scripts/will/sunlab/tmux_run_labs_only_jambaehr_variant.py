from datetime import datetime

date_str = datetime.now().strftime("%Y%m%d")

project_dir = "/home/wp14/PyHealth"
seed = 12
conda_env = "pyhealth2"
ehr_root = "/shared/rsaas/physionet.org/files/mimiciv/2.2"
note_root = "/shared/rsaas/physionet.org/files/mimic-note"
logs_dir = "/home/wp14/logs"
output_dir = "/home/wp14/output"
task = "labs"
cache_dir = f"/home/wp14/pyhealth_cache_{task}"
embedding_dim = 64
heads = 4
jamba_transformer_layers = 2
jamba_mamba_layers = 6
mamba_state_size = 16
mamba_conv_kernel = 4
dropout = 0.1
epochs = 50
batch_size = 32
lr = 1e-3
weight_decay = 1e-5
patience = 5
num_workers = 4
pos_weight = 1.0
dev = False
cuda_visible_devices = "4"
use_old_cache = True
session_name = f"jambaehr_{task}_s{seed}"
wandb_project = "pyhealth-multimodal-labs-only"

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
log_tag = f"jambaehr_{task}_s{seed}_{date_str}"

print(f"""
### STEP 2: Paste this into the tmux session

mkdir -p {log_dir} &&
eval "$(conda shell.bash hook)" &&
conda activate {conda_env} &&
cd {project_dir} &&
export PYTHONPATH={project_dir}:$PYTHONPATH &&
export CUDA_VISIBLE_DEVICES={cuda_visible_devices} &&
export WANDB_PROJECT={wandb_project} &&
export WANDB_NAME={log_tag} &&
{{ echo "[params] seed={seed} task={task} cache_dir={cache_dir} embedding_dim={embedding_dim} heads={heads} jamba_transformer_layers={jamba_transformer_layers} jamba_mamba_layers={jamba_mamba_layers} mamba_state_size={mamba_state_size} mamba_conv_kernel={mamba_conv_kernel} dropout={dropout} epochs={epochs} batch_size={batch_size} lr={lr} weight_decay={weight_decay} patience={patience} num_workers={num_workers} pos_weight={pos_weight} dev={dev}" | tee {log_dir}/{log_tag}.out | tee -a {log_dir}/{log_tag}.err > /dev/null; }} &&
{{ echo "[start] $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a {log_dir}/{log_tag}.out | tee -a {log_dir}/{log_tag}.err > /dev/null; }} &&
python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
    --ehr-root {ehr_root} \\
    --cache-dir {cache_dir} \\
    --task {task}{' --dev' if dev else ''} \\
    --model jambaehr \\
    --embedding-dim {embedding_dim} \\
    --heads {heads} \\
    --jamba-transformer-layers {jamba_transformer_layers} \\
    --jamba-mamba-layers {jamba_mamba_layers} \\
    --mamba-state-size {mamba_state_size} \\
    --mamba-conv-kernel {mamba_conv_kernel} \\
    --dropout {dropout} \\
    --epochs {epochs} \\
    --batch-size {batch_size} \\
    --lr {lr} \\
    --weight-decay {weight_decay} \\
    --patience {patience} \\
    --num-workers {num_workers} \\
    --pos-weight {pos_weight} \\
    --seed {seed} \\
    --output-dir {output_dir} \\
    > >(tee -a {log_dir}/{log_tag}.out) \\
    2> >(tee -a {log_dir}/{log_tag}.err >&2);
{{ echo "[end] $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a {log_dir}/{log_tag}.out | tee -a {log_dir}/{log_tag}.err > /dev/null; }}
""")

# ── Step 3: Detach / reattach / monitor ──────────────────────────────────────
print(f"""
### STEP 3: Detach without killing (Ctrl+b d), then reattach later with

tmux attach -t {session_name}

### To check on it later without attaching:

tail -f {log_dir}/{log_tag}.out

### wandb: run should appear at (look for "wandb: 🚀 View run" in the log)

https://wandb.ai/<entity>/{wandb_project}

### To kill the session when done:

tmux kill-session -t {session_name}
""")

# ── Step 4: Compute token stats (optional) ───────────────────────────────────
token_stats_tag = f"token_stats_notes_labs_s{seed}_{date_str}"

print(f"""
### STEP 4 (optional): Compute token stats (run in a separate shell, no GPU needed)

mkdir -p {log_dir} &&
eval "$(conda shell.bash hook)" &&
conda activate {conda_env} &&
cd {project_dir} &&
python scripts/compute_token_stats.py \\
    --ehr-root {ehr_root} \\
    --note-root {note_root} \\
    --cache-dir {cache_dir} \\
    --task notes_labs{' --dev' if dev else ''} \\
    > >(tee {log_dir}/{token_stats_tag}.out) \\
    2> >(tee {log_dir}/{token_stats_tag}.err >&2)
""")
