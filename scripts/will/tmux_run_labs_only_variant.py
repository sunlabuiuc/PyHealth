project_dir = "/home/wp14/PyHealth"
seed = 12
conda_env = "pyhealth2"
ehr_root = "/shared/rsaas/physionet.org/files/mimiciv/2.2"
cache_dir = "/home/wp14/pyhealth_cache"
logs_dir = "/home/wp14/logs"
output_dir = "/home/wp14/output"
embedding_dim = 128
hidden_dim = 128
rnn_type = "GRU"
rnn_layers = 2
dropout = 0.1
epochs = 50
batch_size = 32
lr = 1e-3
weight_decay = 1e-5
patience = 5
num_workers = 4
dev = False
cuda_visible_devices = "0"
session_name = f"rnn_labs_s{seed}"

# ── Step 0: Clean logs and cache ────────────────────────────────────────────
print(f"""
### STEP 0 (optional): Clean logs and cache

rm -rf {logs_dir}/*
rm -rf {cache_dir}/*
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
log_tag = f"rnn_labs_s{seed}"

print(f"""
### STEP 2: Paste this into the tmux session

mkdir -p {log_dir} &&
eval "$(conda shell.bash hook)" &&
conda activate {conda_env} &&
cd {project_dir} &&
export PYTHONPATH={project_dir}:$PYTHONPATH &&
export CUDA_VISIBLE_DEVICES={cuda_visible_devices} &&
python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
    --ehr-root {ehr_root} \\
    --cache-dir {cache_dir} \\
    --task labs{' --dev' if dev else ''} \\
    --model rnn \\
    --embedding-dim {embedding_dim} \\
    --hidden-dim {hidden_dim} \\
    --rnn-type {rnn_type} \\
    --rnn-layers {rnn_layers} \\
    --dropout {dropout} \\
    --epochs {epochs} \\
    --batch-size {batch_size} \\
    --lr {lr} \\
    --weight-decay {weight_decay} \\
    --patience {patience} \\
    --num-workers {num_workers} \\
    --seed {seed} \\
    --output-dir {output_dir} \\
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
