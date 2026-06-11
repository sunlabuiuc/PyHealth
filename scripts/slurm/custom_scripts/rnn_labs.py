project_dir = "/u/wp14/PyHealth"
seed = 12
account = "jimeng-ic"
partition = "IllinoisComputes-GPU"
time = "06:00:00"
mem = "32G"
gres = "gpu:A100:1"
conda_env = "pyhealth2"
ehr_root = "/projects/illinois/eng/cs/jimeng/physionet.org/files/mimiciv/2.2"
cache_dir = "/u/wp14/pyhealth_cache"
output_dir = "output/rnn_labs"
embedding_dim = 128
hidden_dim = 128
rnn_type = "GRU"
rnn_layers = 2
dropout = 0.1
epochs = 20
batch_size = 32
lr = 1e-3
weight_decay = 1e-5
patience = 5
num_workers = 4
dev = True  # True → srun (interactive, 1hr), False → sbatch (queued, full run)

# ── Step 0: Clean logs (optional) ────────────────────────────────────────────
print(f"""
### STEP 0 (optional): Clean logs

rm -rf logs/dev/rnn_labs_s{seed}.out logs/dev/rnn_labs_s{seed}.err
rm -rf logs/slurm/rnn_labs_s{seed}_*.out logs/slurm/rnn_labs_s{seed}_*.err
""")

# ── Step 1: Reserve resources ─────────────────────────────────────────────────
if dev:
    print(f"""
### STEP 1: Reserve an interactive node

srun \\
    --account={account} \\
    --partition={partition} \\
    --nodes=1 --ntasks=1 --cpus-per-task={num_workers} \\
    --mem={mem} --gres={gres} --time={time} \\
    --pty bash
""")
else:
    print(f"""
### STEP 1: Submit batch job (includes step 2 automatically)

sbatch \\
    --job-name=rnn_labs_s{seed} \\
    --account={account} \\
    --partition={partition} \\
    --nodes=1 --ntasks=1 --cpus-per-task={num_workers} \\
    --mem={mem} --gres={gres} --time={time} \\
    --output=logs/slurm/rnn_labs_s{seed}_%j.out \\
    --error=logs/slurm/rnn_labs_s{seed}_%j.err \\
    --wrap="
        module load miniconda3/24.9.2 &&
        eval "$(conda shell.bash hook)" &&
        conda activate {conda_env} &&
        cd $PROJECT_DIR &&
        export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH &&
        python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
            --ehr-root {ehr_root} \\
            --cache-dir {cache_dir} \\
            --task labs \\
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
            --output-dir {output_dir}
    "
""")

# ── Step 2: Run (only needed for dev/interactive) ─────────────────────────────
print("\n" + "=" * 60 + "\n")
if dev:
    print(f"""
### STEP 2: Once on the node, run

module load miniconda3/24.9.2 &&
eval "$(conda shell.bash hook)" &&
conda activate {conda_env} &&
cd {project_dir} &&
export PYTHONPATH={project_dir}:$PYTHONPATH &&
mkdir -p logs/dev &&
python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
    --ehr-root {ehr_root} \\
    --cache-dir {cache_dir} \\
    --task labs \\
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
    --output-dir {output_dir} --dev \\
    > >(tee logs/dev/rnn_labs_s{seed}.out) \\
    2> >(tee logs/dev/rnn_labs_s{seed}.err >&2)
""")
