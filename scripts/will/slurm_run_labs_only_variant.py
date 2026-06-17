project_dir = "/u/wp14/PyHealth"
seed = 12
account = "jimeng-ic"
partition = "eng-research-gpu"
time = "24:00:00"
mem = "64G"
gres = "gpu:A10:1"
conda_env = "pyhealth2"
ehr_root = "/projects/illinois/eng/cs/jimeng/physionet.org/files/mimiciv/2.2"
cache_dir = "/u/wp14/pyhealth_cache"
output_dir = "output/rnn_labs"
embedding_dim = 128
hidden_dim = 128
rnn_type = "GRU"
rnn_layers = 2
dropout = 0.1
epochs = 5
batch_size = 32
lr = 1e-3 
weight_decay = 1e-5
patience = 5
num_workers = 4
dev = False

# ── Step 0: Clean logs and cache ────────────────────────────────────────────
print(f"""
### STEP 0 (optional): Clean logs and cache

rm -rf {project_dir}/logs/*
rm -rf {cache_dir}/*
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

mkdir -p {project_dir}/logs && sbatch \\
    --job-name=rnn_labs_s{seed} \\
    --account={account} \\
    --partition={partition} \\
    --nodes=1 --ntasks=1 --cpus-per-task={num_workers} \\
    --mem={mem} --gres={gres} --time={time} \\
    --output={project_dir}/logs/rnn_labs_s{seed}_%j.out \\
    --error={project_dir}/logs/rnn_labs_s{seed}_%j.err \\
    --wrap='
        module load miniconda3/24.9.2 &&
        eval "$(conda shell.bash hook)" &&
        conda activate {conda_env} &&
        cd {project_dir} &&
        export PYTHONPATH={project_dir}:$PYTHONPATH &&
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
    '
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
mkdir -p {project_dir}/logs/dev &&
python {project_dir}/examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
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
    > >(tee {project_dir}/logs/dev/rnn_labs_s{seed}.out) \\
    2> >(tee {project_dir}/logs/dev/rnn_labs_s{seed}.err >&2)
""")
