#!/usr/bin/env bash
# Submit 12 Table 2 jobs (heavy models) to IllinoisComputes-GPU partition.
# Reads seeds from the existing manifest so results stay consistent with
# mlp/rnn runs already queued on eng-research-gpu.
#
# Usage:
#   bash scripts/slurm/submit_table2_ic.sh
#
# Before running: cancel the pending eng-research heavy model jobs:
#   scancel <job_ids>   # ehrmamba/transformer/bottleneck/jambaehr jobs only
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
SEED_MANIFEST="${SEED_MANIFEST:-scripts/slurm/table2_random_seeds.txt}"
ACCOUNT="${SLURM_ACCOUNT:-jimeng-ic}"
PARTITION="${SLURM_PARTITION:-IllinoisComputes-GPU}"

cd "${PROJECT_DIR}"
mkdir -p logs/slurm

if [[ ! -f "${SEED_MANIFEST}" ]]; then
    echo "ERROR: Seed manifest not found: ${SEED_MANIFEST}" >&2
    echo "Run submit_table2_random.sh first to generate seeds, then rerun this script." >&2
    exit 1
fi

mapfile -t SEEDS < "${SEED_MANIFEST}"
if [[ "${#SEEDS[@]}" -ne 3 ]]; then
    echo "ERROR: Expected 3 seeds in manifest, got ${#SEEDS[@]}" >&2
    exit 1
fi

echo "Seeds (from manifest): ${SEEDS[*]}"
echo "Partition : ${PARTITION}"
echo "Account   : ${ACCOUNT}"
echo ""

submit_job() {
    local model="$1"
    local seed="$2"
    local time_limit="$3"
    local mem="$4"
    shift 4
    local extra_exports=("$@")   # additional KEY=VALUE strings

    local export_str="ALL,MODEL=${model},SEED=${seed}"
    for kv in "${extra_exports[@]}"; do
        export_str="${export_str},${kv}"
    done

    sbatch \
        --job-name="t2ic_${model}_s${seed}" \
        --account="${ACCOUNT}" \
        --partition="${PARTITION}" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem="${mem}" \
        --gres=gpu:1 \
        --time="${time_limit}" \
        --output="logs/slurm/table2ic_${model}_seed${seed}_%j.out" \
        --error="logs/slurm/table2ic_${model}_seed${seed}_%j.err" \
        --export="${export_str}" \
        scripts/slurm/run_table2.sh
}

# A100 = 80GB VRAM, H200 = 141GB — can run much larger batches than A10 (24GB).
# ehrmamba:              bs=8  (vs 2 on A10)
# transformer:           bs=4  (vs 1 on A10)
# bottleneck_transformer:bs=4  (vs 1 on A10)
# jambaehr:              bs=4  (vs 1 on A10)

echo "Submitting ehrmamba × 3..."
for seed in "${SEEDS[@]}"; do
    job=$(submit_job "ehrmamba" "${seed}" "12:00:00" "32G" "TABLE2_BS_EHRMAMBA=8")
    echo "  ehrmamba seed=${seed} → ${job}"
done

echo "Submitting transformer × 3..."
for seed in "${SEEDS[@]}"; do
    job=$(submit_job "transformer" "${seed}" "18:00:00" "32G" "TABLE2_BS_TRANSFORMER=4")
    echo "  transformer seed=${seed} → ${job}"
done

echo "Submitting bottleneck_transformer × 3..."
for seed in "${SEEDS[@]}"; do
    job=$(submit_job "bottleneck_transformer" "${seed}" "18:00:00" "32G" "TABLE2_BS_BOTTLENECK=4")
    echo "  bottleneck_transformer seed=${seed} → ${job}"
done

echo "Submitting jambaehr × 3..."
for seed in "${SEEDS[@]}"; do
    job=$(submit_job "jambaehr" "${seed}" "18:00:00" "32G" "TABLE2_BS_JAMBAEHR=4")
    echo "  jambaehr seed=${seed} → ${job}"
done

echo ""
echo "12 IC jobs submitted. Monitor with:"
echo "  squeue -u rianatri -p ${PARTITION}"
echo "  tail -f logs/slurm/table2ic_<model>_seed<seed>_<jobid>.out"
