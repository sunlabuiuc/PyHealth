#!/usr/bin/env bash
# Generates 3 random seeds, submits 18 SLURM jobs (1 per model+seed),
# with heavier resource requests for memory-hungry models.
# Optionally chains off a cachewarm job via --dependency=afterok:<jobid>.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
SEED_MANIFEST="${SEED_MANIFEST:-scripts/slurm/table2_random_seeds.txt}"
ACCOUNT="${SLURM_ACCOUNT:-jimeng-cs-eng}"
PARTITION="${SLURM_PARTITION:-eng-research-gpu}"
CACHEWARM_JOB_ID="${1:-}"   # optional: pass cachewarm job ID to chain dependency

cd "${PROJECT_DIR}"
mkdir -p logs/slurm

mapfile -t SEEDS < <(
python3 - <<'PY'
import random
rng = random.SystemRandom()
for seed in rng.sample(range(1, 2_147_483_647), 3):
    print(seed)
PY
)

if [[ "${#SEEDS[@]}" -ne 3 ]]; then
    echo "ERROR: failed to generate 3 random seeds." >&2
    exit 1
fi

printf '%s\n' "${SEEDS[@]}" > "${SEED_MANIFEST}"
echo "Random seeds: ${SEEDS[*]}"
echo "Manifest    : ${SEED_MANIFEST}"

# Dependency string — only set if a cachewarm job ID was passed
DEPEND=""
if [[ -n "${CACHEWARM_JOB_ID}" ]]; then
    DEPEND="--dependency=afterok:${CACHEWARM_JOB_ID}"
    echo "Chaining after cachewarm job: ${CACHEWARM_JOB_ID}"
fi

submit_job() {
    local model="$1"
    local seed="$2"
    local time_limit="$3"
    local mem="$4"

    sbatch \
        --job-name="t2_${model}_s${seed}" \
        --account="${ACCOUNT}" \
        --partition="${PARTITION}" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem="${mem}" \
        --gres=gpu:1 \
        --time="${time_limit}" \
        --output="logs/slurm/table2_${model}_seed${seed}_%j.out" \
        --error="logs/slurm/table2_${model}_seed${seed}_%j.err" \
        --export=ALL,MODEL="${model}",SEED="${seed}" \
        ${DEPEND} \
        scripts/slurm/run_table2.sh
}

# Light models (eng-research-gpu / A10 24GB) — mlp + rnn only.
# ehrmamba and heavy models go to IllinoisComputes-GPU via submit_table2_ic.sh.
for model in mlp rnn; do
    for seed in "${SEEDS[@]}"; do
        job=$(submit_job "${model}" "${seed}" "12:00:00" "32G")
        echo "  Submitted ${model} seed=${seed} → ${job}"
    done
done

echo ""
echo "6 eng-research jobs submitted (mlp × 3, rnn × 3)."
echo "Run submit_table2_ic.sh next for the remaining 12 (ehrmamba + heavy models)."
echo "Monitor with:"
echo "  squeue -u rianatri"
echo "  tail -f logs/slurm/table2_<model>_seed<seed>_<jobid>.out"
