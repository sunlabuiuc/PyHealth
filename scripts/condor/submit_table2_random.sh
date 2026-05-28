#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
SUB_TEMPLATE="${SUB_TEMPLATE:-scripts/condor/batch_table2.sub}"
GENERATED_SUB="${GENERATED_SUB:-scripts/condor/batch_table2.generated.sub}"
SEED_MANIFEST="${SEED_MANIFEST:-scripts/condor/table2_random_seeds.txt}"

cd "${PROJECT_DIR}"

mkdir -p logs/condor

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

S0="${SEEDS[0]}"
S1="${SEEDS[1]}"
S2="${SEEDS[2]}"

# Replace SEED0/SEED1/SEED2 placeholders in the template.
# The template has two queue blocks (heavy/light) each with their own Rank —
# sed replaces placeholders in both blocks in one pass.
sed \
    -e "s/SEED0/${S0}/g" \
    -e "s/SEED1/${S1}/g" \
    -e "s/SEED2/${S2}/g" \
    "${SUB_TEMPLATE}" > "${GENERATED_SUB}"

echo "Generated random seeds: ${SEEDS[*]}"
echo "Seed manifest: ${SEED_MANIFEST}"
echo "Submit file  : ${GENERATED_SUB}"

condor_submit "${GENERATED_SUB}"
