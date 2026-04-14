#!/usr/bin/env bash
# Full e2e: clone repo on CC if needed, set up conda env, sync scripts,
# warm cache, and submit 18 Table 2 training jobs.
#
# Usage:
#   bash run_table2_cc.sh           # normal run
#   SETUP=1 bash run_table2_cc.sh   # force re-run setup even if env exists
#   SETUP=clean bash run_table2_cc.sh  # nuke and rebuild conda env
set -euo pipefail

CC="${CC:-rianatri@cc-login.campuscluster.illinois.edu}"
REMOTE_REPO="${REMOTE_REPO:-/u/rianatri/PyHealth}"
LOCAL_REPO="${LOCAL_REPO:-/Users/saurabhatri/Dev/Multimodal-PyHealth}"
SETUP="${SETUP:-auto}"  # auto | 1 | clean

cd "${LOCAL_REPO}"

echo "[1/5] Validate local files..."
test -f scripts/slurm/run_table2.sh
test -f scripts/slurm/run_cachewarm.sh
test -f scripts/slurm/submit_table2_random.sh
test -f scripts/slurm/setup_cc.sh
bash -n scripts/slurm/run_table2.sh
echo "  OK"

echo "[2/5] Sync repo files to Campus Cluster..."
# Full pyhealth source sync — ensures CC has latest model/processor/task code
rsync -avz --relative \
  pyhealth/ \
  examples/mortality_prediction/unified_embedding_e2e_mimic4.py \
  scripts/condor/warm_table2_cache.py \
  scripts/slurm/run_table2.sh \
  scripts/slurm/run_cachewarm.sh \
  scripts/slurm/submit_table2_random.sh \
  scripts/slurm/setup_cc.sh \
  "${CC}:${REMOTE_REPO}/"

echo "[3/5] Setup conda env on CC (if needed)..."
ssh "${CC}" "REMOTE_REPO='${REMOTE_REPO}' SETUP='${SETUP}' bash -s" <<'EOF'
set -euo pipefail
cd "${REMOTE_REPO}"

module load miniconda3/24.9.2
eval "$(conda shell.bash hook)"

CONDA_ENV="pyhealth2"
ENV_EXISTS=0
conda env list | grep -q "^${CONDA_ENV}[[:space:]]" && ENV_EXISTS=1

if [[ "${SETUP}" == "clean" ]]; then
    echo "  Clean rebuild requested..."
    bash scripts/slurm/setup_cc.sh clean
elif [[ "${SETUP}" == "1" || "${ENV_EXISTS}" == "0" ]]; then
    echo "  Running setup (env missing or SETUP=1)..."
    bash scripts/slurm/setup_cc.sh
else
    echo "  Conda env ${CONDA_ENV} exists, skipping setup."
    echo "  (Run with SETUP=1 to force, SETUP=clean to rebuild.)"
    # Quick smoke test to make sure it's not broken
    conda activate "${CONDA_ENV}"
    python -c "import pyhealth, torch; print(f'  pyhealth OK, torch {torch.__version__}')"
fi
EOF

echo "[4/5] Prepare remote directories + permissions..."
ssh "${CC}" "REMOTE_REPO='${REMOTE_REPO}' bash -s" <<'EOF'
set -euo pipefail
cd "${REMOTE_REPO}"
mkdir -p logs/slurm
chmod +x scripts/slurm/run_table2.sh
chmod +x scripts/slurm/run_cachewarm.sh
chmod +x scripts/slurm/submit_table2_random.sh
chmod +x scripts/slurm/setup_cc.sh
echo "  Permissions OK"
EOF

echo "[5/5] Submit cachewarm + chain 18 training jobs..."
ssh "${CC}" "REMOTE_REPO='${REMOTE_REPO}' bash -s" <<'EOF'
set -euo pipefail
cd "${REMOTE_REPO}"

CACHE_INDEX="/u/${USER}/pyhealth_cache/662e765e-a310-5499-92b3-de524ea984bb/tasks/ClinicalNotesICDLabsMIMIC4_04c5dd00-6fb7-5542-9b0d-c722773fcc42/samples_cdbbc602-34e2-5a41-8643-4c76b08829f6.ld/index.json"

if [[ -f "${CACHE_INDEX}" ]]; then
    echo "  Cache already warm — submitting 18 training jobs directly."
    bash scripts/slurm/submit_table2_random.sh
else
    echo "  Cache cold — submitting cachewarm job first."
    WARM_JOB=$(sbatch scripts/slurm/run_cachewarm.sh | awk '{print $NF}')
    echo "  Cachewarm job ID: ${WARM_JOB}"
    bash scripts/slurm/submit_table2_random.sh "${WARM_JOB}"
    echo ""
    echo "  All 19 jobs queued (1 cachewarm + 18 training)."
    echo "  Training jobs will start automatically after cachewarm finishes."
fi

echo ""
squeue -u rianatri --format="%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null || squeue -u rianatri
EOF

echo ""
echo "Done. Monitor with:"
echo "  ssh ${CC} 'squeue -u rianatri'"
echo "  ssh ${CC} 'tail -f ${REMOTE_REPO}/logs/slurm/table2_<model>_seed<N>_<jobid>.out'"
