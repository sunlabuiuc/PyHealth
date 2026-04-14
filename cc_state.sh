#!/usr/bin/env bash
# cc_state.sh — one-shot local script for Campus Cluster Table 2 management.
#
# Usage:
#   bash cc_state.sh                 # show queue + recent job status
#   bash cc_state.sh resubmit        # sync, cancel all, clean cache, warm, submit all 18
#   bash cc_state.sh cancel          # cancel all pending/running jobs
#   bash cc_state.sh sync            # sync scripts to CC only (no submit)
#   bash cc_state.sh clean-cache     # delete corrupted parquet cache entries
#   bash cc_state.sh results         # print best AUROC/AUPRC/F1 per completed run
#   bash cc_state.sh logs [MODEL]    # tail recent logs (optional model filter)
set -euo pipefail

CC="${CC:-rianatri@cc-login.campuscluster.illinois.edu}"
REMOTE_REPO="${REMOTE_REPO:-/u/rianatri/PyHealth}"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$(dirname "$0")" && pwd)}"
SSH_KEY="${SSH_KEY:-}"   # e.g. SSH_KEY=~/.ssh/id_ed25519 bash cc_state.sh
SSH_OPTS="-o StrictHostKeyChecking=no${SSH_KEY:+ -o IdentitiesOnly=yes -i ${SSH_KEY}}"
CMD="${1:-state}"

ssh_cc() { ssh ${SSH_OPTS} "${CC}" "$@"; }
rsync_cc() {
    rsync -avz --relative \
        -e "ssh ${SSH_OPTS}" \
        "$@" \
        "${CC}:${REMOTE_REPO}/"
}

# ── sync ──────────────────────────────────────────────────────────────────────
do_sync() {
    echo "[sync] Syncing scripts and pyhealth source to CC..."
    cd "${LOCAL_REPO}"
    rsync_cc \
        pyhealth/ \
        examples/mortality_prediction/unified_embedding_e2e_mimic4.py \
        scripts/slurm/run_table2.sh \
        scripts/slurm/run_cachewarm.sh \
        scripts/slurm/submit_table2_random.sh \
        scripts/slurm/submit_table2_ic.sh \
        scripts/slurm/setup_cc.sh \
        scripts/condor/warm_table2_cache.py
    echo "[sync] Done."
}

# ── state ─────────────────────────────────────────────────────────────────────
do_state() {
    echo "=== Queue (rianatri) ==="
    ssh_cc "squeue -u rianatri --format='%.10i %.12P %.22j %.8T %.10M %.6D %R' 2>/dev/null || true"

    echo ""
    echo "=== Recent job outcomes (last 24h) ==="
    ssh_cc "sacct -u rianatri --starttime=now-24hours \
        --format=JobID%15,JobName%25,State%12,ExitCode,Elapsed \
        --noheader 2>/dev/null | grep -v '\.batch\|\.extern' || true"

    echo ""
    echo "=== GPU availability ==="
    ssh_cc "sinfo -p eng-research-gpu,IllinoisComputes-GPU \
        -o '%.20P %.10T %.6D %.15G' 2>/dev/null || true"
}

# ── cancel ────────────────────────────────────────────────────────────────────
do_cancel() {
    echo "[cancel] Cancelling all jobs for rianatri..."
    ssh_cc "scancel -u rianatri 2>/dev/null || true; echo '  Done.'"
}

# ── clean-cache ───────────────────────────────────────────────────────────────
do_clean_cache() {
    echo "[clean-cache] Removing corrupted parquet cache entries on CC..."
    ssh_cc "bash -s" <<'EOF'
CACHE_DIR="/u/${USER}/pyhealth_cache"
echo "  Cache dir: ${CACHE_DIR}"

# Remove any global_event_df.parquet directories that are empty or have 0-byte files
# (these are left behind by failed dask writes)
find "${CACHE_DIR}" -name "global_event_df.parquet" -type d | while read -r d; do
    # Check for empty or 0-byte parquet files inside
    bad=$(find "${d}" -name "*.parquet" -size 0 2>/dev/null | head -1)
    if [[ -n "${bad}" ]] || [[ -z "$(ls -A "${d}" 2>/dev/null)" ]]; then
        echo "  Removing corrupted: ${d}"
        rm -rf "${d}"
    else
        echo "  OK (non-empty): ${d}"
    fi
done

# Also clean up any stale dask temp dirs
rm -rf /u/${USER}/dask_tmp/ 2>/dev/null && echo "  Cleaned dask_tmp" || true
mkdir -p /u/${USER}/dask_tmp
echo "  Done."
EOF
}

# ── resubmit ──────────────────────────────────────────────────────────────────
do_resubmit() {
    do_sync

    echo ""
    echo "[resubmit] Cancelling all jobs..."
    ssh_cc "scancel -u rianatri 2>/dev/null || true; sleep 2"

    echo ""
    do_clean_cache

    echo ""
    echo "[resubmit] Submitting cachewarm + 18 training jobs (training deps on cachewarm)..."
    ssh_cc "REMOTE_REPO='${REMOTE_REPO}' bash -s" <<'EOF'
set -euo pipefail
cd "${REMOTE_REPO}"

mkdir -p /u/${USER}/dask_tmp logs/slurm

# Submit cachewarm job
WARM_JOB=$(sbatch \
    --account=jimeng-cs-eng \
    --partition=eng-research-gpu \
    --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --mem=48G --gres=gpu:1 --time=08:00:00 \
    --job-name=table2_cachewarm \
    --output=logs/slurm/table2_cachewarm_%j.out \
    --error=logs/slurm/table2_cachewarm_%j.err \
    scripts/slurm/run_cachewarm.sh | awk '{print $NF}')
echo "  Cachewarm job: ${WARM_JOB}"

# All 18 jobs → IllinoisComputes-GPU (IC), chained after cachewarm.
# mlp/rnn: BERT encoder OOMs on A10 24GB at bs=16; IC A100/H200 handles bs=16 fine.
DEPEND="--dependency=afterok:${WARM_JOB}"

for model in mlp rnn ehrmamba transformer bottleneck_transformer jambaehr; do
    while IFS= read -r seed; do
        case "${model}" in
            mlp)                   bs_var="TABLE2_BS_MLP=16"      ; tl="6:00:00"  ;;
            rnn)                   bs_var="TABLE2_BS_RNN=16"      ; tl="6:00:00"  ;;
            ehrmamba)              bs_var="TABLE2_BS_EHRMAMBA=8"  ; tl="12:00:00" ;;
            transformer)          bs_var="TABLE2_BS_TRANSFORMER=4"; tl="18:00:00" ;;
            bottleneck_transformer) bs_var="TABLE2_BS_BOTTLENECK=4"; tl="18:00:00" ;;
            jambaehr)             bs_var="TABLE2_BS_JAMBAEHR=4"  ; tl="18:00:00" ;;
        esac
        job=$(sbatch \
            --job-name="t2ic_${model}_s${seed}" \
            --account=jimeng-ic \
            --partition=IllinoisComputes-GPU \
            --nodes=1 --ntasks=1 --cpus-per-task=4 \
            --mem=32G --gres=gpu:1 --time="${tl}" \
            --output="logs/slurm/table2ic_${model}_seed${seed}_%j.out" \
            --error="logs/slurm/table2ic_${model}_seed${seed}_%j.err" \
            --export="ALL,MODEL=${model},SEED=${seed},${bs_var}" \
            ${DEPEND} \
            scripts/slurm/run_table2.sh | awk '{print $NF}')
        echo "  Submitted ${model} seed=${seed} → ${job}"
    done < scripts/slurm/table2_random_seeds.txt
done

echo ""
echo "19 jobs queued (1 cachewarm + 18 training). Queue:"
squeue -u rianatri --format="%.10i %.12P %.22j %.8T %.10M %.6D %R"
EOF
}

# ── results ───────────────────────────────────────────────────────────────────
do_results() {
    echo "[results] Fetching completed results from CC..."
    ssh_cc "REMOTE_REPO='${REMOTE_REPO}' bash -s" <<'EOSSH'
set -euo pipefail
cd "${REMOTE_REPO}"
OUT="output/table2"

if [[ ! -d "${OUT}" ]]; then
    echo "  No output directory found."
    exit 0
fi

# Print header
printf "\n%-35s %8s %8s %8s %8s %s\n" "Run" "AUROC" "AUPRC" "F1" "Acc" "Epochs"
printf '%s\n' "$(printf '%.0s-' {1..80})"

found=0
for d in "${OUT}"/*/; do
    run=$(basename "${d}")
    json="${d}metrics_history.json"
    [[ -f "${json}" ]] || continue
    found=1
    # Extract best val roc_auc (max), and corresponding auprc/f1/acc from that epoch
    python3 - "${json}" "${run}" <<'PY'
import json, sys
path, run = sys.argv[1], sys.argv[2]
with open(path) as f:
    h = json.load(f)

# Handle two possible formats:
# Format A: {"val": [{"roc_auc": 0.8, "epoch": 1, ...}, ...]}
# Format B: [{"epoch": 1, "roc_auc": 0.8, ...}, ...]  (flat list)
# Format C: {"roc_auc": [0.8, 0.9, ...], "pr_auc": [...]}  (dict of lists)
if isinstance(h, list):
    val = h
elif isinstance(h, dict):
    if "val" in h and isinstance(h["val"], list) and h["val"] and isinstance(h["val"][0], dict):
        val = h["val"]
    elif "roc_auc" in h and isinstance(h["roc_auc"], list):
        # dict-of-lists format
        keys = list(h.keys())
        n = len(h[keys[0]])
        val = [{k: h[k][i] for k in keys} for i in range(n)]
    else:
        val = []
else:
    val = []

if not val:
    print(f"{'  '+run:<35} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} -")
    sys.exit(0)

def safe_get(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return 0

best = max(val, key=lambda e: safe_get(e, "roc_auc"))
epoch = safe_get(best, "epoch")
total = len(val)
print(f"{run:<35} {safe_get(best,'roc_auc'):8.4f} {safe_get(best,'pr_auc','auprc'):8.4f} {safe_get(best,'f1'):8.4f} {safe_get(best,'accuracy'):8.4f} {epoch}/{total}")
PY
done

if [[ "${found}" -eq 0 ]]; then
    echo "  No completed results yet (metrics_history.json not found in any run dir)."
fi
EOSSH
}

# ── logs ──────────────────────────────────────────────────────────────────────
do_logs() {
    local filter="${2:-}"
    echo "[logs] Fetching recent log tails from CC (filter: ${filter:-all})..."
    ssh_cc "REMOTE_REPO='${REMOTE_REPO}' FILTER='${filter}' bash -s" <<'EOSSH'
set -euo pipefail
cd "${REMOTE_REPO}"
LOG_DIR="logs/slurm"
if [[ -n "${FILTER}" ]]; then
    mapfile -t LOGS < <(ls -t "${LOG_DIR}"/*"${FILTER}"*.out 2>/dev/null | head -6)
else
    mapfile -t LOGS < <(ls -t "${LOG_DIR}"/*.out 2>/dev/null | head -9)
fi
if [[ "${#LOGS[@]}" -eq 0 ]]; then
    echo "  No log files found."
else
    for f in "${LOGS[@]}"; do
        echo ""
        echo "━━━ ${f} ━━━"
        tail -20 "${f}" 2>/dev/null || echo "  (empty)"
    done
fi
EOSSH
}

# ── dispatch ──────────────────────────────────────────────────────────────────
case "${CMD}" in
    state)        do_state ;;
    sync)         do_sync ;;
    cancel)       do_cancel ;;
    clean-cache)  do_clean_cache ;;
    resubmit)     do_resubmit ;;
    results)      do_results ;;
    logs)         do_logs "$@" ;;
    *)
        echo "Unknown command: ${CMD}"
        echo "Usage: bash cc_state.sh [state|sync|cancel|clean-cache|resubmit|results|logs [MODEL]]"
        exit 1
        ;;
esac
