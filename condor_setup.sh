#!/bin/bash
set -e #sets flag so script stops on any error

ENV_NAME="pyhealth2"

resolve_conda_sh() {
    # 1) explicit override
    if [ -n "${CONDA_SH:-}" ] && [ -f "${CONDA_SH}" ]; then
        echo "${CONDA_SH}"
        return 0
    fi

    # 2) conda on PATH
    if command -v conda >/dev/null 2>&1; then
        local base
        base="$(conda info --base 2>/dev/null || true)"
        if [ -n "${base}" ] && [ -f "${base}/etc/profile.d/conda.sh" ]; then
            echo "${base}/etc/profile.d/conda.sh"
            return 0
        fi
    fi

    # 3) optional module systems (quiet)
    if [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh >/dev/null 2>&1 || true
        if command -v module >/dev/null 2>&1; then
            module load miniconda3 >/dev/null 2>&1 || true
            module load anaconda3 >/dev/null 2>&1 || true
            if command -v conda >/dev/null 2>&1; then
                local mod_base
                mod_base="$(conda info --base 2>/dev/null || true)"
                if [ -n "${mod_base}" ] && [ -f "${mod_base}/etc/profile.d/conda.sh" ]; then
                    echo "${mod_base}/etc/profile.d/conda.sh"
                    return 0
                fi
            fi
        fi
    fi

    # 4) common install locations
    local user_name home_dir
    user_name="${USER:-$(id -un 2>/dev/null || true)}"
    home_dir="${HOME:-/home/${user_name}}"
    local candidates=(
        "${home_dir}/miniconda3/etc/profile.d/conda.sh"
        "/home/${user_name}/miniconda3/etc/profile.d/conda.sh"
        "${home_dir}/anaconda3/etc/profile.d/conda.sh"
        "/home/${user_name}/anaconda3/etc/profile.d/conda.sh"
        "/opt/miniconda3/etc/profile.d/conda.sh"
        "/opt/anaconda3/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
    )
    local c
    for c in "${candidates[@]}"; do
        if [ -f "${c}" ]; then
            echo "${c}"
            return 0
        fi
    done

    # 5) broad filesystem fallback for unknown install layouts
    local found=""
    found="$(find "${home_dir}" /opt /usr/local /shared -maxdepth 6 -type f -path '*/etc/profile.d/conda.sh' 2>/dev/null | head -n 1 || true)"
    if [ -n "${found}" ] && [ -f "${found}" ]; then
        echo "${found}"
        return 0
    fi

    return 1
}

CONDA_SH="$(resolve_conda_sh || true)"
if [ -z "${CONDA_SH}" ] || [ ! -f "${CONDA_SH}" ]; then
    echo "ERROR: conda.sh not found." >&2
    echo "Set it explicitly, e.g.: export CONDA_SH=/path/to/conda.sh" >&2
    exit 1
fi

source "${CONDA_SH}"
eval "$(conda shell.bash hook)"

# fresh install if requested
if [ "${1:-}" == "clean" ]; then
    conda deactivate 2>/dev/null || true
    conda env remove -n ${ENV_NAME} -y 2>/dev/null || true
    echo "Removed env ${ENV_NAME}"
fi

# Create env if missing
if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Environment '${ENV_NAME}' already exists."
else
    conda create -n ${ENV_NAME} python=3.12 -y
fi

conda activate ${ENV_NAME}

# Ensure required runtime deps are present even if env pre-existed.
if ! python -c "import torch" >/dev/null 2>&1; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi
if ! python -c "import pyhealth" >/dev/null 2>&1; then
    pip install -e . #installs the current directory as an editable package, a pointer to the local source code
fi

# Verify
python -c "import pyhealth; print('PyHealth: OK')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
