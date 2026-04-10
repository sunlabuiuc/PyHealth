#!/bin/bash
set -e #sets flag so script stops on any error

ENV_NAME="pyhealth2"

# Load miniconda
if ! command -v conda &> /dev/null; then
    module load miniconda3/24.9.2
    source ~/.bashrc #refreshes the shell environment
fi

eval "$(conda shell.bash hook)"

# fresh install if requested
if [ "$1" == "clean" ]; then
    conda deactivate 2>/dev/null || true
    conda env remove -n ${ENV_NAME} -y 2>/dev/null || true
    echo "Removed env ${ENV_NAME}"
fi

# Create env or skip
if conda info --envs 2>/dev/null | grep -q "${ENV_NAME}"; then
    echo "Environment '${ENV_NAME}' already exists. Skipping. (Run './setup.sh clean' to reset.)"
else
    conda create -n ${ENV_NAME} python=3.12 -y
    conda activate ${ENV_NAME}
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install -e . #installs the current directory as an editable package, a pointer to the local source code
    echo "done"
    exit 0
fi

# Activate and verify
conda activate ${ENV_NAME}
python -c "import pyhealth; print('PyHealth: OK')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
