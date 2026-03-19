#  PyHealth Cluster Environment Setup
#  Usage: ./setup.sh

set -e

ENV_NAME="pyhealth2"
ENV_FILE="pyhealth_cluster_env.yml"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}  PyHealth Cluster Setup${NC}"

#  Load miniconda 
if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}Loading miniconda module...${NC}"
    module load miniconda3/24.9.2
    source ~/.bashrc
fi

#  Create or skip conda env 
if conda info --envs 2>/dev/null | grep -q "${ENV_NAME}"; then
    echo -e "${GREEN}Conda environment '${ENV_NAME}' already exists. Skipping creation.${NC}"
else
    echo -e "${YELLOW}Creating conda environment '${ENV_NAME}'...${NC}"
    if [ ! -f "$ENV_FILE" ]; then
        echo "Error: ${ENV_FILE} not found. Make sure you're in the PyHealth repo root."
        exit 1
    fi
    conda env create -f ${ENV_FILE}
fi

#  Activate env 
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}
echo -e "${GREEN}Activated ${ENV_NAME} (Python: $(python --version))${NC}"

#  Install PyHealth editable if needed 
if python -c "import pyhealth" 2>/dev/null; then
    echo -e "${GREEN}PyHealth already installed. Skipping.${NC}"
else
    echo -e "${YELLOW}Installing PyHealth in editable mode...${NC}"
    pip install -e .
fi

#  Verify 
echo ""
echo -e "${GREEN}Verification:${NC}"
python -c "import pyhealth; print(f'  PyHealth: OK')"
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
echo ""
echo -e "${GREEN}Setup complete!${NC}"