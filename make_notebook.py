import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# TPC Ablation Study - Google Colab\n",
                "\n",
                "**Steps:**\n",
                "1. Enable GPU: Runtime → Change runtime type → T4 GPU\n",
                "2. Run cells 1-3 to setup and install dependencies\n",
                "3. Run cell 4 to mount Google Drive (for saving results)\n",
                "4. Run cell 5 to download MIMIC-IV data (~10-15 min, 9.92 GB)\n",
                "5. Run cell 6 to configure paths\n",
                "6. Run cell 7 to start training (2-4 hours)\n",
                "7. Results save to MyDrive/tpc_ablation_results/"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Clone repository\n",
                "!git clone https://github.com/tarakjc2c/PyHealth.git\n",
                "%cd PyHealth\n",
                "!git checkout pr-1028"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install dependencies\n",
                "!pip install -e . -q\n",
                "!pip install litdata polars pandas dask mne rdkit peft transformers ogb -q"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Mount Google Drive\n",
                "from google.colab import drive\n",
                "drive.mount('/content/drive')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Download MIMIC-IV data from shared Google Drive\n",
                "!pip install gdown -q\n",
                "import os\n",
                "import shutil\n",
                "\n",
                "MIMIC_ROOT = '/content/mimic-iv'\n",
                "GDRIVE_FOLDER_ID = '15vyfKQ6H0g7DVEbI8vAMNhr4fi2lznVn'\n",
                "\n",
                "if not os.path.exists(f'{MIMIC_ROOT}/hosp'):\n",
                "    print('Downloading MIMIC-IV data (9.92 GB, ~15-20 minutes)...')\n",
                "    print('This will download 34 files from shared Google Drive')\n",
                "    !gdown --folder {GDRIVE_FOLDER_ID} -O /content/mimic-iv-download --remaining-ok\n",
                "    \n",
                "    # Find where gdown actually put the files\n",
                "    download_dir = '/content/mimic-iv-download'\n",
                "    if os.path.exists(download_dir):\n",
                "        contents = os.listdir(download_dir)\n",
                "        print(f'Downloaded to: {contents}')\n",
                "        \n",
                "        # Check if nested folder\n",
                "        if len(contents) == 1 and os.path.isdir(f'{download_dir}/{contents[0]}'):\n",
                "            # Move from nested folder\n",
                "            shutil.move(f'{download_dir}/{contents[0]}', MIMIC_ROOT)\n",
                "        else:\n",
                "            # Move entire download dir\n",
                "            shutil.move(download_dir, MIMIC_ROOT)\n",
                "        print(f'✓ Data organized at {MIMIC_ROOT}')\n",
                "else:\n",
                "    print('✓ MIMIC-IV data already exists')\n",
                "\n",
                "# Verify critical files\n",
                "if os.path.exists(f'{MIMIC_ROOT}/hosp'):\n",
                "    hosp_files = len([f for f in os.listdir(f'{MIMIC_ROOT}/hosp') if f.endswith('.csv.gz')])\n",
                "    icu_files = len([f for f in os.listdir(f'{MIMIC_ROOT}/icu') if f.endswith('.csv.gz')])\n",
                "    print(f'✓ hosp/: {hosp_files} files')\n",
                "    print(f'✓ icu/: {icu_files} files')\n",
                "else:\n",
                "    print('✗ ERROR: Data download failed. Check permissions on Google Drive link.')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Fix paths for Colab\n",
                "script = 'examples/length_of_stay/length_of_stay_mimic4_tpc.py'\n",
                "\n",
                "with open(script, 'r') as f:\n",
                "    lines = f.readlines()\n",
                "\n",
                "# Replace path definitions\n",
                "new_lines = []\n",
                "for line in lines:\n",
                "    if 'MIMIC_ROOT = r\"C:' in line:\n",
                "        new_lines.append('MIMIC_ROOT = \"/content/mimic-iv\"\\n')\n",
                "    elif 'CACHE_PATH = r\"C:' in line:\n",
                "        new_lines.append('CACHE_PATH = \"/content/tpc_cache\"\\n')\n",
                "    elif 'OUTPUT_DIR = \"tpc_ablation_results\"' in line:\n",
                "        new_lines.append('OUTPUT_DIR = \"/content/drive/MyDrive/tpc_ablation_results\"\\n')\n",
                "    else:\n",
                "        new_lines.append(line)\n",
                "\n",
                "with open(script, 'w') as f:\n",
                "    f.writelines(new_lines)\n",
                "\n",
                "print('✓ Paths configured for Colab')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run ablation study (2-4 hours)\n",
                "!python examples/length_of_stay/length_of_stay_mimic4_tpc.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Download Results\n",
                "\n",
                "Results are in MyDrive/tpc_ablation_results/\n",
                "Download them with the cell below:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import files\n",
                "import os\n",
                "\n",
                "result_dir = '/content/drive/MyDrive/tpc_ablation_results'\n",
                "for filename in ['ablation_results.json', 'mc_dropout_results.json']:\n",
                "    filepath = os.path.join(result_dir, filename)\n",
                "    if os.path.exists(filepath):\n",
                "        print(f'Downloading: {filename}')\n",
                "        files.download(filepath)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open('examples/length_of_stay/tpc_mimic4_colab.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("✓ Created valid Jupyter notebook")
