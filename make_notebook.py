import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# TPC Ablation Study - Google Colab\n",
                "\n",
                "**IMPORTANT: Reset runtime first if rerunning**\n",
                "- Runtime → Disconnect and delete runtime (to avoid nested folders)\n",
                "\n",
                "**Steps:**\n",
                "1. Enable GPU: Runtime → Change runtime type → T4 GPU → Save\n",
                "2. Run cells 1-3 to setup (clone, install, mount Drive)\n",
                "3. Run cell 4 to download MIMIC-IV data (~15-20 min, 9.92 GB)\n",
                "4. Verify cell 4 shows '✓ READY!' before continuing\n",
                "5. Run cell 5 to configure paths\n",
                "6. Run cell 6 to start training (2-4 hours)\n",
                "7. Results save to MyDrive/tpc_ablation_results/"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Clone repository (only if not already cloned)\n",
                "import os\n",
                "if not os.path.exists('/content/PyHealth'):\n",
                "    !git clone https://github.com/tarakjc2c/PyHealth.git\n",
                "    print('✓ Cloned repository')\n",
                "else:\n",
                "    print('✓ Repository already exists')\n",
                "\n",
                "%cd /content/PyHealth\n",
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
                "!pip install litdata polars pandas dask mne rdkit peft transformers ogb -q\n",
                "import os\n",
                "print(f'✓ Installed in: {os.getcwd()}')"
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
                "# Check if data already exists\n",
                "if os.path.exists(f'{MIMIC_ROOT}/hosp/diagnoses_icd.csv.gz'):\n",
                "    print('✓ MIMIC-IV data already exists')\n",
                "else:\n",
                "    print('Downloading MIMIC-IV data (9.92 GB, ~15-20 minutes)...')\n",
                "    print('Downloading 34 files from shared Google Drive\\n')\n",
                "    \n",
                "    # Download to temporary location\n",
                "    temp_dir = '/content/mimic-download'\n",
                "    !gdown --folder {GDRIVE_FOLDER_ID} -O {temp_dir} --remaining-ok\n",
                "    \n",
                "    print('\\nOrganizing files...')\n",
                "    # Find where gdown put the files\n",
                "    if os.path.exists(temp_dir):\n",
                "        !ls -la {temp_dir}\n",
                "        \n",
                "        # Check for nested folder structure\n",
                "        contents = os.listdir(temp_dir)\n",
                "        print(f'Downloaded contents: {contents}')\n",
                "        \n",
                "        # Look for hosp and icu folders\n",
                "        found = False\n",
                "        for item in contents:\n",
                "            item_path = os.path.join(temp_dir, item)\n",
                "            if os.path.isdir(item_path):\n",
                "                # Check if this folder contains hosp/icu\n",
                "                if os.path.exists(f'{item_path}/hosp') and os.path.exists(f'{item_path}/icu'):\n",
                "                    shutil.move(item_path, MIMIC_ROOT)\n",
                "                    found = True\n",
                "                    break\n",
                "        \n",
                "        # If hosp/icu are directly in temp_dir\n",
                "        if not found and 'hosp' in contents and 'icu' in contents:\n",
                "            shutil.move(temp_dir, MIMIC_ROOT)\n",
                "            found = True\n",
                "        \n",
                "        if found:\n",
                "            print(f'✓ Data organized at {MIMIC_ROOT}')\n",
                "        else:\n",
                "            print('✗ ERROR: Could not find hosp/icu folders in download')\n",
                "            print('Manual check needed - run: !find /content -name \"chartevents.csv.gz\" -type f')\n",
                "    else:\n",
                "        print('✗ ERROR: Download directory not created')\n",
                "\n",
                "# Verify critical files exist\n",
                "print('\\nVerifying data files:')\n",
                "critical_files = [\n",
                "    'hosp/diagnoses_icd.csv.gz',\n",
                "    'hosp/patients.csv.gz',\n",
                "    'icu/chartevents.csv.gz',\n",
                "    'icu/icustays.csv.gz'\n",
                "]\n",
                "\n",
                "all_found = True\n",
                "for f in critical_files:\n",
                "    path = os.path.join(MIMIC_ROOT, f)\n",
                "    if os.path.exists(path):\n",
                "        size_mb = os.path.getsize(path) / (1024*1024)\n",
                "        print(f'✓ {f} ({size_mb:.1f} MB)')\n",
                "    else:\n",
                "        print(f'✗ MISSING: {f}')\n",
                "        all_found = False\n",
                "\n",
                "if all_found:\n",
                "    hosp_files = len([f for f in os.listdir(f'{MIMIC_ROOT}/hosp') if f.endswith('.csv.gz')])\n",
                "    icu_files = len([f for f in os.listdir(f'{MIMIC_ROOT}/icu') if f.endswith('.csv.gz')])\n",
                "    print(f'\\n✓ READY! Found {hosp_files} hosp files, {icu_files} icu files')\n",
                "else:\n",
                "    print('\\n✗ Download incomplete. Try running this cell again.')"
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
