# Running PyHealth on the Campus Cluster

## Quick Start

```bash
git clone https://github.com/Multimodal-PyHealth/PyHealth.git
cd PyHealth
chmod +x setup.sh
./setup.sh
```

## Data Paths

All MIMIC-4 data is under `/projects/illinois/eng/cs/jimeng/physionet.org/files/`:

| Data | Path |
| EHR | `/projects/illinois/eng/cs/jimeng/physionet.org/files/mimiciv/2.2` |
| Clinical Notes | `/projects/illinois/eng/cs/jimeng/physionet.org/files/mimic-note` |
| Chest X-rays | `/projects/illinois/eng/cs/jimeng/physionet.org/files/mimic-cxr-jpg/2.1.0` |

**Important:** `NOTE_ROOT` should be `.../mimic-note` (not `.../mimic-note/note`). The config YAML appends `note/` automatically.

Set `CACHE_DIR` to your own writable directory: `/u/<NetID>/pyhealth_cache`

## Running on a Compute Node

Run on a compute node:
Slurm command example:
```bash
srun --account=jimeng-cs-eng --partition=eng-research-gpu --time=00:10:00 --gres=gpu:1 --pty bash
```

Once on the compute node, re-activate and run:

```bash
module load miniconda3/24.9.2
conda activate pyhealth2
cd ~/PyHealth
python examples/mortality_prediction/multimodal_mimic4.py
```
For a clean install, which reruns the setup proccess by deleting the enviroment packages and conda env and reinstalls them, run:
```./setup.sh clean```