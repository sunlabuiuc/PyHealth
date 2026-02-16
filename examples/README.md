# PyHealth Examples

This directory contains example scripts and notebooks for using PyHealth.

## HALO Synthetic Data Generation

### Google Colab Notebook (No Cluster Required)

**File**: `halo_mimic3_colab.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunlabuiuc/PyHealth/blob/master/examples/halo_mimic3_colab.ipynb)

Train HALO and generate synthetic MIMIC-III data directly in your browser using Google Colab.

**Requirements**:
- Google account (for Colab)
- MIMIC-III access from PhysioNet
- Files: ADMISSIONS.csv, DIAGNOSES_ICD.csv, PATIENTS.csv, patient_ids.txt

**Quick Start**:
1. Open `halo_mimic3_colab.ipynb` in Google Colab
2. Enable GPU (Runtime → Change runtime type → GPU)
3. Run cells in order
4. Upload your MIMIC-III files when prompted
5. Download synthetic data CSV

**Demo vs Production**:
- **Demo** (default): 5 epochs, 1K samples, ~30 min
- **Production**: 80 epochs, 10K samples, ~6-10 hours (change configuration)

**Features**:
- Google Drive integration for persistence
- Resume capability if session times out
- Automatic checkpoint saving
- CSV output format
- Data quality validation

### Cluster Training (SLURM)

**Files**:
- `slurm/train_halo_mimic3.slurm` - Training script
- `slurm/generate_halo_mimic3.slurm` - Generation script
- `halo_mimic3_training.py` - Python training code
- `generate_synthetic_mimic3_halo.py` - Python generation code

For users with access to GPU clusters. See individual script headers for usage.

**Example**:
```bash
# Train
sbatch slurm/train_halo_mimic3.slurm

# Generate
sbatch slurm/generate_halo_mimic3.slurm
```
