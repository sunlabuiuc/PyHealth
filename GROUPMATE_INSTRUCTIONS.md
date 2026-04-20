# TPC Implementation - Setup Instructions for Groupmates

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/tarakjc2c/PyHealth.git
cd PyHealth
git checkout pr-1028
```

### 2. Install Dependencies
```bash
pip install -e .
pip install litdata polars pandas dask[complete] mne rdkit peft transformers accelerate ogb
```

### 3. Download MIMIC-IV Data
The full MIMIC-IV dataset is available here:
https://drive.google.com/drive/folders/15vyfKQ6H0g7DVEbI8vAMNhr4fi2lznVn?usp=sharing

**Option A - Manual Download:**
1. Download the folder from Google Drive
2. Extract to: `C:\cs598\mimic-iv\` (Windows) or `/path/to/mimic-iv/` (Linux/Mac)
3. Verify you have `hosp/` and `icu/` subdirectories

**Option B - Automated Download (requires gdown):**
```bash
pip install gdown
python download_mimic_from_gdrive.py --link "https://drive.google.com/drive/folders/15vyfKQ6H0g7DVEbI8vAMNhr4fi2lznVn?usp=sharing" --output "C:\cs598\mimic-iv"
```

### 4. Run Unit Tests (Quick Validation)
```bash
python -m pytest tests/core/test_tpc.py -v
```
**Expected:** 12/12 tests passing in ~15 seconds

### 5. Run Ablation Study (Full Experiment)
```bash
# Update paths in examples/length_of_stay/length_of_stay_mimic4_tpc.py if needed:
# MIMIC_ROOT = r"C:\cs598\mimic-iv"  # or your path
# CACHE_PATH = r"C:\cs598\.cache_dir"

python examples/length_of_stay/length_of_stay_mimic4_tpc.py
```

**IMPORTANT REQUIREMENTS:**
- **RAM:** 16GB+ recommended (memory error with <8GB due to 3.5GB chartevents file)
- **Time:** First run: 30-60 min (data caching) + 2-4 hours (training 4 configs)
- **Disk:** 10GB free space (9.92GB dataset + cache files)

### 6. Expected Output
After successful run, you'll find:
- `tpc_ablation_results/ablation_results.json` - Performance comparison
- `tpc_ablation_results/mc_dropout_results.json` - Uncertainty estimates
- Console output showing MAE/MSE for 4 configurations

## What This Implements

- **TPC Model:** Temporal Pointwise Convolutional Networks (Rocheteau et al., CHIL 2021)
- **Task:** Remaining length-of-stay prediction on MIMIC-IV
- **Ablation Study:** 4 configurations:
  1. Baseline (3 layers, MSLE loss, 0.3 dropout)
  2. Shallow (2 layers)
  3. MSE Loss (vs. MSLE)
  4. Low Dropout (0.1 vs. 0.3)
- **Novel Extension:** Monte Carlo Dropout uncertainty estimation

## Troubleshooting

### Memory Error
**Problem:** `MemoryError` or `P2P transfer failed` during data loading
**Solution:** 
- Close other applications
- Use machine with 16GB+ RAM
- Or modify script to use `dev=True` (line 303) for smaller subset

### Missing Data Files
**Problem:** `FileNotFoundError` for MIMIC-IV tables
**Solution:** Verify folder structure:
```
C:\cs598\mimic-iv\
├── hosp\
│   ├── admissions.csv.gz
│   ├── chartevents.csv.gz (3.5GB - critical!)
│   └── ... (20 more files)
└── icu\
    ├── icustays.csv.gz
    └── ... (8 more files)
```

### Import Errors
**Problem:** `ImportError: cannot import name 'TPC'`
**Solution:** Reinstall in development mode:
```bash
pip install -e .
```

## Files Changed (PR #1028)

**Core Implementation:**
- `pyhealth/models/tpc.py` - TPC model (900 lines)
- `pyhealth/tasks/length_of_stay_tpc_mimic4.py` - Remaining LoS task (200 lines)
- `pyhealth/models/__init__.py` - Model registration

**Testing:**
- `tests/core/test_tpc.py` - 12 comprehensive unit tests

**Documentation:**
- `docs/api/models/pyhealth.models.tpc.rst`
- `docs/api/tasks/pyhealth.tasks.length_of_stay_tpc_mimic4.rst`
- `docs/api/models.rst`, `docs/api/tasks.rst` - Index updates

**Examples:**
- `examples/length_of_stay/length_of_stay_mimic4_tpc.py` - Full ablation study

**Supporting:**
- `test-resources/core/mimic4demo/icu/d_items.csv.gz` - Demo data support
- `PR1028_PROGRESS_LOG.txt` - Detailed implementation log

## Contact

If you run into issues, check the progress log or ping the group chat.

**Status:** All unit tests passing (12/12). Ablation study ready to run on machines with adequate RAM.
