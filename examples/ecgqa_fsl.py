"""ECG Question Answering with Few-Shot Learning — PyHealth example.

This script demonstrates the full data pipeline for the FSL_ECG_QA
project (Tang et al., CHIL 2025) using PyHealth datasets and tasks.

Pipeline:
    1. PTBXLDataset  → PTBXLResampling task → resampled ECG signals (12, 2500)
    2. ECGQADataset  → ECGQA task (with signal_loader) → multimodal QA samples

For the full meta-learning training loop, see:
    https://github.com/Tang-Jia-Lu/FSL_ECG_QA/blob/main/train.py

Requirements:
    - PTB-XL dataset (https://physionet.org/content/ptb-xl/1.0.3/)
    - ECG-QA data (https://github.com/Tang-Jia-Lu/FSL_ECG_QA/tree/main/ecgqa)
    - pip install wfdb

Authors:
    Jovian Wang (jovianw2@illinois.edu)
    Matthew Pham (mdpham2@illinois.edu)
    Yiyun Wang (yiyunw3@illinois.edu)
"""

import json
import os
import tempfile
from pathlib import Path

import torch
from pyhealth.datasets import PTBXLDataset, ECGQADataset
from pyhealth.tasks import PTBXLResampling, ECGQA

# ---------- Configuration ----------
# Update these paths to match your local setup
PTBXL_ROOT = "/path/to/ptb-xl/1.0.3/"             # contains records500/, records100/
ECGQA_ROOT = "/path/to/ecgqa/ptbxl/paraphrased/"   # contains train/, valid/, test/

# Set to True for a quick test run (loads a small matched subset).
# Set to False to process the full dataset.
DEV_MODE = True


def _load_dev_subset(ptbxl_root, ecgqa_root):
    """Load a small matched subset for quick testing.

    PTBXLDataset dev mode picks 5 random patients from the full 21K
    range. To guarantee every QA sample has a matching signal, this
    helper pre-filters the QA JSON files to only include records whose
    ecg_id appears in the loaded PTB-XL signals, then loads the filtered
    data through ECGQADataset.
    """
    # Load + resample PTB-XL signals (5 in dev mode)
    print("  Loading PTB-XL signals...")
    ptb = PTBXLDataset(root=ptbxl_root, downsampled=False, dev=True)
    signal_ds = ptb.set_task(PTBXLResampling(root=ptbxl_root))
    signal_lookup = {s["record_id"]: s["signal"] for s in signal_ds}
    matched_ecg_ids = set(int(k) for k in signal_lookup.keys())
    print(f"  PTB-XL: {len(signal_lookup)} signals loaded (ecg_ids: {matched_ecg_ids})")

    # Pre-filter QA JSONs to only records with matching ecg_ids
    print("  Filtering QA data to matched ecg_ids...")
    tmp_dir = tempfile.mkdtemp()
    src = Path(ecgqa_root)
    total_kept = 0
    for split in ("train", "valid", "test"):
        dst = Path(tmp_dir) / split
        dst.mkdir()
        split_records = []
        for fpath in sorted((src / split).glob("*.json")):
            with open(fpath) as f:
                records = json.load(f)
            split_records.extend(r for r in records if r["ecg_id"][0] in matched_ecg_ids)
        if not split_records:
            # Write a dummy record so _verify_data passes; it gets filtered
            # out by prepare_metadata (question_type won't start with "single-")
            split_records = [{"ecg_id": [0], "question": "", "answer": [""],
                              "question_type": "dummy", "attribute_type": "",
                              "template_id": 0, "question_id": 0,
                              "sample_id": 0, "attribute": [""]}]
        else:
            total_kept += len(split_records)
        with open(dst / "00.json", "w") as f:
            json.dump(split_records, f)
    print(f"  Kept {total_kept} QA records for {len(matched_ecg_ids)} ecg_ids")

    # Load filtered QA data with signal loader
    def signal_loader(ecg_id):
        return torch.FloatTensor(signal_lookup[ecg_id])

    qa = ECGQADataset(root=tmp_dir)
    samples = qa.set_task(ECGQA(signal_loader=signal_loader))
    print(f"  Created {len(samples)} matched QA samples")
    return samples, signal_lookup


def main():
    if DEV_MODE:
        samples, signal_lookup = _load_dev_subset(PTBXL_ROOT, ECGQA_ROOT)
    else:
        # ---------- Full pipeline ----------
        # Step 1: Load + resample all PTB-XL signals
        print("Loading PTB-XL dataset...")
        ptb = PTBXLDataset(root=PTBXL_ROOT, downsampled=False)
        signal_ds = ptb.set_task(PTBXLResampling(root=PTBXL_ROOT))
        signal_lookup = {s["record_id"]: s["signal"] for s in signal_ds}
        print(f"  Loaded {len(signal_lookup)} signal samples")

        # Step 2: Build signal loader
        def signal_loader(ecg_id: int) -> torch.Tensor:
            return torch.FloatTensor(signal_lookup[ecg_id])

        # Step 3: Load ECG-QA data with signals
        print("Loading ECG-QA dataset...")
        qa = ECGQADataset(root=ECGQA_ROOT)
        samples = qa.set_task(ECGQA(signal_loader=signal_loader))
        print(f"  Created {len(samples)} QA samples")

    # ---------- Inspect a sample ----------
    if len(samples) == 0:
        print("\nNo matched samples found. Check that PTBXL_ROOT and ECGQA_ROOT are correct.")
        return

    sample = samples[0]
    print("\n=== Sample ===")
    print(f"  patient_id:    {sample['patient_id']}")
    print(f"  question:      {sample['question'][:80]}...")
    print(f"  answer:        {sample['answer']}")
    print(f"  question_type: {sample['question_type']}")
    print(f"  episode_class: {sample['episode_class']}")
    if "signal" in sample:
        print(f"  signal shape:  {sample['signal'].shape}")


if __name__ == "__main__":
    main()
