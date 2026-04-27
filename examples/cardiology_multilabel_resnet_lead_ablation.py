"""
Spatial Feature Ablation Study: 12-Lead Clinical ECG vs. 1-Lead Wearable ECG

This script demonstrates how varying the spatial feature dimensions (number of ECG leads)
affects the input shape and predictive framework of a PyHealth model.

1. Clinical Context & Objective
-------------------------------
Standard clinical ECGs utilize 12 distinct leads to capture the electrical activity of the
heart from multiple spatial angles. However, modern wearable devices (like smartwatches)
typically only capture a single lead (equivalent to Lead I). 

This ablation study benchmarks the structural impact of transitioning from a 12-lead to a
1-lead setup. By isolating the 'leads' parameter in the CardiologyMultilabelClassification
task, we evaluate how PyHealth's native ResNet architecture adapts to the loss of spatial
projection vectors.

2. Experimental Setup
---------------------
- Dataset: Synthetic data generated to mimic the PhysioNet/CinC Challenge 2020 format.
- Task Configuration 1 (Baseline): All 12 leads utilized. Input shape is (12, 1250).
- Task Configuration 2 (Ablation): Only Lead I (index 0) utilized. Input shape is (1, 1250).
- Model: PyHealth's native ResNet, initialized dynamically based on the dataset's feature space.

3. Expected Findings
--------------------
While models trained on 1-lead data might maintain robust performance for rhythm-based 
abnormalities (like Atrial Fibrillation), their performance is expected to degrade significantly 
for morphology-based diagnoses that rely on spatial axes, such as Bundle Branch Blocks (LBBB/RBBB) 
or Axis Deviations.
"""

import os
import shutil
import numpy as np
import pandas as pd
from scipy.io import savemat

# PyHealth Imports
from pyhealth.datasets import Cardiology2Dataset, get_dataloader
from pyhealth.tasks import CardiologyMultilabelClassification
from pyhealth.models import ResNet 

def generate_synthetic_data(root_dir: str, num_patients: int = 3):
    """Generates synthetic .mat and .hea files to simulate the PhysioNet dataset."""
    patient_dir = os.path.join(root_dir, "cpsc_2018", "g1")
    os.makedirs(patient_dir, exist_ok=True)
    
    # 164934002 = T wave abnormality, 426783006 = Sinus rhythm
    sample_dx = "426783006,164934002"
    
    for i in range(num_patients):
        mat_path = os.path.join(patient_dir, f"A{i:04d}.mat")
        hea_path = os.path.join(patient_dir, f"A{i:04d}.hea")
        
        # 12 leads, 10 seconds at 500Hz = 5000 samples
        synthetic_signal = np.random.randn(12, 5000) 
        savemat(mat_path, {"val": synthetic_signal})
        
        with open(hea_path, "w") as f:
            f.write(f"A{i:04d} 12 500 5000\n")
            f.write("# Age: 63\n")
            f.write("# Sex: Male\n")
            f.write(f"# Dx: {sample_dx}\n")

def run_ablation_experiment():
    print("Initializing Spatial Feature Ablation Study...")
    SYNTHETIC_ROOT = "/tmp/synthetic_cardiology_data"
    CACHE_DIR = "/tmp/pyhealth_cache_ablation"
    
    if os.path.exists(SYNTHETIC_ROOT):
        shutil.rmtree(SYNTHETIC_ROOT)
    generate_synthetic_data(SYNTHETIC_ROOT)

    results = []
    
    # Define our two configurations for the ablation study
    configs = {
        "12-Lead (Clinical)": list(range(12)),
        "1-Lead (Wearable)": [0]
    }
    
    for setup_name, leads in configs.items():
        # 1. Load Dataset (using dev=True to minimize overhead)
        dataset = Cardiology2Dataset(
            root=SYNTHETIC_ROOT, 
            chosen_dataset=[1, 0, 0, 0, 0, 0], 
            cache_dir=CACHE_DIR,
            dev=True
        )
        
        # 2. Apply Task with varying feature dimensions
        task = CardiologyMultilabelClassification(leads=leads)
        sample_dataset = dataset.set_task(task)
        
        # 3. Initialize PyHealth Dataloader
        dataloader = get_dataloader(sample_dataset, batch_size=2, dev=True)
        batch = next(iter(dataloader))
        
        # 4. Initialize native PyHealth Model
        # ResNet automatically adapts to the feature dimension defined in the dataset
        model = ResNet(
            dataset=sample_dataset,
            feature_keys=["signal"],
            label_key="labels",
            mode="multilabel"
        )
        
        # 5. Forward pass through the PyHealth model
        out = model(**batch)
        
        # Record structural findings
        signal_shape = batch["signal"].shape
        results.append({
            "Configuration": setup_name,
            "Input Channels": signal_shape[1],
            "Batch Input Shape": tuple(signal_shape),
            "Loss Output Type": type(out["loss"]).__name__,
            "Logits Shape": tuple(out["y_prob"].shape),
            "Model Parameters": sum(p.numel() for p in model.parameters())
        })

    # Clean up synthetic data
    shutil.rmtree(SYNTHETIC_ROOT)

    # Output tabular findings
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("ABLATION STUDY: PYHEALTH RESNET FEATURE VARIATION RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    run_ablation_experiment()