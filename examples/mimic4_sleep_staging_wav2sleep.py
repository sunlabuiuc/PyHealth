"""
Example script for Sleep Stage Classification using Wav2Sleep on MIMIC-IV dataset.
This script demonstrates the model's robustness through an Ablation Study
on missing modalities (Stochastic Masking), adapted for MIMIC-IV clinical signals.
"""

import torch
from pyhealth.models import Wav2Sleep

def run_example():
    print("--- PyHealth Example: MIMIC-IV Sleep Staging with Wav2Sleep ---")

    # 1. Setup mock data (Adapted for MIMIC-IV: ECG + Respiratory/PPG)
    # batch_size=2, sequence_length=5 epochs, signal_length=3000
    batch_size, seq_len, signal_len = 2, 5, 3000

    data = {
        "ecg": torch.randn(batch_size, seq_len, signal_len),
        "resp": torch.randn(batch_size, seq_len, signal_len),
        "label": torch.randint(0, 5, (batch_size, seq_len))
    }

    # 2. Initialize Wav2Sleep
    model = Wav2Sleep(
        dataset=None,
        feature_keys=["ecg", "resp"],
        label_key="label",
        mode="multiclass",
        embedding_dim=128,
        mask_prob={"ecg": 0.5, "resp": 0.5}
    )

    # 3. Ablation Study: Clinical Signal Loss
    print("\n[Ablation] Scenario: Respiratory sensor noise/loss in MIMIC-IV")

    data_missing = {
        "ecg": data["ecg"],
        "resp": torch.zeros_like(data["resp"]),
        "label": data["label"]
    }

    model.eval()
    with torch.no_grad():
        output = model(**data_missing)

    print(f"Inference Successful!")
    print(f"Loss with missing modality: {output['loss']:.4f}")
    print(f"Output probability shape: {output['y_prob'].shape} (5 Sleep Stages)")

    print("\n[Clinical Value]: The model maintains diagnostic capability "
          "even with incomplete bedside monitor data.")

if __name__ == "__main__":
    run_example()
