"""Example: train HALO on MIMIC-III and generate synthetic patients.

This example demonstrates:
1. Loading MIMIC-III data
2. Applying the EHRGenerationMIMIC3 task (per-visit ICD-9 code sequences)
3. Creating a SampleDataset with a NestedSequenceProcessor
4. Training the HALO generator with its custom training loop
5. Generating synthetic patients
6. Evaluating the synthetic data with the generative metrics suite
"""

import pandas as pd

from pyhealth.datasets import MIMIC3Dataset, split_by_patient
from pyhealth.metrics.generative import evaluate_synthetic_ehr
from pyhealth.models import HALO
from pyhealth.tasks import EHRGenerationMIMIC3

if __name__ == "__main__":
    # STEP 1: Load MIMIC-III base dataset
    base_dataset = MIMIC3Dataset(
        root="/srv/local/data/MIMIC-III/mimic-iii-clinical-database-1.4",
        tables=["diagnoses_icd"],
        dev=True,
    )

    # STEP 2: Apply the EHR generation task (unconditional, no labels).
    # This task is shared by all generators in pyhealth.models.generators.
    sample_dataset = base_dataset.set_task(EHRGenerationMIMIC3())
    print(f"Total samples: {len(sample_dataset)}")
    print(f"Input schema: {sample_dataset.input_schema}")
    print(f"Output schema: {sample_dataset.output_schema}")

    sample = sample_dataset[0]
    print("\nSample structure:")
    print(f"  Patient ID: {sample['patient_id']}")
    print(f"  Visits tensor shape: {tuple(sample['visits'].shape)}")

    # STEP 3: Split dataset by patient
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )

    # STEP 4: Initialize HALO (small config for the dev subset)
    model = HALO(
        dataset=sample_dataset,
        embed_dim=128,
        n_heads=4,
        n_layers=4,
        n_ctx=48,
        batch_size=16,
        epochs=5,
        lr=1e-4,
        save_dir="./halo_save",
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized with {num_params} parameters")

    # STEP 5: Train with HALO's custom loop (saves best checkpoint to save_dir)
    model.train_model(train_dataset, val_dataset=val_dataset)

    # STEP 6: Generate synthetic patients (one per real training patient).
    synthetic = model.generate(num_samples=len(train_dataset), random_sampling=True)
    print("\nGenerated synthetic patients (first 3):")
    for patient in synthetic[:3]:
        print(f"  {patient['patient_id']}: {len(patient['visits'])} visits")
        print(f"    {patient['visits']}")

    # STEP 7: Evaluate the synthetic data with the generative metrics suite.
    # evaluate_synthetic_ehr expects flat dataframes with one row per code:
    #   columns = [id, time, visit_codes, labels]
    # `labels` is a placeholder -- the utility metric overwrites it with the
    # next-visit prediction target.
    index_to_code = {
        v: k for k, v in sample_dataset.input_processors["visits"].code_vocab.items()
    }

    def real_subset_to_records(subset):
        for sample in subset:
            pid = str(sample["patient_id"])
            visits_tensor = sample["visits"]
            for t, visit in enumerate(visits_tensor.tolist()):
                for idx in visit:
                    code = index_to_code.get(int(idx))
                    if code in (None, "<pad>", "<unk>"):
                        continue
                    yield {"id": pid, "time": t, "visit_codes": code, "labels": 0}

    def synthetic_to_records(patients):
        for p in patients:
            pid = str(p["patient_id"])
            for t, visit in enumerate(p["visits"]):
                for code in visit:
                    yield {"id": pid, "time": t, "visit_codes": code, "labels": 0}

    schema = {"visit_codes": str, "labels": int, "time": int, "id": str}
    train_df = pd.DataFrame(real_subset_to_records(train_dataset)).astype(schema)
    test_df = pd.DataFrame(real_subset_to_records(test_dataset)).astype(schema)
    syn_df = pd.DataFrame(synthetic_to_records(synthetic)).astype(schema)
    print(
        f"\nEval rows -- train: {len(train_df)}, test: {len(test_df)}, "
        f"synthetic: {len(syn_df)}"
    )

    # sample_size / n_bootstraps / n_runs are kept small for the dev subset;
    # raise them when running on the full MIMIC-III cohort.
    results = evaluate_synthetic_ehr(
        train_ehr=train_df,
        test_ehr=test_df,
        syn_ehr=syn_df,
        sample_size=min(30, len(train_dataset), len(test_dataset)),
        mode="lstm",
        metrics="all",
        lstm_params={"embed_dim": 16, "hidden_dim": 16, "batch_size": 16, "epochs": 3},
        n_bootstraps=5,
        n_runs=3,
    )
    print("\nGenerative metrics (mean +/- std):")
    for name, (mean, std) in results.items():
        print(f"  {name:30s} {mean:.4f} +/- {std:.4f}")
