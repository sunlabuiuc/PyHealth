"""Example: train the GPT-2 baseline on MIMIC-III and generate patients.

GPT2 and PromptEHR are token-sequence models: each patient becomes one flat
stream of code ids, ``[BOS] codes_v1 [DELIM] codes_v2 ... [EOS]``. They read
per-visit code *indices*, so this uses EHRSequenceGenerationMIMIC3 -- not the
multi-hot EHRGenerationMIMIC3 that HALO takes (see halo_mimic3.py). Pairing a
model with the wrong task does not raise; it trains on nonsense.

Swap GPT2 for PromptEHR below and the rest of the script is unchanged.

This example demonstrates:
1. Loading MIMIC-III data
2. Applying the EHRSequenceGenerationMIMIC3 task (per-visit ICD-9 code indices)
3. Training GPT2 with its custom training loop
4. Generating synthetic patients
5. Evaluating the synthetic data with the generative metrics suite
"""

from pyhealth.datasets import MIMIC3Dataset, split_by_patient
from pyhealth.metrics.generative import evaluate_synthetic_ehr
from pyhealth.models import GPT2
from pyhealth.tasks import EHRSequenceGenerationMIMIC3, to_evaluation_dataframe

if __name__ == "__main__":
    # STEP 1: Load MIMIC-III. dev=True keeps this to a small subset -- start
    # here, and only drop it once the whole script runs end to end.
    base_dataset = MIMIC3Dataset(
        root="/srv/local/data/MIMIC-III/mimic-iii-clinical-database-1.4",
        tables=["diagnoses_icd"],
        dev=True,
    )

    # STEP 2: Apply the sequence-encoded generation task (no labels).
    sample_dataset = base_dataset.set_task(EHRSequenceGenerationMIMIC3())
    print(f"Total samples: {len(sample_dataset)}")

    sample = sample_dataset[0]
    # (num_visits, max_codes_per_visit) of vocabulary indices, right-padded
    # with <pad> (0) -- NOT a multi-hot vector.
    print(f"Visits tensor shape: {tuple(sample['visits'].shape)}")

    # STEP 3: Split by patient so no patient appears in two splits.
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )

    # STEP 4: Initialize GPT2 (small config for the dev subset).
    model = GPT2(
        dataset=sample_dataset,
        embed_dim=128,
        n_heads=4,
        n_layers=4,
        max_len=256,
        batch_size=16,
        epochs=5,
        lr=1e-4,
        save_dir="./gpt2_save",
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized with {num_params} parameters")

    # STEP 5: Train (saves the best checkpoint to save_dir).
    model.train_model(train_dataset, val_dataset=val_dataset)

    # STEP 6: Generate one synthetic patient per real training patient.
    synthetic = model.generate(num_samples=len(train_dataset))
    print("\nGenerated synthetic patients (first 3):")
    for patient in synthetic[:3]:
        print(f"  {patient['patient_id']}: {len(patient['visits'])} visits")
        print(f"    {patient['visits']}")

    # STEP 7: Evaluate. The metrics want one row per (patient, visit, code);
    # to_evaluation_dataframe produces exactly that from either real records or
    # a generator's output. Real visits are read straight off the index tensor
    # through the processor, which knows how to invert its own encoding.
    processor = sample_dataset.input_processors["visits"]
    index_to_code = {idx: code for code, idx in processor.code_vocab.items()}

    def to_records(subset):
        for item in subset:
            visits = []
            for row in item["visits"]:
                codes = [
                    index_to_code[idx]
                    for idx in processor.visit_code_ids(row)
                    if index_to_code.get(idx) not in (None, "<pad>", "<unk>")
                ]
                if codes:
                    visits.append(codes)
            yield {"visits": visits}

    schema = {"visit_codes": str, "labels": int, "time": int, "id": str}
    train_df = to_evaluation_dataframe(to_records(train_dataset)).astype(schema)
    test_df = to_evaluation_dataframe(to_records(test_dataset)).astype(schema)
    syn_df = to_evaluation_dataframe(synthetic).astype(schema)
    print(
        f"\nEval rows -- train: {len(train_df)}, test: {len(test_df)}, "
        f"synthetic: {len(syn_df)}"
    )

    # Small settings for the dev subset; raise them on the full cohort.
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
