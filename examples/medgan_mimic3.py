"""Example: train MedGAN on MIMIC-III and generate synthetic patients.

MedGAN and CorGAN are bag-of-codes generators: a patient is one multi-hot
vector over the code vocabulary, with no visit axis at all. They read
EHRCodeSetGenerationMIMIC3, which pools every admission's codes into a single
set per patient -- not the per-visit tasks HALO (halo_mimic3.py) or
GPT2/PromptEHR (gpt2_mimic3.py) take.

Swap MedGAN for CorGAN below and the rest of the script is unchanged.

This example demonstrates:
1. Loading MIMIC-III data
2. Applying the EHRCodeSetGenerationMIMIC3 task (one ICD-9 code set per patient)
3. Training MedGAN (autoencoder pre-training, then adversarial training)
4. Generating synthetic patients
5. Evaluating with the privacy metrics

Note on metrics: the utility metric in pyhealth.metrics.generative scores
next-visit prediction, which is meaningless without a visit axis. This example
therefore requests ``metrics="privacy"``. Asking for ``"all"`` here would
produce a utility number that looks real and is not.
"""

from pyhealth.datasets import MIMIC3Dataset, split_by_patient
from pyhealth.metrics.generative import evaluate_synthetic_ehr
from pyhealth.models import MedGAN
from pyhealth.tasks import EHRCodeSetGenerationMIMIC3, to_evaluation_dataframe

if __name__ == "__main__":
    # STEP 1: Load MIMIC-III. dev=True keeps this small -- start here.
    base_dataset = MIMIC3Dataset(
        root="/srv/local/data/MIMIC-III/mimic-iii-clinical-database-1.4",
        tables=["diagnoses_icd"],
        dev=True,
    )

    # STEP 2: Pool each patient's admissions into one code set. min_visits
    # still counts real admissions, so single-admission patients are dropped
    # before the visit axis is collapsed.
    sample_dataset = base_dataset.set_task(EHRCodeSetGenerationMIMIC3())
    print(f"Total samples: {len(sample_dataset)}")

    sample = sample_dataset[0]
    # (vocab_size,) -- one row per patient, not per visit.
    print(f"Visits tensor shape: {tuple(sample['visits'].shape)}")

    # STEP 3: Split by patient.
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )

    # STEP 4: Initialize MedGAN (small config for the dev subset).
    model = MedGAN(
        dataset=sample_dataset,
        latent_dim=32,
        hidden_dim=32,
        discriminator_hidden_dim=64,
        batch_size=32,
        ae_epochs=10,
        gan_epochs=20,
        save_dir="./medgan_save",
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized with {num_params} parameters")

    # STEP 5: Train (autoencoder first, then the GAN).
    model.train_model(train_dataset, val_dataset=val_dataset)

    # STEP 6: Generate one synthetic patient per real training patient.
    synthetic = model.generate(num_samples=len(train_dataset))
    print("\nGenerated synthetic patients (first 3):")
    for patient in synthetic[:3]:
        # visits is a single-element list: the aggregate bag of codes.
        print(f"  {patient['patient_id']}: {len(patient['visits'][0])} codes")

    # STEP 7: Evaluate. MultiHotProcessor keeps its vocabulary in label_vocab
    # (the nested processors call theirs code_vocab), and each row is a
    # multi-hot vector, so the codes present are its nonzero columns.
    processor = sample_dataset.input_processors["visits"]
    index_to_code = {idx: code for code, idx in processor.label_vocab.items()}

    def to_records(subset):
        for item in subset:
            codes = [
                index_to_code[int(col)]
                for col in item["visits"].nonzero(as_tuple=True)[0].tolist()
                if index_to_code.get(int(col)) is not None
            ]
            # One "visit" per patient, matching the generator's output shape.
            yield {"visits": [codes]}

    schema = {"visit_codes": str, "labels": int, "time": int, "id": str}
    train_df = to_evaluation_dataframe(to_records(train_dataset)).astype(schema)
    test_df = to_evaluation_dataframe(to_records(test_dataset)).astype(schema)
    syn_df = to_evaluation_dataframe(synthetic).astype(schema)
    print(
        f"\nEval rows -- train: {len(train_df)}, test: {len(test_df)}, "
        f"synthetic: {len(syn_df)}"
    )

    # privacy only -- see the note at the top of this file.
    results = evaluate_synthetic_ehr(
        train_ehr=train_df,
        test_ehr=test_df,
        syn_ehr=syn_df,
        sample_size=min(30, len(train_dataset), len(test_dataset)),
        metrics="privacy",
        n_bootstraps=5,
    )
    print("\nPrivacy metrics (mean +/- std):")
    for name, (mean, std) in results.items():
        print(f"  {name:30s} {mean:.4f} +/- {std:.4f}")
