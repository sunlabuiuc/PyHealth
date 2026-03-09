"""Train MedGAN on MIMIC-III diagnosis codes and save a checkpoint."""

# 1. Load MIMIC-III dataset
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient
from pyhealth.tasks import medgan_generation_mimic3_fn
from pyhealth.models.generators.medgan import MedGAN

base_dataset = MIMIC3Dataset(
    root="/path/to/mimic3",
    tables=["diagnoses_icd"],
)

# 2. Apply generation task — flattens all ICD codes per patient into a bag-of-codes
sample_dataset = base_dataset.set_task(medgan_generation_mimic3_fn)
print(f"{len(sample_dataset)} patients after filtering")

# 3. Patient-level split — required for generative models to prevent data leakage across splits
train_dataset, val_dataset, _ = split_by_patient(sample_dataset, [0.8, 0.1, 0.1])

# 4. Instantiate and train — reduce epochs for testing; 100+ recommended for quality
model = MedGAN(
    dataset=sample_dataset,
    latent_dim=128,
    hidden_dim=128,
    batch_size=128,
    ae_epochs=100,
    gan_epochs=200,
    save_dir="./medgan_checkpoints/",
)
model.train_model(train_dataset, val_dataset)

# 5. Checkpoint is saved automatically to save_dir by train_model
print("Training complete. Checkpoint saved to ./medgan_checkpoints/")
