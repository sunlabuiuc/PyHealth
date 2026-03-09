"""Generate synthetic MIMIC-III patient records using a trained MedGAN checkpoint."""
import json

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import medgan_generation_mimic3_fn
from pyhealth.models.generators.medgan import MedGAN

# Update this to your local MIMIC-III path before running
MIMIC3_ROOT = "/path/to/mimic3"

# 1. Reconstruct dataset — required to initialise MedGAN's vocabulary from the processor.
base_dataset = MIMIC3Dataset(
    root=MIMIC3_ROOT,
    tables=["diagnoses_icd"],
)
sample_dataset = base_dataset.set_task(medgan_generation_mimic3_fn)

# 2. Instantiate model (training params are unused during generation;
#    they must match your training configuration for checkpoint compatibility).
model = MedGAN(
    dataset=sample_dataset,
    latent_dim=128,
    hidden_dim=128,
    batch_size=128,
    save_dir="./medgan_checkpoints/",
)

# 3. Load trained checkpoint
model.load_model("./medgan_checkpoints/best.pt")

# 4. Generate synthetic patients — each patient is a flat bag-of-codes (no visit structure)
synthetic = model.synthesize_dataset(num_samples=10000)
print(f"Generated {len(synthetic)} synthetic patients")
print(f"Example record: {synthetic[0]}")

# 5. Save to JSON
output_path = "synthetic_medgan_10k.json"
with open(output_path, "w") as f:
    json.dump(synthetic, f, indent=2)
print(f"Saved to {output_path}")
