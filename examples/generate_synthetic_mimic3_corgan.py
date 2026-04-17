"""Generate synthetic MIMIC-III patient records using a trained CorGAN checkpoint."""
import json

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import corgan_generation_mimic3_fn
from pyhealth.models.generators.corgan import CorGAN

# Update this to your local MIMIC-III path before running
MIMIC3_ROOT = "/path/to/mimic3"

# 1. Reconstruct dataset — required to initialise CorGAN's vocabulary from the processor.
# If you trained with different tables=, update this to match exactly.
base_dataset = MIMIC3Dataset(
    root=MIMIC3_ROOT,
    tables=["diagnoses_icd"],
)
sample_dataset = base_dataset.set_task(corgan_generation_mimic3_fn)

# 2. Instantiate model (epochs and training params are unused during generation;
#    they must match your training configuration for checkpoint compatibility).
model = CorGAN(
    dataset=sample_dataset,
    latent_dim=128,
    hidden_dim=128,
    batch_size=128,
    epochs=50,
    save_dir="./corgan_checkpoints/",
)

# 3. Load trained checkpoint
model.load_model("./corgan_checkpoints/corgan_final.pt")

# 4. Generate synthetic patients — each patient is a flat bag-of-codes (no visit structure)
synthetic = model.synthesize_dataset(num_samples=10000)
print(f"Generated {len(synthetic)} synthetic patients")
print(f"Example record: {synthetic[0]}")

# 5. Save to JSON
output_path = "synthetic_corgan_10k.json"
with open(output_path, "w") as f:
    json.dump(synthetic, f, indent=2)
print(f"Saved to {output_path}")
