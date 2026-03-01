"""PromptEHR: Synthetic MIMIC-III Patient Generation.

Load a trained PromptEHR checkpoint and generate synthetic patients.

Reference:
    Wang et al. "PromptEHR: Conditional Electronic Healthcare Records
    Generation with Prompt Learning." EMNLP 2023.
    https://arxiv.org/abs/2211.01761
"""

import json

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.models import PromptEHR
from pyhealth.tasks import promptehr_generation_mimic3_fn

MIMIC3_ROOT = "/srv/local/data/physionet.org/files/mimiciii/1.4"
CHECKPOINT_PATH = "./save/promptehr/checkpoint.pt"
OUTPUT_PATH = "./save/promptehr/synthetic_patients.json"
NUM_SAMPLES = 10_000

# 1. Load dataset + apply task (needed for processor/vocab reconstruction)
dataset = MIMIC3Dataset(
    root=MIMIC3_ROOT,
    tables=["patients", "admissions", "diagnoses_icd"],
    code_mapping={},
)
sample_dataset = dataset.set_task(promptehr_generation_mimic3_fn)

# 2. Load checkpoint
model = PromptEHR(dataset=sample_dataset)
model.load_model(CHECKPOINT_PATH)
print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

# 3. Generate
print(f"Generating {NUM_SAMPLES} synthetic patients...")
synthetic = model.synthesize_dataset(num_samples=NUM_SAMPLES)
print(f"Generated {len(synthetic)} patients")

# 4. Save
with open(OUTPUT_PATH, "w") as f:
    json.dump(synthetic, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")

# Summary stats
avg_visits = sum(len(p["visits"]) for p in synthetic) / len(synthetic)
print(f"Average visits per patient: {avg_visits:.2f}")
