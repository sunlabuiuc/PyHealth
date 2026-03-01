"""PromptEHR: Training on MIMIC-III.

Train PromptEHR for synthetic EHR generation using PyHealth 2.0 API.

Reference:
    Wang et al. "PromptEHR: Conditional Electronic Health Records Generation
    with Prompt Learning." CHIL 2023.
"""

from pyhealth.datasets import MIMIC3Dataset, split_by_patient
from pyhealth.models import PromptEHR
from pyhealth.tasks import promptehr_generation_mimic3_fn

MIMIC3_ROOT = "/srv/local/data/physionet.org/files/mimiciii/1.4"

# 1. Load MIMIC-III
dataset = MIMIC3Dataset(
    root=MIMIC3_ROOT,
    tables=["patients", "admissions", "diagnoses_icd"],
    code_mapping={},
)

# 2. Apply generation task
sample_dataset = dataset.set_task(promptehr_generation_mimic3_fn)
print(f"Patients: {len(sample_dataset)}")
sample_dataset.stat()

# 3. Split
train, val, test = split_by_patient(sample_dataset, [0.8, 0.1, 0.1])

# 4. Initialize model
model = PromptEHR(
    dataset=sample_dataset,
    n_num_features=1,
    cat_cardinalities=[2],
    d_hidden=128,
    prompt_length=1,
    epochs=20,
    batch_size=16,
    lr=1e-5,
    warmup_steps=1000,
    save_dir="./save/promptehr/",
)

# 5. Train
model.train_model(train, val)
print("Training complete. Checkpoint saved to ./save/promptehr/")
