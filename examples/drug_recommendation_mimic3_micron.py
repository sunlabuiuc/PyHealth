from pathlib import Path
import torch

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import MICRON
from pyhealth.trainer import Trainer
from pyhealth.metrics import (
    binary_jaccard_score,
    binary_f1_score,
    binary_precision_recall_curve_auc,
    ddi_rate_score
)

# STEP 1: load data and define the schemas for PyHealth 2.0
base_path = Path("/srv/local/data/physionet.org/files/mimiciii/1.4")  # Update this path
base_dataset = MIMIC3Dataset(
    root=base_path,
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    code_mapping={"NDC": ("ATC", {"level": 3})},  # Map to ATC level 3 codes
    dev=True,  # Set to False for full dataset
)
base_dataset.stat()

# STEP 2: set task and create dataloaders
# Define the schemas for PyHealth 2.0
input_schema = {
    "conditions": "sequence",  # Each visit has a sequence of diagnoses
    "procedures": "sequence",  # Each visit has a sequence of procedures
}
output_schema = {
    "drugs": "multilabel"  # Multi-hot encoded drug prescriptions
}

# Create dataset with schemas
sample_dataset = base_dataset.set_task(
    schema={
        "inputs": input_schema,
        "outputs": output_schema,
    }
)
sample_dataset.stat()

train_dataset, val_dataset, test_dataset = split_by_patient(
    sample_dataset, [0.8, 0.1, 0.1]
)

train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

# STEP 3: define model with PyHealth 2.0 compatible parameters
model = MICRON(
    dataset=sample_dataset,
    embedding_dim=128,  # Dimension for feature embeddings
    hidden_dim=128,    # Dimension for hidden layers
    lam=0.1,          # Weight for reconstruction loss
)


# STEP 4: define trainer with appropriate metrics and train the model

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

metrics = {
    "jaccard": binary_jaccard_score,
    "f1": binary_f1_score,
    "pr_auc": binary_precision_recall_curve_auc,
    "ddi_rate": ddi_rate_score,
}

trainer = Trainer(
    model=model,
    metrics=metrics,
    device=device,
)

trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=5,
    monitor="pr_auc",  # Metric to monitor for early stopping
    monitor_criterion="max",  # We want to maximize PR-AUC
)

# STEP 5: evaluate on test set
test_metrics = trainer.evaluate(test_dataloader)
print("\nTest Set Metrics:")
for metric_name, value in test_metrics.items():
    print(f"{metric_name}: {value:.4f}")

# Optional: Save model and results
# torch.save(model.state_dict(), "micron_model.pt")
# with open("test_results.json", "w") as f:
#     json.dump(test_metrics, f, indent=2)
