"""
Fairness metric testing of PyHealth RNN readmission prediction model
using the Synthetic MIMIC-III dataset. Synthetic sensitive attributes are 
used for baseline auditing purposes and to align with the goals of FAMEWS.
"""
#!pip install fairlearn
#!pip install pyhealth

import numpy as np

from pyhealth.datasets import MIMIC3Dataset, SampleDataset, get_dataloader
from pyhealth.tasks.readmission_prediction import readmission_prediction_mimic3_fn
from pyhealth.models import RNN
from pyhealth.trainer import Trainer
from sklearn.model_selection import train_test_split


dataset = MIMIC3Dataset(
    root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
    tables=["diagnoses_icd", "procedures_icd", "prescriptions"]
)

sample_dataset = dataset.set_task(readmission_prediction_mimic3_fn)

# Manual split for version compatibility
samples = sample_dataset.samples

train_samples, temp_samples = train_test_split(
    samples, test_size=0.3, random_state=42
)
val_samples, test_samples = train_test_split(
    temp_samples, test_size=0.67, random_state=42
)

train_dataset = SampleDataset(train_samples)
val_dataset = SampleDataset(val_samples)
test_dataset = SampleDataset(test_samples)
feature_keys = ["conditions", "procedures", "drugs"]

# Model
model = RNN(
    dataset=train_dataset,
    feature_keys=feature_keys,
    label_key="label",
    mode="binary"
)

train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
val_loader   = get_dataloader(val_dataset, batch_size=64, shuffle=False)
test_loader  = get_dataloader(test_dataset, batch_size=64, shuffle=False)

trainer = Trainer(model=model, metrics = ["accuracy"])

#Train
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=2,
    optimizer_params={"lr": 1e-3},
    load_best_model_at_last=False
)
metrics = trainer.evaluate(test_loader)

# Fairlearn fairness auditing
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score

y_true, y_prob, _ = trainer.inference(test_loader)
y_true = np.array(y_true)
y_prob = np.array(y_prob)

# Synthetic sensitive attributes
sensitive = (np.arange(len(y_true)) % 2)

# Prediction bias injection
y_prob_biased = y_prob.copy()
y_prob_biased[sensitive == 1] += 0.2 
y_prob_biased = np.clip(y_prob_biased, 0, 1)
y_pred_biased = (y_prob_biased > 0.5).astype(int)

acc = accuracy_score(y_true, y_pred_biased)
parity = demographic_parity_difference(
    y_true, y_pred_biased, sensitive_features=sensitive
)
eOdds = equalized_odds_difference(
    y_true, y_pred_biased, sensitive_features=sensitive
)

print("\nFairness Auditing Results")
print("Accuracy:", acc) 
print("Demographic Parity Difference:", parity) 
print("Equalized Odds Difference:", eOdds) 