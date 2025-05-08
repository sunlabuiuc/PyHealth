# src/experiments/pyhealth_mortality.py

import os
import torch
from torch.utils.data import DataLoader

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import InHospitalMortality
from pyhealth.models import RNN
from pyhealth.pipeline import Pipeline

def main():
    # load raw MIMIC-III data
    mimic3 = MIMIC3Dataset(
        root_dir = "src/data/mimic3",        # your local MIMIC-III folder
        tables   = ["ADMISSIONS", "CHARTEVENTS", "LABEVENTS", "DIAGNOSES_ICD"],
        dev      = False                     # use full data
    )

    # in-hospital mortality task
    #   time_window="all" uses all history up to discharge,
    #   min_length=24*60 filters out stays shorter than 24h
    task = InHospitalMortality(
        dataset    = mimic3,
        time_window= "all",
        min_length = 24 * 60
    )

    # get train/val splits
    train_set, val_set, test_set = task.get_splits(
        ratios=(0.7, 0.1, 0.2),
        seed=42
    )

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False, num_workers=4)

    # make an rnn model
    model = RNN(
        input_dim  = task.input_dim,   # automatically inferred
        hidden_dim = 128,
        output_dim = task.output_dim,  # automatically 1 for mortality
        num_layers = 1
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # train and evaluate the model
    pipe = Pipeline(
        model  = model,
        task   = task,
        device = "cuda" if torch.cuda.is_available() else "cpu"
    )

    pipe.fit(
        train_loader,
        eval_loader = val_loader,
        epochs      = 5
    )

    # final test
    metrics = pipe.evaluate(test_loader)
    print("\nPyHealth In-Hospital Mortality Results")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.4f}")

if __name__ == "__main__":
    main()
