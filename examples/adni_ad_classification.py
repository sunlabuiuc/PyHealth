"""Classification of Alzheimer's Disease using the ADNI Dataset.

Author: Bryan Lau (bryan16@illinois.edu)

This script executes the Pyhealth pipeline implementation of the method 
described in the paper "On the Design of Convolutional  Neural Networks 
for Automatic Detection of Alzheimer's Disease" by Liu et al.
(https://arxiv.org/abs/1911.03740).

The pipeline consists of:

- ADNIDataset:
    Pyhealth-compatible dataset for ADNI data

- AlzheimersDiseaseClassification:
    Task to present the necessary features

- NIftIImageProcessor:
    Image processor to load and process NIftI files

- AlzheimersDiseaseCNN:
    Model that replicates the structure described by Liu et al.

This script has been tested against 3182 real MRI brain scan image files 
downloaded from the ADNI dataset, hosted at the Image & Data Archive (IDA) 
at the Laboratory of Neuro Imaging (LONI). Users may apply for access at:

https://adni.loni.usc.edu/data-samples/adni-data/

The required pre-processing parameters are:

- Multiplanar reconstruction (MPR)
- Gradient warping correction (GradWarp)
- B1 non-uniformity correction
- N3 intensity normalization

This implementation aligns with the method described by Liu et al. except 
that the following are omitted for the sake of simplicity:

- Image pre-processing using Clinica (e.g. Dartel template registration)
- Training augmentation, i.e. gaussian blurring and random cropping

"""
import os
import shutil

from pathlib import Path
from torch.optim import SGD

from pyhealth.datasets import ADNIDataset, get_dataloader
from pyhealth.datasets.splitter import split_by_patient  # New splitter
from pyhealth.models import AlzheimersDiseaseCNN
from pyhealth.tasks import AlzheimersDiseaseClassification
from pyhealth.trainer import Trainer

from adni_ad_synthetic_data import create_adni_image

# Initialization
BATCH_SIZE = 4
EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.9
METRICS = ["balanced_accuracy", "accuracy", "f1_macro", "roc_auc_macro_ovr"]
MONITOR = "balanced_accuracy"
NUM_SYNTHETIC_SAMPLES = 30
NUM_WORKERS = 4
SEED = 99

# Path where the ADNI files are located
ADNI_ROOT = "./adni_root"
CACHE_DIR = "./cache"

# Set this flag to:
#   True to generate and use synthetic data
#   False to use real ADNI images that you have downloaded
USE_SYNTHETIC_DATA = True

if __name__ == '__main__':

    # Convert paths
    adni_path = Path(ADNI_ROOT)
    cache_path = Path(CACHE_DIR)

    # Generate synthetic ADNI data
    if USE_SYNTHETIC_DATA:
        adni_path = Path("./adni_synthetic")
        cache_path = Path("./cache_synthetic")

        for path in [adni_path, cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
        
        for i in range(NUM_SYNTHETIC_SAMPLES):
            subject_id = f"002_S_{i:04d}"
            if i < int(NUM_SYNTHETIC_SAMPLES * 0.33):
                group = "CN"
            elif i < int(NUM_SYNTHETIC_SAMPLES * 0.66):
                group = "MCI"
            else:
                group = "AD"
            print(f"Generating synthetic image for {subject_id} ({group})")
            create_adni_image(adni_path, subject_id, group)

    # Instantiate base ADNI dataset
    adni_dataset = ADNIDataset(root=str(adni_path), cache_dir=str(cache_path), dev=False, num_workers=NUM_WORKERS)
    adni_dataset.stats()

    # Set task and obtain samples
    adni_task = AlzheimersDiseaseClassification()
    sample_dataset = adni_dataset.set_task(adni_task)

    # Split data by patient into train/val/test (70/15/15)
    split_ratios = [0.7, 0.15, 0.15]
    train_data, val_data, test_data = split_by_patient(
        sample_dataset, ratios=split_ratios, seed=SEED)

    # Create dataloaders
    train_loader = get_dataloader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = get_dataloader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Instantiate model using samples
    model = AlzheimersDiseaseCNN(
        dataset=sample_dataset, 
        width_factor=4, use_age=True, use_gender=True, norm_method="instance"
    )

    # Instantiate trainer
    trainer = Trainer(
        model=model,
        metrics=METRICS,
        output_path="./output"
    )

    # Train the model
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        epochs=EPOCHS,
        optimizer_class=SGD,
        optimizer_params={"lr": LEARNING_RATE, "momentum": MOMENTUM, "weight_decay": 1e-3},
        max_grad_norm=1.0,
        monitor=MONITOR,
        monitor_criterion="max",
        patience=PATIENCE,
        load_best_model_at_last=True,
    )

    # Evaluate
    scores = trainer.evaluate(test_loader)
    print(f"\nTest scores: {scores}")
