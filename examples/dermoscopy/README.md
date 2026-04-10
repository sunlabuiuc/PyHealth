# Dermoscopy Melanoma Classification & Artifact Ablation

This folder contains a complete pipeline for training and evaluating melanoma classification models (ResNet50, Swin Transformer, DINOv2) on dermoscopy datasets (ISIC 2018, HAM10000, PH2).

It implements the artifact robustness and frequency ablation methodologies detailed in the CHIL 2025 paper: *A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations*.

## Directory Setup

To run these examples, you must have your data structured locally as follows:

    data/
    ├── isic2018/
    │   ├── images/ (contains metadata.csv and .jpg files)
    ├── ham10000/
    │   ├── images/ (contains metadata.csv and .jpg files)
    └── ph2/
        ├── PH2_dataset.txt
        └── PH2 Dataset images/

## Included Scripts

### 1. Model Training (`train_dermoscopy.py`)

Trains a baseline model on the combined ISIC 2018 and HAM10000 datasets using PyHealth's native `Trainer` and `TorchvisionModel` wrappers.

    python train_dermoscopy.py --model dinov2 --mode whole --data_dir /path/to/data --epochs 10

### 2. Trap Set Generation (`generate_artifact_data.py`)

Uses Stable Diffusion + LoRA to synthetically inject clinical artifacts (e.g., rulers, ink, gel bubbles) into the PH2 dataset to create out-of-distribution "Trap Sets".

    python generate_artifact_data.py --data_dir /path/to/data --lora_path /path/to/lora --artifact ruler

### 3. Artifact Robustness Evaluation (`evaluate_artifact_robustness.py`)

Evaluates a trained PyHealth model on the generated Trap Sets and outputs ROC-AUC curves and Confusion Matrices.

    python evaluate_artifact_robustness.py --model dinov2 --mode whole --weights ./output/dinov2_whole/model.pth --data_dir /path/to/data --artifact ruler

### 4. Epoch Ablation Study (`run_epoch_ablation.py`)

Automates the training and periodic evaluation of models across multiple epochs (e.g., 3, 5, 10, 20) to plot how artifact reliance changes over time.

    python run_epoch_ablation.py --model resnet50 --mode whole --data_dir /path/to/data --artifact ruler