# Contributor: [Your Name]
# NetID: [Your NetID]

"""
Artifact Robustness Evaluation for Dermoscopy Models.

Evaluates PyHealth vision models trained on clean data against the synthetically 
corrupted PH2 "Trap Sets" to measure out-of-distribution reliance on clinical artifacts.

Paper: "A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations" (CHIL 2025)
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

from pyhealth.datasets import get_dataloader

# Native PyHealth Dermoscopy Imports
from pyhealth.datasets import DermoscopyDataset
from pyhealth.tasks import DermoscopyMelanomaClassification
from pyhealth.processors import DermoscopyImageProcessor
from pyhealth.models import TorchvisionModel, DINOv2

def generate_visualizations(y_true, y_prob, model_name, mode_name, output_dir, artifact_name):
    """
    Generates and exports the ROC Curve and Confusion Matrix for the evaluation run.
    
    Args:
        y_true (list): Ground truth labels.
        y_prob (list): Predicted probabilities for the positive class.
        model_name (str): The architecture evaluated.
        mode_name (str): The frequency mode used.
        output_dir (str): Directory to save the output plot.
        artifact_name (str): The specific trap set evaluated.
    """
    y_pred = (np.array(y_prob) > 0.5).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc(fpr, tpr):.3f}')
    axes[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes[0].set(xlabel='False Positive Rate', ylabel='True Positive Rate', title=f'ROC Curve ({artifact_name.upper()})')
    axes[0].legend()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set(xlabel='Predicted Label', ylabel='True Label', title='Confusion Matrix')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"results_{model_name}_{mode_name}_{artifact_name}_trap.png"), dpi=300)
    print(f"[*] Visualizations saved to {output_dir}")

def main():
    """Loads a pretrained model and performs inference on the designated Trap Set."""
    parser = argparse.ArgumentParser(description="Evaluate a model on an artifact-injected Trap Set.")
    parser.add_argument('--model', type=str, choices=['resnet50', 'swin', 'dinov2'], required=True, help="Architecture of the loaded model.")
    parser.add_argument('--mode', type=str, choices=['whole', 'high_whole', 'low_whole'], required=True, help="Frequency processor mode.")
    parser.add_argument('--weights', type=str, required=True, help="Path to the trained .pth weights file.")
    parser.add_argument('--data_dir', type=str, required=True, help="Root directory containing the Trap Sets.")
    parser.add_argument('--artifact', type=str, choices=['patches', 'dark_corner', 'ruler', 'ink', 'gel_bubble'], required=True, help="The specific Trap Set to evaluate against.")
    args = parser.parse_args()

    # 1. Load the specific PH2 Trap Set
    dataset = DermoscopyDataset(root=args.data_dir, dataset_name=f"ph2_with_{args.artifact}", dev=False)
    processor = DermoscopyImageProcessor(mode=args.mode)
    task_dataset = dataset.set_task(task=DermoscopyMelanomaClassification, input_processors={"image": processor})
    test_loader = get_dataloader(task_dataset, batch_size=32, shuffle=False)

    # 2. Initialize Model Architecture 
    if args.model == "dinov2":
        model = DINOv2(dataset=task_dataset, model_size="vits14")
    else:
        model_name = "swin_t" if args.model == "swin" else "resnet50"

        model = TorchvisionModel(
            dataset=task_dataset, 
            model_name=model_name,
            model_config={"weights": "DEFAULT"}
        )

    # 3. Load Trained Weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(args.weights, map_location=device, weights_only=True)
    # Handle PyHealth Trainer dictionary wrapping
    model.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict)
    model.to(device).eval()

    # 4. Perform Inference
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(**batch)
            y_true.extend(outputs["y_true"].cpu().numpy())
            y_prob.extend(outputs["y_prob"].cpu().numpy())

    out_dir = os.path.dirname(args.weights)
    generate_visualizations(y_true, y_prob, args.model, args.mode, out_dir, args.artifact)

if __name__ == "__main__":
    main()