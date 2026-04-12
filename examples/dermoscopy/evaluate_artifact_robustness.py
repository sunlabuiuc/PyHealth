# Contributor: [Your Name]
# NetID: [Your NetID]

"""
Artifact Robustness Evaluation Script.

Evaluates trained models against synthetic out-of-distribution artifact datasets (Trap Sets).
Supports dynamic targeting of base datasets (e.g., ph2, isic2018).
"""

import argparse
import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from pyhealth.datasets import get_dataloader
from pyhealth.datasets import DermoscopyDataset
from pyhealth.tasks import DermoscopyMelanomaClassification
from pyhealth.processors import DermoscopyImageProcessor
from pyhealth.models import TorchvisionModel, DINOv2

def load_weights(model, weights_path, device):
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict)
    model.to(device).eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Smart Artifact Evaluation")
    parser.add_argument('--model', type=str, choices=['resnet50', 'swin', 'dinov2'], default='resnet50')
    parser.add_argument('--mode', type=str, choices=[
        'whole', 'lesion', 'background', 'bbox', 'bbox70', 'bbox90', 
        'high_whole', 'high_lesion', 'high_background', 
        'low_whole', 'low_lesion', 'low_background'
    ], default='whole')
    parser.add_argument('--exp_dir', type=str, required=True, help="Path to the parent experiment directory")
    parser.add_argument('--strategy', type=str, choices=['master', 'ensemble', 'fold_average'], default='fold_average')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--eval_dataset', type=str, default='ph2', help="The base dataset the artifacts were generated on (e.g., ph2)")
    parser.add_argument('--artifact', type=str, required=True)
    args = parser.parse_args()

    # 1. Load the Trap Set dynamically based on the requested dataset
    dataset_target = f"{args.eval_dataset}_with_{args.artifact}"
    dataset = DermoscopyDataset(root=args.data_dir, dataset_name=dataset_target, dev=False)
    processor = DermoscopyImageProcessor(mode=args.mode)
    task_dataset = dataset.set_task(task=DermoscopyMelanomaClassification, input_processors={"image": processor})
    test_loader = get_dataloader(task_dataset, batch_size=32, shuffle=False)

    # 2. Initialize Base Architecture
    if args.model == "dinov2":
        base_model = DINOv2(dataset=task_dataset, model_size="vits14")
    else:
        model_name = "swin_t" if args.model == "swin" else "resnet50"
        base_model = TorchvisionModel(dataset=task_dataset, model_name=model_name, model_config={"weights": "DEFAULT"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Locate Weights based on Strategy
    weight_paths = []
    if args.strategy == "master":
        weight_paths.append(os.path.join(args.exp_dir, "master", "best_model.pth"))
    elif args.strategy in ["ensemble", "fold_average"]:
        for i in range(5): 
            weight_paths.append(os.path.join(args.exp_dir, f"fold_{i}", "best_model.pth"))

    print(f"[*] Evaluation Strategy: {args.strategy.upper()} on {dataset_target.upper()} (Loading {len(weight_paths)} models)")

    # 4. Perform Inference
    y_true_all = []
    y_prob_ensemble = None
    fold_aucs = []

    for i, w_path in enumerate(weight_paths):
        if not os.path.exists(w_path):
            raise FileNotFoundError(f"Could not find weights at {w_path}")
            
        model = load_weights(base_model, w_path, device)
        y_prob_current_model = []
        current_y_true = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                res = model(**batch)
                y_prob_current_model.extend(res['y_prob'].cpu().numpy())
                
                if i == 0: y_true_all.extend(res['y_true'].cpu().numpy())
                current_y_true.extend(res['y_true'].cpu().numpy())
        
        # 5a. Strategy: Fold Average (Paper baseline)
        if args.strategy == "fold_average":
            fold_auc = roc_auc_score(current_y_true, y_prob_current_model)
            fold_aucs.append(fold_auc)
            print(f"  -> Fold {i} ROC-AUC: {fold_auc:.4f}")
            
        # Accumulate probabilities for ensembling or master
        if y_prob_ensemble is None:
            y_prob_ensemble = np.array(y_prob_current_model)
        else:
            y_prob_ensemble += np.array(y_prob_current_model)

    # 5b. Output Final Results
    if args.strategy == "fold_average":
        final_auc = np.mean(fold_aucs)
        final_std = np.std(fold_aucs)
        print(f"\n[!] FINAL PAPER METRIC ({args.artifact}): {final_auc:.4f} ± {final_std:.4f}")
    elif args.strategy == "ensemble":
        y_prob_final = y_prob_ensemble / len(weight_paths)
        final_auc = roc_auc_score(y_true_all, y_prob_final)
        print(f"\n[!] FINAL ENSEMBLE ROC-AUC ({args.artifact}): {final_auc:.4f}")
    elif args.strategy == "master":
        final_auc = roc_auc_score(y_true_all, y_prob_ensemble)
        print(f"\n[!] FINAL MASTER ROC-AUC ({args.artifact}): {final_auc:.4f}")

if __name__ == "__main__":
    main()