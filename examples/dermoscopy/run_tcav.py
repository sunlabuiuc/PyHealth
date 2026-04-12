# Contributor: [Your Name]
# NetID: [Your NetID]

"""
Testing with Concept Activation Vectors (TCAV) for PyHealth.

Calculates how much a human-understandable concept (like "Rulers") influences 
the latent space representations of the model's predictions. Translates the 
authors' SGDClassifier logic into a dynamic PyTorch forward hook.
"""

import argparse
import os
import torch
import numpy as np
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

from pyhealth.datasets import DermoscopyDataset, get_dataloader
from pyhealth.tasks import DermoscopyMelanomaClassification
from pyhealth.processors import DermoscopyImageProcessor

# Import your PyHealth-wrapped models
from train_dermoscopy import MelanomaClassifier
from dinov2 import DINOv2

def load_weights(model, weights_path, device):
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict)
    model.to(device).eval()
    return model

def extract_activations(model, target_layer, dataloader, device):
    activations, labels = [], []
    def hook_fn(module, input, output):
        activations.append(input[0].view(input[0].size(0), -1).detach().cpu().numpy())
        
    handle = target_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Activations", leave=False):
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            model(**batch_gpu)
            labels.extend(batch['melanoma'].cpu().numpy())
            
    handle.remove()
    return np.concatenate(activations), np.array(labels)

def compute_tcav_score(cav_vector, target_activations):
    projections = np.dot(target_activations, cav_vector.T)
    return np.mean(projections > 0)

def main():
    parser = argparse.ArgumentParser(description="TCAV Explainable AI")
    parser.add_argument('--model', type=str, choices=['resnet50', 'swin', 'dinov2'], required=True)
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--artifact', type=str, required=True)
    args = parser.parse_args()

    print("="*60)
    print(f"TCAV ANALYSIS: {args.model.upper()} vs '{args.artifact.upper()}' CONCEPT")
    print("="*60)

    processor = DermoscopyImageProcessor(mode="whole")
    DermoscopyMelanomaClassification.output_schema = {} 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data (Concept, Random, and Domain)
    concept_ds = DermoscopyDataset(root=args.data_dir, dataset_name=f"ph2_with_{args.artifact}", dev=False)
    concept_task = concept_ds.set_task(task_fn=DermoscopyMelanomaClassification, input_processors={"image": processor})
    concept_loader = get_dataloader(concept_task, batch_size=32, shuffle=False)

    random_ds = DermoscopyDataset(root=args.data_dir, dataset_name="ph2", dev=False)
    random_task = random_ds.set_task(task_fn=DermoscopyMelanomaClassification, input_processors={"image": processor})
    random_loader = get_dataloader(random_task, batch_size=32, shuffle=False)

    domain_ds = DermoscopyDataset(root=args.data_dir, dataset_name="isic2018", dev=False)
    domain_task = domain_ds.set_task(task_fn=DermoscopyMelanomaClassification, input_processors={"image": processor})
    domain_loader = get_dataloader(domain_task, batch_size=32, shuffle=False)

    print(f"[*] Initializing {args.model.upper()}...")
    if args.model == "dinov2":
        base_model = DINOv2(dataset=domain_task, feature_keys=["image"], label_key="melanoma", mode="binary")
        target_layer = base_model.fc
    else:
        base_model = MelanomaClassifier(dataset=domain_task, feature_keys=["image"], label_key="melanoma", mode="binary", arch=args.model)
        target_layer = base_model.classifier

    weight_path = os.path.join(args.exp_dir, "fold_0", "best_model.pth")
    if not os.path.exists(weight_path):
        weight_path = os.path.join(args.exp_dir, "master", "best_model.pth") 
    model = load_weights(base_model, weight_path, device)

    print(f"[*] Extracting Latent Features from Classification Layer...")
    concept_acts, _ = extract_activations(model, target_layer, concept_loader, device)
    random_acts, _ = extract_activations(model, target_layer, random_loader, device)
    domain_acts, domain_labels = extract_activations(model, target_layer, domain_loader, device)

    print("[*] Training Concept Activation Vector (CAV)...")
    X = np.concatenate([concept_acts, random_acts])
    y = np.concatenate([np.ones(len(concept_acts)), np.zeros(len(random_acts))]) 
    
    X = X.reshape(X.shape[0], -1) 
    cav_classifier = SGDClassifier(alpha=0.01, max_iter=1000, random_state=42)
    cav_classifier.fit(X, y)
    cav_vector = cav_classifier.coef_

    benign_acts = domain_acts[domain_labels == 0]
    malignant_acts = domain_acts[domain_labels == 1]

    benign_tcav = compute_tcav_score(cav_vector, benign_acts.reshape(benign_acts.shape[0], -1))
    malignant_tcav = compute_tcav_score(cav_vector, malignant_acts.reshape(malignant_acts.shape[0], -1))

    print("\n" + "="*50)
    print(" TCAV INTERPRETABILITY RESULTS")
    print("="*50)
    print(f" Concept: {args.artifact.upper()}")
    print(f" -> Influence on BENIGN Predictions:    {benign_tcav * 100:.2f}%")
    print(f" -> Influence on MALIGNANT Predictions: {malignant_tcav * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()