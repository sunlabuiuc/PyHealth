# Contributor: [Your Name]
# NetID: [Your NetID]

"""
Testing with Concept Activation Vectors (TCAV) for PyHealth.

Calculates how much a human-understandable concept (like "Rulers") influences 
the latent space representations of the model's predictions. Translates the 
authors' SGDClassifier logic into a dynamic PyTorch forward hook.

Reference: Kim, B., et al. (2018). Interpretability Beyond Feature Attribution: 
Quantitative Testing with Concept Activation Vectors (TCAV). ICML.
"""

import argparse
import os
import torch
import numpy as np
import logging
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

from pyhealth.datasets import get_dataloader

from pyhealth.datasets import DermoscopyDataset
from pyhealth.tasks import DermoscopyMelanomaClassification
from pyhealth.processors import DermoscopyImageProcessor
from pyhealth.models import DINOv2
from train_dermoscopy import MelanomaClassifier, setup_dynamic_logging

def load_weights(model, weights_path, device):
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict)
    model.to(device).eval()
    return model

def extract_activations(model, target_layer, dataloader, device):
    """
        Registers a forward hook to extract latent representations from a specific layer.

        Args:
            model (nn.Module): The trained PyTorch model.
            target_layer (nn.Module): The specific layer to attach the forward hook to (e.g., model.fc).
            dataloader (DataLoader): PyHealth dataloader containing the target images.
            device (torch.device): Compute device (CPU or CUDA).

        Returns:
            tuple: A tuple containing:
                - activations (np.ndarray): Flattened latent feature vectors.
                - labels (np.ndarray): Ground truth binary labels for the dataset.
    """
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

def main():
    parser = argparse.ArgumentParser(description="TCAV")
    parser.add_argument('--model', type=str, choices=['resnet50', 'swin', 'dinov2'], required=True)
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default=None, help="Parent log directory to save session output logs (defaults to dermoscopy_logs in home directory)")
    parser.add_argument('--train_datasets', nargs='+', default=['isic2018'])
    parser.add_argument('--eval_dataset', type=str, default='ph2', help="The base dataset to evaluate concepts on")
    parser.add_argument('--artifact', type=str, default=None, help="Optional diffusion artifact to use as the concept.")
    args = parser.parse_args()

    if args.artifact:
        concept_target = f"{args.eval_dataset}_with_{args.artifact}"
        run_details = f"{args.model}_{args.eval_dataset}_with_{args.artifact}"
    else:
        concept_target = args.eval_dataset
        run_details = f"{args.model}_{args.eval_dataset}_baseline_sanity_check"
        print("[!] WARNING: No artifact provided. TCAV will compare the eval dataset against itself as a baseline sanity check.")

    # Start Dynamic Logging
    # Strip PyHealth's redundant default console handlers so only custom logger is used for the session logs
    logging.getLogger("pyhealth").handlers.clear()
    setup_dynamic_logging(args.log_dir, "tcav", run_details)

    processor = DermoscopyImageProcessor(mode="whole")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[*] Loading Dataset for TCAV...")
    # TCAV requires the Train sets, the Base Eval set, and the Concept Target/Trap Set
    # dict.fromkeys automatically handles the duplicates if concept_target == args.eval_dataset
    all_datasets = list(dict.fromkeys(args.train_datasets + [args.eval_dataset, concept_target]))
    dynamic_dataset_name = "_".join(all_datasets)

    dataset = DermoscopyDataset(
        root=args.data_dir, 
        datasets=all_datasets, 
        dataset_name=dynamic_dataset_name,
        cache_dir=os.path.join(args.data_dir, ".cache")
    )
    
    # Extract specific datasets using Task class filter
    dataset.task = None
    concept_task = dataset.set_task(DermoscopyMelanomaClassification([concept_target]), input_processors={"image": processor})
    concept_loader = get_dataloader(concept_task, batch_size=32, shuffle=False)

    dataset.task = None
    random_task = dataset.set_task(DermoscopyMelanomaClassification([args.eval_dataset]), input_processors={"image": processor})
    random_loader = get_dataloader(random_task, batch_size=32, shuffle=False)

    dataset.task = None
    domain_task = dataset.set_task(DermoscopyMelanomaClassification(args.train_datasets), input_processors={"image": processor})
    domain_loader = get_dataloader(domain_task, batch_size=32, shuffle=False)

    if args.model == "dinov2":
        base_model = DINOv2(dataset=dataset, feature_keys=["image"], label_key="melanoma", mode="binary")
        target_layer = base_model.fc
    else:
        base_model = MelanomaClassifier(dataset=dataset, feature_keys=["image"], label_key="melanoma", mode="binary", arch=args.model)
        target_layer = base_model.model.fc if hasattr(base_model, 'model') else base_model.classifier

    weight_path = os.path.join(args.exp_dir, "fold_0", "best.ckpt")
    if not os.path.exists(weight_path): weight_path = os.path.join(args.exp_dir, "master", "best.ckpt") 
    model = load_weights(base_model, weight_path, device)

    print(f"[*] Extracting Latent Features from {target_layer.__class__.__name__}...")
    concept_acts, _ = extract_activations(model, target_layer, concept_loader, device)
    random_acts, _ = extract_activations(model, target_layer, random_loader, device)
    domain_acts, domain_labels = extract_activations(model, target_layer, domain_loader, device)

    print("[*] Training CAV...")
    X = np.concatenate([concept_acts, random_acts])
    y = np.concatenate([np.ones(len(concept_acts)), np.zeros(len(random_acts))]) 
    
    cav_classifier = SGDClassifier(alpha=0.01, max_iter=1000, random_state=42)
    cav_classifier.fit(X.reshape(X.shape[0], -1), y)
    cav_vector = cav_classifier.coef_

    benign_acts = domain_acts[domain_labels == 0].reshape(-1, X.shape[1])
    malignant_acts = domain_acts[domain_labels == 1].reshape(-1, X.shape[1])

    b_score = np.mean(np.dot(benign_acts, cav_vector.T) > 0)
    m_score = np.mean(np.dot(malignant_acts, cav_vector.T) > 0)

    print(f"\n[!] TCAV INFLUENCE ({args.artifact.upper()}):")
    print(f"    BENIGN:    {b_score * 100:.2f}%")
    print(f"    MALIGNANT: {m_score * 100:.2f}%")

if __name__ == "__main__":
    main()