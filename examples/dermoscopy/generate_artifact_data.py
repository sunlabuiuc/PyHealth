# Contributor: [Your Name]
# NetID: [Your NetID]

"""
Synthetic Artifact Generation for Dermoscopy Trap Sets.

This script uses Stable Diffusion and LoRA (Low-Rank Adaptation) to synthetically 
inject clinical artifacts (e.g., rulers, ink, gel bubbles) into the PH2 dataset, 
creating out-of-distribution "Trap Sets".

Paper: "A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations" (CHIL 2025)
Stable Diffusion Reference: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022)
"""

import os
import argparse
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionInpaintPipeline

# PyHealth Imports
from pyhealth.datasets import DermoscopyDataset

# Maps human-readable artifact names to specific LoRA trigger tokens used in the CHIL 2025 study
ARTIFACT_MAP = {
    "patches": "olis",
    "dark_corner": "lun",
    "ruler": "dits",
    "ink": "httr",
    "gel_bubble": "sown"
}

def main():
    """Parses arguments, loads the PH2 dataset, and applies diffusion-based artifact injection."""
    parser = argparse.ArgumentParser(description="Generate synthetic dermoscopy artifacts using Stable Diffusion.")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the root data directory containing the 'ph2' folder.")
    parser.add_argument('--lora_path', type=str, required=True, help="Path to the pre-trained LoRA weights for artifact generation.")
    parser.add_argument('--artifact', type=str, choices=list(ARTIFACT_MAP.keys()), required=True, help="The type of artifact to synthetically inject.")
    args = parser.parse_args()

    token = ARTIFACT_MAP[args.artifact]

    print("="*60)
    print(f"PHASE 1: Generating Synthetic Artifacts (PH2 Trap Set: {args.artifact.upper()})")
    print("="*60)

    # 1. Load clean PH2 data using PyHealth's native dataset loader
    dataset = DermoscopyDataset(
        root=args.data_dir, 
        dataset_name="ph2", 
        dev=False
    )
    
    # 2. Setup output directories for the new Trap Set
    output_dir = os.path.join(args.data_dir, f"ph2_with_{args.artifact}", "images")
    mask_dir = os.path.join(args.data_dir, f"ph2_with_{args.artifact}", "masks")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # 3. Initialize Stable Diffusion Inpainting Pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    pipeline.load_lora_weights(args.lora_path)

    print(f"[*] Processing {len(dataset.patients)} images...")
    
    records = []
    for patient_id, patient in tqdm(dataset.patients.items()):
        event = patient.get_events("dermoscopy")[0]
        
        img_path = event.attr_dict["image_path"]
        mask_path = event.attr_dict["mask_path"]
        label = int(float(event.attr_dict["label"]))
        
        # Load and resize images to match Stable Diffusion's native 512x512 resolution
        raw_img = Image.open(img_path).convert("RGB").resize((512, 512))
        
        # Invert the lesion mask so Diffusion draws in the BACKGROUND (around the lesion), not on it
        binary_mask = Image.open(mask_path).convert("L").resize((512, 512))
        inverted_mask = Image.eval(binary_mask, lambda x: 255 - x)

        # Generate the artifact using the specific LoRA trigger token
        prompt = f"a dermoscopic image of {token} benign"
        augmented_img = pipeline(
            prompt=prompt,
            image=raw_img,
            mask_image=inverted_mask,
            strength=0.8,
            guidance_scale=10.0
        ).images[0]
        
        new_img_name = f"{patient_id}_artifact.jpg"
        new_img_path = os.path.join(output_dir, new_img_name)
        augmented_img.save(new_img_path)
        
        # Track metadata for PyHealth's dataset indexer
        records.append({
            "isic_id": f"{patient_id}_artifact",
            "diagnosis_1": "Malignant" if label == 1 else "Benign"
        })

    # Export the standard metadata.csv expected by PyHealth
    csv_path = os.path.join(output_dir, "metadata.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"[*] Generation complete! Dataset saved to {output_dir}")

if __name__ == "__main__":
    main()