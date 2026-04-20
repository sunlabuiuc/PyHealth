# Contributor: [Your Name]
# NetID: [Your NetID]

"""
Synthetic Artifact Generation for Dermoscopy Trap Sets.

This script utilizes Stable Diffusion Inpainting with LoRA (Low-Rank Adaptation) 
to synthetically inject clinical artifacts (e.g., rulers, ink, gel bubbles) into 
datasets, recreating the 'Trap Sets' used in Jin (2025).

Paper: "A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations" (CHIL 2025)
Citation here
Reference: Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR. 
(Model: runwayml/stable-diffusion-inpainting).

- Fully integrated with DermoscopyDataset for automatic metadata caching and path resolution.
- Automatically generates the 'metadata.csv' required by `_prepare_trap_set` in DermoscopyDataset 
  to ensure seamless loading during model evaluation.
"""

import os
import argparse
import pandas as pd
import torch
import logging
from PIL import Image
from pathlib import Path
from diffusers import StableDiffusionInpaintPipeline

from pyhealth.datasets import DermoscopyDataset
from pyhealth.tasks import DermoscopyMelanomaClassification

from train_dermoscopy import setup_dynamic_logging

ARTIFACT_MAP = {
    "dark-corner": "lun",
    "gel-bubble": "sown",
    "ink": "httr",
    "patches": "olis",
    "ruler": "dits"
}

# The authors' finely-tuned strength parameters for masked blending
STRENGTH_MAP = {
    "dark-corner": 0.75, 
    "gel-bubble": 0.60, 
    "ink": 0.70, 
    "patches": 0.85, 
    "ruler": 0.65
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the parent dermoscopy_data directory")
    parser.add_argument('--source_dataset', type=str, default="ph2", help="Which base dataset to inject artifacts into (e.g., ph2)")
    parser.add_argument('--artifact_type', type=str, choices=list(ARTIFACT_MAP.keys()), required=True)
    parser.add_argument('--log_dir', type=str, default=None, help="Parent log directory to save session output logs (defaults to dermoscopy_logs in home directory)")
    parser.add_argument('--lora_weights_dir', type=str, default=None, help="Parent directory of trained LoRAs  (defaults to artifact_loras_weights in home directory)")
    args = parser.parse_args()

    run_details = f"{args.source_dataset}_with_{args.artifact_type}"
    # START DYNAMIC LOGGING
    # Strip PyHealth's redundant default console handlers so only custom logger is used for the session logs
    logging.getLogger("pyhealth").handlers.clear()
    setup_dynamic_logging(args.log_dir, "generate_artifacts", run_details)

    if args.lora_weights_dir is None:
        lora_weights_dir = Path.home() / "artifact_lora_weights"
    else:
        lora_weights_dir = Path(args.lora_weights_dir)

    lora_weights_dir = str(lora_weights_dir)

    lora_path = os.path.join(lora_weights_dir, args.artifact_type, "pytorch_lora_weights.safetensors")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA weights not found at {lora_path}. Did you run the bash script?")

    token = ARTIFACT_MAP[args.artifact_type]
    strength = STRENGTH_MAP[args.artifact_type]
    
    # Read the unified CSV directly with Pandas
    # We DO NOT initialize the PyHealth dataset here to prevent it from overwriting the master CSV.
    # To build dataset first, can run train_dermoscopy.py or run this command (with data directory absolute path):
    # python -c "import os; from pyhealth.datasets import DermoscopyDataset; DermoscopyDataset(root='path/to/data', cache_dir=os.path.join('path/to/data', '.cache'))"
    csv_path = os.path.join(args.data_dir, "dermoscopy-metadata-pyhealth.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Master metadata CSV not found at {csv_path}. Please initialize the dataset once to build it.")
        
    df_meta = pd.read_csv(csv_path)
    
    # Filter only the rows for the dataset we are currently injecting (e.g., ph2)
    target_df = df_meta[df_meta["source_dataset"] == args.source_dataset]

    if len(target_df) == 0:
        raise ValueError(f"No records found for dataset '{args.source_dataset}' in the master CSV.")

    # Setup the output directory structure specifically for dermoscopy.py's `_prepare_trap_set`
    trap_set_name = f"{args.source_dataset}_with_{args.artifact_type}"
    output_img_dir = os.path.join(args.data_dir, trap_set_name, "images")
    os.makedirs(output_img_dir, exist_ok=True)

    # Initialize Stable Diffusion 1.5 Inpainting Pipeline
    print(f"[*] Loading Stable Diffusion v1.5 with {args.artifact_type} LoRA...")
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    
    pipeline.load_lora_weights(lora_path)

    print(f"[*] Generating Trap Set '{trap_set_name}' ({len(target_df)} images) at Strength {strength}...")

    # Data tracking to build the PyHealth-compatible metadata.csv
    generated_metadata = []

    # 4. Generate the Trap Set by iterating through the filtered DataFrame
    for _, row in target_df.iterrows():
        patient_id = row["patient_id"]
        img_path = row["image_path"]
        mask_path = row["mask_path"]
        label = row["label"] # 0 for benign, 1 for malignant
        
        diagnosis = "malignant" if label == 1 else "benign"
        
        if not os.path.exists(mask_path):
            print(f"[-] Missing mask for {patient_id}, skipping...")
            continue

        # Load and resize images to SD 1.5 standard (512x512)
        raw_img = Image.open(img_path).convert("RGB").resize((512, 512))
        inverted_mask = Image.eval(Image.open(mask_path).convert("L").resize((512, 512)), lambda x: 255 - x)

        prompt = f"a dermoscopic image of {token} {diagnosis}"
        
        # Run Inpainting
        augmented_img = pipeline(
            prompt=prompt, 
            image=raw_img, 
            mask_image=inverted_mask, 
            strength=strength, 
            guidance_scale=10.0,
            num_inference_steps=50
        ).images[0]

        # Save Image
        # If original patient_id from PH2 is "ph2_IMD002", we strip the prefix so it aligns cleanly
        clean_id = patient_id.replace(f"{args.source_dataset}_", "")
        new_img_name = f"{clean_id}.jpg"
        augmented_img.save(os.path.join(output_img_dir, new_img_name))

        # Track metadata for PyHealth
        generated_metadata.append({
            "image_id": clean_id,
            "diagnosis_1": "Malignant" if label == 1 else "Benign"
        })

    # Save the metadata.csv directly into the images folder for `_prepare_trap_set` to find
    df_meta = pd.DataFrame(generated_metadata)
    csv_out_path = os.path.join(output_img_dir, "metadata.csv")
    df_meta.to_csv(csv_out_path, index=False)

    print(f"[+] Generation complete! Metadata saved to {csv_out_path}.")
    print(f"[+] You can now evaluate this by passing '{trap_set_name}' to your evaluation scripts!")

if __name__ == "__main__":
    main()