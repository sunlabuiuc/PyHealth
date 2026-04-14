# Contributor: [Your Name]
# NetID: [Your NetID]

"""
Synthetic Artifact Generation for Dermoscopy Trap Sets.

This script utilizes Stable Diffusion Inpainting with LoRA (Low-Rank Adaptation) 
to synthetically inject clinical artifacts (e.g., rulers, ink, gel bubbles) into 
datasets, successfully recreating the 'Trap Sets' used in Jin (2025).

Paper: "A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations" (CHIL 2025)
Reference: Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR. 
(Model: runwayml/stable-diffusion-inpainting).
"""
# No PyHealth dependencies needed for generation.
# Identical underlying generation as the main pipeline, outputs CSV format

import os
import argparse
import torch
import pandas as pd
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

from train_dermoscopy import setup_dynamic_logging

ARTIFACT_MAP = {"dark-corner": "lun",
                "gel-bubble": "sown",
                "ink": "httr",
                "patches": "olis",
                "ruler": "dits"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='ph2')
    parser.add_argument('--lora_path', type=str, required=True)
    parser.add_argument('--artifact', type=str, choices=list(ARTIFACT_MAP.keys()), required=True)
    args = parser.parse_args()

    setup_dynamic_logging("data_prep", f"{args.dataset}_with_{args.artifact}")

    token = ARTIFACT_MAP[args.artifact]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    pipeline.load_lora_weights(args.lora_path)
    pipeline.enable_attention_slicing()

    input_images_dir = os.path.join(args.data_dir, args.dataset, "images")
    input_masks_dir = os.path.join(args.data_dir, args.dataset, "masks")
    
    output_dataset_name = f"{args.dataset}_with_{args.artifact}"
    output_dir = os.path.join(args.data_dir, output_dataset_name, "images")
    os.makedirs(output_dir, exist_ok=True)

    records = []
    valid_images = [f for f in os.listdir(input_images_dir) if f.endswith('.jpg') or f.endswith('.bmp')]
    
    print(f"[*] Generating Trap Set ({len(valid_images)} images) for Architecture...")
    
    for img_name in valid_images:
        img_path = os.path.join(input_images_dir, img_name)
        mask_name = img_name.replace(".jpg", ".bmp") 
        mask_path = os.path.join(input_masks_dir, mask_name)
        if not os.path.exists(mask_path): continue
            
        raw_img = Image.open(img_path).convert("RGB").resize((512, 512))
        inverted_mask = Image.eval(Image.open(mask_path).convert("L").resize((512, 512)), lambda x: 255 - x)

        augmented_img = pipeline(prompt=f"a dermoscopic image of {token} benign", image=raw_img, mask_image=inverted_mask, strength=0.8, guidance_scale=10.0).images[0]
        
        patient_id = img_name.split('.')[0]
        new_img_name = f"{patient_id}_artifact.jpg"
        augmented_img.save(os.path.join(output_dir, new_img_name))
        
        records.append({"isic_id": f"{patient_id}_artifact", "diagnosis_1": "Benign"})

    # Save a CSV so `prepare_metadata` dynamic scanner can pick it up instantly
    pd.DataFrame(records).to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    print(f"[SUCCESS] Trap Set {output_dataset_name} created successfully!")

if __name__ == "__main__":
    main()