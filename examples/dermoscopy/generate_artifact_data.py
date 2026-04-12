# Contributor: [Your Name]
# NetID: [Your NetID]

"""
Synthetic Artifact Generation for Dermoscopy Trap Sets.

This script utilizes Stable Diffusion Inpainting with LoRA (Low-Rank Adaptation) 
to synthetically inject clinical artifacts (e.g., rulers, ink, gel bubbles) into 
datasets, successfully recreating the 'Trap Sets' used in Jin (2025).

Paper: "A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations" (CHIL 2025)
"""

import os
import argparse
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

# Maps human-readable artifact names to specific LoRA trigger tokens used in the CHIL 2025 study
ARTIFACT_MAP = {
    "patches": "olis",
    "dark_corner": "lun",
    "ruler": "dits",
    "ink": "httr",
    "gel_bubble": "sown"
}

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dermoscopy artifacts using Stable Diffusion.")
    parser.add_argument('--data_dir', type=str, required=True, help="Base directory containing dataset folders.")
    parser.add_argument('--dataset', type=str, default='ph2', help="The source dataset to poison (e.g., ph2).")
    parser.add_argument('--lora_path', type=str, required=True, help="Path to the directory containing trained LoRA weights for the artifact.")
    parser.add_argument('--artifact', type=str, choices=list(ARTIFACT_MAP.keys()), required=True)
    args = parser.parse_args()

    token = ARTIFACT_MAP[args.artifact]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[*] Initializing Stable Diffusion Pipeline for {args.artifact.upper()} injection...")
    
    # 1. Initialize the Base Pipeline
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipeline = pipeline.to(device)
    
    # 2. Inject the LoRA weights (Critical Fix for proper generation)
    print(f"[*] Loading LoRA weights from {args.lora_path}...")
    pipeline.load_lora_weights(args.lora_path)
    pipeline.enable_attention_slicing()

    # 3. Setup Directories
    input_images_dir = os.path.join(args.data_dir, args.dataset, "images")
    input_masks_dir = os.path.join(args.data_dir, args.dataset, "masks")
    
    output_dataset_name = f"{args.dataset}_with_{args.artifact}"
    output_dir = os.path.join(args.data_dir, output_dataset_name, "images")
    os.makedirs(output_dir, exist_ok=True)

    # Note: A true robust implementation would parse the metadata.csv here, 
    # but this loop handles raw image iteration directly.
    valid_images = [f for f in os.listdir(input_images_dir) if f.endswith('.jpg') or f.endswith('.bmp')]
    
    print(f"[*] Generating Trap Set ({len(valid_images)} images) into {output_dir}...")
    
    for img_name in valid_images:
        img_path = os.path.join(input_images_dir, img_name)
        mask_name = img_name.replace(".jpg", ".bmp") # Assumes standard mask naming
        mask_path = os.path.join(input_masks_dir, mask_name)
        
        if not os.path.exists(mask_path):
            print(f"[!] Warning: Mask for {img_name} not found. Skipping...")
            continue
            
        raw_img = Image.open(img_path).convert("RGB").resize((512, 512))
        
        # Invert the mask: We want Stable Diffusion to draw in the background, not on the lesion!
        binary_mask = Image.open(mask_path).convert("L").resize((512, 512))
        inverted_mask = Image.eval(binary_mask, lambda x: 255 - x)

        prompt = f"a dermoscopic image of {token} benign"
        
        augmented_img = pipeline(
            prompt=prompt,
            image=raw_img,
            mask_image=inverted_mask,
            strength=0.8,
            guidance_scale=10.0
        ).images[0]
        
        new_img_name = img_name.replace(".jpg", "_artifact.jpg").replace(".bmp", "_artifact.jpg")
        new_img_path = os.path.join(output_dir, new_img_name)
        augmented_img.save(new_img_path)
        
    print(f"\n[SUCCESS] {output_dataset_name} Trap Set successfully generated!")
    print("Please ensure you generate a matching metadata.csv for PyHealth dataloading!")

if __name__ == "__main__":
    main()