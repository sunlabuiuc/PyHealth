"""DreamBooth + LoRA fine-tuning on ISIC 2018 dermoscopic artifact images.

Replicates the diffusion model training from:
  Jin et al. "A Study of Artifacts on Melanoma Classification under
  Diffusion-Based Perturbations", CHIL 2025.

Method
------
For each artifact type, fine-tune Stable Diffusion 1.5 with DreamBooth and
LoRA to associate a rare token with that artifact's visual appearance.  The
trained LoRA adapters are later loaded into an SD inpainting pipeline to
augment PH2 images (see ph2_diffusion_sd.py).

Training setup (matches paper §3.3)
------------------------------------
  Base model   : runwayml/stable-diffusion-v1-5
  LoRA targets : UNet attention projections + CLIP text-encoder projections
  LoRA rank    : 64   (paper: 64)
  LoRA alpha   : 32   (paper: 32)
  Epochs       : 4    (paper: 3–5)
  LR           : 1e-4 (paper: 1e-4)
  Batch size   : 2    (paper: 2)
  Resolution   : 512  (paper: 512)
  Prior weight : 0.3  (paper: 0.3)
  Prior images : 200  (paper: 200)
  Seed         : 0    (paper: 0)
  Precision    : fp16

Rare tokens (paper Table / §3.3)
---------------------------------
  patches      → olis
  dark_corner  → lun
  ruler        → dits
  ink          → httr
  gel_bubble   → sown

Instance prompt : "a dermoscopic image of {token} {class}"
Class prompt    : "a dermoscopic image of {class}"

Instance image selection
------------------------
We sample --n_instance images per artifact from the Bissoto et al. (2020)
artifact annotations included in isic-artifact-metadata-pyhealth.csv.
Single-artifact images are preferred (cleanest signal); if fewer than
--n_instance such images exist the remaining slots are filled from
multi-artifact images that contain the target artifact.

Outputs
-------
  ~/lora_checkpoints/{artifact}/unet/         — LoRA adapter (PEFT format)
  ~/lora_checkpoints/{artifact}/text_encoder/ — LoRA adapter (PEFT format)
  ~/lora_checkpoints/{artifact}/prior_images/ — 200 generated class images

Usage
-----
  # Train one artifact
  pixi run -e base python examples/isic2018_artifacts_train_lora_sd15.py --artifact gel_bubble

  # Train all five artifacts sequentially
  pixi run -e base python examples/isic2018_artifacts_train_lora_sd15.py --artifact all

  # Quick smoke test (3 instance images, 1 epoch, 10 prior images)
  pixi run -e base python examples/isic2018_artifacts_train_lora_sd15.py --artifact ink --test

Optional flags
--------------
  --artifact    {hair,dark_corner,ruler,gel_bubble,ink,patches,all}
  --n_instance  Number of instance images per artifact (default 50)
  --epochs      Training epochs (default 4)
  --lr          Learning rate (default 1e-4)
  --rank        LoRA rank (default 64)
  --alpha       LoRA alpha (default 32)
  --prior_weight Prior preservation loss weight (default 0.3)
  --n_prior     Prior images to generate (default 200)
  --output_dir  Checkpoint root (default ~/lora_checkpoints)
  --data_csv    Artifact annotation CSV
  --image_dir   ISIC image directory
  --model       Base SD model (default runwayml/stable-diffusion-v1-5)
  --test        Smoke-test mode: 3 images, 1 epoch, 10 prior images
"""

import argparse
import csv
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RARE_TOKENS = {
    "patches":     "olis",
    "dark_corner": "lun",
    "ruler":       "dits",
    "ink":         "httr",
    "gel_bubble":  "sown",
    "hair":        "helo",   # not in paper; we define our own rare token
}

ALL_ARTIFACTS = list(RARE_TOKENS.keys())

UNET_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
TEXT_ENCODER_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def sample_instance_images(artifact: str, n: int, data_csv: str, image_dir: str,
                            seed: int = 0) -> list:
    """Return up to n image paths for images labelled with `artifact`.

    Prefers single-artifact images (cleanest visual signal); fills remaining
    slots with multi-artifact images that contain the target artifact.
    """
    rng = random.Random(seed)
    rows = []
    with open(data_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    artifact_cols = ["dark_corner", "hair", "gel_bubble", "ruler", "ink", "patches"]

    single, multi = [], []
    for row in rows:
        if int(row[artifact]) != 1:
            continue
        path = row.get("path") or os.path.join(image_dir, row["image"])
        if not os.path.exists(path):
            path = os.path.join(image_dir, row["image"])
        if not os.path.exists(path):
            continue
        n_arts = sum(int(row[a]) for a in artifact_cols if a in row)
        label = row.get("label_string", "benign")
        if n_arts == 1:
            single.append((path, label))
        else:
            multi.append((path, label))

    rng.shuffle(single)
    rng.shuffle(multi)
    combined = (single + multi)[:n]
    print(f"  {artifact}: {len(single)} single-artifact, {len(multi)} multi; "
          f"selected {len(combined)} / {n} requested")
    return combined   # list of (path, label)


def generate_prior_images(pipe, class_prompts: list, out_dir: Path, seed: int = 0):
    """Generate prior class images using the unmodified base model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.jpg"))
    if len(existing) >= len(class_prompts):
        print(f"  Prior images already exist ({len(existing)}), skipping generation.")
        return [str(p) for p in existing[:len(class_prompts)]]

    print(f"  Generating {len(class_prompts)} prior images…")
    generator = torch.Generator("cuda").manual_seed(seed)
    paths = []
    for i, prompt in enumerate(class_prompts):
        out_path = out_dir / f"prior_{i:04d}.jpg"
        if out_path.exists():
            paths.append(str(out_path))
            continue
        result = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
            height=512,
            width=512,
        ).images[0]
        result.save(out_path, quality=95)
        paths.append(str(out_path))
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(class_prompts)}")
    return paths


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


class DreamBoothDataset(Dataset):
    """Yields (instance_pixel, instance_prompt, prior_pixel, prior_prompt)."""

    def __init__(self, instance_items, instance_prompt_fn,
                 prior_paths, prior_prompt):
        self.instance_items = instance_items          # list of (path, label)
        self.instance_prompt_fn = instance_prompt_fn  # fn(label) -> str
        self.prior_paths = prior_paths
        self.prior_prompt = prior_prompt

    def __len__(self):
        return max(len(self.instance_items), len(self.prior_paths))

    def __getitem__(self, idx):
        inst_path, label = self.instance_items[idx % len(self.instance_items)]
        prior_path = self.prior_paths[idx % len(self.prior_paths)]

        inst_img = Image.open(inst_path).convert("RGB")
        prior_img = Image.open(prior_path).convert("RGB")

        return {
            "instance_pixel":  IMAGE_TRANSFORMS(inst_img),
            "instance_prompt": self.instance_prompt_fn(label),
            "prior_pixel":     IMAGE_TRANSFORMS(prior_img),
            "prior_prompt":    self.prior_prompt,
        }


def collate_fn(batch):
    return {
        "instance_pixels":  torch.stack([b["instance_pixel"] for b in batch]),
        "instance_prompts": [b["instance_prompt"] for b in batch],
        "prior_pixels":     torch.stack([b["prior_pixel"] for b in batch]),
        "prior_prompts":    [b["prior_prompt"] for b in batch],
    }


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------

def add_lora(model, target_modules: list, rank: int, alpha: int):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(model, config)


def encode_prompts(tokenizer, text_encoder, prompts: list, device):
    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(tokens.input_ids.to(device))[0]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_artifact(args, artifact: str):
    from diffusers import (
        DDPMScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    device = torch.device("cuda")
    token = RARE_TOKENS[artifact]
    out_dir = Path(args.output_dir) / artifact
    out_dir.mkdir(parents=True, exist_ok=True)
    unet_dir = out_dir / "unet"
    te_dir = out_dir / "text_encoder"

    if unet_dir.exists() and te_dir.exists():
        print(f"\n[{artifact}] LoRA already trained, skipping.")
        return

    print(f"\n{'='*60}")
    print(f"Training LoRA for artifact: {artifact}  (token: '{token}')")
    print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # 1. Sample instance images
    # -----------------------------------------------------------------------
    n_inst = 3 if args.test else args.n_instance
    instance_items = sample_instance_images(
        artifact, n_inst, args.data_csv, args.image_dir, seed=args.seed
    )
    if not instance_items:
        print(f"  WARNING: no images found for {artifact}, skipping.")
        return

    # -----------------------------------------------------------------------
    # 2. Build prompts
    # -----------------------------------------------------------------------
    def instance_prompt(label: str) -> str:
        return f"a dermoscopic image of {token} {label}"

    def class_prompt(label: str) -> str:
        return f"a dermoscopic image of {label}"

    labels_used = list({lbl for _, lbl in instance_items})

    # -----------------------------------------------------------------------
    # 3. Generate prior images using unmodified base pipeline
    # -----------------------------------------------------------------------
    n_prior = 10 if args.test else args.n_prior
    prior_prompts = []
    for i in range(n_prior):
        lbl = "benign" if i < n_prior // 2 else "malignant"
        prior_prompts.append(class_prompt(lbl))

    pipe_txt2img = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe_txt2img.set_progress_bar_config(disable=True)

    prior_dir = out_dir / "prior_images"
    prior_paths = generate_prior_images(
        pipe_txt2img, prior_prompts, prior_dir, seed=args.seed
    )
    del pipe_txt2img
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 4. Load model components
    # -----------------------------------------------------------------------
    print("  Loading model components…")
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")

    # Load VAE separately in fp16 (frozen, not trained)
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae",
                                         torch_dtype=torch.float16).to(device)
    vae.requires_grad_(False)

    # -----------------------------------------------------------------------
    # 5. Add LoRA adapters
    # -----------------------------------------------------------------------
    text_encoder = add_lora(text_encoder, TEXT_ENCODER_LORA_TARGETS,
                            args.rank, args.alpha).to(device)
    unet = add_lora(unet, UNET_LORA_TARGETS, args.rank, args.alpha).to(device)

    # Cast trainable LoRA params to fp32, rest to fp16
    for name, param in unet.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
        else:
            param.data = param.data.half()
    for name, param in text_encoder.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
        else:
            param.data = param.data.half()

    # -----------------------------------------------------------------------
    # 6. Dataset & dataloader
    # -----------------------------------------------------------------------
    prior_label_list = ["benign" if i < n_prior // 2 else "malignant"
                        for i in range(len(prior_paths))]
    # Use a single class prompt (mixed) for simplicity
    dataset = DreamBoothDataset(
        instance_items=instance_items,
        instance_prompt_fn=instance_prompt,
        prior_paths=prior_paths,
        prior_prompt="a dermoscopic image of benign",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate_fn,
                        num_workers=0, drop_last=True)

    # -----------------------------------------------------------------------
    # 7. Optimizer
    # -----------------------------------------------------------------------
    trainable = (
        [p for p in unet.parameters() if p.requires_grad] +
        [p for p in text_encoder.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  betas=(0.9, 0.999), weight_decay=1e-2)

    epochs = 1 if args.test else args.epochs
    total_steps = epochs * math.ceil(len(dataset) / args.batch_size)
    print(f"  Instance: {len(instance_items)}  Prior: {len(prior_paths)}  "
          f"Steps: {total_steps}")

    # -----------------------------------------------------------------------
    # 8. Training loop
    # -----------------------------------------------------------------------
    scaler = torch.cuda.amp.GradScaler()
    global_step = 0

    for epoch in range(epochs):
        unet.train()
        text_encoder.train()
        epoch_loss = 0.0

        for batch in loader:
            # Concatenate instance + prior along batch dim for a single fwd pass
            pixels = torch.cat([
                batch["instance_pixels"].to(device, dtype=torch.float16),
                batch["prior_pixels"].to(device, dtype=torch.float16),
            ])
            prompts = batch["instance_prompts"] + batch["prior_prompts"]

            with torch.cuda.amp.autocast():
                # Encode images to latents
                latents = vae.encode(pixels).latent_dist.sample() * 0.18215

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (bsz,),
                    device=device, dtype=torch.long
                )
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                # Text conditioning
                enc = encode_prompts(tokenizer, text_encoder, prompts, device)

                # UNet prediction
                pred = unet(noisy_latents, timesteps, enc).sample

                # Split instance vs prior halves
                half = bsz // 2
                pred_inst, pred_prior = pred[:half], pred[half:]
                noise_inst, noise_prior = noise[:half], noise[half:]

                if scheduler.config.prediction_type == "epsilon":
                    target_inst, target_prior = noise_inst, noise_prior
                else:
                    target_inst = scheduler.get_velocity(
                        latents[:half], noise_inst, timesteps[:half])
                    target_prior = scheduler.get_velocity(
                        latents[half:], noise_prior, timesteps[half:])

                loss_inst = F.mse_loss(pred_inst.float(), target_inst.float())
                loss_prior = F.mse_loss(pred_prior.float(), target_prior.float())
                loss = loss_inst + args.prior_weight * loss_prior

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

        avg = epoch_loss / max(1, len(loader))
        print(f"  Epoch {epoch+1}/{epochs}  loss={avg:.4f}")

    # -----------------------------------------------------------------------
    # 9. Save LoRA weights
    # -----------------------------------------------------------------------
    unet.save_pretrained(str(unet_dir))
    text_encoder.save_pretrained(str(te_dir))
    print(f"  Saved LoRA adapters → {out_dir}")

    del unet, text_encoder, vae
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="DreamBooth + LoRA training for dermoscopic artifacts"
    )
    p.add_argument("--artifact", default="all",
                   choices=ALL_ARTIFACTS + ["all"],
                   help="Artifact to train (default: all)")
    p.add_argument("--n_instance", type=int, default=50,
                   help="Instance images per artifact (default 50)")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--prior_weight", type=float, default=0.3)
    p.add_argument("--n_prior", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir",
                   default=os.path.expanduser("~/lora_checkpoints"))
    p.add_argument("--data_csv",
                   default=os.path.expanduser(
                       "~/isic2018_data/isic-artifact-metadata-pyhealth.csv"))
    p.add_argument("--image_dir",
                   default=os.path.expanduser(
                       "~/isic2018_data/ISIC2018_Task1-2_Training_Input"))
    p.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--test", action="store_true",
                   help="Smoke-test: 3 images, 1 epoch, 10 prior images")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    artifacts = ALL_ARTIFACTS if args.artifact == "all" else [args.artifact]
    print(f"Training LoRA for: {artifacts}")
    print(f"Instance images: {3 if args.test else args.n_instance}  "
          f"Epochs: {1 if args.test else args.epochs}  "
          f"Prior: {10 if args.test else args.n_prior}")

    for artifact in artifacts:
        train_artifact(args, artifact)

    print("\nAll done.")


if __name__ == "__main__":
    main()
