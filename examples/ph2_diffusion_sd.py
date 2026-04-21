"""PH2 Dermoscopic Artifact Augmentation via Stable Diffusion Inpainting.

Adds synthetic artifacts to the 200-image PH2 dataset using a two-step pipeline:

  1. Mask generation — each artifact function returns the original image
     unchanged plus a binary mask that defines *where* the artifact should
     appear (hair paths, edge strips, bubble circles, corner gradients, etc.)
  2. SD inpainting — the model generates the artifact appearance from scratch
     inside the mask, guided purely by a text prompt. No programmatic drawing
     is blended into the image.

Supported artifact types
------------------------
  hair         — Bezier-path mask, SD generates hair strands         (paper: excluded)
  dark_corner  — Radial peripheral mask, SD generates dark vignette  (paper: ✓)
  ruler        — Edge-strip mask, SD generates ruler tick marks       (paper: ✓)
  gel_bubble   — Circular masks, SD generates gel bubble discs        (paper: ✓)
  ink          — Small ellipse masks, SD generates ink dot marks      (paper: ✓)
  patches      — Edge-strip mask, SD generates colour-checker swatches (paper: ✓)

Outputs
-------
  ~/ph2_augmented/
    clean/          — resized originals (512×512) with no artifact
    hair/           — hair artifact variants
    dark_corner/    — dark corner vignette variants
    ruler/          — ruler mark variants
    gel_bubble/     — gel bubble variants
    ink/            — ink marking variants
    patches/        — colour calibration patch variants
    augmented_metadata.csv   — columns: image_id, path, diagnosis, artifact

Usage
-----
  # All artifact types (GPU required — all types use SD inpainting)
  pixi run -e base python examples/ph2_diffusion_sd.py \\
      --src ~/ph2/PH2-dataset-master \\
      --out ~/ph2_augmented

  # Specific artifacts
  pixi run -e base python examples/ph2_diffusion_sd.py \\
      --artifacts gel_bubble ink dark_corner

  [--model runwayml/stable-diffusion-inpainting]
  [--n_aug 1]       # augmented copies per image per artifact type
  [--test 3]        # only process first N images (smoke test)
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# Artifact generators
# ---------------------------------------------------------------------------

def _bezier_point(p0, p1, p2, t):
    """Quadratic Bezier interpolation."""
    return (
        (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0],
        (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1],
    )


def make_hair_overlay(img: Image.Image, n_strands: int = 12, seed: int = 0):
    """Draw dark Bezier hair strands as seeds; mask those paths for SD refinement."""
    rng = random.Random(seed)
    w, h = img.size
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)

    for _ in range(n_strands):
        x0, y0 = rng.randint(0, w), rng.randint(0, h)
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        x1, y1 = rng.randint(0, w), rng.randint(0, h)
        thickness = rng.randint(2, 4)
        darkness = rng.randint(5, 25)
        hair_col = (darkness, darkness, darkness)
        pts = [_bezier_point((x0, y0), (cx, cy), (x1, y1), t / 60) for t in range(61)]
        pts_int = [(int(p[0]), int(p[1])) for p in pts]
        draw.line(pts_int, fill=hair_col, width=thickness)
        mdraw.line(pts_int, fill=255, width=thickness + 8)

    mask = mask.filter(ImageFilter.MaxFilter(5))
    return overlay, mask


def make_dark_corner_overlay(img: Image.Image):
    """Apply a mild dark vignette; mask the affected region for SD refinement."""
    w, h = img.size
    arr = np.array(img).astype(np.float32)
    cx, cy = w / 2, h / 2
    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt(((x_idx - cx) / (w / 2)) ** 2 + ((y_idx - cy) / (h / 2)) ** 2)
    # Gentler onset at 0.80, softer falloff
    vignette = np.clip(1.0 - np.maximum(0, dist - 0.80) * 1.5, 0, 1)
    arr *= vignette[:, :, None]
    overlay = Image.fromarray(arr.astype(np.uint8))
    mask = Image.fromarray((255 * (1 - vignette)).astype(np.uint8))
    return overlay, mask


def make_ruler_overlay(img: Image.Image, seed: int = 0):
    """Draw a semi-transparent ruler strip offset from one edge; mask for SD refinement."""
    rng = random.Random(seed)
    w, h = img.size
    overlay = img.copy()
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)

    edge = rng.choice(["top", "bottom", "left", "right"])
    n_ticks = rng.randint(10, 20)
    tick_len_major = rng.randint(18, 28)
    tick_len_minor = tick_len_major // 2
    strip_w = tick_len_major + 16
    offset = 10
    tick_color = (30, 30, 30)

    # Semi-transparent background strip (alpha=120 ≈ 47% opaque)
    strip_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(strip_layer)

    if edge in ("top", "bottom"):
        y_bg = offset if edge == "top" else h - offset - strip_w
        sdraw.rectangle([0, y_bg, w, y_bg + strip_w], fill=(240, 240, 240, 120))
        base = overlay.convert("RGBA")
        base.alpha_composite(strip_layer)
        overlay = base.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for i in range(n_ticks):
            x = int(w * (i + 1) / (n_ticks + 1))
            ln = tick_len_major if i % 5 == 0 else tick_len_minor
            draw.line([(x, y_bg + 4), (x, y_bg + 4 + ln)], fill=tick_color, width=2)
        mdraw.rectangle([0, y_bg, w, y_bg + strip_w], fill=255)
    else:
        x_bg = offset if edge == "left" else w - offset - strip_w
        sdraw.rectangle([x_bg, 0, x_bg + strip_w, h], fill=(240, 240, 240, 120))
        base = overlay.convert("RGBA")
        base.alpha_composite(strip_layer)
        overlay = base.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for i in range(n_ticks):
            y = int(h * (i + 1) / (n_ticks + 1))
            ln = tick_len_major if i % 5 == 0 else tick_len_minor
            draw.line([(x_bg + 4, y), (x_bg + 4 + ln, y)], fill=tick_color, width=2)
        mdraw.rectangle([x_bg, 0, x_bg + strip_w, h], fill=255)

    return overlay, mask


def make_gel_bubble_overlay(img: Image.Image, n_bubbles: int = 5, seed: int = 0):
    """Draw semi-transparent bubble fills as seeds; mask those regions for SD refinement."""
    rng = random.Random(seed)
    w, h = img.size
    max_r = max(5, min(w, h) // 12)   # cap radius to ~1/12 of shortest side
    overlay = img.copy()
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)

    bubble_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bdraw = ImageDraw.Draw(bubble_layer)

    for _ in range(n_bubbles):
        r = rng.randint(max(5, max_r // 2), max_r)
        cx = rng.randint(r, max(r + 1, w - r))
        cy = rng.randint(r, max(r + 1, h - r))
        bdraw.ellipse([cx - r, cy - r, cx + r, cy + r],
                      fill=(210, 220, 235, 100), outline=(100, 120, 160, 255), width=2)
        mdraw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

    base = overlay.convert("RGBA")
    base.alpha_composite(bubble_layer)
    overlay = base.convert("RGB")

    mask = mask.filter(ImageFilter.MaxFilter(3))
    return overlay, mask


def make_ink_overlay(img: Image.Image, n_marks: int = 5, seed: int = 0):
    """Draw ink marks biased toward lines; mask those spots for SD refinement."""
    import math
    rng = random.Random(seed)
    w, h = img.size
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)

    for _ in range(n_marks):
        cx = rng.randint(int(0.1 * w), int(0.9 * w))
        cy = rng.randint(int(0.1 * h), int(0.9 * h))
        # Bias: 60% line, 25% cross, 15% dot
        roll = rng.random()
        style = "line" if roll < 0.60 else ("cross" if roll < 0.85 else "dot")
        ink_col = (rng.randint(0, 20), rng.randint(0, 20), rng.randint(80, 140))
        r = rng.randint(16, 36)
        if style == "dot":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ink_col)
            mdraw.ellipse([cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8], fill=255)
        elif style == "cross":
            draw.line([(cx - r, cy), (cx + r, cy)], fill=ink_col, width=3)
            draw.line([(cx, cy - r), (cx, cy + r)], fill=ink_col, width=3)
            mdraw.ellipse([cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8], fill=255)
        else:
            angle = rng.uniform(0, math.pi)
            x1, y1 = int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))
            x2, y2 = int(cx - r * math.cos(angle)), int(cy - r * math.sin(angle))
            draw.line([(x1, y1), (x2, y2)], fill=ink_col, width=4)
            mdraw.ellipse([cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8], fill=255)

    mask = mask.filter(ImageFilter.MaxFilter(3))
    return overlay, mask


def make_patches_overlay(img: Image.Image, seed: int = 0):
    """Draw round colour-calibration stickers half-cut-off at a corner edge; mask for SD."""
    rng = random.Random(seed)
    w, h = img.size
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)

    colours = [
        (210, 180, 170), (170, 200, 180), (180, 180, 210),
        (200, 195, 170), (190, 175, 200), (170, 205, 210),
    ]
    rng.shuffle(colours)
    r   = max(10, min(w, h) // 8)
    gap = max(4, r // 4)
    n_patches = 1

    # Always start from bottom-left corner, march right
    # cy = h places centre on edge so upper half is visible
    cy = h
    cx = r  # first circle starts at x=r from left
    for i in range(n_patches):
        if cx - r >= w:
            break
        col = colours[i % len(colours)]
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col,
                     outline=(20, 20, 20), width=1)
        mdraw.ellipse([cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4], fill=255)
        cx += 2 * r + gap

    return overlay, mask


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

# Rare tokens used during LoRA training (matches isic2018_artifacts_train_lora_sd15.py)
RARE_TOKENS = {
    "patches":     "olis",
    "dark_corner": "lun",
    "ruler":       "dits",
    "ink":         "httr",
    "gel_bubble":  "sown",
    "hair":        "helo",
}

# Inpainting strength per artifact (paper §3.3; hair not in paper → 0.75)
ARTIFACT_STRENGTH = {
    "patches":     0.60,
    "dark_corner": 0.55,
    "ruler":       0.50,
    "ink":         0.45,
    "gel_bubble":  0.52,
    "hair":        0.55,
}

# Fallback descriptive prompts used when no LoRA is available
FALLBACK_PROMPTS = {
    "hair": (
        "dermoscopy skin lesion image with thin dark hair strands crossing the lesion, "
        "medical imaging, high detail"
    ),
    "ruler": (
        "dermoscopy skin lesion image with white ruler measurement marks along the edge, "
        "medical imaging, calibration scale"
    ),
    "gel_bubble": (
        "dermoscopy skin lesion image with transparent gel air bubbles on the surface, "
        "bright circular reflections, medical imaging"
    ),
    "ink": (
        "dermoscopy skin lesion image with small dark ink markings and dots on the skin, "
        "clinical annotation marks, medical imaging"
    ),
    "dark_corner": (
        "dermoscopy skin lesion image with dark vignette corners, black edges fading "
        "toward the center, medical imaging"
    ),
    "patches": (
        "dermoscopy skin lesion image with colour calibration patches along the border, "
        "round circular colour swatches, medical imaging"
    ),
}


def lora_prompt(artifact: str, diagnosis: str) -> str:
    """Build the DreamBooth instance prompt using the trained rare token."""
    token = RARE_TOKENS[artifact]
    label = "malignant" if "melanoma" in diagnosis else "benign"
    return f"a dermoscopic image of {token} {label}"


def load_pipeline(model_id: str):
    from diffusers import StableDiffusionInpaintPipeline

    print(f"Loading inpainting model: {model_id}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_lora_pipeline(model_id: str, artifact: str, lora_dir: str):
    """Load a fresh SD inpainting pipeline with LoRA adapters applied.

    Loads a new pipeline instance per artifact to avoid PEFT adapter
    accumulation warnings when swapping adapters on the same model.
    Returns (pipe, use_lora).
    """
    from diffusers import StableDiffusionInpaintPipeline
    from peft import PeftModel

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    unet_path = os.path.join(lora_dir, artifact, "unet")
    te_path   = os.path.join(lora_dir, artifact, "text_encoder")

    if os.path.isdir(unet_path):
        print(f"  Loading LoRA for {artifact} from {lora_dir}/{artifact}/")
        pipe.unet         = PeftModel.from_pretrained(pipe.unet, unet_path).to("cuda", dtype=torch.float16)
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, te_path).to("cuda", dtype=torch.float16)
        return pipe, True
    else:
        print(f"  [no LoRA] {artifact}: using base model + fallback prompt")
        return pipe, False


def inpaint(pipe, image: Image.Image, mask: Image.Image,
            prompt: str, strength: float) -> Image.Image:
    """Run SD inpainting at 512×512, resize result back to original size."""
    orig_size = image.size
    img_512  = image.resize((512, 512), Image.LANCZOS)
    mask_512 = mask.resize((512, 512), Image.NEAREST)

    result = pipe(
        prompt=prompt,
        image=img_512,
        mask_image=mask_512,
        num_inference_steps=30,
        guidance_scale=10.0,
        strength=strength,
    ).images[0]

    return result.resize(orig_size, Image.LANCZOS)


def augment_image(pipe, img: Image.Image, artifact: str, aug_idx: int,
                  diagnosis: str = "benign", use_lora: bool = False) -> Image.Image:
    """Generate one augmented copy of img with the given artifact type.

    When use_lora=True the DreamBooth rare-token prompt and paper-matched
    guidance/strength values are used; otherwise falls back to the
    descriptive prompt with the same strength.
    """
    seed = aug_idx * 1000 + hash(artifact) % 1000

    if artifact == "hair":
        img_in, mask = make_hair_overlay(img, n_strands=12, seed=seed)
    elif artifact == "ruler":
        img_in, mask = make_ruler_overlay(img, seed=seed)
    elif artifact == "gel_bubble":
        img_in, mask = make_gel_bubble_overlay(img, n_bubbles=5, seed=seed)
    elif artifact == "ink":
        img_in, mask = make_ink_overlay(img, n_marks=4, seed=seed)
    elif artifact == "dark_corner":
        img_in, mask = make_dark_corner_overlay(img)
    elif artifact == "patches":
        img_in, mask = make_patches_overlay(img, seed=seed)
    else:
        raise ValueError(f"Unknown artifact: {artifact}")

    prompt   = lora_prompt(artifact, diagnosis) if use_lora else FALLBACK_PROMPTS[artifact]
    strength = ARTIFACT_STRENGTH[artifact]
    return inpaint(pipe, img_in, mask, prompt, strength)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PH2 artifact augmentation via SD inpainting")
    p.add_argument("--src", default=os.path.expanduser("~/ph2/PH2-dataset-master"))
    p.add_argument("--out", default=os.path.expanduser("~/ph2_augmented"))
    p.add_argument("--model", default="runwayml/stable-diffusion-inpainting")
    p.add_argument("--lora_dir", default=os.path.expanduser("~/lora_checkpoints"),
                   help="Root directory of trained LoRA adapters (default ~/lora_checkpoints)")
    p.add_argument("--n_aug", type=int, default=1, help="Augmented copies per artifact type")
    p.add_argument("--test", type=int, default=0, help="Only process first N images (0=all)")
    p.add_argument("--artifacts", nargs="+",
                   default=["hair", "dark_corner", "ruler", "gel_bubble", "ink", "patches"])
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    out = Path(args.out)

    # Load metadata CSV
    meta_path = src / "PH2_simple_dataset.csv"
    if not meta_path.exists():
        meta_path = src / "ph2_metadata_pyhealth.csv"
    rows = []
    with open(meta_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    id_col   = "image_name" if "image_name" in rows[0] else "image_id"
    diag_col = "diagnosis"
    images_dir = src / "images"

    if args.test:
        rows = rows[: args.test]

    out.mkdir(parents=True, exist_ok=True)
    (out / "clean").mkdir(exist_ok=True)
    for art in args.artifacts:
        (out / art).mkdir(exist_ok=True)

    # Build list of valid (img_id, diagnosis, img_path) tuples
    valid_rows = []
    for row in rows:
        img_id   = row[id_col]
        diagnosis = row[diag_col].lower().replace(" ", "_")
        img_path  = images_dir / f"{img_id}.jpg"
        if not img_path.exists():
            print(f"  [skip] {img_id}: image not found")
            continue
        valid_rows.append((img_id, diagnosis, img_path))

    results = []  # (image_id, path, diagnosis, artifact)

    # -----------------------------------------------------------------------
    # Save clean copies
    # -----------------------------------------------------------------------
    for img_id, diagnosis, img_path in valid_rows:
        clean_path = out / "clean" / f"{img_id}.jpg"
        if not clean_path.exists():
            Image.open(img_path).convert("RGB").save(clean_path)
        results.append((img_id, str(clean_path), diagnosis, "clean"))

    # -----------------------------------------------------------------------
    # Process one artifact at a time: load fresh pipeline+LoRA, run all images
    # -----------------------------------------------------------------------
    for artifact in args.artifacts:
        pipe, use_lora = load_lora_pipeline(args.model, artifact, args.lora_dir)

        for img_id, diagnosis, img_path in valid_rows:
            for i in range(args.n_aug):
                suffix   = f"_aug{i}" if args.n_aug > 1 else ""
                out_name = f"{img_id}{suffix}.jpg"
                out_path = out / artifact / out_name

                if out_path.exists():
                    print(f"  [cached] {img_id} {artifact}{suffix}")
                else:
                    print(f"  Processing {img_id} → {artifact}{suffix} ...",
                          end=" ", flush=True)
                    img = Image.open(img_path).convert("RGB")
                    aug = augment_image(pipe, img, artifact, aug_idx=i,
                                        diagnosis=diagnosis, use_lora=use_lora)
                    aug.save(out_path)
                    print("✓")

                results.append((img_id, str(out_path), diagnosis, artifact))

        # Free GPU memory before loading next artifact's pipeline
        del pipe
        torch.cuda.empty_cache()

    # Write metadata CSV
    csv_path = out / "augmented_metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "path", "diagnosis", "artifact"])
        writer.writerows(results)

    print(f"\nDone. {len(results)} records → {csv_path}")


if __name__ == "__main__":
    main()
