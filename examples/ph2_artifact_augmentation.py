"""PH2 Dermoscopic Artifact Augmentation via Stable Diffusion Inpainting.

Adds synthetic artifacts (hair, dark_corner, ruler) to the 200-image PH2
dataset using a two-step pipeline:

  1. Programmatic mask/overlay generation (Bezier hair strands, vignette, ruler marks)
  2. Stable Diffusion inpainting to harmonise the artifact into the image

Outputs
-------
  ~/ph2_augmented/
    clean/          — resized originals (512×512) with no artifact
    hair/           — hair artifact variants
    dark_corner/    — dark corner vignette variants
    ruler/          — ruler mark variants
    augmented_metadata.csv   — columns: image_id, path, diagnosis, artifact, label

Usage
-----
  pixi run -e base python examples/ph2_artifact_augmentation.py \\
      --src ~/ph2/PH2-dataset-master \\
      --out ~/ph2_augmented \\
      [--model runwayml/stable-diffusion-inpainting] \\
      [--n_aug 1]            # augmented copies per image per artifact type
      [--test 3]             # only process first N images (smoke test)
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
    """Return (overlaid_image, mask) with dark hair strands drawn on img.

    The mask marks every pixel touched by a strand (white = inpaint here).
    """
    rng = random.Random(seed)
    w, h = img.size
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)

    for _ in range(n_strands):
        # Random quadratic Bezier strand
        x0, y0 = rng.randint(0, w), rng.randint(0, h)
        cx, cy = rng.randint(0, w), rng.randint(0, h)  # control point
        x1 = rng.randint(0, w)
        y1 = rng.randint(0, h)
        thickness = rng.randint(1, 3)
        color = rng.randint(10, 50)  # very dark

        pts = [_bezier_point((x0, y0), (cx, cy), (x1, y1), t / 60) for t in range(61)]
        pts_int = [(int(p[0]), int(p[1])) for p in pts]
        draw.line(pts_int, fill=(color, color, color), width=thickness)
        mdraw.line(pts_int, fill=255, width=thickness + 4)  # slightly wider mask

    # Dilate mask a bit for inpainting context
    mask = mask.filter(ImageFilter.MaxFilter(5))
    return overlay, mask


def make_dark_corner_overlay(img: Image.Image):
    """Apply a soft dark vignette. Returns (overlaid, mask).

    Dark corner doesn't need inpainting — it's deterministic.
    """
    w, h = img.size
    arr = np.array(img).astype(np.float32)
    cx, cy = w / 2, h / 2
    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt(((x_idx - cx) / (w / 2)) ** 2 + ((y_idx - cy) / (h / 2)) ** 2)
    # Vignette: fade to black at corners, onset ~0.7 of half-diagonal
    vignette = np.clip(1.0 - np.maximum(0, dist - 0.7) * 2.0, 0, 1)
    arr *= vignette[:, :, None]
    overlaid = Image.fromarray(arr.astype(np.uint8))
    mask = Image.fromarray((255 * (1 - vignette)).astype(np.uint8))
    return overlaid, mask


def make_ruler_overlay(img: Image.Image, seed: int = 0):
    """Draw ruler tick marks along one edge. Returns (overlaid, mask)."""
    rng = random.Random(seed)
    w, h = img.size
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)

    edge = rng.choice(["top", "bottom", "left", "right"])
    n_ticks = rng.randint(8, 20)
    tick_color = (rng.randint(180, 255),) * 3  # light/white marks
    tick_len = rng.randint(8, 20)
    thickness = 2

    if edge in ("top", "bottom"):
        y0 = 5 if edge == "top" else h - 5 - tick_len
        for i in range(n_ticks):
            x = int(w * (i + 1) / (n_ticks + 1))
            ln = tick_len if i % 5 == 0 else tick_len // 2
            draw.line([(x, y0), (x, y0 + ln)], fill=tick_color, width=thickness)
            mdraw.line([(x, y0), (x, y0 + ln)], fill=255, width=thickness + 6)
    else:
        x0 = 5 if edge == "left" else w - 5 - tick_len
        for i in range(n_ticks):
            y = int(h * (i + 1) / (n_ticks + 1))
            ln = tick_len if i % 5 == 0 else tick_len // 2
            draw.line([(x0, y), (x0 + ln, y)], fill=tick_color, width=thickness)
            mdraw.line([(x0, y), (x0 + ln, y)], fill=255, width=thickness + 6)

    mask = mask.filter(ImageFilter.MaxFilter(5))
    return overlay, mask


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

ARTIFACT_PROMPTS = {
    "hair": (
        "dermoscopy skin lesion image with thin dark hair strands crossing the lesion, "
        "medical imaging, high detail"
    ),
    "ruler": (
        "dermoscopy skin lesion image with white ruler measurement marks along the edge, "
        "medical imaging, calibration scale"
    ),
    "dark_corner": None,  # pure programmatic, no inpainting needed
}


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


def inpaint(pipe, image: Image.Image, mask: Image.Image, prompt: str) -> Image.Image:
    """Run SD inpainting at 512×512, resize result back to original size."""
    orig_size = image.size
    img_512 = image.resize((512, 512), Image.LANCZOS)
    mask_512 = mask.resize((512, 512), Image.NEAREST)

    result = pipe(
        prompt=prompt,
        image=img_512,
        mask_image=mask_512,
        num_inference_steps=30,
        guidance_scale=7.5,
        strength=0.85,
    ).images[0]

    return result.resize(orig_size, Image.LANCZOS)


def augment_image(pipe, img: Image.Image, artifact: str, aug_idx: int) -> Image.Image:
    """Generate one augmented copy of img with the given artifact type."""
    seed = aug_idx * 1000 + hash(artifact) % 1000

    if artifact == "hair":
        overlaid, mask = make_hair_overlay(img, n_strands=12, seed=seed)
        return inpaint(pipe, overlaid, mask, ARTIFACT_PROMPTS["hair"])

    elif artifact == "ruler":
        overlaid, mask = make_ruler_overlay(img, seed=seed)
        return inpaint(pipe, overlaid, mask, ARTIFACT_PROMPTS["ruler"])

    elif artifact == "dark_corner":
        result, _ = make_dark_corner_overlay(img)
        return result

    else:
        raise ValueError(f"Unknown artifact: {artifact}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PH2 artifact augmentation via SD inpainting")
    p.add_argument("--src", default=os.path.expanduser("~/ph2/PH2-dataset-master"))
    p.add_argument("--out", default=os.path.expanduser("~/ph2_augmented"))
    p.add_argument("--model", default="runwayml/stable-diffusion-inpainting")
    p.add_argument("--n_aug", type=int, default=1, help="Augmented copies per artifact type")
    p.add_argument("--test", type=int, default=0, help="Only process first N images (0=all)")
    p.add_argument("--artifacts", nargs="+", default=["hair", "dark_corner", "ruler"])
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

    # Normalise column names
    id_col = "image_name" if "image_name" in rows[0] else "image_id"
    diag_col = "diagnosis"

    images_dir = src / "images"
    if args.test:
        rows = rows[: args.test]

    # Load SD pipeline (skip for dark_corner-only runs)
    need_sd = any(a in args.artifacts for a in ["hair", "ruler"])
    pipe = load_pipeline(args.model) if need_sd else None

    out.mkdir(parents=True, exist_ok=True)
    (out / "clean").mkdir(exist_ok=True)
    for art in args.artifacts:
        (out / art).mkdir(exist_ok=True)

    results = []  # (image_id, path, diagnosis, artifact)

    for row in rows:
        img_id = row[id_col]
        diagnosis = row[diag_col].lower().replace(" ", "_")
        img_path = images_dir / f"{img_id}.jpg"
        if not img_path.exists():
            print(f"  [skip] {img_id}: image not found")
            continue

        img = Image.open(img_path).convert("RGB")

        # Save clean copy
        clean_path = out / "clean" / f"{img_id}.jpg"
        img.save(clean_path)
        results.append((img_id, str(clean_path), diagnosis, "clean"))

        for artifact in args.artifacts:
            for i in range(args.n_aug):
                suffix = f"_aug{i}" if args.n_aug > 1 else ""
                out_name = f"{img_id}{suffix}.jpg"
                out_path = out / artifact / out_name

                if out_path.exists():
                    print(f"  [cached] {img_id} {artifact}{suffix}")
                else:
                    print(f"  Processing {img_id} → {artifact}{suffix} ...", end=" ", flush=True)
                    aug = augment_image(pipe, img, artifact, aug_idx=i)
                    aug.save(out_path)
                    print("✓")

                results.append((img_id, str(out_path), diagnosis, artifact))

    # Write metadata CSV
    csv_path = out / "augmented_metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "path", "diagnosis", "artifact"])
        writer.writerows(results)

    print(f"\nDone. {len(results)} records → {csv_path}")


if __name__ == "__main__":
    main()
