"""End-to-end demo on real medical data (MSD Task04 Hippocampus).

This script exercises the full dataset + task pipeline on a small
publicly-available slice of the Medical Segmentation Decathlon
Hippocampus dataset (3 patients, MR volumes, ~36 axial slices each,
with instance labels: 1 = anterior, 2 = posterior hippocampus).

Unlike ``lidc_retina_unet_detection_retinaunet.py``, which uses a
synthetic corpus for reproducible ablation, this script runs against
**real acquired volumes with real segmentation masks**, and emits a
visualization with ground-truth bounding boxes overlaid so you can
eyeball that the mask-to-box pipeline behaves sensibly on non-toy data.

Data prep
---------
The example expects data already laid out on disk at::

    examples/data/hippocampus/
      patient_367/{volume.npy, mask.npy}
      patient_304/{volume.npy, mask.npy}
      patient_204/{volume.npy, mask.npy}

These were generated from the public MSD Task04 tar::

    curl -L -o Task04_Hippocampus.tar \\
      https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar

followed by extracting ``imagesTr/labelsTr`` for patients 367/304/204,
loading with nibabel, and saving as numpy arrays. The conversion script
is committed inline in the repo's development history (see
``IMPLEMENTATION_NOTES.md``); re-running it is a one-liner.

Run
---
::

    PYTHONPATH=. python examples/hippocampus_retina_unet_demo.py

Writes a figure to ``examples/figures/hippocampus_demo.png``.
"""

from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pyhealth.datasets.retina_unet_ct_dataset import RetinaUNetCTDataset
from pyhealth.tasks.retina_unet_detection import RetinaUNetDetectionTask


DATA_ROOT = Path(__file__).parent / "data" / "hippocampus"
FIG_PATH = Path(__file__).parent / "figures" / "hippocampus_demo.png"


def pick_slices(processed, per_patient: int = 3):
    """Return a list of (patient_id, processed_sample) for visualization.

    Selects up to ``per_patient`` slices per patient, preferring slices
    with the most detected boxes (usually both anterior + posterior
    visible).
    """
    by_patient = {}
    for sample in processed:
        pid = sample["patient_id"]
        by_patient.setdefault(pid, []).append(sample)

    chosen = []
    for pid, samples in by_patient.items():
        samples.sort(key=lambda s: (-s["boxes"].shape[0], s["slice_idx"]))
        chosen.extend((pid, s) for s in samples[:per_patient])
    return chosen


def draw_panel(ax, sample, patient_id):
    img = sample["image"].squeeze(-1)
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_title(
        f"{patient_id}  slice={sample['slice_idx']}  "
        f"boxes={sample['boxes'].shape[0]}",
        fontsize=8,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # Instance IDs from the raw mask tell us anterior (1) vs posterior (2);
    # the task-level `labels` is all-1 (single-class), so we recover the
    # color from the mask directly.
    mask = sample["mask"]
    instance_ids = [i for i in np.unique(mask) if i != 0]
    palette = {1: "#e74c3c", 2: "#3498db"}  # anterior red, posterior blue

    for inst_id in instance_ids:
        ys, xs = np.where(mask == inst_id)
        if len(xs) == 0:
            continue
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        rect = Rectangle(
            (x_min - 0.5, y_min - 0.5),
            x_max - x_min + 1,
            y_max - y_min + 1,
            linewidth=1.2,
            edgecolor=palette.get(inst_id, "#ffffff"),
            facecolor="none",
        )
        ax.add_patch(rect)


def main():
    if not DATA_ROOT.is_dir():
        raise SystemExit(
            f"Expected data at {DATA_ROOT} — see the docstring for prep steps."
        )

    ds = RetinaUNetCTDataset(
        root=str(DATA_ROOT),
        skip_empty_slices=True,
    )

    print(f"Loaded {ds.stats()} from {DATA_ROOT}")
    print(f"Patients: {list(ds.iter_patients())}")

    processed = ds.set_task(RetinaUNetDetectionTask())
    total_boxes = sum(s["boxes"].shape[0] for s in processed)
    print(f"Task produced {total_boxes} boxes across {len(processed)} slices")

    per_patient_counts = {}
    for s in processed:
        per_patient_counts.setdefault(s["patient_id"], 0)
        per_patient_counts[s["patient_id"]] += s["boxes"].shape[0]
    for pid, n in sorted(per_patient_counts.items()):
        print(f"  {pid}: {n} boxes")

    picks = pick_slices(processed, per_patient=3)
    n = len(picks)
    rows = 3
    cols = int(np.ceil(n / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.atleast_2d(axes).ravel()

    for ax, (pid, sample) in zip(axes, picks):
        draw_panel(ax, sample, pid)
    for ax in axes[len(picks):]:
        ax.axis("off")

    fig.suptitle(
        "RetinaUNetCTDataset + RetinaUNetDetectionTask\n"
        "MSD Task04 Hippocampus — red=anterior (label 1), blue=posterior (label 2)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"Wrote {FIG_PATH}")


if __name__ == "__main__":
    main()
