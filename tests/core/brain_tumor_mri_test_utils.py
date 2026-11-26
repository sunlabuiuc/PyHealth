"""Synthetic Brain Tumor MRI image tree shared by the dataset and task tests."""

import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def _write_image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 255, (16, 16), dtype=np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


def build_brain_tumor_root(root: Path) -> Path:
    """Ten images, two of them copies of others.

    - ``Training/glioma_tumor/g_copy.jpg`` is a copy of the Testing image
      ``Testing/glioma_tumor/t_g.jpg`` (a cross-split duplicate).
    - ``Training/no_tumor/n_copy.jpg`` copies ``Training/no_tumor/n1.jpg``.
    """
    images = {
        "Training/no_tumor/n1.jpg": 1,
        "Training/no_tumor/n_copy.jpg": 1,
        "Training/glioma_tumor/g1.jpg": 2,
        "Training/glioma_tumor/g_copy.jpg": 3,
        "Training/meningioma_tumor/m1.jpg": 4,
        "Training/pituitary_tumor/p1.jpg": 5,
        "Testing/no_tumor/t_n.jpg": 6,
        "Testing/glioma_tumor/t_g.jpg": 3,
        "Testing/meningioma_tumor/t_m.jpg": 7,
        "Testing/pituitary_tumor/t_p.png": 8,
    }
    for relative, seed in images.items():
        target = root / relative
        source = next(
            (root / other for other, s in images.items()
             if s == seed and (root / other).exists()),
            None,
        )
        if source is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        else:
            _write_image(target, seed)
    (root / "Training" / "no_tumor" / "notes.txt").write_text("not an image")
    return root
