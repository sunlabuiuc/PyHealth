# Contributor: Aayan Verma (aayanv2)
# Contributor: Bryson Lee (brysonl3)
# Contributor: Xin Zhang (xinz13)
# Paper: A Study of Artifacts on Melanoma Classification under Diffusion-Based
# Perturbations (Jin and Ghassemi, CHIL 2025)
# https://proceedings.mlr.press/v287/jin25b.html
r"""Artifact sensitivity of a ResNet-50 melanoma classifier on PH2.

Trains a ResNet-50 on each of the paper's 12 dataset modes of PH2 (the default run
uses the six region and box modes; ``--views`` selects others), then tests it
with hand-drawn artifacts added. This reproduces, on PH2, the input-disruption
experiment of Jin and Ghassemi, "A Study of Artifacts on Melanoma Classification
under Diffusion-Based Perturbations" (CHIL 2025,
https://proceedings.mlr.press/v287/jin25b.html), and extends it with a control
the paper leaves as future work.

What the script does
--------------------
1. **Reproduction (Table 2, source column).** For each dataset mode ("view")
   in ``--views`` it trains an ImageNet-pretrained ResNet-50 (Adam, lr 1e-4,
   batch size 32, 10 epochs, 224 px, as in the paper) with 5-fold
   cross-validation on PH2 and reports the mean and standard deviation of the
   test AUROC across folds. Table 2 of the paper gives, for comparison, PH2
   source AUROCs of 0.975 (whole), 0.952 (lesion), 0.884 (background),
   0.880 (bbox), 0.845 (bbox70) and 0.865 (bbox90).
2. **Extension A: artifact sensitivity with hand-drawn artifacts.** The paper
   adds artifacts with diffusion models. Here each trained model is also tested
   on its test fold with a hand-drawn dark corner, gel bubble, ink ring, patch
   or ruler (``PH2ArtifactClassification(artifact=...)``). The artifact is
   never drawn on a lesion pixel, so the lesion is provably unchanged (the
   script checks this), and the result does not depend on a generative model
   that could alter the lesion.
3. **Extension B: uncertainty.** The paper reports only the standard deviation
   over 5 folds. We also pool every image's out-of-fold prediction and report
   a paired bootstrap 95% confidence interval of ``AUROC(original) -
   AUROC(artifact)``; an interval that excludes 0 means the artifact really
   moves the classifier.

Results on the real PH2 archive
-------------------------------
One run, seed 0, ImageNet-pretrained ResNet-50, 224 px, 10 epochs, 5-fold CV
(Apple GPU, about 50 minutes, run with ``--views`` listing all 12 modes, as in
the Usage section below). Test AUROC, mean +- std over
folds, next to the paper's Table 2 (PH2 source column; the paper's std is
0.01-0.11)::

    view              ours           paper   diff
    whole             0.965 +- 0.034  0.975  -0.010
    lesion            0.930 +- 0.039  0.952  -0.022
    background        0.907 +- 0.049  0.884  +0.023
    bbox              0.885 +- 0.031  0.880  +0.005
    bbox70            0.829 +- 0.059  0.845  -0.016
    bbox90            0.713 +- 0.065  0.865  -0.152
    whole_high        0.590 +- 0.051  0.903  -0.313
    lesion_high       0.948 +- 0.056  0.877  +0.071
    background_high   0.846 +- 0.051  0.908  -0.062
    whole_low         0.966 +- 0.023  0.925  +0.041
    lesion_low        0.947 +- 0.044  0.935  +0.012
    background_low    0.848 +- 0.031  0.890  -0.042

Reading the comparison:

- 8 of the 12 modes are within 0.05 of the paper, and the paper's headline
  ordering holds: whole > lesion > background > bbox > bbox70. Whole-image
  training beats lesion-only and background-only, as the paper claims.
- Two modes differ a lot. ``whole_high`` is far lower (0.59 vs 0.90): the paper
  itself says its PH2 high-pass results are probably inflated by overfitting on
  200 images, and our value matches its ISIC finding that a classifier
  "struggles to learn from high-frequency signals". ``bbox90`` is lower and
  falls steadily with the box size, while the paper's rises from bbox70 to
  bbox90. We cannot tell why without its code; the paper does not say how the
  box is masked or how the high-pass is scaled, and we chose black fill and a
  0.5 shift.
- Our ``lesion_high`` (0.948) is far above ``whole_high`` (0.590). A likely
  reason, not tested here: filtering after masking turns the lesion border into
  a bright outline, which hands the model the lesion's shape.
- This is one seed on 200 images; most differences under 0.05 are inside one
  fold std.

Extension A (hand-drawn artifacts), ``whole`` view, AUROC drop when the artifact
is added to every test image (paper's Table 4, PH2 column, diffusion artifacts,
in brackets): ink 0.028 [0.016], dark corner 0.012 [-0.003], ruler 0.009
[0.009], patches 0.007 [-0.001], gel bubble 0.003 [0.002]. Ink is the largest
drop in both, but the drops are small. In the ``whole`` view (Extension B) only
``ink`` (CI +0.002 to +0.043) clearly excludes 0; ``ruler`` barely does (CI
+0.0004 to +0.018), and the other three include 0. Across all views, 8 of the 45
informative intervals exclude 0, against about 2 expected by chance, so only the
repeated ink result should be trusted. The ink ring is drawn around the lesion,
so it is partly more disruptive by construction. The ``lesion`` views change by
exactly 0.000 under every artifact, confirming the lesion is untouched.

Usage
-----
Real PH2 (download it from the authors; see
:class:`~pyhealth.datasets.PH2Dataset`)::

    python examples/derm/ph2_artifact_classification_resnet50.py \
        --root /path/to/PH2Dataset --pretrained

This trains the six default modes (whole, lesion, background, bbox, bbox70, bbox90).
The 12-row table above needs all 12 modes::

    python examples/derm/ph2_artifact_classification_resnet50.py \
        --root /path/to/PH2Dataset --pretrained --views whole lesion background \
        bbox bbox70 bbox90 whole_high lesion_high background_high whole_low \
        lesion_low background_low

Self-contained smoke test, with a synthetic PH2-like archive in which rulers
appear only on malignant images and the lesions are uninformative (a planted
shortcut). A classifier trained on
it should lose AUROC when rulers are added to every image::

    python examples/derm/ph2_artifact_classification_resnet50.py --quick

Notes: 224 px images cost about 600 KB each per view and artifact in the task
cache, so the default run (6 views x 6 renderings) needs about 4 GB of disk (twice
that for all 12 views), and a CPU run takes hours; use ``--views``, ``--artifacts``,
``--image-size`` and ``--epochs`` to shrink it. ``--quick`` uses 64 px images,
ResNet-18, 15 epochs and the ``whole`` view, and finishes in about a minute on a
CPU. Expected: AUROC 1.0 on the original images and a clear drop (about 0.7)
with rulers added.
"""

import argparse
import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from pyhealth.datasets import PH2Dataset, SampleDataset, get_dataloader
from pyhealth.models import TorchvisionModel
from pyhealth.tasks import PH2ArtifactClassification
from pyhealth.tasks.dermoscopy_artifact_views import ARTIFACTS, VIEWS, add_artifact
from pyhealth.trainer import Trainer

DEFAULT_VIEWS = ("whole", "lesion", "background", "bbox", "bbox70", "bbox90")


# --------------------------------------------------------------------------
# Synthetic PH2-like archive (only used when --root is not given)
# --------------------------------------------------------------------------
def make_synthetic_ph2(root: Path, n_images: int = 90, seed: int = 0) -> None:
    """Write a small PH2-shaped archive whose only class signal is a ruler.

    Lesions are identical dark ellipses on skin-coloured noise for every class,
    so the lesion carries no information. Every melanoma, and no nevus, gets a
    ruler. A classifier that scores well on this archive can only be reading
    the ruler, so adding rulers to all test images must hurt it: a built-in
    check that the artifact-sensitivity test detects a real shortcut.

    Called by :func:`main` when ``--root`` is not given.

    Args:
        root: Empty folder to write the archive into.
        n_images: Number of images; one in three is a melanoma.
        seed: Seed for the image noise and the lesion sizes.
    """
    rng = np.random.default_rng(seed)
    size = 96
    yy, xx = np.mgrid[0:size, 0:size]
    rows = []
    for i in range(n_images):
        name = f"IMD{i + 1:03d}"
        malignant = i % 3 == 2  # one melanoma in three, like PH2's 40/200
        radius = rng.uniform(14, 24, 2)
        mask = (
            ((yy - size / 2) / radius[0]) ** 2 + ((xx - size / 2) / radius[1]) ** 2
        ) < 1
        skin = np.array([0.85, 0.65, 0.55]) + rng.normal(0, 0.03, (size, size, 3))
        lesion_colour = np.array([0.40, 0.25, 0.18])
        image = np.where(mask[..., None], lesion_colour, skin).astype(np.float32)
        image += rng.normal(0, 0.02, image.shape).astype(np.float32)
        image = np.clip(image, 0, 1)
        if malignant:
            image = add_artifact(image, mask, "ruler", seed=i)
        folder = root / "PH2 Dataset images" / name
        (folder / f"{name}_Dermoscopic_Image").mkdir(parents=True)
        (folder / f"{name}_lesion").mkdir(parents=True)
        Image.fromarray((image * 255).astype(np.uint8)).save(
            folder / f"{name}_Dermoscopic_Image" / f"{name}.bmp"
        )
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            folder / f"{name}_lesion" / f"{name}_lesion.bmp"
        )
        rows.append([name, "", 2 if malignant else int(i % 2)])
    pd.DataFrame(
        [["Image Name", "Histological Diagnosis", "Clinical Diagnosis"]] + rows
    ).to_csv(root / "PH2_dataset.csv", index=False, header=False)


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
def build_samples(
    dataset: PH2Dataset,
    view: str,
    artifact: str | None,
    args: argparse.Namespace,
) -> tuple[SampleDataset, dict[str, int]]:
    """Render every image under one view/artifact and index it by image id.

    Called by :func:`main` once per view and artifact. The index lets the
    original and artifact versions of a test image be matched by id, not by
    position.

    Args:
        dataset: The PH2 dataset.
        view: One of the 12 dataset modes.
        artifact: Artifact to draw, or None for the original images.
        args: Parsed command line (uses ``image_size`` and ``seed``).

    Returns:
        The processed samples, and a dict mapping each image id to its index.
    """
    task = PH2ArtifactClassification(
        view=view, artifact=artifact, image_size=args.image_size, seed=args.seed
    )
    samples = dataset.set_task(task)
    ids = [samples[i]["image_id"] for i in range(len(samples))]
    return samples, {image_id: i for i, image_id in enumerate(ids)}


def train_model(train_samples: SampleDataset, args: argparse.Namespace) -> Trainer:
    """Fine-tune a torchvision model on one fold, with the paper's optimiser.

    Called by :func:`main` once per fold. Trains for a fixed number of epochs
    with Adam and no validation set, as the paper does.

    Args:
        train_samples: Training images of this fold (original, no artifact).
        args: Parsed command line (uses ``model``, ``pretrained``, ``device``,
            ``batch_size``, ``epochs`` and ``lr``).

    Returns:
        The trainer holding the fine-tuned model.
    """
    model = TorchvisionModel(
        dataset=train_samples,
        model_name=args.model,
        model_config={"weights": "DEFAULT" if args.pretrained else None},
    )
    trainer = Trainer(model=model, device=args.device, enable_logging=False)
    trainer.train(
        train_dataloader=get_dataloader(train_samples, args.batch_size, shuffle=True),
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
    )
    return trainer


def predict(
    trainer: Trainer,
    samples: SampleDataset,
    index_of: dict[str, int],
    image_ids: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    """Return P(malignant) for ``image_ids``, in that order.

    Called by :func:`main` for the original images and for each artifact.

    Args:
        trainer: Trainer holding the fitted model.
        samples: Samples of one view and artifact.
        index_of: Maps an image id to its index in ``samples``.
        image_ids: Ids of the images to score.
        batch_size: Batch size for inference.

    Returns:
        Float array with one malignancy probability per id in ``image_ids``.
    """
    subset = samples.subset([index_of[i] for i in image_ids])
    _, y_prob, _ = trainer.inference(get_dataloader(subset, batch_size))
    subset.close()
    return np.asarray(y_prob).reshape(-1)


def paired_bootstrap_delta(
    y: np.ndarray,
    p_original: np.ndarray,
    p_perturbed: np.ndarray,
    n: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap 95% CI of AUROC(original) - AUROC(perturbed) over images.

    Both AUROCs are computed on the same resampled images, so the interval
    reflects the change itself, not the spread between folds. Called by
    :func:`main` once per view and artifact.

    Args:
        y: Binary labels, one per image.
        p_original: Out-of-fold probabilities on the original images.
        p_perturbed: Out-of-fold probabilities on the artifact images.
        n: Number of bootstrap resamples.
        seed: Seed for the resampling.

    Returns:
        The 2.5th and 97.5th percentile of the AUROC difference. An interval
        that excludes 0 means the artifact changes the classifier.
    """
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        pick = rng.integers(0, len(y), len(y))
        if len(set(y[pick])) < 2:  # a resample needs both classes
            continue
        deltas.append(
            roc_auc_score(y[pick], p_original[pick])
            - roc_auc_score(y[pick], p_perturbed[pick])
        )
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def check_lesion_is_untouched(
    dataset: PH2Dataset, args: argparse.Namespace, n_checked: int = 5
) -> None:
    """Confirm that every artifact leaves the lesion pixels bit-identical.

    Called by :func:`main` before training. Renders the ``lesion`` view with
    and without each artifact and asserts the two are equal.

    Args:
        dataset: The PH2 dataset.
        args: Parsed command line (uses ``artifacts``, ``image_size``, ``seed``).
        n_checked: Number of images to check.

    Raises:
        AssertionError: If an artifact changes a lesion pixel.
    """
    probe = {
        a: PH2ArtifactClassification("lesion", a, args.image_size, args.seed)
        for a in (None, *args.artifacts)
    }
    patients = dataset.unique_patient_ids[:n_checked]
    for patient_id in patients:
        (event,) = dataset.get_patient(patient_id).get_events(event_type="ph2")
        reference = probe[None].render(event.image_path, event.mask_path, patient_id)
        for artifact in args.artifacts:
            changed = probe[artifact].render(
                event.image_path, event.mask_path, patient_id
            )
            assert np.array_equal(reference, changed), (patient_id, artifact)
    print(
        f"Lesion check: {len(args.artifacts)} artifacts leave the lesion of "
        f"{len(patients)} images unchanged."
    )


def main(args: argparse.Namespace) -> None:
    """Run the artifact-sensitivity experiment and print the results table.

    For every view: train once per fold on the original images, test on the
    original and every artifact version, then report the fold AUROCs and the
    paired bootstrap interval of the AUROC change. Writes a CSV when
    ``--output-dir`` is given.

    Args:
        args: Parsed command line, from :func:`parse_args`.
    """
    torch.manual_seed(args.seed)
    workdir = Path(args.workdir)
    root = args.root
    if root is None:
        root = str(workdir / "synthetic_ph2")
        make_synthetic_ph2(Path(root), seed=args.seed)
        print(f"No --root given: using a synthetic PH2-like archive at {root}")
    dataset = PH2Dataset(root=root, cache_dir=str(workdir / "cache"))
    check_lesion_is_untouched(dataset, args)

    labels = {}
    base, base_index = build_samples(dataset, "whole", None, args)
    for image_id, i in base_index.items():
        labels[image_id] = int(float(base[i]["malignant"]))
    base.close()
    image_ids = np.array(sorted(labels))
    y = np.array([labels[i] for i in image_ids])
    print(f"{len(y)} images, {y.sum()} malignant, {args.folds}-fold CV")

    folds = list(
        StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(
            image_ids, y
        )
    )
    renderings = (None, *args.artifacts)
    records = []
    for view in args.views:
        samples = {}
        for artifact in renderings:
            samples[artifact] = build_samples(dataset, view, artifact, args)
        fold_auc = {a: [] for a in renderings}
        oof = {a: np.zeros(len(y)) for a in renderings}
        for fold, (train_idx, test_idx) in enumerate(folds):
            train_set, train_index = samples[None]
            train = train_set.subset([train_index[i] for i in image_ids[train_idx]])
            trainer = train_model(train, args)
            train.close()
            test_ids = image_ids[test_idx]
            for artifact in renderings:
                sset, sindex = samples[artifact]
                prob = predict(trainer, sset, sindex, test_ids, args.batch_size)
                oof[artifact][test_idx] = prob
                fold_auc[artifact].append(roc_auc_score(y[test_idx], prob))
            print(
                f"[{view}] fold {fold + 1}/{args.folds} AUROC {fold_auc[None][-1]:.3f}"
            )
        for artifact in renderings:
            row = {
                "view": view,
                "artifact": artifact or "original",
                "auroc_mean": float(np.mean(fold_auc[artifact])),
                "auroc_std": float(np.std(fold_auc[artifact])),
            }
            if artifact is not None:
                low, high = paired_bootstrap_delta(y, oof[None], oof[artifact])
                row["delta_auroc_ci95"] = (float(low), float(high))
            records.append(row)
        for sset, _ in samples.values():
            sset.close()

    table = pd.DataFrame(records)
    print(
        "\nAUROC over folds (mean +- std); delta = original - artifact, "
        "with a paired-bootstrap 95% CI over pooled out-of-fold predictions"
    )
    for row in records:
        text = (
            f"{row['view']:<12} {row['artifact']:<12} "
            f"{row['auroc_mean']:.3f} +- {row['auroc_std']:.3f}"
        )
        if "delta_auroc_ci95" in row:
            low, high = row["delta_auroc_ci95"]
            text += f"   delta CI95 [{low:+.3f}, {high:+.3f}]"
        print(text)
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "ph2_artifact_sensitivity.csv", index=False)
        (out / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))
        print(f"Saved results to {out}")


def parse_args() -> argparse.Namespace:
    """Parse the command line; ``--quick`` only changes the defaults.

    Returns:
        The parsed arguments, plus ``workdir``, a fresh temporary folder that
        holds the task caches and the synthetic archive.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=None, help="PH2 folder; omit for synthetic")
    parser.add_argument(
        "--views", nargs="+", default=list(DEFAULT_VIEWS), choices=VIEWS
    )
    parser.add_argument(
        "--artifacts", nargs="+", default=list(ARTIFACTS), choices=ARTIFACTS
    )
    parser.add_argument("--model", default="resnet50")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="start from ImageNet weights (downloads them)",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--quick", action="store_true", help="tiny, fast settings for a smoke test"
    )
    if parser.parse_known_args()[0].quick:  # smaller defaults; flags still win
        parser.set_defaults(
            model="resnet18",
            image_size=64,
            epochs=15,
            folds=3,
            views=["whole"],
        )
    args = parser.parse_args()
    args.workdir = tempfile.mkdtemp(prefix="ph2_example_")
    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    finally:  # the task caches are large and only needed during the run
        shutil.rmtree(args.workdir, ignore_errors=True)
