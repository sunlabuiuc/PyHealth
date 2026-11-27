"""MEDFAIR on HAM10000: ERM vs. subgroup resampling, three selection rules.

The HAM10000 experiment of MEDFAIR (Zong, Yang and Hospedales, "MEDFAIR:
Benchmarking Fairness for Medical Imaging", ICLR 2023,
https://arxiv.org/abs/2210.01725). A ResNet-18 learns to tell malignant (MEL,
AKIEC) from benign lesions with plain ERM and with subgroup resampling, for
every learning rate and seed. Per method, a learning rate is then chosen
three ways from validation scores averaged over seeds: best overall AUC,
smallest distance to optimal (DTO) and minimax Pareto, the latter judged on
the group that is worst under ERM, as in MEDFAIR. The table gives the test
scores of each choice as mean (std) over seeds.

Without ``--root`` the script writes a small synthetic ISIC 2018 layout to a
temporary directory and runs the whole pipeline as a smoke test; its numbers
mean nothing. On the real data (open access, see ``HAM10000Dataset``; put the
Dataverse ``HAM10000_metadata.csv`` next to the ISIC 2018 folders):

    python examples/derm/medfair_ham10000.py --root /path/to/isic2018_task3 \\
        --cache-dir /path/to/cache --attribute sex --epochs 15

Compare ``--attribute sex`` with the "HAM10000 / Sex" rows of MEDFAIR's
appendix Table A9 (arXiv:2210.01725v2; ERM and resampling, minimax Pareto)
and Table A8 (ERM under each selection rule). Its age results use four
groups taken from the five bins of its Table A2 (0-20 to 80+) in a way the
paper does not state, so they are not directly comparable to
``--attribute age_group``.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from pyhealth.datasets import HAM10000Dataset, PatientSplit, get_dataloader
from pyhealth.metrics.fairness_selection import dto_select, minimax_pareto_select
from pyhealth.metrics.group_fairness import (
    best_f1_threshold,
    group_fairness_metrics_fn,
)
from pyhealth.models import TorchvisionModel
from pyhealth.sampler import get_group_balanced_dataloader
from pyhealth.tasks import HAM10000MalignancyClassification
from pyhealth.trainer import Trainer

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
REPORTED = ["auc", "worst_auc", "auc_gap"]


def write_synthetic_ham10000(root: Path, n_lesions: int, seed: int) -> None:
    """Writes random images in the ISIC 2018 Task 3 layout plus Dataverse
    metadata. Malignant images are redder, more so for men, so the subgroups
    differ in difficulty."""
    rng = np.random.default_rng(seed)
    image_dir = root / "ISIC2018_Task3_Training_Input"
    gt_dir = root / "ISIC2018_Task3_Training_GroundTruth"
    image_dir.mkdir(parents=True)
    gt_dir.mkdir()
    rows = []
    for i in range(n_lesions):
        image = f"ISIC_{i:07d}"
        diagnosis = CLASSES[rng.integers(len(CLASSES))]
        sex = "male" if rng.random() < 0.55 else "female"
        malignant = diagnosis in ("MEL", "AKIEC")
        red = 90 + (100 if malignant else 0) * (1.0 if sex == "male" else 0.4)
        pixels = rng.normal([red, 90, 120], 30, size=(32, 32, 3)).clip(0, 255)
        Image.fromarray(pixels.astype(np.uint8)).save(image_dir / f"{image}.jpg")
        rows.append(
            {
                "image": image,
                "lesion_id": f"HAM_{i:07d}",
                "diagnosis": diagnosis,
                "age": float(rng.choice(np.arange(5, 90, 5))),
                "sex": sex,
            }
        )
    frame = pd.DataFrame(rows)
    onehot = {c: (frame["diagnosis"] == c).astype(float) for c in CLASSES}
    pd.DataFrame({"image": frame["image"], **onehot}).to_csv(
        gt_dir / f"{gt_dir.name}.csv", index=False
    )
    frame.assign(diagnosis_confirm_type="histo")[
        ["image", "lesion_id", "diagnosis_confirm_type"]
    ].to_csv(root / "ISIC2018_Task3_Training_LesionGroupings.csv", index=False)
    frame.rename(columns={"image": "image_id", "diagnosis": "dx"}).assign(
        dx_type="histo", localization="back"
    ).to_csv(root / "HAM10000_metadata.csv", index=False)


def train_one(train, val, train_groups, args, method: str, lr: float, seed: int):
    torch.manual_seed(seed)
    if method == "resampling":
        loader = get_group_balanced_dataloader(
            train, args.batch_size, groups=train_groups, seed=seed
        )
    else:
        loader = get_dataloader(train, args.batch_size, shuffle=True, seed=seed)
    model = TorchvisionModel(
        dataset=train,
        model_name="resnet18",
        model_config={"weights": "DEFAULT" if args.pretrained else None},
    )
    trainer = Trainer(model=model, metrics=["roc_auc"], enable_logging=False)
    trainer.train(
        train_dataloader=loader,
        val_dataloader=get_dataloader(val, args.batch_size),
        epochs=args.epochs,
        optimizer_params={"lr": lr},
        weight_decay=1e-4,
        monitor="loss",
        monitor_criterion="min",
        patience=5,
    )
    return trainer


def score_run(trainer, val, test, groups, batch_size) -> dict[str, dict]:
    """Val and test metrics, the test set at the threshold tuned on val."""
    y_val, p_val, _ = trainer.inference(get_dataloader(val, batch_size))
    y_test, p_test, _ = trainer.inference(get_dataloader(test, batch_size))
    threshold = best_f1_threshold(y_val.ravel(), p_val.ravel())
    return {
        "val": group_fairness_metrics_fn(
            y_val, p_val, groups["val"], threshold=threshold
        ),
        "test": group_fairness_metrics_fn(
            y_test, p_test, groups["test"], threshold=threshold
        ),
    }


def seed_stats(runs: list[dict], part: str) -> tuple[pd.Series, pd.Series]:
    frame = pd.DataFrame([r[part] for r in runs])
    return frame.mean(), frame.std(ddof=0)


def select(settings: dict, names: list[str], ref: int | None) -> dict[str, float]:
    """Learning rate picked by each rule from seed-averaged val scores."""
    lrs = list(settings)
    means = [seed_stats(settings[lr], "val")[0] for lr in lrs]
    aucs = np.array([[m.get(f"auc_{g}", np.nan) for g in names] for m in means])
    return {
        "overall": lrs[int(np.argmax([m["auc"] for m in means]))],
        "dto": lrs[dto_select(aucs)],
        "minimax": lrs[minimax_pareto_select(aucs, group=ref)],
    }


def defined_groups(settings: dict, names: list[str]) -> list[str]:
    """Groups with a val AUC in every run; selection needs all of them."""
    runs = [r for rs in settings.values() for r in rs]
    return [
        g
        for g in names
        if all(np.isfinite(r["val"].get(f"auc_{g}", np.nan)) for r in runs)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", help="ISIC 2018 Task 3 root; synthetic if unset")
    parser.add_argument("--cache-dir", help="PyHealth cache; temporary if unset")
    parser.add_argument("--attribute", default="sex", choices=["sex", "age_group"])
    parser.add_argument("--age-bins", type=int, nargs="+", default=[60])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        root = args.root
        if root is None:
            root = Path(tmp) / "isic2018"
            write_synthetic_ham10000(root, n_lesions=400, seed=0)
            args.image_size, args.epochs, args.pretrained = 32, 2, False
            args.lrs, args.seeds = [1e-3, 1e-4], [0, 1]
        elif not (Path(root) / "HAM10000_metadata.csv").exists():
            raise SystemExit(
                f"No HAM10000_metadata.csv in {root}; the task needs its age and "
                "sex columns (see HAM10000Dataset)."
            )
        dataset = HAM10000Dataset(
            root=str(root), cache_dir=args.cache_dir or str(Path(tmp) / "cache")
        )
        task = HAM10000MalignancyClassification(
            image_size=args.image_size, age_bins=args.age_bins
        )
        train, val, test = dataset.set_task(
            task, split=PatientSplit((0.8, 0.1, 0.1), seed=0)
        )
        # Reading a field decodes the whole cached sample, so collect the
        # groups in one pass per split and reuse them for every run.
        groups = {
            name: [part[i][args.attribute] for i in range(len(part))]
            for name, part in (("train", train), ("val", val), ("test", test))
        }
        names = ["F", "M"] if args.attribute == "sex" else task.age_groups

        results = {}
        for method in ("erm", "resampling"):
            results[method] = {
                lr: [
                    score_run(
                        train_one(train, val, groups["train"], args, method, lr, s),
                        val,
                        test,
                        groups,
                        args.batch_size,
                    )
                    for s in args.seeds
                ]
                for lr in args.lrs
            }

        defined = [set(defined_groups(r, names)) for r in results.values()]
        kept = [g for g in names if all(g in d for d in defined)]
        if not kept:
            raise SystemExit("No group has a val AUC in every run; nothing to select.")
        if len(kept) < len(names):
            print(
                f"Selecting on {kept}; no val AUC for {sorted(set(names) - set(kept))}."
            )
        erm_means = [seed_stats(runs, "val")[0] for runs in results["erm"].values()]
        erm_best = np.array([[m[f"auc_{g}"] for g in kept] for m in erm_means]).max(0)
        ref = int(np.argmin(erm_best))

        rows = []
        for method, settings in results.items():
            for rule, lr in select(settings, kept, ref).items():
                mean, std = seed_stats(settings[lr], "test")
                rows.append(
                    {
                        "method": method,
                        "selection": rule,
                        "lr": lr,
                        **{k: f"{mean[k]:.4f} ({std[k]:.4f})" for k in REPORTED},
                    }
                )
        print(f"Minimax reference group (worst under ERM): {kept[ref]}")
        print(pd.DataFrame(rows).to_string(index=False))
        for part in (train, val, test):
            part.close()


if __name__ == "__main__":
    main()
