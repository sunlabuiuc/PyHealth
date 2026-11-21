"""Early Fusion on MedMod in-hospital mortality (MIMIC-IV + MIMIC-CXR).

Authors: Yuqing Hu (yhu672@gatech.edu), Hanh Trang Do (hdo61@gatech.edu)

Paper:
    Elsharief et al. "MedMod: Multimodal Benchmark for Medical Prediction Tasks
    with Electronic Health Records and Chest X-Ray Scans." CHIL 2025.
    https://proceedings.mlr.press/v287/elsharief25a.html

Trains :class:`~pyhealth.models.EHRCXREarlyFusion`: an EHR LSTM and a CXR
ResNet-34, both loaded from MedFuse's released checkpoints and frozen, so only
the projection and the classifier train. Training follows MedFuse
(``scripts/mortality/train/early.sh``, ``trainers/fusion_trainer.py``): Adam,
lr 9.515e-05, betas (0.9, 0.9), batch 16, up to 50 epochs, the learning rate
halved when the validation loss stops falling for 10 epochs, validation before
each epoch, the best epoch by validation AUROC, and early stopping after 15
epochs without improvement. The data loader gives image paths, so the images
of each batch are loaded and transformed here, with augmentation in training.

Smoke test on synthetic data (no MIMIC, no checkpoints; scores mean nothing):

    python examples/mimic4_medmod_ihm_early_fusion.py --synthetic 300 --epochs 2

Real data (credentialed MIMIC-IV v1.0 and MIMIC-CXR-JPG, plus the MedFuse
checkpoints from https://github.com/nyuad-cai/MedFuse, Git LFS):

    python examples/mimic4_medmod_ihm_early_fusion.py \\
        --ehr-root /data/mimiciv/1.0 --cxr-root /data/mimic-cxr-jpg/2.0.0 \\
        --image-dir /data/mimic-cxr-jpg/2.0.0/resized --cache-dir /data/cache \\
        --ehr-weights checkpoints/mortality/uni_ehr_all/best_checkpoint.pth.tar \\
        --cxr-weights checkpoints/cxr_rad_full/best_checkpoint.pth.tar

Ablation: ``--train-encoders`` also trains both encoders (the paper's Joint
Fusion) instead of freezing them.

MedSigLIP extension: ``--embeddings FILE`` (written by
``python -m pyhealth.datasets.medmod.medsiglip``) replaces the CXR ResNet-34
with the stored embeddings of a frozen MedSigLIP image encoder: the model is
built with ``cxr_backbone=nn.Identity()`` and ``cxr_feature_dim=1152``, each
batch's image input is looked up by image file name, and ``--cxr-weights`` is
not used. The stored vectors are fixed, so there is no image augmentation.

Results (test set): pending a real-data run of this script.
Paper: AUROC 0.842 +- 0.004, AUPRC 0.515 +- 0.020.

Deviation: frozen encoders stay in eval mode during training; MedFuse sets the
whole model, frozen encoders included, to train mode.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from pyhealth.datasets import get_dataloader
from pyhealth.datasets.medmod.image_presets import get_preset, load_cxr
from pyhealth.datasets.medmod.medsiglip import load_embeddings, lookup_embeddings
from pyhealth.models import EHRCXREarlyFusion

# sha256 of MedFuse's Git LFS checkpoint files.
EHR_SHA256 = "c545d8d536882948f25b4dd8a136c43da49fd10d4519e8b24b821109ad22f552"
CXR_SHA256 = "a02628fcec408cc59ac1464015ddcf87b98840e37812563ce648ddce11b3bf90"


def load_checkpoint(path: str, sha256: str) -> dict:
    """Checks a MedFuse checkpoint file and returns its state dict.

    The file also stores numpy scalars, which torch's restricted loader
    (``weights_only=True``) accepts once they are allowed explicitly.
    """
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"{path}: not MedFuse's released checkpoint (sha256)")
    from numpy._core.multiarray import scalar

    allowed = [
        (scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        type(np.dtype("float64")),
        codecs.encode,
    ]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint["state_dict"]


def ehr_state_dict(state: dict) -> dict:
    """MedFuse's EHR LSTM is two 1-layer LSTMs, ``ehr_model.layer{k}.*_l0``.

    In one 2-layer ``nn.LSTM`` they are ``*_l{k}``.
    """
    weights = {}
    for key, value in state.items():
        if key.startswith("ehr_model.layer"):
            layer, name = key[len("ehr_model.layer") :].split(".", 1)
            weights[name.replace("_l0", f"_l{layer}")] = value
    return weights


def cxr_state_dict(state: dict) -> dict:
    """MedFuse's ``cxr_model.vision_backbone`` is a ResNet-34 without ``fc``."""
    prefix = "cxr_model.vision_backbone."
    return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}


def load_data(args: argparse.Namespace):
    """Returns the train, validation and test samples of the fixed MedMod split."""
    if args.synthetic:
        from pyhealth.datasets import split_by_medmod
        from pyhealth.datasets.medmod.synthetic import create_synthetic_samples

        samples = create_synthetic_samples(
            hours_zero="keep", n_stays=args.synthetic, seed=args.seed
        )
        return split_by_medmod(samples)
    from pyhealth.datasets import MedModMIMIC4Dataset, MedModSplit
    from pyhealth.tasks import MedModInHospitalMortality

    dataset = MedModMIMIC4Dataset(
        ehr_root=args.ehr_root, cxr_root=args.cxr_root, cache_dir=args.cache_dir
    )
    task = MedModInHospitalMortality(hours_zero="keep", image_dir=args.image_dir)
    return dataset.set_task(task, split=MedModSplit())


def add_images(batch: dict, transform, table=None) -> dict:
    """Loads and transforms the images of a batch into ``batch["cxr"]``, or,
    with ``table`` (``--embeddings``), looks up their stored embeddings."""
    if table is not None:
        batch["cxr"] = lookup_embeddings(batch["image_path"], *table)
        return batch
    images = [transform(load_cxr(path)) for path in batch["image_path"]]
    batch["cxr"] = torch.stack(images)
    return batch


@torch.no_grad()
def evaluate(model, samples, transform, table=None) -> tuple[float, float, float]:
    """Returns AUROC, AUPRC and the mean loss on ``samples``."""
    model.eval()
    y_true, y_prob, losses = [], [], []
    for batch in get_dataloader(samples, batch_size=64):
        output = model(**add_images(batch, transform, table))
        losses.append(float(output["loss"]) * len(batch["image_path"]))
        y_true.append(output["y_true"].cpu().reshape(-1))
        y_prob.append(output["y_prob"].cpu().reshape(-1))
    y, p = torch.cat(y_true).numpy(), torch.cat(y_prob).numpy()
    return roc_auc_score(y, p), average_precision_score(y, p), sum(losses) / len(y)


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", type=int, default=0, help="synthetic stays")
    parser.add_argument("--ehr-root", help="MIMIC-IV v1.0 root")
    parser.add_argument("--cxr-root", help="MIMIC-CXR-JPG root")
    parser.add_argument("--image-dir", help="resized MIMIC-CXR images")
    parser.add_argument("--cache-dir", help="PyHealth cache directory")
    parser.add_argument("--ehr-weights", help="MedFuse uni_ehr_all checkpoint")
    parser.add_argument("--cxr-weights", help="MedFuse cxr_rad_full checkpoint")
    parser.add_argument("--train-encoders", action="store_true")
    parser.add_argument("--embeddings", help="MedSigLIP .npz instead of a ResNet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=9.515e-05)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    # MPS only with --device mps: it crashes on some macOS and torch versions.
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    real = (args.ehr_root, args.cxr_root, args.image_dir, args.cache_dir)
    if not args.synthetic and not all(real):
        parser.error(
            "set --synthetic N, or all of --ehr-root, --cxr-root, "
            "--image-dir and --cache-dir"
        )
    if args.embeddings and args.cxr_weights:
        parser.error("--cxr-weights is for the ResNet, not --embeddings")
    return args


def main() -> None:
    """Trains Early Fusion and prints the test AUROC and AUPRC."""
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # the image presets draw from numpy's RNG

    train, val, test = load_data(args)
    print(f"stays train/val/test: {len(train)}/{len(val)}/{len(test)}")
    weights = {}
    if args.ehr_weights:
        state = load_checkpoint(args.ehr_weights, EHR_SHA256)
        weights["ehr_state_dict"] = ehr_state_dict(state)
    if args.cxr_weights:
        state = load_checkpoint(args.cxr_weights, CXR_SHA256)
        weights["cxr_state_dict"] = cxr_state_dict(state)
    table, image_encoder = None, {}
    if args.embeddings:
        table = load_embeddings(args.embeddings)[:2]
        image_encoder = {
            "cxr_backbone": nn.Identity(),
            "cxr_feature_dim": table[1].shape[1],
        }
    if len(weights) + bool(table) < 2:
        print("Encoders without MedFuse weights are random: smoke test only.")
    model = EHRCXREarlyFusion(
        train, freeze_encoders=not args.train_encoders, **weights, **image_encoder
    ).to(args.device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr, betas=(0.9, 0.9))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10
    )
    train_tf, eval_tf = (
        get_preset("early_fusion", train=True),
        get_preset("early_fusion"),
    )

    best_auroc, best_state, waited = -1.0, None, 0
    for epoch in range(args.epochs + 1):
        # As MedFuse: validate before each epoch; the last pass only validates.
        val_auroc, _, val_loss = evaluate(model, val, eval_tf, table)
        if val_auroc > best_auroc:
            best_auroc, waited = val_auroc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
        print(f"epoch {epoch}: val AUROC {val_auroc:.4f}")
        if epoch == args.epochs or waited >= args.patience:
            break
        model.train()
        loader = get_dataloader(train, batch_size=args.batch_size, shuffle=True)
        for batch in loader:
            loss = model(**add_images(batch, train_tf, table))["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(val_loss)

    model.load_state_dict(best_state)
    auroc, auprc, _ = evaluate(model, test, eval_tf, table)
    print("RESULT", json.dumps({"test_auroc": auroc, "test_auprc": auprc}))


if __name__ == "__main__":
    main()
