"""JointSeqImage on PhysioNet 2012 in-hospital mortality (AAAI 2025 setup).

Reproduces the P12 experiment of Chen et al., "Integrating Sequence and Image
Modeling in Irregular Medical Time Series Through Self-Supervised Learning"
(AAAI 2025): Raindrop's 8:1:1 splits of the 11,988 stays, minority-class
upsampling, 8 epochs at batch size 32, focal loss, a Swin-Base pretrained on
ImageNet-21k reading 384 x 384 line-graph images, and AUROC / AUPRC on the
test set at the epoch with the best validation AUROC.

The splits are Raindrop's published ones (Zhang et al., ICLR 2022; MIT
licence), downloaded from its repository at a fixed commit; ViTST and the
paper use the same. Split ``n`` is run with seed ``n``.

Usage::

    # A few minutes on a laptop: 1,000 stays, a tiny random Swin, one epoch.
    python examples/joint_seq_image_physionet2012_mortality.py \
        --root data/physionet2012 --tiny

    # One run of the paper's setting (split 1).
    python examples/joint_seq_image_physionet2012_mortality.py \
        --root data/physionet2012 --split 1

The PhysioNet 2012 files are downloaded into ``--root`` when missing (open
access).

Results (Raindrop splits 1-3, Swin-Base ImageNet-21k at 224 x 224 instead of
384, the paper's three-step training, 8 epochs, Apple M4 Pro, about 2.4 h per
split)::

    split   best epoch   test AUROC   test AUPRC
    1       1            82.6         44.7
    2       1            82.2         50.4
    3       0            82.2         40.0
    mean +- sd           82.3 +- 0.2  45.0 +- 5.2
    paper (384 x 384, 5 splits)   86.0 +- 0.3  50.4 +- 2.1

Validation AUROC peaks in the first two epochs on every split, then falls.
The paper's appendix with its learning rates is not public, so ``--lr`` and
``--swin-lr`` are guesses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import PhysioNet2012Dataset, get_dataloader
from pyhealth.metrics.binary import binary_metrics_fn
from pyhealth.models import JointSeqImage
from pyhealth.models.joint_seq_image import SWIN_BASE_IN22K
from pyhealth.tasks import (
    PhysioNet2012Mortality,
    physionet2012_mortality_statistics,
    split_by_record_ids,
)

RAINDROP = (
    "https://raw.githubusercontent.com/mims-harvard/Raindrop/"
    "892eb5734e84aa8d18476c6a8975b55b2f30e1d1/P12data/"
)
# SHA-256 of Raindrop's files at that commit, checked before loading them.
RAINDROP_SHA256 = {
    "processed_data/arr_outcomes.npy": "eb9b35208d3b0189b301b0f93ddbab60241a7e72a6a75825388968297393d886",
    "splits/phy12_split1.npy": "61cf02844ff685aa5469a31f606482279d434079b6b29c7691b66e34ad98cc89",
    "splits/phy12_split2.npy": "0415c7010a24573a29540e59f89543089ec67d81b598239c922fff90d73e2cbe",
    "splits/phy12_split3.npy": "8f13eeb58365f8637c8af56d8b2f4b25c65bd535b52f2504dbf33a828b1289fa",
    "splits/phy12_split4.npy": "ea90042db1036593713311556ede7971cbc349b847463aaefa9277321b0d35da",
    "splits/phy12_split5.npy": "b00ac2fcfcf508c0cc15ee69142ce9ec9a3918b0c91f517260a32d008e7be3f6",
}
TINY_SWIN = {"embed_dim": 16, "depths": [1, 1], "num_heads": [1, 2], "window_size": 4}


def raindrop_split(split: int, cache: Path) -> tuple[list[str], list[str], list[str]]:
    """RecordIDs of Raindrop's ``phy12_split{split}`` train / val / test sets.

    The split files index Raindrop's ``arr_outcomes.npy``, whose first column
    is the RecordID of each of the 11,988 stays.
    """
    cache.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in (f"splits/phy12_split{split}.npy", "processed_data/arr_outcomes.npy"):
        dest = cache / name.replace("/", "_")
        if not dest.exists():
            urllib.request.urlretrieve(RAINDROP + name, dest)
        if hashlib.sha256(dest.read_bytes()).hexdigest() != RAINDROP_SHA256[name]:
            raise ValueError(f"{dest} does not match Raindrop's file; delete it and rerun.")
        files[name] = dest
    record_ids = np.load(files["processed_data/arr_outcomes.npy"])[:, 0].astype(int)
    # The split file is a pickled tuple of three index arrays; its checksum
    # was verified above, so unpickling it is safe.
    parts = np.load(files[f"splits/phy12_split{split}.npy"], allow_pickle=True)
    return tuple([str(record_ids[i]) for i in part] for part in parts)


def to_tensors(samples) -> dict[str, torch.Tensor]:
    """All samples as stacked tensors (P12 fits in memory: ~370 MB)."""
    batch = next(iter(get_dataloader(samples, batch_size=len(samples))))
    return {k: batch[k] for k in ("x", "x_time", "mortality")}


def batches(data: dict[str, torch.Tensor], order: torch.Tensor, size: int):
    for start in range(0, len(order), size):
        index = order[start : start + size]
        yield {k: v[index] for k, v in data.items()}


@torch.no_grad()
def evaluate(model: JointSeqImage, data: dict[str, torch.Tensor], size: int) -> dict:
    model.eval()
    probs = [
        model(**batch)["y_prob"].cpu()
        for batch in batches(data, torch.arange(len(data["mortality"])), size)
    ]
    y_prob = torch.cat(probs).view(-1).numpy()
    y_true = data["mortality"].view(-1).numpy()
    return binary_metrics_fn(y_true, y_prob, metrics=["roc_auc", "pr_auc"])


def upsampled_order(labels: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Every negative once and the positives repeated to match, shuffled."""
    labels = labels.view(-1)
    pos, neg = torch.nonzero(labels == 1).view(-1), torch.nonzero(labels == 0).view(-1)
    repeats = max(len(neg) // max(len(pos), 1), 1)
    order = torch.cat([neg, pos.repeat(repeats)])
    return order[torch.randperm(len(order), generator=generator)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="PhysioNet 2012 folder.")
    parser.add_argument("--cache-dir", default=None, help="PyHealth cache folder.")
    parser.add_argument("--output", default="joint_seq_image_p12", help="Results folder.")
    parser.add_argument("--split", type=int, default=1, help="Raindrop split, 1-5.")
    parser.add_argument("--seed", type=int, default=None, help="Defaults to --split.")
    parser.add_argument("--image-type", default="line")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--image-model", default=SWIN_BASE_IN22K)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="All but the Swin.")
    parser.add_argument("--swin-lr", type=float, default=2e-5)
    parser.add_argument(
        "--single-pass", action="store_true",
        help="Run the generator once per batch (faster; see JointSeqImage.train_step).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--tiny", action="store_true", help="Smoke test on the first 1,000 stays.")
    args = parser.parse_args()
    seed = args.split if args.seed is None else args.seed
    torch.manual_seed(seed)
    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    dataset = PhysioNet2012Dataset(
        root=args.root, download=True, cache_dir=args.cache_dir, dev=args.tiny
    )
    train_ids, val_ids, test_ids = raindrop_split(args.split, output / "raindrop")
    present = set(dataset.unique_patient_ids)
    mean, std = physionet2012_mortality_statistics(
        dataset, [i for i in train_ids if i in present]
    )
    samples = dataset.set_task(PhysioNet2012Mortality(mean=mean, std=std, outlier_sd=5.0))
    parts = split_by_record_ids(samples, train_ids, val_ids, test_ids)
    train, val, test = (to_tensors(part) for part in parts)
    print(f"train / val / test stays: {len(train['mortality'])} / "
          f"{len(val['mortality'])} / {len(test['mortality'])}")

    model = JointSeqImage(
        samples,
        image_type=args.image_type,
        image_size=64 if args.tiny else args.image_size,
        image_model=None if args.tiny else args.image_model,
        swin_config=TINY_SWIN if args.tiny else None,
    ).to(device)
    swin = [p for n, p in model.named_parameters() if n.startswith("image_encoder.")]
    rest = [
        p for n, p in model.named_parameters()
        if not n.startswith(("image_encoder.", "discriminator."))
    ]
    optimizer = torch.optim.AdamW(
        [{"params": swin, "lr": args.swin_lr}, {"params": rest, "lr": args.lr}]
    )
    d_optimizer = torch.optim.AdamW(model.discriminator.parameters(), lr=args.lr)
    g_optimizer = (
        None if args.single_pass
        else torch.optim.AdamW(model.generator_parameters(), lr=args.lr)
    )

    generator = torch.Generator().manual_seed(seed)
    epochs = 1 if args.tiny else args.epochs
    best, history = None, []
    checkpoint = output / f"split{args.split}_seed{seed}.pt"
    for epoch in range(epochs):
        start = time.time()
        order = upsampled_order(train["mortality"], generator)
        losses = []
        for step, batch in enumerate(batches(train, order, args.batch_size)):
            losses.append(model.train_step(batch, optimizer, d_optimizer, g_optimizer))
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(order) // args.batch_size} "
                      f"{losses[-1]}", flush=True)
        model.update_clusters()
        val_scores = evaluate(model, val, 2 * args.batch_size)
        record = {
            "epoch": epoch,
            "minutes": (time.time() - start) / 60,
            "loss": {k: float(np.mean([l[k] for l in losses])) for k in losses[0]},
            "val": val_scores,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if best is None or val_scores["roc_auc"] > best["val"]["roc_auc"]:
            best = record
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    test_scores = evaluate(model, test, 2 * args.batch_size)
    result = {
        "split": args.split, "seed": seed, "best_epoch": best["epoch"],
        "val": best["val"], "test": test_scores, "history": history,
        "args": vars(args),
    }
    (output / f"split{args.split}_seed{seed}.json").write_text(json.dumps(result, indent=2))
    print("test", json.dumps(test_scores))


if __name__ == "__main__":
    main()
