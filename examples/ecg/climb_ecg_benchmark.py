"""CLIMB's ECG benchmark: ECG-JEPA and a Transformer on four 12-lead datasets.

Reproduces the ECG rows of Table 25 of Dai et al., "CLIMB: Data Foundations
for Large Scale Multimodal Clinical Foundation Models" (ICML 2025,
https://arxiv.org/abs/2503.07667): one model is trained on the training sets
of PTB-XL, CPSC 2018, Georgia and Chapman-Shaoxing together, on CLIMB's seven
classes, and scored on each validation set (CLIMB has no test split) with
macro AUROC, sensitivity and specificity.

- Labels and the 80/20 splits come from CLIMB's split files, downloaded from
  its repository at a fixed commit and checked against their SHA-256 (CLIMB's
  code has no licence, so nothing of it is copied here; its splits are random
  and unseeded, so they cannot be regenerated).
- ``--model ecg_jepa`` fine-tunes :class:`~pyhealth.models.ECGJEPA` from a
  published checkpoint (``--checkpoint``, from
  https://github.com/sehunfromdaegu/ECG_JEPA) on 8 leads x 2,500 samples
  (10 s at 250 Hz).
- ``--model transformer`` trains CLIMB's Transformer baseline (dim 256, 12
  blocks, 8 heads, 25-sample segments of all 12 leads at 500 Hz) from scratch;
  it is written here from the paper's description.

Data (open, CC BY 4.0): the ``training/`` folder of the PhysioNet/CinC
Challenge 2020 or 2021 (``cpsc_2018``, ``georgia``, ``ptb-xl``) for
:class:`~pyhealth.datasets.CardiologyDataset`, and the Challenge 2021
``chapman_shaoxing`` folder (or PhysioNet ``ecg-arrhythmia``) for
:class:`~pyhealth.datasets.ChapmanShaoxingDataset`.

Usage::

    # A few minutes: dev mode (1,000 patients per dataset object, so PTB-XL, CPSC and
    # Georgia share one limit and Chapman-Shaoxing has its own), tiny models, one epoch.
    python examples/ecg/climb_ecg_benchmark.py --root .../training \\
        --chapman-root .../training/chapman_shaoxing --tiny

    python examples/ecg/climb_ecg_benchmark.py --root .../training \\
        --chapman-root .../training/chapman_shaoxing --model ecg_jepa \\
        --checkpoint multiblock_epoch100.pth --epochs 3 --max-hours 2.15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from pyhealth.datasets import CardiologyDataset, ChapmanShaoxingDataset, get_dataloader
from pyhealth.models import ECGJEPA
from pyhealth.tasks import (
    CLIMB_ECG_CLASSES,
    CLIMBECGClassification,
    read_climb_ecg_splits,
)
from pyhealth.tasks.climb_ecg import ECG_LEADS

CLIMB = (
    "https://raw.githubusercontent.com/DDVD233/CLIMB/"
    "0f767b1ea168810b998981078dc27e7f9a4e4675/src/datasets/ecg/"
)
SPLIT_SHA256 = {
    "ptbxl": "e48f07e91d6b2616707483bfff3863d4c3bd02866274ccf935d8363269713ba9",
    "CPSC": "e008e2ba52280c568d986fa34e4a46c751c633cb67afb73e367ce890015440b1",
    "Ga": "6e47ca14de188d0a5086e2759f8019d5560a529516c39f1ad3183722c3ec306a",
    "ChapmanShaoxing": "b3ec97578d4f780c7f01c61383194a8780734b286523c8411e2c685795406be9",
}
# Record-name prefix of each dataset, in the column order of Table 25.
DATASETS = {"HR": "PTB-XL", "JS": "Chapman", "A": "CPSC", "E": "Georgia"}
PAPER = {  # Table 25, AUROC
    "ecg_jepa": {"PTB-XL": 0.906, "Chapman": 0.858, "CPSC": 0.979, "Georgia": 0.767, "overall": 0.877},
    "transformer": {"PTB-XL": 0.785, "Chapman": 0.797, "CPSC": 0.579, "Georgia": 0.671, "overall": 0.708},
}


def climb_labels(cache: Path) -> dict[str, tuple[str, list[str]]]:
    """Split and classes of every recording, from CLIMB's four split files."""
    cache.mkdir(parents=True, exist_ok=True)
    labels = {}
    for name, sha in SPLIT_SHA256.items():
        path = cache / f"{name}_splits.csv"
        if not path.exists():
            urllib.request.urlretrieve(f"{CLIMB}{name}_splits.csv", path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError(f"{path} is not CLIMB's file; delete it and rerun.")
        labels.update(read_climb_ecg_splits(path))
    return labels


def dataset_of(record_id: str) -> str:
    return DATASETS[record_id.rstrip("0123456789")]


def load_tensors(samples) -> dict:
    """All samples in memory: float16 signals, labels, dataset and split."""
    parts = {"signal": [], "label": [], "dataset": [], "train": []}
    for batch in get_dataloader(samples, batch_size=256):
        parts["signal"].append(batch["signal"].half())
        parts["label"].append(batch["label"].float())
        parts["dataset"] += [dataset_of(r) for r in batch["record_id"]]
        parts["train"] += [s == "train" for s in batch["split"]]
    return {
        "signal": torch.cat(parts["signal"]),
        "label": torch.cat(parts["label"]),
        "dataset": np.array(parts["dataset"]),
        "train": torch.tensor(parts["train"]),
    }


def build_samples(args, labels, task):
    """Sample datasets of the three Challenge sources and of Chapman-Shaoxing."""
    common = {"cache_dir": args.cache_dir, "dev": args.tiny}
    cardiology = CardiologyDataset(
        root=args.root, sources=["ptb-xl", "cpsc_2018", "georgia"], **common
    )
    chapman = ChapmanShaoxingDataset(root=args.chapman_root, subset="climb", **common)
    return [cardiology.set_task(task), chapman.set_task(task)]


class SegmentTransformer(nn.Module):
    """CLIMB's Transformer baseline, from the paper's description.

    The signal (``[B, leads, length]``) is cut into segments of ``segment``
    samples of all leads, each linearly embedded into one token; a class
    token and learned position embeddings are added, ``depth`` pre-norm
    blocks follow, and the mean token is classified.
    """

    def __init__(self, leads, length, n_out, dim=256, depth=12, heads=8, mlp=512, segment=25):
        super().__init__()
        self.segment = segment
        self.embed = nn.Linear(leads * segment, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, length // segment + 1, dim))
        layer = nn.TransformerEncoderLayer(
            dim, heads, mlp, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        tokens = self.embed(x.reshape(b, c, n // self.segment, self.segment).permute(0, 2, 1, 3).reshape(b, n // self.segment, -1))
        h = torch.cat([self.cls.expand(b, -1, -1), tokens], dim=1) + self.pos
        return self.head(self.blocks(h).mean(dim=1))


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Macro AUROC, and macro sensitivity / specificity at 0.5, over classes
    that have both positives and negatives."""
    keep = [k for k in range(y.shape[1]) if 0 < y[:, k].sum() < len(y)]
    pred = p >= 0.5
    sens = [pred[y[:, k] == 1, k].mean() for k in keep]
    spec = [(~pred[y[:, k] == 0, k]).mean() for k in keep]
    return {
        "auroc": float(np.mean([roc_auc_score(y[:, k], p[:, k]) for k in keep])),
        "sens": float(np.mean(sens)),
        "spec": float(np.mean(spec)),
    }


@torch.no_grad()
def evaluate(logits_fn, data, index, device, batch_size):
    probs = []
    for start in range(0, len(index), batch_size):
        x = data["signal"][index[start : start + batch_size]].to(device).float()
        probs.append(torch.sigmoid(logits_fn(x)).float().cpu())
    p = torch.cat(probs).numpy()
    y = data["label"][index].numpy()
    names = data["dataset"][index.numpy()]
    present = [name for name in DATASETS.values() if (names == name).any()]
    out = {name: scores(y[names == name], p[names == name]) for name in present}
    # Table 25's "Overall" is the mean over the datasets.
    out["overall"] = {m: float(np.mean([out[n][m] for n in present])) for m in ("auroc", "sens", "spec")}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="Challenge training/ folder.")
    parser.add_argument("--chapman-root", required=True, help="Chapman-Shaoxing records.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", default="climb_ecg")
    parser.add_argument("--model", choices=["ecg_jepa", "transformer"], default="ecg_jepa")
    parser.add_argument("--checkpoint", default=None, help="ECG-JEPA checkpoint (.pth).")
    parser.add_argument("--freeze-encoder", action="store_true", help="Linear probe.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None,
                        help="Default: 1e-3 (transformer, CLIMB); 1e-5 (ecg_jepa).")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup", type=float, default=0.1, help="Fraction of steps.")
    parser.add_argument("--max-hours", type=float, default=None, help="Stop training after this.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tiny", action="store_true", help="Smoke test: dev mode (1,000 patients per dataset object), tiny models, one epoch.")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = args.device or (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    jepa = args.model == "ecg_jepa"
    lr = args.lr or (1e-5 if jepa else 1e-3)

    labels = climb_labels(output / "climb_splits")
    task = (
        CLIMBECGClassification(labels)
        if jepa
        else CLIMBECGClassification(labels, num_samples=5000, leads=ECG_LEADS)
    )
    parts = build_samples(args, labels, task)
    loaded = [load_tensors(part) for part in parts]
    data = {k: (np.concatenate if k == "dataset" else torch.cat)([d[k] for d in loaded]) for k in loaded[0]}
    train_idx = torch.nonzero(data["train"]).view(-1)
    val_idx = torch.nonzero(~data["train"]).view(-1)
    counts = {n: int((data["dataset"] == n).sum()) for n in DATASETS.values()}
    print(f"recordings {counts}; train {len(train_idx)}, val {len(val_idx)}", flush=True)

    n_out = len(CLIMB_ECG_CLASSES)
    if jepa:
        tiny = {"dim": 32, "depth": 2, "num_heads": 2} if args.tiny else {}
        model = ECGJEPA(parts[0], checkpoint=None if args.tiny else args.checkpoint,
                        freeze_encoder=args.freeze_encoder, **tiny).to(device)

        def logits_fn(x):
            return model.fc(model.embed(x))
    else:
        tiny = {"dim": 32, "depth": 2, "heads": 2, "mlp": 64} if args.tiny else {}
        model = SegmentTransformer(len(ECG_LEADS), 5000, n_out, **tiny).to(device)
        logits_fn = model
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    epochs = 1 if args.tiny else args.epochs
    steps = epochs * math.ceil(len(train_idx) / args.batch_size)
    warmup = max(1, int(args.warmup * steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(args.seed)
    start, history, step, stopped = time.time(), [], 0, False
    for epoch in range(epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=generator)]
        losses = []
        for i in range(0, len(order), args.batch_size):
            index = order[i : i + args.batch_size]
            x = data["signal"][index].to(device).float()
            loss = loss_fn(logits_fn(x), data["label"][index].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            step += 1
            if step % 100 == 0:
                rate = step * args.batch_size / (time.time() - start)
                print(f"epoch {epoch} step {step}/{steps} loss {np.mean(losses[-100:]):.4f} "
                      f"{rate:.1f} samples/s", flush=True)
            if args.max_hours and time.time() - start > args.max_hours * 3600:
                stopped = True
                break
        model.eval()
        val = evaluate(logits_fn, data, val_idx, device, 2 * args.batch_size)
        record = {"epoch": epoch, "steps": step, "minutes": (time.time() - start) / 60,
                  "loss": float(np.mean(losses)), "val": val, "stopped_early": stopped}
        history.append(record)
        print(json.dumps(record), flush=True)
        if stopped:
            break

    name = f"{args.model}{'_probe' if args.freeze_encoder else ''}_seed{args.seed}"
    result = {"final": history[-1]["val"], "paper_auroc": PAPER[args.model], "history": history,
              "args": vars(args), "lr": lr, "counts": counts}
    (output / f"{name}.json").write_text(json.dumps(result, indent=2))
    print("final AUROC", {k: round(v["auroc"], 3) for k, v in history[-1]["val"].items()})
    print("paper AUROC", PAPER[args.model])


if __name__ == "__main__":
    main()
