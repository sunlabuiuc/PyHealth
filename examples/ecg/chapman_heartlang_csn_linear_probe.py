"""HeartLang linear probe on HeartLang's CSN benchmark (Chapman-Shaoxing).

Reproduces the CSN setup of HeartLang (Jin et al., ICLR 2025): the frozen
pretrained HeartLang encoder with a linear head trained on the 38 CSN labels,
using HeartLang's own train/val/test split.

Needs:

- the open PhysioNet release (CC BY 4.0) from
  https://physionet.org/content/ecg-arrhythmia/1.0.0/, with ``--root`` at the
  version folder holding ``WFDBRecords/``;
- a checkout of https://github.com/PKUDigitalHealth/HeartLang, with
  ``--split-dir`` at its ``datasets/dataset_preprocess/CSN`` folder (the
  ``chapman_{train,val,test}.csv`` record lists);
- ``pip install "pyhealth[ptbxl]"`` (wfdb for QRS detection) and network
  access to download the pretrained HeartLang weights.

Example::

    python examples/ecg/chapman_heartlang_csn_linear_probe.py \\
        --root /data/ecg-arrhythmia/1.0.0 \\
        --split-dir HeartLang/datasets/dataset_preprocess/CSN

The hyperparameters follow HeartLang's ``base_csn_linear_prob.sh`` (100
epochs, 10 warm-up epochs, batch 256, learning rate 5e-3, weight decay 0.05).
"""

from __future__ import annotations

import argparse
import math

import torch
from sklearn.metrics import roc_auc_score

from pyhealth.datasets import ChapmanShaoxingDataset, get_dataloader
from pyhealth.models import HeartLang
from pyhealth.tasks import ChapmanShaoxingHeartLangClassification
from pyhealth.tasks.chapman_heartlang import split_by_heartlang_csn


@torch.no_grad()
def features(model, dataset, batch_size):
    """Frozen HeartLang embeddings and multi-hot labels of a dataset."""
    model.eval()
    xs, ys = [], []
    for batch in get_dataloader(dataset, batch_size=batch_size, shuffle=False):
        xs.append(model.embed(**batch).cpu())
        ys.append(batch["labels"].float())
    return torch.cat(xs), torch.cat(ys)


def macro_auroc(y, logits):
    """Macro AUROC in %, over the classes with both outcomes present."""
    y, p = y.numpy(), torch.sigmoid(logits).numpy()
    keep = (y.sum(0) > 0) & (y.sum(0) < len(y))
    return 100 * roc_auc_score(y[:, keep], p[:, keep], average="macro")


def linear_probe(model, splits, args, device):
    """Trains a linear head on frozen features; returns (val, test) AUROC."""
    torch.manual_seed(args.seed)
    (xtr, ytr), (xva, yva), (xte, yte) = (
        features(model, d, args.batch_size) for d in splits
    )
    head = torch.nn.Linear(xtr.shape[1], ytr.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.05)
    steps = math.ceil(len(xtr) / args.batch_size)
    total, warmup = args.epochs * steps, min(10, args.epochs) * steps

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best_val, best_state = -1.0, None
    for _ in range(args.epochs):
        head.train()
        for idx in torch.randperm(len(xtr)).split(args.batch_size):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                head(xtr[idx].to(device)), ytr[idx].to(device)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            schedule.step()
        head.eval()
        with torch.no_grad():
            val = macro_auroc(yva, head(xva.to(device)).cpu())
        if val > best_val:
            best_val = val
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    with torch.no_grad():
        test = macro_auroc(yte, head(xte.to(device)).cpu())
    return best_val, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, help="ecg-arrhythmia/1.0.0 folder")
    parser.add_argument("--split-dir", required=True, help="HeartLang CSN CSV folder")
    parser.add_argument("--label-match", default="heartlang",
                        choices=["heartlang", "exact"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ChapmanShaoxingDataset(root=args.root)
    samples = dataset.set_task(
        ChapmanShaoxingHeartLangClassification(label_match=args.label_match)
    )
    splits = split_by_heartlang_csn(samples, args.split_dir)
    print("train={} val={} test={}".format(*(len(s) for s in splits)))

    model = HeartLang(samples, pretrained=True, freeze_backbone=True).to(device)
    val, test = linear_probe(model, splits, args, device)
    print(f"CSN macro AUROC: val={val:.2f} test={test:.2f}")


if __name__ == "__main__":
    main()
