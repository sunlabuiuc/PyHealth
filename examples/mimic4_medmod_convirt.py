"""MedMod ConVIRT on MIMIC-IV + MIMIC-CXR: contrastive pretraining of an EHR
LSTM and a chest X-ray ResNet-34, then a linear probe for in-hospital
mortality (MedMod, Elsharief et al., CHIL 2025; paper Table 4: AUROC 0.847).

Synthetic data (no MIMIC access; a few minutes on CPU):

    python examples/mimic4_medmod_convirt.py --synthetic

Real data, two steps:

    # 1. pretraining on MedMod's phenotyping train/val stays, whole ICU stay
    python examples/mimic4_medmod_convirt.py pretrain \\
        --extracted-root /data/medmod_extracted \\
        --cxr-root /data/mimic-cxr-jpg/2.0.0 \\
        --image-dir /data/mimic-cxr-jpg/2.0.0/resized --out runs/convirt
    # 2. linear probe on in-hospital mortality, from pretraining epoch 65
    python examples/mimic4_medmod_convirt.py probe \\
        --ehr-root /data/mimiciv/1.0 --cxr-root /data/mimic-cxr-jpg/2.0.0 \\
        --image-dir /data/mimic-cxr-jpg/2.0.0/resized \\
        --cache-dir /data/pyhealth_cache --checkpoint runs/convirt/epoch_065.pt

``--extracted-root`` is the output of MedMod's extraction scripts (MIMIC-IV
benchmark + ``create_phenotyping``); ``--image-dir`` holds the images resized
by ``pyhealth.datasets.medmod.resize_cxr``.

Pretraining follows MedMod's released settings (src/scripts/pretrain/
simclr.sh; src/models/selfsupervised/models/simclr_model.py): 300 epochs,
batch 256 with drop_last, temperature 0.01, LARS (lr 0.6044, momentum 0.9,
weight decay 1e-4) with a 10-epoch linear warmup then
CosineAnnealingWarmRestarts(T_0=500), both stepped once per epoch, seed 84305,
MedMod's simclrv2 augmentation; every epoch's weights are kept. MedMod's
early stopping (patience 30 on val loss) is not applied. The probe follows
MedMod's linear eval (src/scripts/lineareval/mortality/multimodal.sh;
models/finetune.py) with the encoders frozen, as in the paper's Table 4:
AdamW lr 0.001, batch 256, at most 200 epochs, early stopping on val AUROC
(patience 30), train augmentation; MedMod reports the final epoch.

MedSigLIP extension: ``--embeddings FILE`` (written by
``python -m pyhealth.datasets.medmod.medsiglip``) replaces the ResNet-34 with
the stored embeddings of a frozen MedSigLIP image encoder, in both steps:
ConVIRT is built with ``cxr_backbone=nn.Identity()`` and
``cxr_feature_dim=1152``, and each batch's image input is looked up by image
file name. The stored vectors are fixed, so there is no image augmentation;
pass the same file to both steps.

    python examples/mimic4_medmod_convirt.py pretrain ... \\
        --embeddings /data/medsiglip_448.npz --out runs/convirt_medsiglip
    python examples/mimic4_medmod_convirt.py probe ... \\
        --embeddings /data/medsiglip_448.npz \\
        --checkpoint runs/convirt_medsiglip/epoch_065.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.optim.optimizer import Optimizer, required

from pyhealth.datasets import get_dataloader, split_by_medmod
from pyhealth.datasets.medmod.image_presets import get_preset, load_cxr
from pyhealth.datasets.medmod.medsiglip import load_embeddings, lookup_embeddings
from pyhealth.models import ConVIRT, ConVIRTLinearProbe


class LARS(Optimizer):
    """LARS from lightning-flash 0.7.5 (flash/core/optimizers/lars.py,
    Apache-2.0), the optimizer MedMod imports (simclr_model.py:16). Same
    arithmetic; the zero-norm test is a ``torch.where`` (no device sync)."""

    def __init__(self, params, lr=required, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False, trust_coefficient=0.001,
                 eps=1e-8):
        defaults = {"lr": lr, "momentum": momentum, "dampening": dampening,
                    "weight_decay": weight_decay, "nesterov": nesterov}
        self.eps = eps
        self.trust_coefficient = trust_coefficient
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            wd, momentum = group["weight_decay"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if wd != 0:
                    p_norm, g_norm = torch.norm(p), torch.norm(d_p)
                    lars_lr = (p_norm / (g_norm + p_norm * wd + self.eps)
                               * self.trust_coefficient)
                    scaled = d_p.add(p, alpha=wd) * lars_lr
                    d_p = torch.where((p_norm != 0) & (g_norm != 0), scaled, d_p)
                if momentum != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - group["dampening"])
                    d_p = d_p.add(buf, alpha=momentum) if group["nesterov"] else buf
                p.add_(d_p, alpha=-group["lr"])


def default_device() -> str:
    """CUDA if available, else CPU; MPS only with ``--device mps``, since it
    crashes on some macOS and torch versions."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def with_images(batch: dict, transform, table=None) -> dict:
    """Adds the image input: the loaded, transformed images or, with ``table``
    (``--embeddings``), their stored MedSigLIP embeddings."""
    if table is not None:
        batch["cxr"] = lookup_embeddings(batch["image_path"], *table)
    else:
        images = [transform(load_cxr(p)) for p in batch["image_path"]]
        batch["cxr"] = torch.stack(images)
    return batch


def build_convirt(dataset, args) -> ConVIRT:
    """ConVIRT with a ResNet, or reading stored embeddings (``--embeddings``)."""
    if args.table is None:
        return ConVIRT(dataset, cxr_backbone=args.backbone,
                       temperature=args.temperature)
    return ConVIRT(dataset, cxr_backbone=nn.Identity(),
                   cxr_feature_dim=args.table[1].shape[1],
                   temperature=args.temperature)


def pretrain(dataset, out: Path, args) -> None:
    """ConVIRT pretraining with MedMod's settings; resumes from out/last.pt."""
    train, val, _ = split_by_medmod(dataset)
    print(f"pretraining stays: train {len(train)}, val {len(val)}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # the presets draw from numpy's global RNG
    model = build_convirt(dataset, args).to(dev)
    opt = LARS(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # created in MedMod's order (simclr_model.py:88-93)
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 500)
    warmup = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / 10)
    history, start = [], 0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        cosine.load_state_dict(ck["cosine"])
        warmup.load_state_dict(ck["warmup"])
        history, start = ck["history"], ck["epoch"] + 1
        print(f"resumed after epoch {ck['epoch']}", flush=True)
    train_tf, eval_tf = get_preset("convirt", train=True), get_preset("convirt")

    def run(part, epoch, training):
        model.train(training)
        loader = get_dataloader(part, batch_size=args.batch_size, shuffle=training,
                                seed=args.seed, drop_last=training)
        loader.epoch = epoch + 1  # shuffle order depends on (seed, epoch)
        losses, sizes = [], []
        for batch in loader:
            batch = with_images(batch, train_tf if training else eval_tf, args.table)
            with torch.set_grad_enabled(training):
                loss = model(**batch)["loss"]
                if training:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            losses.append(loss.detach())
            sizes.append(len(batch["image_path"]))
        per_batch = torch.stack(losses).float().cpu().numpy()
        return float((per_batch * np.array(sizes)).sum() / sum(sizes))

    for epoch in range(start, args.epochs):
        lr = opt.param_groups[0]["lr"]
        torch.manual_seed(args.seed + epoch)
        np.random.seed(args.seed + epoch)
        train_loss = run(train, epoch, True)
        (warmup if epoch < 9 else cosine).step()  # simclr_model.py:151-155
        with torch.no_grad():
            val_loss = run(val, epoch, False)
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                        "val_loss": val_loss})
        print(json.dumps(history[-1]), flush=True)
        torch.save({"epoch": epoch, "model": model.state_dict()},
                   out / f"epoch_{epoch:03d}.pt")
        tmp = out / "last.tmp"
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": opt.state_dict(), "cosine": cosine.state_dict(),
                    "warmup": warmup.state_dict(), "history": history}, tmp)
        os.replace(tmp, out / "last.pt")
    best = min(history, key=lambda h: h["val_loss"])
    print(f"done; lowest val loss {best['val_loss']:.4f} at epoch {best['epoch']}")


def probe(train, val, test, checkpoint: Path, args) -> dict:
    """Linear probe on the frozen pretrained encoders, MedMod's linear eval."""
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    convirt = build_convirt(train, args)
    convirt.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    model = ConVIRTLinearProbe(train, convirt).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=0.001)
    train_tf, eval_tf = get_preset("convirt", train=True), get_preset("convirt")

    @torch.no_grad()
    def auroc_auprc(part):
        model.eval()
        ys, ps = [], []
        for batch in get_dataloader(part, batch_size=args.batch_size):
            out = model(**with_images(batch, eval_tf, args.table))
            ys.append(out["y_true"].float().cpu().reshape(-1))
            ps.append(out["y_prob"].float().cpu().reshape(-1))
        y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
        return float(roc_auc_score(y, p)), float(average_precision_score(y, p))

    best, waited = -math.inf, 0
    for epoch in range(args.probe_epochs):
        torch.manual_seed(args.seed + epoch)
        np.random.seed(args.seed + epoch)
        model.train()
        loader = get_dataloader(train, batch_size=args.batch_size, shuffle=True,
                                seed=args.seed, drop_last=True)
        loader.epoch = epoch + 1
        for batch in loader:
            loss = model(**with_images(batch, train_tf, args.table))["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_auroc, _ = auroc_auprc(val)
        print(json.dumps({"epoch": epoch, "val_auroc": val_auroc}), flush=True)
        if round(val_auroc, 4) > best + 0.001:  # MedMod: EarlyStopping(0.001, 30)
            best, waited = round(val_auroc, 4), 0
        else:
            waited += 1
            if waited >= 30:
                break
    auroc, auprc = auroc_auprc(test)
    result = {"test_auroc": auroc, "test_auprc": auprc, "epochs": epoch + 1}
    print("TEST", json.dumps(result))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", nargs="?", choices=["pretrain", "probe"])
    ap.add_argument("--synthetic", action="store_true",
                    help="both steps on small synthetic data (no MIMIC)")
    ap.add_argument("--extracted-root")
    ap.add_argument("--ehr-root")
    ap.add_argument("--cxr-root")
    ap.add_argument("--image-dir")
    ap.add_argument("--cache-dir")
    ap.add_argument("--out", default="runs/convirt")
    ap.add_argument("--checkpoint")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--probe-epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.6044)
    ap.add_argument("--temperature", type=float, default=0.01)
    ap.add_argument("--backbone", default="resnet34")
    ap.add_argument("--embeddings",
                    help="MedSigLIP .npz: use stored image embeddings, not a ResNet")
    ap.add_argument("--seed", type=int, default=84305)
    ap.add_argument("--device", default=default_device())
    args = ap.parse_args()
    args.table = load_embeddings(args.embeddings)[:2] if args.embeddings else None

    if args.synthetic:
        from pyhealth.datasets.medmod.synthetic import (
            create_synthetic_pretraining_samples,
            create_synthetic_samples,
        )

        args.epochs, args.probe_epochs, args.batch_size = 2, 2, 32
        out = Path(tempfile.mkdtemp(prefix="convirt_synthetic_"))
        pretrain(create_synthetic_pretraining_samples(n_stays=96), out, args)
        train, val, test = split_by_medmod(create_synthetic_samples("drop", 200))
        probe(train, val, test, out / "epoch_001.pt", args)
    elif args.step == "pretrain":
        from pyhealth.datasets.medmod.pretraining import (
            create_pretraining_dataset,
            load_pretraining_samples,
        )

        samples, funnel = load_pretraining_samples(
            args.extracted_root, args.cxr_root, args.image_dir)
        print("funnel:", json.dumps(funnel), flush=True)
        pretrain(create_pretraining_dataset(samples), Path(args.out), args)
    elif args.step == "probe":
        from pyhealth.datasets import MedModMIMIC4Dataset, MedModSplit
        from pyhealth.tasks import MedModInHospitalMortality

        dataset = MedModMIMIC4Dataset(ehr_root=args.ehr_root, cxr_root=args.cxr_root,
                                      cache_dir=args.cache_dir)
        task = MedModInHospitalMortality(hours_zero="drop", image_dir=args.image_dir)
        train, val, test = dataset.set_task(task, split=MedModSplit())
        probe(train, val, test, Path(args.checkpoint), args)
    else:
        ap.error("give a step (pretrain or probe) or --synthetic")


if __name__ == "__main__":
    main()
