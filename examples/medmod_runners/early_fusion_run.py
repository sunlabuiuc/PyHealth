"""Train EHRCXREarlyFusion on the real IHM `keep` set, MedFuse settings.

    cd "<repo>"
    PYTHONPATH="$PWD" python -m examples.medmod_runners.early_fusion_run \\
        --data-root /path/to/mimic_data \\
        --ehr-weights .../medfuse_weights/uni_ehr_all/best_checkpoint.pth.tar \\
        --cxr-weights .../medfuse_weights/cxr_rad_full/best_checkpoint.pth.tar

Frozen pretrained encoders from MedFuse's released checkpoints:
EHR = mortality/uni_ehr_all (LSTM 2 x 256), CXR = cxr_rad_full (ResNet-34,
`cxr_model.vision_backbone.*`). Trained: projection + classifier.

MedFuse scripts/mortality/train/early.sh + trainers/fusion_trainer.py:
Adam lr 9.515e-05, betas (0.9, 0.9) (beta_1 default), batch 16, 50 epochs,
ReduceLROnPlateau(factor 0.5, patience 10) on val loss, validation before each
epoch, best epoch by val AUROC, stop after 15 epochs without a better one
(--patience default). Images: get_preset("early_fusion", train=True) for
train, eval preset otherwise.

MedSigLIP extension (--embeddings <.npz from pyhealth.datasets.medmod.medsiglip>):
the CXR ResNet-34 is replaced by the stored frozen MedSigLIP embeddings
(``cxr_backbone=nn.Identity()``, ``cxr_feature_dim`` = their size), looked up
per batch; no image augmentation; --cxr-weights not used; default tag
early_fusion_medsiglip_<commit> (``bundle`` instead of the commit when run
from the Colab code zip); the file's SHA-256 and settings go in the
report.

Known deviation: the model keeps its frozen encoders in eval mode; MedFuse
calls model.train() on everything (fusion_trainer.py:300), so its frozen
ResNet updates BatchNorm statistics and dropout hits the EHR features.

Aggregates only: metrics, counts, timings. Report saved after every epoch to
<data-root>/medmod_ihm_results/<tag>.json (default tag early_fusion_<commit>),
with the code version (``provenance.py``).

Checkpoints: every --checkpoint-every epochs (default 10) the trained layers,
optimizer, scheduler, best epoch, early-stopping counter and random states go
to <tag>_checkpoint.pt, so a continued run gives the same result as an
uninterrupted one. Run the same command again after a disconnect: a finished
run stops at once, an interrupted run continues from its checkpoint when the
settings and code are the same, and otherwise the run starts fresh.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

EHR_SHA256 = "c545d8d536882948f25b4dd8a136c43da49fd10d4519e8b24b821109ad22f552"
CXR_SHA256 = "a02628fcec408cc59ac1464015ddcf87b98840e37812563ce648ddce11b3bf90"
PAPER = {"auroc": (0.842, 0.004), "auprc": (0.515, 0.020)}  # MedMod Table 4


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_checkpoint(path: Path, sha: str) -> dict:
    """Restricted torch load (weights_only) after a sha256 check."""
    if _sha256(path) != sha:
        raise ValueError(f"{path.name}: sha256 differs from MedFuse's LFS pointer")
    from numpy._core.multiarray import scalar

    allowed = [
        (scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        type(np.dtype("float64")),
        codecs.encode,
    ]
    with torch.serialization.safe_globals(allowed):
        ck = torch.load(path, map_location="cpu", weights_only=True)
    return ck["state_dict"]


def ehr_state_dict(sd: dict) -> dict:
    """MedFuse ehr_model.layer{k}.*_l0 -> nn.LSTM(num_layers=2) *_l{k}."""
    out = {}
    for k, v in sd.items():
        if k.startswith("ehr_model.layer"):
            layer, name = k[len("ehr_model.layer") :].split(".", 1)
            out[name.replace("_l0", f"_l{layer}")] = v
    return out


def cxr_state_dict(sd: dict) -> dict:
    """MedFuse cxr_model.vision_backbone.* -> torchvision ResNet without fc."""
    p = "cxr_model.vision_backbone."
    return {k[len(p) :]: v for k, v in sd.items() if k.startswith(p)}


def bootstrap_ci(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a.append(roc_auc_score(y[i], p[i]))
        b.append(average_precision_score(y[i], p[i]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))], [
        float(np.percentile(b, 2.5)),
        float(np.percentile(b, 97.5)),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ehr-weights", required=True)
    ap.add_argument("--cxr-weights", help="required unless --embeddings")
    ap.add_argument("--embeddings", help="MedSigLIP .npz instead of the ResNet")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=9.515e-05)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    default_dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        default_dev = "cuda"
    ap.add_argument("--device", default=default_dev)
    ap.add_argument("--results-dir", default=None,
                    help="default <data-root>/medmod_ihm_results (Colab: Drive)")
    ap.add_argument("--tag", default=None,
                    help="default early_fusion_<commit>, or early_fusion_bundle")
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="save a checkpoint every N epochs")
    args = ap.parse_args()
    if bool(args.cxr_weights) == bool(args.embeddings):
        ap.error("give --cxr-weights (ResNet) or --embeddings (MedSigLIP), not both")

    from torch import nn

    from pyhealth.datasets import get_dataloader
    from pyhealth.datasets.medmod.image_presets import get_preset, load_cxr
    from pyhealth.datasets.medmod.medsiglip import lookup_embeddings
    from pyhealth.models import EHRCXREarlyFusion

    from .samples import MedModPaths, build_samples, make_task

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # the presets draw from numpy's global RNG
    dev = torch.device(args.device)
    from .provenance import (
        code_label, code_version, load_embedding_table, report_code,
        rng_state, same_code, save_atomic, set_rng_state,
    )

    code = code_version("early_fusion_run.py", "samples.py")
    report_code(code)
    kind = "early_fusion_medsiglip" if args.embeddings else "early_fusion"
    tag = args.tag or f"{kind}_{code_label(code)}"
    results = Path(args.results_dir or Path(args.data_root) / "medmod_ihm_results")
    out_path = results / f"{tag}.json"
    ckpt_path = results / f"{tag}_checkpoint.pt"
    if out_path.exists() and "test" in json.loads(out_path.read_text()):
        print(f"{tag} already finished", flush=True)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = MedModPaths.from_root(args.data_root)
    _, (train, val, test) = build_samples(
        paths, make_task(paths, "keep"), args.num_workers
    )
    ehr_sd = ehr_state_dict(load_checkpoint(Path(args.ehr_weights), EHR_SHA256))
    table, emb_record = None, None
    if args.embeddings:
        table, emb_record = load_embedding_table(args.embeddings)
        image = {"cxr_backbone": nn.Identity(), "cxr_feature_dim": table[1].shape[1]}
    else:
        cxr_sd = cxr_state_dict(load_checkpoint(Path(args.cxr_weights), CXR_SHA256))
        image = {"cxr_state_dict": cxr_sd}
    model = EHRCXREarlyFusion(
        train, ehr_state_dict=ehr_sd, freeze_encoders=True, **image
    ).to(dev)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr, betas=(0.9, 0.9))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=10, mode="min"
    )
    tr_tf, ev_tf = get_preset("early_fusion", train=True), get_preset("early_fusion")

    def add_images(batch, tf):
        if table is not None:  # stored embeddings: no transform
            batch["cxr"] = lookup_embeddings(batch["image_path"], *table)
        else:
            batch["cxr"] = torch.stack([tf(load_cxr(p)) for p in batch["image_path"]])
        return batch

    @torch.no_grad()
    def evaluate(part):
        model.eval()
        ys, ps, losses = [], [], []
        for batch in get_dataloader(part, batch_size=64, shuffle=False):
            out = model(**add_images(batch, ev_tf))
            losses.append(float(out["loss"]) * len(batch["image_path"]))
            ps.append(out["y_prob"].float().cpu().reshape(-1))
            ys.append(out["y_true"].float().cpu().reshape(-1))
        y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
        return y, p, sum(losses) / len(y)

    report = {
        "settings": {k: v for k, v in vars(args).items()
                     if k not in ("data_root", "results_dir")},
        "code": code,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "trainable_params": sum(p.numel() for p in trainable),
        "paper_table4": PAPER,
        "embeddings": emb_record,
        "deviation": "frozen encoders kept in eval mode (MedFuse: train mode)",
        "epochs": [],
    }
    best_auroc, best_state, best_epoch, waited = -1.0, None, -1, 0
    trained = [k for k, p in model.named_parameters() if p.requires_grad]
    start = 0
    # continue from the checkpoint of an interrupted run with the same
    # settings and code; otherwise start fresh
    ck = (torch.load(ckpt_path, map_location="cpu", weights_only=True)
          if ckpt_path.exists() else None)
    if ck is not None:
        keep = ("resume", "checkpoint_every", "device", "num_workers")
        old = {k: v for k, v in ck["settings"].items() if k not in keep}
        new = {k: v for k, v in report["settings"].items() if k not in keep}
        if old != new or not same_code(ck["code"], code):
            print("checkpoint from other settings or code: starting fresh", flush=True)
            ck = None
    if ck is not None:
        # only the projection and the classifier are trained; the frozen
        # encoders stay as loaded, so the checkpoint keeps the trained part
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        best_auroc, best_epoch = ck["best_auroc"], ck["best_epoch"]
        waited = ck["waited"]
        if ck["best_trained"] is not None:
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_state.update(ck["best_trained"])
        report["epochs"] = ck["epochs"]
        set_rng_state(ck["rng"])
        start = ck["epoch"] + 1
        print(f"resuming after epoch {ck['epoch']}", flush=True)
    for epoch in range(start, args.epochs):
        t = time.time()
        y, p, val_loss = evaluate(val)  # validate first, as MedFuse does
        val_auroc = float(roc_auc_score(y, p))
        if val_auroc > best_auroc:
            best_auroc, best_epoch, waited = val_auroc, epoch, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            waited += 1
        if waited >= args.patience:
            print(f"early stop before epoch {epoch}", flush=True)
            break
        model.train()
        losses = []
        for batch in get_dataloader(train, batch_size=args.batch_size, shuffle=True):
            loss = model(**add_images(batch, tr_tf))["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step(val_loss)
        row = {
            "epoch": epoch,
            "val_auroc_before": val_auroc,
            "val_loss_before": val_loss,
            "train_loss": float(np.mean(losses)),
            "lr": opt.param_groups[0]["lr"],
            "seconds": round(time.time() - t, 1),
        }
        report["epochs"].append(row)
        print(json.dumps(row), flush=True)
        out_path.write_text(json.dumps(report, indent=2))
        if args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0:
            state = model.state_dict()
            save_atomic({
                "epoch": epoch, "settings": report["settings"], "code": code,
                "model": {k: state[k] for k in trained},
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "best_auroc": best_auroc, "best_epoch": best_epoch,
                "waited": waited,
                "best_trained": None if best_state is None
                else {k: best_state[k] for k in trained},
                "epochs": report["epochs"], "rng": rng_state(),
            }, ckpt_path)

    # final validation after the last epoch, then test the best epoch
    y, p, _ = evaluate(val)
    if roc_auc_score(y, p) > best_auroc:
        best_auroc, best_epoch = float(roc_auc_score(y, p)), len(report["epochs"])
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
    model.load_state_dict(best_state)
    y, p, _ = evaluate(test)
    auroc_ci, auprc_ci = bootstrap_ci(y, p)
    report["best"] = {"epoch": best_epoch, "val_auroc": best_auroc}
    report["test"] = {
        "n": int(len(y)),
        "deaths": int(y.sum()),
        "auroc": float(roc_auc_score(y, p)),
        "auroc_ci95": auroc_ci,
        "auprc": float(average_precision_score(y, p)),
        "auprc_ci95": auprc_ci,
    }
    out_path.write_text(json.dumps(report, indent=2))
    print("BEST", json.dumps(report["best"]), "TEST", json.dumps(report["test"]))


if __name__ == "__main__":
    main()
