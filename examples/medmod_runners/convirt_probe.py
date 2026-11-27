"""Linear probe of a pretrained ConVIRT on MedMod in-hospital mortality.

    cd "<repo>"
    PYTHONPATH="$PWD" python -u -m examples.medmod_runners.convirt_probe \\
        --data-root /path/to/mimic_data \\
        --checkpoint <results>/convirt_resnet34/epoch_NNN.pt

Model: ``ConVIRTLinearProbe``: the pretrained encoders, frozen and in
eval mode; one linear layer on the concatenated encoder outputs (EHR 128 +
image 512, MedMod ``--fusion_layer 0``, parser.py:76; fusion_models.py:86-93,
120-128) is the only trained part. This is the true linear probe of the
paper's Table 4 (0.847). MedMod's released script passes ``--finetune``,
which also trains the encoders (fusion_models.py:123-128, finetune.py
configure_optimizers): not done here.

Settings from MedMod's linear eval (src/scripts/lineareval/mortality/
multimodal.sh; src/models/selfsupervised/models/finetune.py;
src/models/selfsupervised/trainers/finetune.py):
* pretraining checkpoint: the one passed with --checkpoint. The Colab
  notebook passes the epoch with the highest validation AUROC, as the paper
  describes;
* AdamW lr 0.001 (torch defaults otherwise: weight decay 0.01), batch 256,
  train loader shuffled with drop_last (data/utils.py:390), at most 200
  epochs; MedMod builds a MultiStepLR but, with manual optimization, never
  steps it, so the lr stays 0.001;
* validation after every epoch; EarlyStopping on val AUROC (rounded to 4
  decimals as MedMod logs it), min_delta 0.001, patience 30;
* MedMod tests the weights at the end of training (``trainer.test(model)``
  after ``fit``; its ModelCheckpoint has no monitor): ``test_final``. The
  best-val-AUROC epoch's test result is reported too (``test_best_val``);
* images: get_preset("convirt", train=True) on the train split (MedMod passes
  ``--transforms_cxr simclrv2``, utils.py:114-129), eval preset otherwise;
  so the frozen image features differ every epoch (--no-augment computes
  them once instead, much faster);
* EHR: the IHM ``drop`` set, 48 x 76;
* seed 84305 (trainers/finetune.py:27).

MedSigLIP extension (--embeddings <.npz>, for a checkpoint pretrained with
--embeddings): the image encoder reads the stored frozen MedSigLIP
embeddings (``nn.Identity()``); no augmentation is possible, so the features
are computed once (as --no-augment); default tag
probe_medsiglip_<commit>_<checkpoint name>; the file's SHA-256 and settings go
in the report.

Aggregates only. Report: <data-root>/medmod_ihm_results/<tag>.json (default
probe_<commit>_<checkpoint name>; ``bundle`` instead of the commit when run
from the Colab code zip), with the code version (``provenance.py``).

Checkpoints: every --checkpoint-every epochs (default 10) the linear layer,
optimizer, best epoch and early-stopping counters go to <tag>_checkpoint.pt.
Every epoch re-seeds itself, so a continued run equals an uninterrupted one.
Run the same command again after a disconnect: a finished run stops at
once, an interrupted run continues from its checkpoint when the settings and
code are the same, and otherwise the run starts fresh.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

PAPER = {"table4_linear_probe_auroc": [0.847, 0.002],
         "table8_auroc": 0.813}  # the paper's tables disagree: report both


def bootstrap_ci(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a.append(roc_auc_score(y[i], p[i]))
        b.append(average_precision_score(y[i], p[i]))
    return ([float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
            [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])


def scores(y, p) -> dict:
    auroc_ci, auprc_ci = bootstrap_ci(y, p)
    return {"n": int(len(y)), "deaths": int(y.sum()),
            "auroc": float(roc_auc_score(y, p)), "auroc_ci95": auroc_ci,
            "auprc": float(average_precision_score(y, p)), "auprc_ci95": auprc_ci}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="a convirt_pretrain checkpoint "
                         "(epoch_NNN.pt, best_val_loss.pt...)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--min-delta", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=84305)
    ap.add_argument("--embeddings", help="MedSigLIP .npz (pretrained with it too)")
    ap.add_argument("--no-augment", action="store_true",
                    help="no train augmentation; features computed once")
    ap.add_argument("--num-workers", type=int, default=4)
    default_dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        default_dev = "cuda"
    ap.add_argument("--device", default=default_dev)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--results-dir", default=None,
                    help="default <data-root>/medmod_ihm_results (Colab: Drive)")
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="save a probe checkpoint every N epochs")
    args = ap.parse_args()

    from torch import nn

    from pyhealth.datasets import get_dataloader
    from pyhealth.datasets.medmod.image_presets import get_preset, load_cxr
    from pyhealth.datasets.medmod.medsiglip import lookup_embeddings
    from pyhealth.models import ConVIRT, ConVIRTLinearProbe

    from .provenance import (
        code_label, code_version, load_embedding_table, report_code,
        same_code, save_atomic,
    )
    from .samples import MedModPaths, build_samples, make_task

    code = code_version("convirt_probe.py", "samples.py")
    report_code(code)
    ckpt_path = Path(args.checkpoint)
    kind = "probe_medsiglip" if args.embeddings else "probe"
    tag = args.tag or f"{kind}_{code_label(code)}_{ckpt_path.stem}"
    results = Path(args.results_dir or Path(args.data_root) / "medmod_ihm_results")
    results.mkdir(parents=True, exist_ok=True)
    out_path = results / f"{tag}.json"
    probe_ckpt = results / f"{tag}_checkpoint.pt"
    if out_path.exists() and "test_final" in json.loads(out_path.read_text()):
        print(f"{tag} already finished", flush=True)
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # the presets draw from numpy's global RNG
    dev = torch.device(args.device)

    paths = MedModPaths.from_root(args.data_root)
    _, (train, val, test) = build_samples(paths, make_task(paths, "drop"),
                                          args.num_workers)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    pre = ck.get("settings", {})
    if bool(pre.get("embeddings")) != bool(args.embeddings):
        raise SystemExit("pass --embeddings exactly when the checkpoint was "
                         "pretrained with --embeddings")
    table, emb_record = None, None
    if args.embeddings:
        table, emb_record = load_embedding_table(args.embeddings)
        image = {"cxr_backbone": nn.Identity(), "cxr_feature_dim": table[1].shape[1]}
    else:
        image = {"cxr_backbone": pre.get("backbone", "resnet34")}
    convirt = ConVIRT(train, temperature=pre.get("temperature", 0.01), **image)
    convirt.load_state_dict(ck["model"])
    probe = ConVIRTLinearProbe(train, convirt).to(dev)
    del convirt
    trainable = [p for p in probe.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    tr_tf, ev_tf = get_preset("convirt", train=True), get_preset("convirt")

    def with_images(batch, tf):
        if table is not None:  # stored embeddings: no transform
            batch["cxr"] = lookup_embeddings(batch["image_path"], *table)
        else:
            batch["cxr"] = torch.stack([tf(load_cxr(p)) for p in batch["image_path"]])
        return batch

    @torch.no_grad()
    def predict(part):
        probe.eval()
        ys, ps = [], []
        for batch in get_dataloader(part, batch_size=args.batch_size, shuffle=False):
            out = probe(**with_images(batch, ev_tf))
            ps.append(out["y_prob"].float().cpu().reshape(-1))
            ys.append(out["y_true"].float().cpu().reshape(-1))
        return torch.cat(ys).numpy(), torch.cat(ps).numpy()

    fixed = None
    if args.no_augment or table is not None:  # fixed inputs: features once
        fixed = {}
        for name, part in (("train", train), ("val", val), ("test", test)):
            loader = (with_images(b, ev_tf) for b in
                      get_dataloader(part, batch_size=args.batch_size, shuffle=False))
            fixed[name] = probe.extract_features(loader)

    def train_epoch(epoch):
        probe.train()
        losses = []
        if fixed is not None:
            x, y = fixed["train"]
            g = torch.Generator().manual_seed(args.seed + epoch)
            order = torch.randperm(len(x), generator=g)
            n_full = len(x) // args.batch_size * args.batch_size  # drop_last
            for i in range(0, n_full, args.batch_size):
                idx = order[i:i + args.batch_size]
                logit = probe.linear(x[idx].to(dev))
                loss = probe.label_processor.loss(logit, y[idx].to(dev))
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
            return float(np.mean(losses))
        loader = get_dataloader(train, batch_size=args.batch_size, shuffle=True,
                                seed=args.seed, drop_last=True)
        loader.epoch = epoch + 1
        for batch in loader:
            loss = probe(**with_images(batch, tr_tf))["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        return float(np.mean(losses))

    def evaluate(name, part):
        if fixed is None:
            return predict(part)
        x, y = fixed[name]
        probe.eval()
        with torch.no_grad():
            p = probe.label_processor.activation(probe.linear(x.to(dev)))
            p = p.float().cpu().reshape(-1)
        return y.float().reshape(-1).numpy(), p.numpy()

    report = {
        "settings": {k: v for k, v in vars(args).items()
                     if k not in ("data_root", "results_dir")},
        "code": code,
        "pretraining": {"checkpoint": ckpt_path.name, "epoch": ck.get("epoch"),
                        "code": ck.get("code"), "settings": pre},
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "trainable_params": sum(p.numel() for p in trainable),
        "paper": PAPER,
        "embeddings": emb_record,
        "epochs": [],
    }
    best = {"val_auroc": -1.0, "epoch": -1, "state": None}
    stop_best, waited = -np.inf, 0
    start = 0
    # continue from the checkpoint of an interrupted run with the same
    # settings and code; otherwise start fresh
    pc = (torch.load(probe_ckpt, map_location="cpu", weights_only=True)
          if probe_ckpt.exists() else None)
    if pc is not None:
        keep = ("resume", "checkpoint_every", "device", "num_workers")
        old = {k: v for k, v in pc["settings"].items() if k not in keep}
        new = {k: v for k, v in report["settings"].items() if k not in keep}
        if old != new or not same_code(pc["code"], code):
            print("checkpoint from other settings or code: starting fresh", flush=True)
            pc = None
    if pc is not None:
        # the encoders are frozen; only the linear layer is trained
        probe.linear.load_state_dict(pc["linear"])
        opt.load_state_dict(pc["optimizer"])
        best, stop_best, waited = pc["best"], pc["stop_best"], pc["waited"]
        report["epochs"] = pc["epochs"]
        start = pc["epoch"] + 1  # every epoch re-seeds torch and numpy itself
        print(f"resuming after epoch {pc['epoch']}", flush=True)
    for epoch in range(start, args.epochs):
        t = time.time()
        torch.manual_seed(args.seed + epoch)
        np.random.seed(args.seed + epoch)
        train_loss = train_epoch(epoch)
        y, p = evaluate("val", val)
        val_auroc = float(roc_auc_score(y, p))
        if val_auroc > best["val_auroc"]:
            best = {"val_auroc": val_auroc, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in probe.linear.state_dict().items()}}
        row = {"epoch": epoch, "train_loss": train_loss, "val_auroc": val_auroc,
               "seconds": round(time.time() - t, 1)}
        report["epochs"].append(row)
        print(json.dumps(row), flush=True)
        out_path.write_text(json.dumps(report, indent=2))
        monitored = round(val_auroc, 4)  # MedMod logs np.round(auroc, 4)
        if monitored > stop_best + args.min_delta:
            stop_best, waited = monitored, 0
        else:
            waited += 1
            if waited >= args.patience:
                print(f"early stop after epoch {epoch}", flush=True)
                break
        if args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0:
            save_atomic({"epoch": epoch, "settings": report["settings"],
                         "code": code, "linear": probe.linear.state_dict(),
                         "optimizer": opt.state_dict(), "best": best,
                         "stop_best": float(stop_best), "waited": waited,
                         "epochs": report["epochs"]}, probe_ckpt)

    y, p = evaluate("test", test)
    report["test_final"] = {"epoch": report["epochs"][-1]["epoch"], **scores(y, p)}
    probe.linear.load_state_dict(best["state"])
    y, p = evaluate("test", test)
    report["test_best_val"] = {"epoch": best["epoch"], "val_auroc": best["val_auroc"],
                               **scores(y, p)}
    out_path.write_text(json.dumps(report, indent=2))
    print("TEST final", json.dumps(report["test_final"]))
    print("TEST best-val", json.dumps(report["test_best_val"]))


if __name__ == "__main__":
    main()
