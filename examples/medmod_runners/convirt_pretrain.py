"""ConVIRT pretraining on the whole-stay MedMod pretraining set, MedMod settings.

Run from the repo root, or on Colab from the code zip made by
``colab_bundle.py``:

    cd "<repo>"
    PYTHONPATH="$PWD" python -u -m examples.medmod_runners.convirt_pretrain \\
        --data-root /path/to/mimic_data --tag convirt_resnet34

Code provenance: the run records the git commit of the ``pyhealth`` it
imports (or, on Colab, the time the code zip was made) and the SHA-256 of this
script, ``pretrain_samples.py`` and ``provenance.py``, and notes uncommitted
changes.

Run the same command again after a disconnect: a finished run stops at
once, an interrupted run continues from its checkpoint when the settings and
code are the same, and otherwise the run starts fresh.

Data: the verified pretraining set of ``pretrain_samples`` (7,755 train / 882
val stays, whole ICU stays, 2 to 2,085 hourly bins). Batches are random (no
length bucketing); the collate zero-pads at the end and returns the true
lengths, and the LSTM packs with them (MedMod src/models/encoders.py:44-45).

MedMod settings (src/scripts/pretrain/simclr.sh; src/models/selfsupervised/
models/simclr_model.py; src/parser.py):
* 300 epochs, batch 256, temperature 0.01, ResNet-34, LSTM 1 x 128,
  projection 128 (the ConVIRT defaults in pyhealth.models);
* LARS (lightning-flash 0.7.5, copied below) lr 0.6044, momentum 0.9,
  weight decay 1e-4 (parser.py:56) on all parameters (simclr_model.py:86);
* schedule (simclr_model.py:63, 90-93, 151-155): LambdaLR warmup
  (epoch + 1) / 10, stepped at the end of epochs 0-8, then
  CosineAnnealingWarmRestarts(T_0=500) stepped at the end of every later
  epoch; the lr starts at 0.06044, reaches 0.6044 at epoch 9 and is ~0.23 at
  epoch 299 (T_0 = 500 epochs, so the cosine never reaches its minimum);
* train loader shuffled with drop_last (data/utils.py:390): 30 steps/epoch;
  val loader batch 256, no shuffle, no drop_last, after every train epoch;
* images: get_preset("convirt", train=True) / get_preset("convirt") (MedMod
  simclrv2, pixels in [0, 1]), so ConVIRT runs with augment_cxr=False and no
  cxr_mean/std;
* seed 84305 (simclr_trainer.py:27).

MedSigLIP extension (--embeddings <.npz from pyhealth.datasets.medmod.medsiglip>):
the ResNet-34 is replaced by the stored frozen MedSigLIP embeddings
(``cxr_backbone=nn.Identity()``, ``cxr_feature_dim`` = their size), looked up
per batch (no image decoding, no augmentation); everything else as above. The
file's SHA-256 and settings go in the report.

Not copied: MedMod's EarlyStopping (val_loss_epoch, min_delta 0.001,
patience 30; simclr_model.py:244). The run goes the full 300 epochs and
records the epoch at which that rule would have stopped. Epochs are counted
from 0, as in MedMod's checkpoint names (Lightning's ``current_epoch``).

Reproducibility: the model is initialized after torch.manual_seed(seed);
epoch e uses shuffle order (seed, e) and re-seeds torch and numpy with
seed + e before its augmentations, which run in the main thread in batch
order (decoding runs in a thread pool). A resumed run therefore sees the
same batches and augmentations as an uninterrupted one. MPS kernels are not
guaranteed bit-identical between runs.

Outputs (aggregates only: losses, lr, timings, counts):
* <data-root>/medmod_ihm_results/<tag>_epochs.jsonl: one line per epoch;
* <data-root>/medmod_ihm_results/<tag>.json: settings + all epochs + summary;
* <data-root>/medmod_ihm_results/<tag>/: last.pt and best_val_loss.pt (full:
  model, optimizer, schedulers, history; last.pt is for continuing) and
  epoch_NNN.pt for every epoch (weights only, ~86 MB each, ~26 GB for 300;
  MedMod also keeps every epoch, simclr_model.py:232-239; --keep-every 10
  keeps every 10th). The weights are trained on MIMIC data, so keep
  them in your own storage and do not share them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required

SETTINGS_THAT_MUST_MATCH = (
    "epochs", "batch_size", "lr", "momentum", "weight_decay", "temperature",
    "warmup_epochs", "cosine_t0", "seed", "backbone", "embeddings",
)


class LARS(Optimizer):
    """lightning-flash 0.7.5 flash/core/optimizers/lars.py (Apache-2.0), as
    imported by MedMod (simclr_model.py:16). Same arithmetic; the
    ``p_norm != 0 and g_norm != 0`` test is a torch.where so that MPS does
    not synchronize on every parameter."""

    def __init__(self, params, lr=required, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False, trust_coefficient=0.001,
                 eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        self.eps = eps
        self.trust_coefficient = trust_coefficient
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    p_norm = torch.norm(p)
                    g_norm = torch.norm(d_p)
                    lars_lr = (
                        p_norm / (g_norm + p_norm * weight_decay + self.eps)
                        * self.trust_coefficient
                    )
                    scaled = d_p.add(p, alpha=weight_decay) * lars_lr
                    d_p = torch.where((p_norm != 0) & (g_norm != 0), scaled, d_p)
                if momentum != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    d_p = d_p.add(buf, alpha=momentum) if group["nesterov"] else buf
                p.add_(d_p, alpha=-group["lr"])


def make_schedulers(opt, warmup_epochs: int, t0: int):
    """MedMod's two schedulers, created in MedMod's order (simclr_model.py:
    88-93: "the order ... affects the initial starting learning rate")."""
    main = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, t0)
    warmup = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda epoch: (epoch + 1) / warmup_epochs
    )
    return main, warmup


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--results-dir",
                    help="where logs and checkpoints go (default "
                         "<data-root>/medmod_ihm_results; on Colab: a Drive folder)")
    ap.add_argument("--tag", default="convirt_pretrain",
                    help="name of the outputs in <data-root>/medmod_ihm_results")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.6044)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.01)
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--cosine-t0", type=int, default=500)
    ap.add_argument("--seed", type=int, default=84305)
    ap.add_argument("--backbone", default="resnet34")
    ap.add_argument("--embeddings", help="MedSigLIP .npz instead of the ResNet")
    ap.add_argument("--keep-every", type=int, default=1,
                    help="keep the weights of every N-th epoch (MedMod: every epoch, "
                         "simclr_model.py:232-239); 0 keeps none")
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1),
                    help="image decoding")
    default_dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        default_dev = "cuda"
    ap.add_argument("--device", default=default_dev)
    args = ap.parse_args()

    from .provenance import (
        code_version, load_embedding_table, report_code, same_code,
    )

    code = code_version("convirt_pretrain.py", "pretrain_samples.py")
    report_code(code)

    from torch import nn

    from pyhealth.datasets import get_dataloader, split_by_medmod
    from pyhealth.datasets.medmod.image_presets import get_preset, load_cxr
    from pyhealth.datasets.medmod.medsiglip import lookup_embeddings
    from pyhealth.models import ConVIRT

    from .pretrain_samples import VERIFIED, build_sample_dataset, load_raw

    results = Path(args.results_dir or Path(args.data_root) / "medmod_ihm_results")
    ckpt_dir = results / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_jsonl = results / f"{args.tag}_epochs.jsonl"
    report_path = results / f"{args.tag}.json"
    if report_path.exists():
        done = json.loads(report_path.read_text())
        if done.get("epochs_done", 0) >= args.epochs:
            print(f"{args.tag} already finished", flush=True)
            return
    settings = {k: v for k, v in vars(args).items()
                if k not in ("data_root", "results_dir", "resume",
                             "threads", "device")}
    table, emb_record = None, None
    if args.embeddings:
        table, emb_record = load_embedding_table(args.embeddings)
        settings["embeddings"] = emb_record["sha256"]  # the file, not its path

    t_load = time.time()
    raw, _ = load_raw(Path(args.data_root))
    dataset = build_sample_dataset(raw)
    del raw
    train, val, _ = split_by_medmod(dataset)
    counts = {"train": len(train), "val": len(val)}
    if counts != VERIFIED:
        raise SystemExit(f"counts {counts} differ from the verified {VERIFIED}")
    print(f"data: {counts} in {time.time() - t_load:.0f} s", flush=True)

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if table is not None:
        image = {"cxr_backbone": nn.Identity(), "cxr_feature_dim": table[1].shape[1]}
    else:
        image = {"cxr_backbone": args.backbone}
    model = ConVIRT(dataset, temperature=args.temperature, **image).to(dev)
    opt = LARS(model.parameters(), lr=args.lr, momentum=args.momentum,
               weight_decay=args.weight_decay)
    main_sched, warm_sched = make_schedulers(opt, args.warmup_epochs, args.cosine_t0)

    history, start = [], 0
    best = {"val_loss": math.inf, "epoch": -1}
    stop_rule = {"best": math.inf, "wait": 0, "would_stop_at": None}
    # continue from last.pt of an interrupted run with the same settings and
    # code; otherwise start fresh
    ck = (torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=True)
          if (ckpt_dir / "last.pt").exists() else None)
    if ck is not None:
        diff = {k for k in SETTINGS_THAT_MUST_MATCH
                if ck["settings"].get(k) != settings[k]}
        if diff or not same_code(ck.get("code", {}), code):
            print("checkpoint from other settings or code: starting fresh", flush=True)
            ck = None
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        main_sched.load_state_dict(ck["main_sched"])
        warm_sched.load_state_dict(ck["warm_sched"])
        history, best, stop_rule = ck["history"], ck["best"], ck["stop_rule"]
        start = ck["epoch"] + 1
        print(f"resumed after epoch {ck['epoch']} (best val loss "
              f"{best['val_loss']:.4f} at epoch {best['epoch']})", flush=True)
    elif log_jsonl.exists():
        log_jsonl.unlink()  # the epoch log of an unfinished run, started again

    n_params = sum(p.numel() for p in model.parameters())
    tr_tf, ev_tf = get_preset("convirt", train=True), get_preset("convirt")
    decode_pool = ThreadPoolExecutor(args.threads)
    prefetch = ThreadPoolExecutor(1)

    def decoded(loader):
        """Batches with their images decoded in the background, one ahead."""
        def job(batch):
            return batch, list(decode_pool.map(load_cxr, batch["image_path"]))
        it = iter(loader)
        nxt = next(it, None)
        pending = prefetch.submit(job, nxt) if nxt is not None else None
        while pending is not None:
            batch, images = pending.result()
            nxt = next(it, None)
            pending = prefetch.submit(job, nxt) if nxt is not None else None
            yield batch, images

    def run(part, epoch, training):
        model.train(training)
        loader = get_dataloader(part, batch_size=args.batch_size,
                                shuffle=training, seed=args.seed,
                                drop_last=training)
        loader.epoch = epoch + 1  # order (seed, epoch); litdata counts from 1
        tf = tr_tf if training else ev_tf
        losses, sizes = [], []
        batches = ((b, b["image_path"]) for b in loader) if table else decoded(loader)
        for batch, images in batches:
            if table is not None:  # stored embeddings: nothing to decode
                batch["cxr"] = lookup_embeddings(batch["image_path"], *table)
            else:
                batch["cxr"] = torch.stack([tf(img) for img in images])  # main thread
            with torch.set_grad_enabled(training):
                loss = model(**batch)["loss"]
                if training:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            losses.append(loss.detach())
            sizes.append(len(images))
        per_batch = torch.stack(losses).float().cpu().numpy()
        w = np.array(sizes, dtype=float)
        weighted = float((per_batch * w).sum() / w.sum())
        return weighted, float(per_batch.mean()), len(sizes)

    for epoch in range(start, args.epochs):
        t0 = time.time()
        lr = opt.param_groups[0]["lr"]
        torch.manual_seed(args.seed + epoch)
        np.random.seed(args.seed + epoch)
        train_loss, _, steps = run(train, epoch, True)
        t_train = time.time() - t0
        # MedMod steps a scheduler at the end of each epoch (simclr_model.py:152-155)
        (warm_sched if epoch < args.warmup_epochs - 1 else main_sched).step()
        with torch.no_grad():
            val_loss, val_loss_batch_mean, _ = run(val, epoch, False)

        improved = val_loss < best["val_loss"]
        if improved:
            best = {"val_loss": val_loss, "epoch": epoch}
        # MedMod's EarlyStopping rule, recorded only (simclr_model.py:244)
        if val_loss < stop_rule["best"] - 0.001:
            stop_rule["best"], stop_rule["wait"] = val_loss, 0
        else:
            stop_rule["wait"] += 1
            if stop_rule["wait"] >= 30 and stop_rule["would_stop_at"] is None:
                stop_rule["would_stop_at"] = epoch

        row = {
            "epoch": epoch, "lr": lr, "train_loss": train_loss, "train_steps": steps,
            "val_loss": val_loss, "val_loss_batch_mean": val_loss_batch_mean,
            "best_epoch": best["epoch"], "train_s": round(t_train, 1),
        }
        history.append(row)
        ck = {
            "epoch": epoch, "settings": settings, "code": code,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(), "main_sched": main_sched.state_dict(),
            "warm_sched": warm_sched.state_dict(), "history": history,
            "best": best, "stop_rule": stop_rule,
        }
        save_atomic(ck, ckpt_dir / "last.pt")
        if improved:
            shutil.copyfile(ckpt_dir / "last.pt", ckpt_dir / "best_val_loss.pt")
        # weights only, as MedMod's per-epoch checkpoints (save_weights_only)
        if args.keep_every and (epoch + 1) % args.keep_every == 0:
            save_atomic({"epoch": epoch, "settings": settings, "code": code,
                         "model": model.state_dict()},
                        ckpt_dir / f"epoch_{epoch:03d}.pt")
        row["epoch_s"] = round(time.time() - t0, 1)
        with open(log_jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")
        mean_s = np.mean([h.get("epoch_s", row["epoch_s"]) for h in history[-10:]])
        eta_h = (args.epochs - 1 - epoch) * mean_s / 3600
        print(json.dumps(row) + f"  ETA {eta_h:.1f} h", flush=True)
        report = {
            "settings": settings, "code": code, "counts": counts,
            "embeddings": emb_record,
            "parameters": n_params,
            "device": str(dev), "epochs_done": epoch + 1, "best": best,
            "medmod_early_stop_would_stop_at_epoch": stop_rule["would_stop_at"],
            "epochs": history,
        }
        report_path.write_text(json.dumps(report, indent=2))
    print("done", json.dumps({"epochs_done": len(history), "best": best,
                              "medmod_early_stop_would_stop_at_epoch":
                              stop_rule["would_stop_at"]}), flush=True)


if __name__ == "__main__":
    main()
