#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KaggleERN_finetune_EEGPT.py
==========================

Fine-tune an EEGPT-like encoder on KaggleERN (INRIA BCI Challenge) windows stored as pickles.

Expected preprocessed format (per file):
    {"signal": np.ndarray(C, T), "label": 0/1, "epoch_id": str}

Expected folder structure:
    <DATA_ROOT>/
      train/
        *.pickle
      val/
        *.pickle
      test/
        *.pickle

Placeholders you MUST set (via CLI args):
    --data_root: path to your preprocessed KaggleERN window pickles (train/val/test)
    --ckpt_path: path to EEGPT pretrained .ckpt file

## Pretrained Models

You can download pretrained models here:

- [EEG_large](https://figshare.com/s/e37df4f8a907a866df4b) (in the 'Files/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt')
  trained on mixed dataset (58-channels, 256Hz, 4s time length EEG) using patch size 64.

For downstream tasks, you should place it into `checkpoint` folder as file name
"checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt". To use the model, simply load the checkpoint and
pass it to the `EEGPTClassifier` class in "downstream/Modules/models/EEGPT_mcae_finetune.py".

Note:
- This script is self-contained (does NOT require the EEGPT repo code), but it follows the same
  checkpoint file naming convention for convenience.
- If your checkpoint key-prefix differs (e.g., "model.encoder."), the loader tries common prefixes.

Example:
    python examples/eeg/KaggleERN_finetune_EEGPT.py \
        --data_root /path/to/processed_kaggle_ern/s42_n56-eegpt \
        --ckpt_path  checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt \
        --output_dir outputs/kaggleern_eegpt \
        --epochs 50 --batch_size 256 --lr 4e-4 --weight_decay 1e-3
"""

from __future__ import annotations

import os
import glob
import math
import time
import pickle
import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Repro / utilities
# -----------------------------
def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_str() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


@torch.no_grad()
def compute_metrics_binary(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    y_true: shape (N,), values in {0,1}
    y_prob: shape (N,), probability of class 1
    """
    y_pred = (y_prob >= 0.5).astype(np.int32)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = float((tp + tn) / max(1, len(y_true)))
    tpr = float(tp / max(1, (tp + fn)))
    tnr = float(tn / max(1, (tn + fp)))
    bacc = float(0.5 * (tpr + tnr))

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")

    return {"acc": acc, "bacc": bacc, "auc": auc}


# -----------------------------
# Dataset: read pickles
# -----------------------------
class PickleWindowDataset(Dataset):
    """
    Each pickle file is a dict:
        {"signal": np.ndarray(C, T), "label": 0/1, "epoch_id": str}
    """
    def __init__(self, folder: str):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        self.paths = sorted(glob.glob(os.path.join(folder, "*.pickle"))) \
                   + sorted(glob.glob(os.path.join(folder, "*.pkl"))) \
                   + sorted(glob.glob(os.path.join(folder, "*.pql")))
        if len(self.paths) == 0:
            raise RuntimeError(f"No pickle files found in: {folder}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        with open(p, "rb") as f:
            obj = pickle.load(f)

        x = np.asarray(obj["signal"], dtype=np.float32)  # (C, T)
        y = int(obj["label"])
        # sanitize
        x = np.nan_to_num(x, posinf=0.0, neginf=0.0)

        x = torch.from_numpy(x)  # (C, T)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


def class_stats(folder: str) -> Tuple[int, int]:
    paths = sorted(glob.glob(os.path.join(folder, "*.pickle"))) \
          + sorted(glob.glob(os.path.join(folder, "*.pkl"))) \
          + sorted(glob.glob(os.path.join(folder, "*.pql")))
    n_pos, n_tot = 0, 0
    for p in paths:
        with open(p, "rb") as f:
            y = int(pickle.load(f)["label"])
        n_pos += int(y == 1)
        n_tot += 1
    return n_pos, n_tot


def make_loaders(data_root: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")
    test_dir  = os.path.join(data_root, "test")

    n_pos, n_tot = class_stats(train_dir)
    n_neg = max(1, n_tot - n_pos)
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32)

    ds_train = PickleWindowDataset(train_dir)
    ds_val   = PickleWindowDataset(val_dir)
    ds_test  = PickleWindowDataset(test_dir)

    def _collate(batch):
        xs, ys = zip(*batch)
        # xs: list[(C,T)] -> (B,C,T)
        return torch.stack(xs, 0), torch.stack(ys, 0)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=_collate)
    loader_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, collate_fn=_collate)
    loader_test  = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, collate_fn=_collate)

    return loader_train, loader_val, loader_test, pos_weight


def peek_input_shape(loader: DataLoader) -> Tuple[int, int]:
    x, _ = next(iter(loader))
    return int(x.shape[1]), int(x.shape[2])  # (C, T)


# -----------------------------
# EEGPT-like encoder (minimal)
# -----------------------------
def temporal_interpolation(x: torch.Tensor, target_len: int) -> torch.Tensor:
    # x: (B, C, T)
    if x.shape[-1] == target_len:
        return x
    return torch.nn.functional.interpolate(x, size=target_len, mode="linear", align_corners=False)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(58, 1024), patch_size=64, patch_stride=None, embed_dim=512):
        super().__init__()
        C, T = img_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if patch_stride is None:
            self.num_patches = (C, T // patch_size)
        else:
            self.num_patches = (C, ((T - patch_size) // patch_stride + 1))
        self.proj = nn.Conv2d(
            1, embed_dim,
            kernel_size=(1, patch_size),
            stride=(1, patch_size if patch_stride is None else patch_stride),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = x.unsqueeze(1)                # (B, 1, C, T)
        x = self.proj(x).transpose(1, 3)  # (B, T_p, C, D)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = attn_drop

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False
        )
        y = y.transpose(1, 2).contiguous().view(B, N, D)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class EEGTransformerEncoder(nn.Module):
    """
    Minimal EEGPT-like encoder.
    Input:
        x: (B, C, T)
        chan_ids: (1, C) int64
    Output:
        z: (B, N_time_patches, embed_num, embed_dim)
    """
    def __init__(
        self,
        img_size=(19, 512),
        patch_size=64,
        patch_stride=32,
        embed_dim=512,
        embed_num=4,
        depth=3,
        num_heads=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_num = embed_num

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, patch_stride=patch_stride, embed_dim=embed_dim)
        _, n_time = self.patch_embed.num_patches
        self.n_time = n_time

        self.blocks = nn.ModuleList([Block(embed_dim, num_heads=num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))
        nn.init.trunc_normal_(self.summary_token, std=0.02)

        self.chan_embed = nn.Embedding(128, embed_dim)
        nn.init.trunc_normal_(self.chan_embed.weight, std=0.02)

    @staticmethod
    def _sinusoidal_pos_emb(n: int, d: int, device, dtype) -> torch.Tensor:
        pos = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)  # (n,1)
        i = torch.arange(d, device=device, dtype=dtype).unsqueeze(0)    # (1,d)
        div = torch.exp((-(2 * (i // 2)) * math.log(10000.0) / d))
        pe = pos * div
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.view(1, n, 1, d)  # (1,n,1,d)

    def forward(self, x: torch.Tensor, chan_ids: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # (B, N, C, D)
        B, N, C, D = x.shape

        x = x + self.chan_embed(chan_ids.to(x.device).long()).unsqueeze(0)  # (1,1,C,D)
        x = x + self._sinusoidal_pos_emb(N, D, x.device, x.dtype)           # (1,N,1,D)

        x = x.flatten(0, 1)  # (B*N, C, D)
        summary = self.summary_token.expand(x.shape[0], -1, -1)  # (B*N, embed_num, D)
        x = torch.cat([x, summary], dim=1)  # (B*N, C+embed_num, D)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x[:, -self.embed_num:, :])     # keep summary tokens
        x = x.view(B, N, self.embed_num, D)
        return x


class EEGPTBinaryClassifier(nn.Module):
    """
    KaggleERN binary classification head on top of an EEGPT-like encoder.

    This keeps your pickle format unchanged and keeps the pipeline KaggleERN-only.
    """
    def __init__(
        self,
        in_channels: int,
        in_seq_len: int,
        eegpt_seq_len: int = 512,
        patch_size: int = 64,
        patch_stride: int = 32,
        embed_dim: int = 512,
        embed_num: int = 4,
        depth: int = 3,
        num_heads: int = 8,
        proj_channels: int = 19,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.eegpt_seq_len = eegpt_seq_len
        self.proj_channels = proj_channels

        # project input channels -> 19 (like common EEG downstream setting)
        self.chan_proj = nn.Conv1d(in_channels, proj_channels, kernel_size=1, bias=False)

        self.encoder = EEGTransformerEncoder(
            img_size=(proj_channels, eegpt_seq_len),
            patch_size=patch_size,
            patch_stride=patch_stride,
            embed_dim=embed_dim,
            embed_num=embed_num,
            depth=depth,
            num_heads=num_heads,
        )

        # N_time_patches for (T=512, patch=64, stride=32) => 15
        n_time = ((eegpt_seq_len - patch_size) // patch_stride) + 1
        self.n_time = n_time

        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_num * embed_dim, 16)
        self.fc2 = nn.Linear(n_time * 16, 2)

        self.register_buffer("chan_ids", torch.arange(proj_channels).view(1, -1).long())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)

        # basic stabilization (keep it minimal + general)
        x = x / 10.0
        x = x - x.mean(dim=-2, keepdim=True)  # subtract mean across channels
        x = torch.nan_to_num(x, posinf=0.0, neginf=0.0)

        # interpolate to EEGPT expected time length
        x = temporal_interpolation(x, self.eegpt_seq_len)

        # project channels to 19
        x = self.chan_proj(x)

        z = self.encoder(x, self.chan_ids)     # (B, N, E, D)
        h = z.flatten(2)                       # (B, N, E*D)
        h = self.fc1(self.drop(h))             # (B, N, 16)
        h = h.flatten(1)                       # (B, N*16)
        logits = self.fc2(h)                   # (B, 2)
        return logits


# -----------------------------
# Checkpoint loading (robust)
# -----------------------------
def safe_load_ckpt(ckpt_path: str) -> Dict[str, Any]:
    """
    Try common checkpoint formats:
      - Lightning: {"state_dict": {...}}
      - raw state_dict
    """
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unrecognized ckpt format: {type(obj)}")


def extract_encoder_state(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Extract encoder.* weights with common prefixes.
    Returns a dict to load into model.encoder (keys without leading 'encoder.').
    """
    candidates = ("encoder.", "model.encoder.", "target_encoder.", "module.encoder.", "model.module.encoder.")
    enc_state: Dict[str, torch.Tensor] = {}

    for k, v in sd.items():
        for pref in candidates:
            if k.startswith(pref):
                new_k = k[len(pref):]
                enc_state[new_k] = v
                break

    return enc_state


def maybe_resize_chan_embed(model: EEGTransformerEncoder, enc_state: Dict[str, torch.Tensor]) -> None:
    """
    If checkpoint chan_embed differs in shape, resize model.chan_embed.
    """
    key = "chan_embed.weight"
    if key not in enc_state:
        return
    w = enc_state[key]
    ckpt_n, ckpt_dim = int(w.shape[0]), int(w.shape[1])
    cur_n, cur_dim = int(model.chan_embed.weight.shape[0]), int(model.chan_embed.weight.shape[1])
    if (ckpt_n, ckpt_dim) == (cur_n, cur_dim):
        return

    device = model.chan_embed.weight.device
    dtype = model.chan_embed.weight.dtype
    new_emb = nn.Embedding(ckpt_n, ckpt_dim).to(device=device, dtype=dtype)
    with torch.no_grad():
        new_emb.weight.copy_(w.to(device=device, dtype=dtype))
    model.chan_embed = new_emb


def load_pretrained_encoder(model: EEGPTBinaryClassifier, ckpt_path: str) -> None:
    sd = safe_load_ckpt(ckpt_path)
    enc_state = extract_encoder_state(sd)
    if len(enc_state) == 0:
        print(f"[warn] No encoder weights found in ckpt: {ckpt_path}")
        return

    maybe_resize_chan_embed(model.encoder, enc_state)
    missing, unexpected = model.encoder.load_state_dict(enc_state, strict=False)
    print("[ckpt] encoder loaded.")
    if len(missing) > 0:
        print("[ckpt] missing keys:", missing[:30], "..." if len(missing) > 30 else "")
    if len(unexpected) > 0:
        print("[ckpt] unexpected keys:", unexpected[:30], "..." if len(unexpected) > 30 else "")


# -----------------------------
# Train / eval
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true, y_prob = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        prob1 = torch.softmax(logits, dim=1)[:, 1]
        y_true.append(y.detach().cpu().numpy())
        y_prob.append(prob1.detach().cpu().numpy())

    y_true = np.concatenate(y_true).astype(np.int32)
    y_prob = np.concatenate(y_prob).astype(np.float32)
    return compute_metrics_binary(y_true, y_prob)


def train_one_run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("[env] device:", device)
    print("[env] data_root:", args.data_root)
    print("[env] ckpt_path:", args.ckpt_path)
    print("[env] output_dir:", args.output_dir)

    ensure_dir(args.output_dir)

    # Data
    train_loader, val_loader, test_loader, pos_weight = make_loaders(
        args.data_root, args.batch_size, args.num_workers
    )
    in_C, in_T = peek_input_shape(train_loader)
    print(f"[data] detected input shape: C={in_C}, T={in_T}")
    n_pos, n_tot = class_stats(os.path.join(args.data_root, "train"))
    print(f"[data] train samples={n_tot}, pos={n_pos}, neg={max(1, n_tot-n_pos)}, pos_weight={pos_weight.item():.4f}")

    # Model
    model = EEGPTBinaryClassifier(
        in_channels=in_C,
        in_seq_len=in_T,
        eegpt_seq_len=args.eegpt_seq_len,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        embed_dim=args.embed_dim,
        embed_num=args.embed_num,
        depth=args.depth,
        num_heads=args.num_heads,
        proj_channels=args.proj_channels,
        dropout=args.dropout,
    ).to(device)

    # Load pretrained encoder weights (optional but recommended)
    if args.ckpt_path and os.path.isfile(args.ckpt_path):
        load_pretrained_encoder(model, args.ckpt_path)
    else:
        print("[warn] ckpt_path not found; training from scratch:", args.ckpt_path)

    # Optionally warmup (freeze encoder)
    if args.warmup_epochs > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print(f"[train] warmup: freeze encoder for {args.warmup_epochs} epochs")

    # Loss (class imbalance)
    # CrossEntropyLoss expects logits (B,2) and target (B,)
    class_weights = torch.tensor([1.0, float(pos_weight.item())], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=args.pct_start,
        anneal_strategy="cos",
    )

    scaler = torch.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val_bacc = -1.0
    best_path = os.path.join(args.output_dir, "checkpoint-best.pth")
    last_path = os.path.join(args.output_dir, "checkpoint-last.pth")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        # Unfreeze after warmup
        if args.warmup_epochs > 0 and epoch == args.warmup_epochs + 1:
            for p in model.encoder.parameters():
                p.requires_grad = True
            print("[train] warmup finished: encoder unfrozen")

        running_loss, n_batches = 0.0, 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(args.amp and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            if not torch.isfinite(loss):
                print("[warn] non-finite loss, skipping batch")
                continue

            scaler.scale(loss).backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.detach().cpu().item())
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        val_metrics = evaluate(model, val_loader, device)
        dur = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | val_bacc={val_metrics['bacc']:.4f} | val_auc={val_metrics['auc']:.4f} | "
            f"{dur:.1f}s"
        )

        # Save last
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "optim": optimizer.state_dict(), "args": vars(args)},
            last_path
        )

        # Save best by val_bacc
        if val_metrics["bacc"] > best_val_bacc:
            best_val_bacc = val_metrics["bacc"]
            torch.save(
                {"epoch": epoch, "best_val_bacc": best_val_bacc, "model": model.state_dict(), "optim": optimizer.state_dict(), "args": vars(args)},
                best_path
            )
            print(f"  -> Saved BEST (val_bacc={best_val_bacc:.4f})")

    # Load best and test
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"[test] loaded best checkpoint: {best_path}")

    test_metrics = evaluate(model, test_loader, device)
    print("========== TEST ==========")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
    print("==========================")

    # Export final weights
    export_path = os.path.join(args.output_dir, "kaggleern_eegpt_finetuned.pth")
    torch.save(model.state_dict(), export_path)
    print("[export] saved fine-tuned weights to:", export_path)


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("KaggleERN EEGPT fine-tuning example")

    # REQUIRED by user
    p.add_argument("--data_root", type=str, required=True,
                   help="Path to preprocessed KaggleERN pickles root containing train/val/test subfolders.")
    p.add_argument("--ckpt_path", type=str, default="checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt",
                   help="Path to EEGPT pretrained checkpoint (.ckpt).")

    # output / runtime
    p.add_argument("--output_dir", type=str, default=f"outputs/kaggleern_eegpt_{now_str()}",
                   help="Where to save logs/checkpoints/exports.")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--pct_start", type=float, default=0.2)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA only).")
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--warmup_epochs", type=int, default=0,
                   help="Freeze encoder for first N epochs, then unfreeze.")

    # model hyperparams (keep defaults aligned with common EEGPT downstream settings)
    p.add_argument("--eegpt_seq_len", type=int, default=512)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--patch_stride", type=int, default=32)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--embed_num", type=int, default=4)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--proj_channels", type=int, default=19)
    p.add_argument("--dropout", type=float, default=0.5)

    return p


def main():
    args = build_parser().parse_args()
    train_one_run(args)


if __name__ == "__main__":
    main()
