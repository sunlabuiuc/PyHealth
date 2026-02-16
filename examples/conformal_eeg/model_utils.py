"""Shared helpers for TUEV conformal scripts: model choice (ContraWR vs TFM-Tokenizer) and STFT dataset wrapper.

TFM runs use the same experimental protocol as ContraWR (split_seed, run_seeds, alpha,
ratios, fixed test set) so results are directly comparable.

TFM loading options:
- Single checkpoint: --tfm-checkpoint (full model or tokenizer-only; use {seed} for per-seed).
- Two checkpoints (pretrained tokenizer + finetuned classifier): --tfm-tokenizer-checkpoint
  and --tfm-classifier-checkpoint (use {seed} in classifier path for per-seed). Use
  --tfm-skip-train to run calibration + inference only.

Scripts support --dataset tuev|tuab (same protocol; TUAB uses binary task, TUEV uses multiclass).
"""

from __future__ import annotations

import numpy as np
import torch

from pyhealth.models import ContraWR, TFMTokenizer


def compute_stft(
    signal_1d: np.ndarray,
    n_fft: int = 128,
    hop_length: int = 64,
    center: bool = False,
) -> np.ndarray:
    """Compute magnitude STFT for a 1D signal. Returns (n_freq, n_time) float32.
    Use center=False for TFM so n_time = (L - n_fft) // hop_length + 1, matching
    the tokenizer's temporal conv (kernel 200, stride 100).
    """
    if signal_1d.ndim != 1:
        signal_1d = np.asarray(signal_1d).mean(axis=0)
    signal_1d = np.asarray(signal_1d, dtype=np.float32)
    t = torch.from_numpy(signal_1d).unsqueeze(0)
    stft = torch.stft(
        t,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
        center=center,
    )
    mag = stft.abs().squeeze(0).numpy()
    return mag.astype(np.float32)


class AddSTFTDataset:
    """Wraps a TUEV task dataset to add 'stft' and convert 'signal' to 1D (mean over channels) for TFM-Tokenizer."""

    def __init__(self, base, n_fft: int = 128, hop_length: int = 64):
        self._base = base
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.input_schema = {**getattr(base, "input_schema", {}), "stft": "tensor"}
        self.output_schema = getattr(base, "output_schema", {})

    @property
    def output_processors(self):
        """Forward so BaseModel.get_output_size() works (TFMTokenizer/ContraWR)."""
        return getattr(self._base, "output_processors", {})

    @property
    def input_processors(self):
        """Forward in case the model reads input_processors."""
        return getattr(self._base, "input_processors", {})

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, i: int):
        sample = dict(self._base[i])
        signal = sample["signal"]
        if np.ndim(signal) == 2:
            signal_1d = np.asarray(signal, dtype=np.float32).mean(axis=0)
        else:
            signal_1d = np.asarray(signal, dtype=np.float32).flatten()
        # Return tensors so get_dataloader's collate stacks them (not list)
        sample["signal"] = torch.from_numpy(signal_1d)
        # center=False so n_time = (L - n_fft)//hop_length + 1, matching TFM temporal conv
        stft_np = compute_stft(
            signal_1d, self.n_fft, self.hop_length, center=False
        )
        # TFM tokenizer expects 100 freq bins; crop if we used n_fft=200 (101 bins)
        if stft_np.shape[0] > 100:
            stft_np = stft_np[:100]
        sample["stft"] = torch.from_numpy(stft_np)
        return sample

    def subset(self, indices):
        return AddSTFTDataset(self._base.subset(indices), self.n_fft, self.hop_length)

    def set_shuffle(self, shuffle: bool) -> None:
        """Forward to base dataset so get_dataloader() works (pyhealth.datasets.utils)."""
        if hasattr(self._base, "set_shuffle"):
            self._base.set_shuffle(shuffle)


def _load_tfm_checkpoint(model, checkpoint_path: str, device: str):
    """Load TFM checkpoint: full model state_dict or tokenizer-only (legacy)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    if not isinstance(state, dict):
        model.load_pretrained_weights(checkpoint_path, map_location=device)
        return
    keys = list(state.keys())
    if any(str(k).startswith("tokenizer.") or str(k).startswith("classifier.") for k in keys):
        model.load_state_dict(state, strict=False)
        print(f"  Loaded full model from {checkpoint_path}")
    else:
        model.load_pretrained_weights(checkpoint_path, map_location=device)


def _load_tfm_classifier_checkpoint(model, checkpoint_path: str, device: str):
    """Load classifier-only checkpoint into model.classifier. Handles keys with or without 'classifier.' prefix."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    if not isinstance(state, dict):
        return
    keys = list(state.keys())
    if any(str(k).startswith("classifier.") for k in keys):
        model.load_state_dict(state, strict=False)
        print(f"  Loaded classifier from {checkpoint_path}")
    else:
        model.classifier.load_state_dict(state, strict=False)
        print(f"  Loaded classifier from {checkpoint_path}")


def get_model(args, sample_dataset, device: str):
    """Build ContraWR or TFMTokenizer from args.model. Use sample_dataset (possibly AddSTFTDataset for TFM).
    Loading options (TFM):
    - args.tfm_checkpoint: single path (full model or tokenizer-only).
    - args.tfm_tokenizer_checkpoint + args.tfm_classifier_checkpoint: pretrained tokenizer + per-seed finetuned classifier.
    """
    if getattr(args, "model", "contrawr").lower() == "tfm":
        model = TFMTokenizer(
            dataset=sample_dataset,
            n_freq=100,
            emb_size=getattr(args, "tfm_emb_size", 64),
            code_book_size=getattr(args, "tfm_code_book_size", 8192),
        )
        model = model.to(device)
        tokenizer_ckpt = getattr(args, "tfm_tokenizer_checkpoint", None)
        classifier_ckpt = getattr(args, "tfm_classifier_checkpoint", None)
        single_ckpt = getattr(args, "tfm_checkpoint", None)
        if tokenizer_ckpt and classifier_ckpt:
            model.load_pretrained_weights(tokenizer_ckpt, map_location=device)
            _load_tfm_classifier_checkpoint(model, classifier_ckpt, device)
        elif single_ckpt:
            _load_tfm_checkpoint(model, single_ckpt, device)
        if getattr(args, "tfm_freeze_tokenizer", False):
            for p in model.tokenizer.parameters():
                p.requires_grad = False
        return model
    else:
        model = ContraWR(dataset=sample_dataset, n_fft=getattr(args, "n_fft", 128))
        return model.to(device)
