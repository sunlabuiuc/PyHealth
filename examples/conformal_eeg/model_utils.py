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


RESAMPLING_RATE = 200  # TFM-Tokenizer standard; n_fft=200, hop_length=100, 100 freq bins


def get_stft_torch(X: torch.Tensor, resampling_rate: int = RESAMPLING_RATE) -> torch.Tensor:
    """Per-channel magnitude STFT matching TFM-Tokenizer repo. Input (B, C, T) -> output (B, C, 100, T')."""
    B, C, T = X.shape
    x_temp = X.reshape(B * C, T)
    window = torch.hann_window(resampling_rate, device=X.device, dtype=X.dtype)
    stft_complex = torch.stft(
        x_temp,
        n_fft=resampling_rate,
        hop_length=resampling_rate // 2,
        window=window,
        onesided=True,
        return_complex=True,
        center=False,
    )
    # (B*C, n_fft//2+1, T') -> take first 100 freq bins
    x_stft_temp = torch.abs(stft_complex)[:, : resampling_rate // 2, :]
    x_stft_temp = x_stft_temp.reshape(B, C, resampling_rate // 2, -1)
    return x_stft_temp


class AddSTFTDataset:
    """Wraps a TUEV/TUAB task dataset to add per-channel 'stft' for TFM-Tokenizer.
    Keeps 'signal' as (C, T); adds 'stft' as (C, 100, T') with n_fft=200, hop_length=100.
    Matches the original TFM-Tokenizer training pipeline (16 token sequences per sample).
    """

    def __init__(self, base, n_fft: int = 200, hop_length: int = 100):
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
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        # Normalize by 95th percentile of |signal| per channel (axis=-1), matching TFM training
        scale = np.quantile(
            np.abs(signal), q=0.95, axis=-1, method="linear", keepdims=True
        ) + 1e-8
        signal = signal / scale
        # signal (C, T) -> tensor
        signal_t = torch.from_numpy(signal)
        sample["signal"] = signal_t
        # Per-channel STFT: (1, C, T) -> (1, C, 100, T')
        stft = get_stft_torch(signal_t.unsqueeze(0), resampling_rate=self.n_fft)
        sample["stft"] = stft.squeeze(0)
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
