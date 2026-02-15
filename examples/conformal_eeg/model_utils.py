"""Shared helpers for TUEV conformal scripts: model choice (ContraWR vs TFM-Tokenizer) and STFT dataset wrapper."""

from __future__ import annotations

import numpy as np
import torch

from pyhealth.models import ContraWR, TFMTokenizer


def compute_stft(signal_1d: np.ndarray, n_fft: int = 128, hop_length: int = 64) -> np.ndarray:
    """Compute magnitude STFT for a 1D signal. Returns (n_freq, n_time) float32."""
    if signal_1d.ndim != 1:
        signal_1d = np.asarray(signal_1d).mean(axis=0)
    signal_1d = np.asarray(signal_1d, dtype=np.float32)
    # Use torch.stft for consistency with typical n_fft/hop_length semantics
    t = torch.from_numpy(signal_1d).unsqueeze(0)
    stft = torch.stft(t, n_fft=n_fft, hop_length=hop_length, return_complex=True)
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
        stft_np = compute_stft(signal_1d, self.n_fft, self.hop_length)
        sample["stft"] = torch.from_numpy(stft_np)
        return sample

    def subset(self, indices):
        return AddSTFTDataset(self._base.subset(indices), self.n_fft, self.hop_length)

    def set_shuffle(self, shuffle: bool) -> None:
        """Forward to base dataset so get_dataloader() works (pyhealth.datasets.utils)."""
        if hasattr(self._base, "set_shuffle"):
            self._base.set_shuffle(shuffle)


def get_model(args, sample_dataset, device: str):
    """Build ContraWR or TFMTokenizer from args.model. Use sample_dataset (possibly AddSTFTDataset for TFM)."""
    if getattr(args, "model", "contrawr").lower() == "tfm":
        n_fft = getattr(args, "n_fft", 128)
        n_freq = n_fft // 2 + 1
        model = TFMTokenizer(
            dataset=sample_dataset,
            n_freq=n_freq,
            emb_size=getattr(args, "tfm_emb_size", 64),
            code_book_size=getattr(args, "tfm_code_book_size", 8192),
        )
    else:
        model = ContraWR(dataset=sample_dataset, n_fft=getattr(args, "n_fft", 128))
    return model.to(device)
