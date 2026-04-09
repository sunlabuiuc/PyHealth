import torch
import pytest

from pyhealth.models.wav2sleep import SIGNAL_TO_SAMPLES_PER_EPOCH, SignalEncoders


B, T, D = 2, 2, 64


def _batch(signal: str) -> torch.Tensor:
    """Synthetic [B, T * samples_per_epoch] for one signal name."""
    s = SIGNAL_TO_SAMPLES_PER_EPOCH[signal]
    return torch.randn(B, T * s, dtype=torch.float32)


def test_signal_encoders_wav2sleep_contract_dict_and_shapes():
    """MVT 1: Output dict matches inputs; every tensor [B,T,D] and aligned; finite for finite in."""
    signal_encoder_map = {"ECG": "ecg_cnn", "PPG": "ppg_cnn"}
    enc = SignalEncoders(
        signal_encoder_map=signal_encoder_map,
        feature_dim=D,
        activation="gelu",
    )
    x = {"ECG": _batch("ECG"), "PPG": _batch("PPG")}
    out = enc(x)

    assert set(out.keys()) == set(x.keys())
    for k in x:
        assert out[k].shape == (B, T, D)
    assert torch.isfinite(out["ECG"]).all() and torch.isfinite(out["PPG"]).all()


def test_signal_encoders_wav2sleep_partial_modalities_same_contract():
    """MVT 2: Only a subset of modalities; same [B,T,D] contract (mirrors missing channels in Wav2Sleep)."""
    signal_encoder_map = {"ECG": "ecg_cnn", "PPG": "ppg_cnn", "ABD": "resp_cnn"}
    enc = SignalEncoders(
        signal_encoder_map=signal_encoder_map,
        feature_dim=D,
        activation="gelu",
    )
    x = {"ECG": _batch("ECG")}  # only one modality
    out = enc(x)

    assert set(out.keys()) == {"ECG"}
    assert out["ECG"].shape == (B, T, D)
    assert torch.isfinite(out["ECG"]).all()