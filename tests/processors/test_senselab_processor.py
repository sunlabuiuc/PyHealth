"""Tests for senselab-based processors.

These tests use synthetic audio (generated with torch) to avoid
requiring real audio files or network downloads.
"""
import pytest
import torch


def make_sine_wav(tmp_path, sr=16000, duration=2.0):
    """Write a sine wave to a temp WAV file and return its path."""
    import torchaudio
    t = torch.linspace(0, duration, int(sr * duration))
    waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
    path = tmp_path / "test.wav"
    torchaudio.save(str(path), waveform, sr)
    return path


# --- SenselabVADProcessor ---

def test_vad_processor_returns_tensor(tmp_path):
    pytest.importorskip("senselab")
    from pyhealth.processors.senselab_processor import SenselabVADProcessor

    p = SenselabVADProcessor()
    wav = make_sine_wav(tmp_path)
    result = p.process(str(wav))
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 2  # (channels, samples)


def test_vad_processor_file_not_found():
    pytest.importorskip("senselab")
    from pyhealth.processors.senselab_processor import SenselabVADProcessor

    p = SenselabVADProcessor()
    with pytest.raises(FileNotFoundError):
        p.process("/nonexistent/audio.wav")


def test_vad_processor_is_not_token():
    pytest.importorskip("senselab")
    from pyhealth.processors.senselab_processor import SenselabVADProcessor

    assert SenselabVADProcessor().is_token() is False


# --- SenselabEGeMAPSProcessor ---

def test_egemaps_processor_output_shape(tmp_path):
    pytest.importorskip("senselab")
    from pyhealth.processors.senselab_processor import SenselabEGeMAPSProcessor

    p = SenselabEGeMAPSProcessor()
    wav = make_sine_wav(tmp_path)
    result = p.process(str(wav))
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 1
    assert result.shape[0] == 88  # eGeMAPS has 88 features


def test_egemaps_processor_is_not_token():
    pytest.importorskip("senselab")
    from pyhealth.processors.senselab_processor import SenselabEGeMAPSProcessor

    assert SenselabEGeMAPSProcessor().is_not_token() is False


# --- SenselabEmbeddingProcessor ---

def test_embedding_processor_wavlm_shape(tmp_path):
    pytest.importorskip("senselab")
    from pyhealth.processors.senselab_processor import SenselabEmbeddingProcessor

    p = SenselabEmbeddingProcessor(model="wavlm")
    wav = make_sine_wav(tmp_path)
    result = p.process(str(wav))
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 1


def test_embedding_processor_invalid_model():
    pytest.importorskip("senselab")
    from pyhealth.processors.senselab_processor import SenselabEmbeddingProcessor

    with pytest.raises(ValueError, match="Unsupported model"):
        SenselabEmbeddingProcessor(model="gpt5_audio")


def test_embedding_processor_import_error(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "senselab", None)
    if "pyhealth.processors.senselab_processor" in sys.modules:
        del sys.modules["pyhealth.processors.senselab_processor"]
    with pytest.raises(ImportError, match="pip install senselab"):
        from pyhealth.processors.senselab_processor import SenselabEmbeddingProcessor
        SenselabEmbeddingProcessor()
