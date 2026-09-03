"""
Senselab Processor Tutorial
============================
Demonstrates PyHealth's senselab processor wrappers for voice
biomarker extraction: VAD, eGeMAPS features, and speaker embeddings.

Use cases include depression detection, Parkinson's monitoring,
and cognitive assessment through voice.

Requires: pip install senselab
"""
import os
import tempfile

import torch
import torchaudio


def make_test_wav(path: str, sr: int = 16000, duration: float = 3.0) -> None:
    """Write a synthetic sine-wave audio file for demonstration."""
    t = torch.linspace(0, duration, int(sr * duration))
    waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
    torchaudio.save(path, waveform, sr)


with tempfile.TemporaryDirectory() as tmpdir:
    audio_path = os.path.join(tmpdir, "sample.wav")
    make_test_wav(audio_path)

    # ------------------------------------------------------------------
    # 1. SenselabVADProcessor
    #    Strips silence and noise, returns speech-only waveform tensor.
    # ------------------------------------------------------------------
    from pyhealth.processors import SenselabVADProcessor

    vad = SenselabVADProcessor(sample_rate=16000)
    speech = vad.process(audio_path)
    print(f"VAD output shape:     {speech.shape}")
    # Expected: torch.Size([1, N]) where N <= original samples

    # ------------------------------------------------------------------
    # 2. SenselabEGeMAPSProcessor
    #    Extracts 88 hand-crafted eGeMAPS acoustic features widely used
    #    in clinical voice research.
    # ------------------------------------------------------------------
    from pyhealth.processors import SenselabEGeMAPSProcessor

    egemaps = SenselabEGeMAPSProcessor(sample_rate=16000, apply_vad=True)
    features = egemaps.process(audio_path)
    print(f"eGeMAPS output shape: {features.shape}")
    # Expected: torch.Size([88])

    # ------------------------------------------------------------------
    # 3. SenselabEmbeddingProcessor
    #    Extracts deep neural speaker embeddings using WavLM or
    #    ECAPA-TDNN for voice biomarker tasks.
    # ------------------------------------------------------------------
    from pyhealth.processors import SenselabEmbeddingProcessor

    # WavLM embedding (768-dim)
    wavlm = SenselabEmbeddingProcessor(model="wavlm", sample_rate=16000)
    embedding = wavlm.process(audio_path)
    print(f"WavLM embedding shape: {embedding.shape}")
    # Expected: torch.Size([768])

    # ECAPA-TDNN embedding (192-dim)
    ecapa = SenselabEmbeddingProcessor(model="ecapa", sample_rate=16000)
    embedding_ecapa = ecapa.process(audio_path)
    print(f"ECAPA embedding shape: {embedding_ecapa.shape}")
    # Expected: torch.Size([192])
