from pathlib import Path
from typing import Any, Literal, Optional, Union

import torch

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("senselab_vad")
class SenselabVADProcessor(FeatureProcessor):
    """Voice Activity Detection (VAD) processor using senselab.

    Removes non-speech segments from an audio file before returning
    a waveform tensor. Useful as a preprocessing step before feature
    extraction in voice biomarker pipelines (e.g. depression detection,
    Parkinson's monitoring).

    Args:
        sample_rate: Target sample rate in Hz. Defaults to 16000
            (required by most senselab VAD models).
        return_tensor: If True, returns a torch.Tensor (channels, samples).
            If False, returns a list of senselab Audio segments.
            Defaults to True.

    Raises:
        ImportError: If senselab is not installed.
        FileNotFoundError: If the audio file path does not exist.

    Example:
        >>> processor = SenselabVADProcessor(sample_rate=16000)
        >>> tensor = processor.process("path/to/audio.wav")
        >>> tensor.shape
        torch.Size([1, N])
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        return_tensor: bool = True,
    ) -> None:
        try:
            import senselab  # noqa: F401
        except ImportError:
            raise ImportError(
                "SenselabVADProcessor requires senselab. "
                "Install it with: pip install senselab"
            )
        self.sample_rate = sample_rate
        self.return_tensor = return_tensor

    def process(self, value: Union[str, Path]) -> Any:
        """Run VAD on a single audio file and return speech-only audio.

        Args:
            value: Path to the audio file.

        Returns:
            torch.Tensor of shape (1, N) containing concatenated speech
            segments, or a list of senselab Audio objects if
            return_tensor=False.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        import torchaudio
        from senselab.audio.data_structures import Audio
        from senselab.audio.tasks.preprocessing.preprocessing import (
            resample_audios,
        )
        from senselab.audio.tasks.voice_activity_detection.api import (
            detect_voice_activity_in_audios,
        )

        audio_path = Path(value)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, orig_sr = torchaudio.load(audio_path)
        audio = Audio(waveform=waveform, sampling_rate=orig_sr)

        # Resample to target rate
        [audio] = resample_audios([audio], resample_rate=self.sample_rate)

        # Run VAD — returns list of Audio objects (speech segments only)
        speech_segments = detect_voice_activity_in_audios([audio])

        if not self.return_tensor:
            return speech_segments

        if not speech_segments:
            return torch.zeros(1, 0)

        waveforms = [seg.waveform for seg in speech_segments]
        combined = torch.cat(waveforms, dim=-1)
        return combined

    def is_token(self) -> bool:
        return False


@register_processor("senselab_egemaps")
class SenselabEGeMAPSProcessor(FeatureProcessor):
    """eGeMAPS feature extraction processor using senselab.

    Extracts the extended Geneva Minimalistic Acoustic Parameter Set
    (eGeMAPS) — 88 hand-crafted acoustic features widely used in
    clinical voice research for conditions such as depression,
    Parkinson's disease, and cognitive assessment.

    Args:
        sample_rate: Target sample rate in Hz. Defaults to 16000.
        apply_vad: Whether to apply Voice Activity Detection before
            extraction. Defaults to True.

    Raises:
        ImportError: If senselab is not installed.
        FileNotFoundError: If the audio file path does not exist.

    Example:
        >>> processor = SenselabEGeMAPSProcessor()
        >>> features = processor.process("path/to/audio.wav")
        >>> features.shape
        torch.Size([88])
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        apply_vad: bool = True,
    ) -> None:
        try:
            import senselab  # noqa: F401
        except ImportError:
            raise ImportError(
                "SenselabEGeMAPSProcessor requires senselab. "
                "Install it with: pip install senselab"
            )
        self.sample_rate = sample_rate
        self.apply_vad = apply_vad

    def process(self, value: Union[str, Path]) -> torch.Tensor:
        """Extract eGeMAPS features from a single audio file.

        Args:
            value: Path to the audio file.

        Returns:
            torch.Tensor of shape (88,) containing eGeMAPS features.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        import torchaudio
        from senselab.audio.data_structures import Audio
        from senselab.audio.tasks.features_extraction.opensmile import (
            extract_features_from_audios_with_opensmile,
        )
        from senselab.audio.tasks.preprocessing.preprocessing import (
            resample_audios,
        )
        from senselab.audio.tasks.voice_activity_detection.api import (
            detect_voice_activity_in_audios,
        )

        audio_path = Path(value)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, orig_sr = torchaudio.load(audio_path)
        audio = Audio(waveform=waveform, sampling_rate=orig_sr)
        [audio] = resample_audios([audio], resample_rate=self.sample_rate)

        if self.apply_vad:
            segments = detect_voice_activity_in_audios([audio])
            if segments:
                combined = torch.cat([s.waveform for s in segments], dim=-1)
                audio = Audio(waveform=combined, sampling_rate=self.sample_rate)

        features_df = extract_features_from_audios_with_opensmile([audio])
        feature_tensor = torch.tensor(
            features_df.values[0], dtype=torch.float32
        )
        return feature_tensor

    def is_token(self) -> bool:
        return False


@register_processor("senselab_embedding")
class SenselabEmbeddingProcessor(FeatureProcessor):
    """Deep speech embedding processor using senselab (WavLM / ECAPA-TDNN).

    Extracts dense neural embeddings from speech audio using pretrained
    models. These embeddings capture speaker identity and paralinguistic
    features useful for voice biomarker tasks.

    Args:
        model: Embedding model to use. One of ``"wavlm"`` or ``"ecapa"``.
            Defaults to ``"wavlm"``.
        sample_rate: Target sample rate in Hz. Defaults to 16000.
        apply_vad: Whether to apply VAD before embedding. Defaults to True.

    Raises:
        ImportError: If senselab is not installed.
        ValueError: If an unsupported model name is provided.
        FileNotFoundError: If the audio file path does not exist.

    Example:
        >>> processor = SenselabEmbeddingProcessor(model="wavlm")
        >>> embedding = processor.process("path/to/audio.wav")
        >>> embedding.shape
        torch.Size([768])
    """

    SUPPORTED_MODELS = ("wavlm", "ecapa")

    def __init__(
        self,
        model: Literal["wavlm", "ecapa"] = "wavlm",
        sample_rate: int = 16000,
        apply_vad: bool = True,
    ) -> None:
        try:
            import senselab  # noqa: F401
        except ImportError:
            raise ImportError(
                "SenselabEmbeddingProcessor requires senselab. "
                "Install it with: pip install senselab"
            )
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{model}'. "
                f"Choose from: {self.SUPPORTED_MODELS}"
            )
        self.model = model
        self.sample_rate = sample_rate
        self.apply_vad = apply_vad

    def process(self, value: Union[str, Path]) -> torch.Tensor:
        """Extract a deep embedding from a single audio file.

        Args:
            value: Path to the audio file.

        Returns:
            torch.Tensor of shape (D,) where D is the embedding dimension
            of the chosen model (768 for WavLM, 192 for ECAPA-TDNN).

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        import torchaudio
        from senselab.audio.data_structures import Audio
        from senselab.audio.tasks.preprocessing.preprocessing import (
            resample_audios,
        )
        from senselab.audio.tasks.speaker_embeddings.api import (
            extract_speaker_embeddings_from_audios,
        )
        from senselab.audio.tasks.voice_activity_detection.api import (
            detect_voice_activity_in_audios,
        )

        audio_path = Path(value)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, orig_sr = torchaudio.load(audio_path)
        audio = Audio(waveform=waveform, sampling_rate=orig_sr)
        [audio] = resample_audios([audio], resample_rate=self.sample_rate)

        if self.apply_vad:
            segments = detect_voice_activity_in_audios([audio])
            if segments:
                combined = torch.cat([s.waveform for s in segments], dim=-1)
                audio = Audio(waveform=combined, sampling_rate=self.sample_rate)

        if self.model == "wavlm":
            model_spec = {"path": "microsoft/wavlm-base-plus"}
        else:
            model_spec = {"path": "speechbrain/spkrec-ecapa-voxceleb"}

        embeddings = extract_speaker_embeddings_from_audios(
            [audio], model=model_spec
        )
        return embeddings[0].squeeze(0)

    def is_token(self) -> bool:
        return False
