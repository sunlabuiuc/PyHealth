from pathlib import Path
from typing import Any, List, Optional, Union
import torch

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("audio")
class AudioProcessor(FeatureProcessor):
    """Feature processor for loading audio from disk and converting them to tensors.

    Args:
        sample_rate: Desired output sample rate. If None, keeps original sample rate.
            Defaults to 4000.
        duration: Desired duration in seconds. If None, keeps original duration.
            If shorter than audio, truncates. If longer, pads with zeros.
            Defaults to 20.0.
        to_mono: Whether to convert stereo audio to mono. Defaults to True.
        normalize: Whether to normalize audio values to [-1, 1]. Defaults to False.
        mean: Precomputed mean for normalization. Defaults to None.
        std: Precomputed std for normalization. Defaults to None.
        n_mels: Number of mel filterbanks. If provided, converts to mel spectrogram.
            Defaults to None (keeps waveform).
        n_fft: Size of FFT for spectrogram. Defaults to 400.
        hop_length: Length of hop between STFT windows. Defaults to None.

    Raises:
        ValueError: If normalization parameters are inconsistent.
    """

    def __init__(
        self,
        sample_rate: Optional[int] = 4000,  # BMD-HS original sample rate
        duration: Optional[float] = 20.0,  # maximum duration of recordings in BMD-HS
        to_mono: bool = True,
        normalize: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        n_mels: Optional[int] = None,
        n_fft: int = 400,
        hop_length: Optional[int] = None,
    ) -> None:
        try:
            import torchaudio
        except ImportError:
            raise ImportError(
                "AudioProcessor requires torchaudio. "
                "Install it with: pip install torchaudio"
            )

        self.sample_rate = sample_rate
        self.duration = duration
        self.to_mono = to_mono
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        if not self.normalize and (self.mean is not None or self.std is not None):
            raise ValueError(
                "Mean and std are provided but normalize is set to False. "
                "Either provide normalize=True, or remove mean and std."
            )

    def process(self, value: Union[str, Path]) -> Any:
        """Process a single audio path into a transformed tensor.

        Args:
            value: Path to audio file as string or Path object.

        Returns:
            Transformed audio tensor. Shape depends on parameters:
            - Waveform: (channels, samples)
            - Mel spectrogram: (channels, n_mels, time)

        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
        import torchaudio
        import torchaudio.transforms as T

        audio_path = Path(value)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        waveform, orig_sample_rate = torchaudio.load(audio_path)

        # Resample if needed
        if self.sample_rate is not None and orig_sample_rate != self.sample_rate:
            resampler = T.Resample(orig_sample_rate, self.sample_rate)
            waveform = resampler(waveform)
            current_sample_rate = self.sample_rate
        else:
            current_sample_rate = orig_sample_rate

        # Convert to mono if needed
        if self.to_mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Adjust duration if specified
        if self.duration is not None:
            target_length = int(self.duration * current_sample_rate)
            current_length = waveform.shape[1]

            if current_length > target_length:
                # Truncate
                waveform = waveform[:, :target_length]
            elif current_length < target_length:
                # Pad with zeros
                padding = target_length - current_length
                waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Convert to mel spectrogram if specified
        if self.n_mels is not None:
            mel_transform = T.MelSpectrogram(
                sample_rate=current_sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
            waveform = mel_transform(waveform)

        # Normalize if specified
        if self.normalize:
            if self.mean is None:
                self.mean = waveform.mean()
            if self.std is None:
                self.std = waveform.std()
            waveform = (waveform - self.mean) / self.std

        return waveform

    def __repr__(self) -> str:
        return (
            f"AudioProcessor(sample_rate={self.sample_rate}, "
            f"duration={self.duration}, to_mono={self.to_mono}, "
            f"normalize={self.normalize}, mean={self.mean}, std={self.std}, "
            f"n_mels={self.n_mels}, n_fft={self.n_fft}, "
            f"hop_length={self.hop_length})"
        )
