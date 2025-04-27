from typing import Tuple
import logging


import numpy as np
import torch
import mne

from . import register_processor
from .base_processor import FeatureProcessor

logger = logging.getLogger(__name__)

@register_processor("signal")
class SignalProcessor(FeatureProcessor):

    def __init__(self, sampling_rate: int = 200, sample_size: int = 4):
        """Feature processor for loading and processing EEG signals.

        Args:
        sampling_rate: The sampling rate of the EEG signal. Defaults to 200.
        sample_size: The number of seconds per sample. Defaults to 4.
        """
        # Configurable sampling rate
        self.sampling_rate = sampling_rate
        self.sample_size = sample_size
        self.segment_size = sampling_rate * sample_size
        self.size = None
        self.segment_cache = {
            "file_path": None,
            "segments": None,
        }

    def process(self, value: Tuple[float, str]) -> torch.Tensor:
        """Process a single sample of an edf file into a transformed tensor.

        Args:
            value: Path to edf file as string or Path object, and the index of 
            the sample segment.

        Returns:
            Transformed tensor.
        """
        segment_idx, file_path = value

        if self.segment_cache["file_path"] != file_path:

            edf = mne.io.read_raw_edf(file_path, preload=True, verbose="warning")

            edf.resample(self.sampling_rate, npad="auto")

            num_channels = edf.info['nchan']
            if num_channels < 32:
                pad = np.zeros((32 - num_channels, edf.n_times))
                data = np.concatenate((edf.get_data()[:num_channels], pad), axis=0)
            else:
                data = edf.get_data()[:32]

            num_segments = int(data.shape[1] / self.segment_size)           
            segments = np.zeros((num_segments, 32, self.segment_size))

            for i in range(num_segments):
                start_idx = i * self.segment_size
                end_idx = start_idx + self.segment_size
                if end_idx <= data.shape[1]:  
                    segments[i] = data[:, start_idx:end_idx]

            self.segment_cache["file_path"] = file_path
            self.segment_cache["segments"] = segments

        segment = self.segment_cache["segments"][segment_idx]

        if self.size is None:
            self.size = segment.shape[1]

        return torch.tensor(segment, dtype=torch.float32)

    