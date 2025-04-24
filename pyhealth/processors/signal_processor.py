from typing import Any, List, Tuple
import logging


import numpy as np
import torch
import mne

from . import register_processor
from .base_processor import FeatureProcessor

logger = logging.getLogger(__name__)

@register_processor("signal")
class SignalProcessor(FeatureProcessor):

    def __init__(self, sampling_rate: 200, sample_size: int = 4):
        # Configurable sampling rate
        self.sampling_rate = sampling_rate
        self.sample_size = sample_size
        self.size = None

    def process(self, value: Tuple[float, float, str]) -> torch.Tensor:
        start_time, stop_time, file_path = value

        logger.info(f"Processing file: {file_path}")

        #1. open edf file
        edf = mne.io.read_raw_edf(file_path, preload=True)

        #2. Resample the signal to sampling rate
        edf.resample(self.sampling_rate, npad="auto")

        #3. Get the first 32 channels, if there are less than 32 pad the signal with zeros
        #   to 32 channels
        num_channels = edf.info['nchan']
        if num_channels < 32:
            pad = np.zeros((32 - num_channels, edf.n_times))
            data = np.concatenate((edf.get_data()[:num_channels], pad), axis=0)
        else:
            data = edf.get_data()[:32]

        logger.info(f"Signal shape: {data.shape}")

        #4. chunk the signal into an array of 4s segments 
        
        segment_size = self.sampling_rate * self.sample_size
        logger.info(f"Segment size: {segment_size}")

        num_segments = int(data.shape[1] / segment_size)
        logger.info(f"Number of segments: {num_segments}")
        
        segments = np.zeros((num_segments, 32, segment_size))
        for i in range(num_segments):
            segments[i] = data[:, i * segment_size:(i + 1) * segment_size]
        
        #5. get all of the segments from a start time and stop time

        start_segment = int(start_time / self.sample_size)
        stop_segment = int(stop_time / self.sample_size)
        segments = segments[start_segment:stop_segment]

        logger.info(f"Start segment: {start_segment}")
        logger.info(f"Stop segment: {stop_segment}")
        logger.info(f"Segments shape: {segments.shape}")

        #6. convert the segments to a tensor and return it
        if self.size is None:
            self.size = segments.shape[1]

        return torch.tensor(segments, dtype=torch.float32)

    