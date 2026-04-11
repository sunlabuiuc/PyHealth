import numpy as np
from scipy.signal import resample, butter, filtfilt

class BiosignalFeaturizer:

    def __init__(self, target_length=256, lowcut=0.5, highcut=40.0, sampling_rate=100):
        self.target_length = target_length
        self.lowcut = lowcut
        self.highcut = highcut
        self.sampling_rate = sampling_rate

    def _bandpass_filter(self, signal):
        """Apply a bandpass filter to the signal."""
        nyquist = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(1, [low, high], btype="band")
        return filtfilt(b, a, signal)

    def encode(self, signal):
        """
        Encodes the biosignal into a fixed-length feature representation.

        Args:
            signal (np.array): 1D array of raw biosignal data.

        Returns:
            np.array: Processed and fixed-length representation of the biosignal.
        """
        # Apply bandpass filtering
        filtered_signal = self._bandpass_filter(signal)

        # Resample to target length for uniformity
        resampled_signal = resample(filtered_signal, self.target_length)

        # Normalize the signal
        normalized_signal = (resampled_signal - np.mean(resampled_signal)) / np.std(resampled_signal)

        return normalized_signal

if __name__ == "__main__":
    # Example biosignal (ECG or other) with 1000 sample points
    sample_signal = np.sin(np.linspace(0, 10, 1000)) + 0.1 * np.random.randn(1000)
    featurizer = BiosignalFeaturizer()
    print(featurizer)
    print(type(featurizer))
    print(featurizer.encode(sample_signal))
