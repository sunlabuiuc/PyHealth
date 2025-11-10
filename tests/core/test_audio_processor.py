import unittest
import tempfile
import os
from pathlib import Path
import torch
import numpy as np

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

from pyhealth.processors.audio_processor import AudioProcessor


@unittest.skipUnless(TORCHAUDIO_AVAILABLE, "torchaudio not available")
class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_rate = 4000

        # Create a mono audio file (1 second)
        mono_waveform = torch.randn(1, self.sample_rate)  # 1 channel, 1 second
        self.mono_path = os.path.join(self.temp_dir, "test_mono.wav")
        torchaudio.save(self.mono_path, mono_waveform, self.sample_rate)

        # Create a stereo audio file (1 second)
        stereo_waveform = torch.randn(2, self.sample_rate)  # 2 channels, 1 second
        self.stereo_path = os.path.join(self.temp_dir, "test_stereo.wav")
        torchaudio.save(self.stereo_path, stereo_waveform, self.sample_rate)

        # Create a short audio file (0.5 seconds)
        short_waveform = torch.randn(1, self.sample_rate // 2)
        self.short_path = os.path.join(self.temp_dir, "test_short.wav")
        torchaudio.save(self.short_path, short_waveform, self.sample_rate)

        # Create audio with different sample rate
        diff_sr = 8000
        diff_waveform = torch.randn(1, diff_sr)
        self.diff_sr_path = os.path.join(self.temp_dir, "test_diff_sr.wav")
        torchaudio.save(self.diff_sr_path, diff_waveform, diff_sr)

    def tearDown(self):
        # Clean up temporary files
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_init_default(self):
        processor = AudioProcessor()
        self.assertEqual(processor.sample_rate, 4000)
        self.assertEqual(processor.duration, 20.0)
        self.assertTrue(processor.to_mono)
        self.assertFalse(processor.normalize)
        self.assertIsNone(processor.mean)
        self.assertIsNone(processor.std)
        self.assertIsNone(processor.n_mels)
        self.assertEqual(processor.n_fft, 400)
        self.assertIsNone(processor.hop_length)

    def test_init_custom(self):
        processor = AudioProcessor(
            sample_rate=2000,
            duration=2.0,
            to_mono=False,
            normalize=True,
            mean=0.0,
            std=1.0,
            n_mels=80,
            n_fft=512,
            hop_length=160,
        )
        self.assertEqual(processor.sample_rate, 2000)
        self.assertEqual(processor.duration, 2.0)
        self.assertFalse(processor.to_mono)
        self.assertTrue(processor.normalize)
        self.assertEqual(processor.mean, 0.0)
        self.assertEqual(processor.std, 1.0)
        self.assertEqual(processor.n_mels, 80)
        self.assertEqual(processor.n_fft, 512)
        self.assertEqual(processor.hop_length, 160)

    def test_init_normalize_without_mean_std(self):
        processor = AudioProcessor(normalize=True)
        # Should not raise error, and mean/std should be None initially
        self.assertIsNone(processor.mean)
        self.assertIsNone(processor.std)

    def test_init_mean_std_without_normalize(self):
        with self.assertRaises(ValueError):
            AudioProcessor(mean=0.0, std=1.0)

    def test_process_mono_default(self):
        processor = AudioProcessor(duration=None)
        result = processor.process(self.mono_path)

        # Should be tensor
        self.assertIsInstance(result, torch.Tensor)
        # Should be mono (1 channel)
        self.assertEqual(result.shape[0], 1)
        # Should be 1 second at 16000 Hz
        self.assertEqual(result.shape[1], 4000)

    def test_process_stereo_to_mono(self):
        processor = AudioProcessor(duration=None, to_mono=True)
        result = processor.process(self.stereo_path)

        # Should convert to mono
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 4000)

    def test_process_stereo_keep_stereo(self):
        processor = AudioProcessor(duration=None, to_mono=False)
        result = processor.process(self.stereo_path)

        # Should keep stereo
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 4000)

    def test_process_resample(self):
        processor = AudioProcessor(sample_rate=8000, duration=None)
        result = processor.process(self.mono_path)

        # Should be resampled
        self.assertEqual(result.shape[1], 8000)  # 0.5 seconds at 8000 Hz

    def test_process_different_input_sr(self):
        processor = AudioProcessor(sample_rate=16000, duration=None)
        result = processor.process(self.diff_sr_path)

        # Should be resampled to 16000 Hz
        self.assertEqual(result.shape[1], 16000)

    def test_process_duration_truncate(self):
        processor = AudioProcessor(duration=0.5)  # 0.5 seconds
        result = processor.process(self.mono_path)

        # Should be truncated
        self.assertEqual(result.shape[1], 2000)  # 0.5 * 4000

    def test_process_duration_pad(self):
        processor = AudioProcessor(duration=2.0)  # 2 seconds
        result = processor.process(self.short_path)

        # Should be padded
        self.assertEqual(result.shape[1], 8000)  # 2.0 * 4000

    def test_process_mel_spectrogram(self):
        processor = AudioProcessor(n_mels=80, hop_length=160)
        result = processor.process(self.mono_path)

        # Should be mel spectrogram
        self.assertEqual(result.shape[0], 1)  # 1 channel
        self.assertEqual(result.shape[1], 80)  # n_mels
        # Time dimension depends on hop_length

    def test_process_normalize(self):
        processor = AudioProcessor(normalize=True, mean=0.0, std=1.0)
        result = processor.process(self.mono_path)

        # Should be normalized
        self.assertIsInstance(result, torch.Tensor)
        # Values should be normalized (roughly)
        self.assertTrue(torch.all(result >= -10))  # Allow some tolerance
        self.assertTrue(torch.all(result <= 10))

    def test_process_normalize_compute_mean_std(self):
        processor = AudioProcessor(duration=None, normalize=True)
        result = processor.process(self.mono_path)

        # Should be normalized, and mean/std should now be computed
        self.assertIsNotNone(processor.mean)
        self.assertIsNotNone(processor.std)
        self.assertIsInstance(result, torch.Tensor)
        # Values should be normalized (roughly)
        self.assertTrue(torch.all(result >= -10))  # Allow some tolerance
        self.assertTrue(torch.all(result <= 10))

    def test_process_invalid_path(self):
        processor = AudioProcessor()
        with self.assertRaises(FileNotFoundError):
            processor.process("/nonexistent/path/audio.wav")

    def test_process_path_object(self):
        processor = AudioProcessor(duration=None)
        result = processor.process(Path(self.mono_path))

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 4000)

    def test_repr(self):
        processor = AudioProcessor()
        repr_str = repr(processor)
        self.assertIn("AudioProcessor", repr_str)
        self.assertIn("sample_rate=4000", repr_str)
        self.assertIn("to_mono=True", repr_str)
        self.assertIn("normalize=False", repr_str)


if __name__ == "__main__":
    unittest.main()
