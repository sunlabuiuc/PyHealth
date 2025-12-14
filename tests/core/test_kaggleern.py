import os
import glob
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd

from pyhealth.datasets import KaggleERNDataset

# If you keep the preprocess config as a dataclass in the module, import it here.
# If you renamed it, update this import accordingly.
try:
    from pyhealth.datasets.kaggleern import KaggleERNPreprocessConfig
except Exception:
    KaggleERNPreprocessConfig = None


def _has_mne() -> bool:
    try:
        import mne  # noqa: F401
        return True
    except Exception:
        return False


class TestKaggleERNDataset(unittest.TestCase):
    """Test cases for KaggleERNDataset (INRIA BCI Challenge / KaggleERN)."""

    def setUp(self):
        """Create a minimal synthetic KaggleERN-like raw dataset folder."""
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

        # Expected raw structure
        train_dir = os.path.join(self.root, "train")
        os.makedirs(train_dir, exist_ok=True)

        # --- Create one synthetic raw CSV ---
        # We generate enough feedback events for stratified split: 20 epochs, 10 per class.
        sfreq = 200
        chunk_size_sec = 3.0
        epoch_samples = int(chunk_size_sec * sfreq)

        n_events = 20
        n_samples = epoch_samples * (n_events + 1)  # ensure last epoch fits
        t = np.arange(n_samples) / sfreq

        # keep channels small for speed; your pipeline should be channel-count agnostic
        n_ch = 8
        ch_names = [f"Ch{i+1:02d}" for i in range(n_ch)]

        df = pd.DataFrame(
            {
                "Time": t,
                "FeedBackEvent": np.zeros(n_samples, dtype=int),
                "EOG": np.zeros(n_samples, dtype=float),
            }
        )
        # synthetic EEG
        for i, ch in enumerate(ch_names):
            df[ch] = np.sin(2 * np.pi * 10 * t + i * 0.1).astype(np.float32)

        # feedback events at 0, 600, 1200, ...
        event_indices = [k * epoch_samples for k in range(n_events)]
        for idx in event_indices:
            df.loc[idx, "FeedBackEvent"] = 1

        raw_csv_path = os.path.join(train_dir, "Data_001.csv")
        df.to_csv(raw_csv_path, index=False)

        # --- Create TrainLabels.csv ---
        # epoch_id format must match your code:
        # prefix = filename.replace("Data_", "").replace(".csv", "") -> "001"
        # epoch_id = f"{prefix}_FB{fb_idx+1:03d}" -> "001_FB001", ...
        labels = []
        for i in range(1, n_events + 1):
            epoch_id = f"001_FB{i:03d}"
            y = 0 if i <= (n_events // 2) else 1  # 10 zeros + 10 ones
            labels.append({"IdFeedBack": epoch_id, "Prediction": y})
        pd.DataFrame(labels).to_csv(os.path.join(self.root, "TrainLabels.csv"), index=False)

        # --- Create ChannelsLocation.csv (minimal, required by verifier) ---
        chloc = pd.DataFrame({"channel": ch_names, "x": 0.0, "y": 0.0, "z": 0.0})
        chloc.to_csv(os.path.join(self.root, "ChannelsLocation.csv"), index=False)

        # Instantiate dataset
        self.dataset = KaggleERNDataset(root=self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dataset_initialization(self):
        """Test that KaggleERNDataset initializes with minimal raw structure."""
        self.assertIsInstance(self.dataset, KaggleERNDataset)
        self.assertTrue(os.path.isdir(os.path.join(self.root, "train")))
        self.assertTrue(os.path.isfile(os.path.join(self.root, "TrainLabels.csv")))
        self.assertTrue(os.path.isfile(os.path.join(self.root, "ChannelsLocation.csv")))

    def test_missing_required_files(self):
        """Test that missing required files raise an error."""
        with tempfile.TemporaryDirectory() as tmp2:
            os.makedirs(os.path.join(tmp2, "train"), exist_ok=True)
            # No TrainLabels.csv / ChannelsLocation.csv
            with self.assertRaises((FileNotFoundError, ValueError, RuntimeError)):
                _ = KaggleERNDataset(root=tmp2)

    @unittest.skipUnless(_has_mne(), "mne is not installed; skipping preprocessing smoke test.")
    def test_preprocess_epochs_smoke(self):
        """Smoke test: preprocessing generates train/val/test pickles in the expected format."""
        if KaggleERNPreprocessConfig is None:
            self.skipTest("KaggleERNPreprocessConfig not importable; update test import to match your module.")

        out_root = os.path.join(self.root, "processed_kaggleern_test")

        cfg = KaggleERNPreprocessConfig(
            root=self.root,
            output_root=out_root,
            # keep defaults matching your EEGPT-like pipeline:
            chunk_size_sec=3.0,
            line_noise_hz=50.0,
            random_seed=42,
            min_epochs_per_file=0,  # do not flag wired files in this minimal test
            pipeline="eegpt",       # if you later remove "pipeline", keep it as default in your dataclass
        )

        ret = self.dataset.preprocess_epochs(cfg)
        self.assertIn("wired_files", ret)

        # Verify output folders exist
        train_dir = os.path.join(out_root, "train")
        val_dir = os.path.join(out_root, "val")
        test_dir = os.path.join(out_root, "test")
        self.assertTrue(os.path.isdir(train_dir))
        self.assertTrue(os.path.isdir(val_dir))
        self.assertTrue(os.path.isdir(test_dir))

        # Verify some pickles exist
        train_pkl = sorted(glob.glob(os.path.join(train_dir, "*.pickle")))
        val_pkl = sorted(glob.glob(os.path.join(val_dir, "*.pickle")))
        test_pkl = sorted(glob.glob(os.path.join(test_dir, "*.pickle")))
        self.assertGreater(len(train_pkl), 0)
        self.assertGreater(len(val_pkl), 0)
        self.assertGreater(len(test_pkl), 0)

        # Verify pickle schema
        one = train_pkl[0]
        with open(one, "rb") as f:
            obj = pickle.load(f)

        self.assertIn("signal", obj)
        self.assertIn("label", obj)
        self.assertIn("epoch_id", obj)

        signal = obj["signal"]
        label = obj["label"]
        epoch_id = obj["epoch_id"]

        self.assertIsInstance(signal, np.ndarray)
        self.assertEqual(signal.ndim, 2)  # (C, T)
        self.assertTrue(isinstance(label, (int, np.integer)))
        self.assertIn(int(label), (0, 1))
        self.assertIsInstance(epoch_id, str)
        self.assertTrue(epoch_id.startswith("001_FB"))

    @unittest.skipUnless(_has_mne(), "mne is not installed; skipping preprocessing determinism test.")
    def test_preprocess_deterministic_split(self):
        """Optional: ensure split sizes are deterministic under the fixed seed."""
        if KaggleERNPreprocessConfig is None:
            self.skipTest("KaggleERNPreprocessConfig not importable; update test import to match your module.")

        out_root = os.path.join(self.root, "processed_kaggleern_test2")

        cfg = KaggleERNPreprocessConfig(
            root=self.root,
            output_root=out_root,
            chunk_size_sec=3.0,
            random_seed=42,
            min_epochs_per_file=0,
            pipeline="eegpt",
        )
        _ = self.dataset.preprocess_epochs(cfg)

        n_train = len(glob.glob(os.path.join(out_root, "train", "*.pickle")))
        n_val = len(glob.glob(os.path.join(out_root, "val", "*.pickle")))
        n_test = len(glob.glob(os.path.join(out_root, "test", "*.pickle")))

        # We created 20 labeled epochs; your code does 80/10/10 split -> 16/2/2
        self.assertEqual(n_train, 16)
        self.assertEqual(n_val, 2)
        self.assertEqual(n_test, 2)


if __name__ == "__main__":
    unittest.main()
