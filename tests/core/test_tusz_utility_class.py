import unittest
from unittest.mock import patch, MagicMock
import torch

from pyhealth.tasks.tusz_utility_class import TUSZHelper, TUSZSignalHeader


class TestTUSZUtilityClass(unittest.TestCase):

    def setUp(self):
        self.helper = TUSZHelper(
            sample_rate=200,
            feature_sample_rate=50,
            label_type="csv",
            eeg_type="bipolar",
            min_binary_slicelength=1,
            min_binary_edge_seiz=1,
        )

       
        self.fake_headers = [
            {"label": label, "sample_frequency": 200}
            for label in [
                'EEG FP1','EEG FP2','EEG F3','EEG F4','EEG F7','EEG F8',
                'EEG C3','EEG C4','EEG CZ','EEG T3','EEG T4',
                'EEG P3','EEG P4','EEG O1','EEG O2','EEG T5','EEG T6','EEG PZ','EEG FZ'
            ]
        ]

        self.signal_headers = TUSZSignalHeader(self.fake_headers)

        self.signals = [torch.randn(1000).numpy() for _ in range(19)]



    @patch("pandas.read_csv")
    def test_full_pipeline(self, mock_read_csv):

        # fake label file
        mock_read_csv.return_value = MagicMock(
            iterrows=lambda: iter([
                (0, {"start_time": 0, "stop_time": 2, "label": "bckg"}),
                (1, {"start_time": 2, "stop_time": 4, "label": "gnsz"}),
            ])
        )


        # skip_file must be False

        self.assertFalse(
            self.helper.skip_file("file", self.signal_headers)
        )


        # pipeline execution

        y = self.helper.process_label("fake_file")
        self.assertTrue(len(y) > 0)

        resampled = self.helper.resample(
            "file", self.signals, self.signal_headers
        )

        self.assertIsInstance(resampled, list)

        y2 = self.helper.transform_labels_with_resampled_signals(
            resampled, y
        )

        self.assertTrue(len(y2) > 0)

        raws, labels, names = self.helper.segment_signals(
            y2, resampled, is_seiz_patient=True
        )

        self.assertIsInstance(raws, list)
        self.assertIsInstance(labels, list)
        self.assertIsInstance(names, list)

        # convert + bipolar test
        
        if len(raws) > 0:
            y1, y2b, y3 = self.helper.convert_labels(labels)

            self.assertEqual(len(raws), len(y1))

            bipolar = self.helper.create_bipolar_signals(
                raws[0]
            )

            self.assertEqual(bipolar.shape[0], 20)


if __name__ == "__main__":
    unittest.main()