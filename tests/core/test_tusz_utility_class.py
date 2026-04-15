import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

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

        # Correct labels (IMPORTANT)
        self.valid_labels = [
            'EEG FP1','EEG FP2','EEG F3','EEG F4','EEG F7','EEG F8',
            'EEG C3','EEG C4','EEG CZ','EEG T3','EEG T4',
            'EEG P3','EEG P4','EEG O1','EEG O2','EEG T5','EEG T6','EEG PZ','EEG FZ'
        ]

        self.valid_headers = [
            {"label": label, "sample_frequency": 200}
            for label in self.valid_labels
        ]

        self.invalid_headers = [
            {"label": f"EEG {i}", "sample_frequency": 200}
            for i in range(19)
        ]

        self.signals = [np.random.randn(1000) for _ in range(19)]

    # (logic-focused)

    @patch("pandas.read_csv")
    def test_full_logic(self, mock_read_csv):


        # skip_file TRUE case

        bad_header = TUSZSignalHeader(self.invalid_headers)
        self.assertTrue(
            self.helper.skip_file("file", bad_header)
        )


        # skip_file FALSE case

        good_header = TUSZSignalHeader(self.valid_headers)
        self.assertFalse(
            self.helper.skip_file("file", good_header)
        )


        # mock labels

        mock_read_csv.return_value = MagicMock(
            iterrows=lambda: iter([
                (0, {"start_time": 0, "stop_time": 2, "label": "bckg"}),
                (1, {"start_time": 2, "stop_time": 4, "label": "gnsz"}),
            ])
        )


        # process_label

        y = self.helper.process_label("fake_file")
        self.assertTrue(len(y) > 0)


        # resample

        resampled = self.helper.resample("file", self.signals, good_header)
        self.assertEqual(len(resampled), 19)


        # transform labels

        y2 = self.helper.transform_labels_with_resampled_signals(
            resampled, y
        )
        self.assertTrue(len(y2) > 0)


        # segmentation

        raws, labels, names = self.helper.segment_signals(
            y2, resampled, is_seiz_patient=True
        )

        self.assertIsInstance(raws, list)
        self.assertIsInstance(labels, list)
        self.assertIsInstance(names, list)


        # convert labels

        if len(raws) > 0:
            y1, y2b, y3 = self.helper.convert_labels(labels)

            self.assertEqual(len(y1), len(labels))
            self.assertTrue(torch.is_tensor(y1[0]))


            # bipolar signals

            bipolar = self.helper.create_bipolar_signals(
                torch.tensor(raws[0], dtype=torch.float32)
            )

            self.assertEqual(bipolar.shape[0], 20)


if __name__ == "__main__":
    unittest.main()