import unittest
from unittest.mock import patch, MagicMock
import torch

from pyhealth.tasks.tusz_task import TUSZTask



# Dummy structures for testing

class DummyEvent:
    def __init__(self):
        self.signal_file = "fake_path.edf"


class DummyPatient:
    def __init__(self):
        self.patient_id = "p0"

    def get_events(self):
        return [DummyEvent()]



# Test Class

class TestTUSZTask(unittest.TestCase):

    def setUp(self):
        self.task = TUSZTask()
        self.patient = DummyPatient()

        # ---- Mock helper methods (core pipeline) ----
        self.task.helper.skip_file = MagicMock(return_value=False)
        self.task.helper.process_label = MagicMock(return_value=[0, 1])
        self.task.helper.is_seizure_patient = MagicMock(return_value=True)

        self.task.helper.resample = MagicMock(
            return_value=[torch.randn(16, 1000)]
        )

        self.task.helper.transform_labels_with_resampled_signals = MagicMock(
            return_value=[0, 1]
        )

        self.task.helper.segment_signals = MagicMock(
            return_value=(
                [torch.randn(16, 1000)],  # signals
                [0],                      # labels
                ["non_seiz"]              # label names
            )
        )

        self.task.helper.convert_labels = MagicMock(
            return_value=([0], [0], [0])
        )

        self.task.helper.create_bipolar_signals = MagicMock(
            return_value=torch.randn(16, 1000)
        )


    # 1. EDF Reading

    @patch("pyhealth.tasks.tusz_task.TUSZSignalHeader")
    @patch("pyhealth.tasks.tusz_task.highlevel.read_edf")
    def test_read_edf_called(self, mock_read_edf, mock_header):
        mock_read_edf.return_value = ([[0.0]*1000], [{}], {})
        mock_header.return_value = MagicMock()

        self.task(self.patient)

        mock_read_edf.assert_called_once()

    # 2. Skip File Logic

    @patch("pyhealth.tasks.tusz_task.TUSZSignalHeader")
    @patch("pyhealth.tasks.tusz_task.highlevel.read_edf")
    def test_skip_file(self, mock_read_edf, mock_header):
        mock_read_edf.return_value = ([[0.0]*1000], [{}], {})
        mock_header.return_value = MagicMock()

        self.task.helper.skip_file.return_value = True

        samples = self.task(self.patient)

        self.assertEqual(samples, [])

    # 3. Process Label Called

    @patch("pyhealth.tasks.tusz_task.TUSZSignalHeader")
    @patch("pyhealth.tasks.tusz_task.highlevel.read_edf")
    def test_process_label_called(self, mock_read_edf, mock_header):
        mock_read_edf.return_value = ([[0.0]*1000], [{}], {})
        mock_header.return_value = MagicMock()

        self.task(self.patient)

        self.task.helper.process_label.assert_called_once()

    # 4. Resample Called
    
    @patch("pyhealth.tasks.tusz_task.TUSZSignalHeader")
    @patch("pyhealth.tasks.tusz_task.highlevel.read_edf")
    def test_resample_called(self, mock_read_edf, mock_header):
        mock_read_edf.return_value = ([[0.0]*1000], [{}], {})
        mock_header.return_value = MagicMock()

        self.task(self.patient)

        self.task.helper.resample.assert_called_once()

    # 5. Label Transformation
    
    @patch("pyhealth.tasks.tusz_task.TUSZSignalHeader")
    @patch("pyhealth.tasks.tusz_task.highlevel.read_edf")
    def test_transform_labels(self, mock_read_edf, mock_header):
        mock_read_edf.return_value = ([[0.0]*1000], [{}], {})
        mock_header.return_value = MagicMock()

        self.task(self.patient)

        self.task.helper.transform_labels_with_resampled_signals.assert_called_once()

    # 6. Segmentation

    @patch("pyhealth.tasks.tusz_task.TUSZSignalHeader")
    @patch("pyhealth.tasks.tusz_task.highlevel.read_edf")
    def test_segmentation(self, mock_read_edf, mock_header):
        mock_read_edf.return_value = ([[0.0]*1000], [{}], {})
        mock_header.return_value = MagicMock()

        self.task(self.patient)

        self.task.helper.segment_signals.assert_called_once()

    # 7. Final Output Structure

    @patch("pyhealth.tasks.tusz_task.TUSZSignalHeader")
    @patch("pyhealth.tasks.tusz_task.highlevel.read_edf")
    def test_sample_output(self, mock_read_edf, mock_header):
        mock_read_edf.return_value = ([[0.0]*1000], [{}], {})
        mock_header.return_value = MagicMock()

        samples = self.task(self.patient)

        self.assertTrue(len(samples) > 0)

        sample = samples[0]

        self.assertIn("patient_id", sample)
        self.assertIn("signal", sample)
        self.assertIn("label", sample)
        self.assertIn("label_bitgt_1", sample)
        self.assertIn("label_bitgt_2", sample)
        self.assertIn("label_name", sample)

        self.assertIsInstance(sample["signal"], torch.Tensor)


if __name__ == "__main__":
    unittest.main()