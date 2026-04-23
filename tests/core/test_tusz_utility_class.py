"""
Unit tests for the TUSZSignalHeader, and TUSZHelper class.

Author:
    Fernando Kenji Sakabe (fks@illinois.edu), 
    Jesica Hirsch (jesicah2@illinois.edu), 
    Jung-Jung Hsieh (jhsieh8@illinois.edu)
"""
import unittest
import numpy as np
import torch
import random

from pyhealth.tasks.tusz_utility_class import TUSZHelper, TUSZSignalHeader


class TestTUSZUtilityClass(unittest.TestCase):

    def setUp(self):
        self.helper = TUSZHelper(
            sample_rate            = 200,
            feature_sample_rate    = 50,
            label_type             = "csv",
            eeg_type               = "bipolar",
            min_binary_slicelength = 1,
            min_binary_edge_seiz   = 1,
        )

        self.valid_labels = [
            'EEG FP1','EEG FP2','EEG F3','EEG F4','EEG F7','EEG F8',
            'EEG C3','EEG C4','EEG CZ','EEG T3','EEG T4',
            'EEG P3','EEG P4','EEG O1','EEG O2','EEG T5','EEG T6','EEG PZ','EEG FZ'
        ]
        self.valid_headers = [
            {"label": label, "sample_frequency": 200}
            for label in self.valid_labels
        ]
        suffixes = ['le', 'ar', 'ar-a']
        self.valid_headers_2 = [
            {"label": f"{label}-{random.choice(suffixes)}", "sample_frequency": 200}
            for label in self.valid_labels
        ]
        self.invalid_headers = [
            {"label": f"EEG {i}", "sample_frequency": 200}
            for i in range(19)
        ]
        self.invalid_headers_2 = [
            {"label": "EEG FP1", "sample_frequency": 300}
        ]

        self.good_header = TUSZSignalHeader(self.valid_headers)
        self.bad_header = TUSZSignalHeader(self.invalid_headers)

        self.signals = [np.random.randn(1000) for _ in range(19)]
        
        # paths to csvs
        self.normal_path = "test-resources/core/tusz/train/aaaaaaac/s001_t001/01_tcp_ar/aaaaaaac_s001_t001.edf"
        self.normal_csv = ".".join(self.normal_path.split(".")[:-1])
        self.seizure_path = "test-resources/core/tusz/eval/aaaaaaab/s002_t002/02_tcp_le/aaaaaaab_s002_t002.edf"
        self.seizure_csv = ".".join(self.seizure_path.split(".")[:-1])

        # resampled
        self.resampled = self.helper.resample("file", self.signals, self.good_header)


    def test_skip_file(self):
        # skip_file TRUE case

        # invalid label
        self.assertTrue(
            self.helper.skip_file("file", self.bad_header)
        )

        # invalid sample rate
        bad_header = TUSZSignalHeader(self.invalid_headers_2)
        self.assertTrue(
            self.helper.skip_file("file", bad_header)
        )

        # skip_file FALSE case

        self.assertFalse(
            self.helper.skip_file("file", self.good_header)
        )

        good_header = TUSZSignalHeader(self.valid_headers_2)
        self.assertFalse(
            self.helper.skip_file("file", good_header)
        )
    

    def test_is_seizure_patient(self):
        self.assertFalse(
            self.helper.is_seizure_patient(self.normal_path)
        )

        self.assertTrue(
            self.helper.is_seizure_patient(self.seizure_path)
        )


    def test_process_label(self):
        y = self.helper.process_label(self.normal_csv)
        self.assertEqual(y[0], '0')
        self.assertEqual(y[-1], '0')

        y = self.helper.process_label(self.seizure_csv)
        self.assertNotEqual(y[0], '0')
        self.assertNotEqual(y[-1], '0')


    def test_resample(self):
        self.assertEqual(len(self.resampled), len(self.signals))

        resampled = self.helper.resample("file", self.signals, self.bad_header)
        self.assertEqual(len(resampled), 0)


    def test_transform_labels(self):
        y = self.helper.process_label(self.normal_csv)
        y2 = self.helper.transform_labels_with_resampled_signals(
            self.resampled, y
        )
        self.assertEqual(y2[0], '0')
        self.assertEqual(y2[-1], '0')

        y = self.helper.process_label(self.seizure_csv)
        y2 = self.helper.transform_labels_with_resampled_signals(
            self.resampled, y
        )
        self.assertEqual(y2[0], '2')
        self.assertEqual(y2[-1], '2')


    def test_segment(self):
        # case 1: 0_patT
        y = self.helper.process_label(self.normal_csv)
        y2 = self.helper.transform_labels_with_resampled_signals(
            self.resampled, y
        )
        raws, labels, names = self.helper.segment_signals(
            y2, self.resampled, is_seiz_patient=True
        )

        self.assertIsInstance(raws, list)
        self.assertIsInstance(labels, list)
        self.assertIsInstance(names, list)

        self.assertEqual(labels[0][0], '0')
        self.assertEqual(labels[0][-1], '0')
        self.assertEqual(names[0], '0_patT')
        self.assertEqual(names[-1], '0_patT')

        self.assertEqual(len(raws), len(labels))
        self.assertEqual(len(raws), len(names))


        # case 2: 0_patF
        raws, labels, names = self.helper.segment_signals(
            y2, self.resampled, is_seiz_patient=False
        )
        self.assertEqual(labels[0][0], '0')
        self.assertEqual(labels[0][-1], '0')
        self.assertEqual(names[0], '0_patF')
        self.assertEqual(names[-1], '0_patF')

        self.assertEqual(len(raws), len(labels))
        self.assertEqual(len(raws), len(names))


        # case 3: beg + middle
        len_y = len(y)
        y = '0' + '4'*(len_y-1)
        y2 = self.helper.transform_labels_with_resampled_signals(
            self.resampled, y
        )
        raws, labels, names = self.helper.segment_signals(
            y2, self.resampled, is_seiz_patient=True
        )

        self.assertEqual(labels[0][0], '0')
        self.assertEqual(labels[0][-1], '2')
        self.assertIn('beg', names[0])
        self.assertIn('middle', names[-1])

        self.assertEqual(len(raws), len(labels))
        self.assertEqual(len(raws), len(names))


        # case 4: end
        y = '4' + '0'*(len_y-1)
        y2 = self.helper.transform_labels_with_resampled_signals(
            self.resampled, y
        )
        raws, labels, names = self.helper.segment_signals(
            y2, self.resampled, is_seiz_patient=True
        )

        self.assertEqual(labels[0][0], '2')
        self.assertEqual(labels[0][-1], '0')
        self.assertIn('end', names[0])
        self.assertEqual(names[-1], '0_patT')

        self.assertEqual(len(raws), len(labels))
        self.assertEqual(len(raws), len(names))

        
        # case 5: whole
        y = '0' + '4' + '0'*(len_y-2)
        y2 = self.helper.transform_labels_with_resampled_signals(
            self.resampled, y
        )
        raws, labels, names = self.helper.segment_signals(
            y2, self.resampled, is_seiz_patient=True
        )

        self.assertEqual(labels[0][0], '0')
        self.assertEqual(labels[0][1], '2')
        self.assertEqual(labels[0][-1], '0')
        self.assertIn('whole', names[0])
        self.assertEqual(names[-1], '0_patT')

        self.assertEqual(len(raws), len(labels))
        self.assertEqual(len(raws), len(names))


    def test_convert_labels(self):
        # case 1: normal
        labels = ['0', '0', '0']
        y1, y2, y3 = self.helper.convert_labels(labels)

        self.assertIsInstance(y1, list)
        self.assertIsInstance(y2, list)
        self.assertIsInstance(y3, list)

        self.assertTrue(torch.is_tensor(y1[0]))
        self.assertEqual(len(y1), len(labels))

        self.assertEqual(y1[0], 0)
        self.assertEqual(y2[0], 0)
        self.assertEqual(y3[0], 0)


        # case 2: seizure 1
        labels = ['2', '2', '2']
        y1, y2, y3 = self.helper.convert_labels(labels)

        self.assertEqual(len(y1), len(labels))
        self.assertListEqual(y1, [2, 2, 2])
        self.assertEqual(y2[0], self.helper.binary_target1[int(labels[0])])
        self.assertEqual(y3[0], self.helper.binary_target2[int(labels[0])])


        # case 3: seizure 2
        labels = ['8']
        y1, y2, y3 = self.helper.convert_labels(labels)

        self.assertEqual(len(y1), len(labels))
        self.assertListEqual(y1, [8])
        self.assertEqual(y2[0], self.helper.binary_target1[int(labels[0])])
        self.assertEqual(y3[0], self.helper.binary_target2[int(labels[0])])


        # case 4: empty
        labels = []
        y1, y2, y3 = self.helper.convert_labels(labels)

        self.assertEqual(len(y1), len(labels))
        self.assertListEqual(y1, [])
        self.assertListEqual(y2, [])
        self.assertListEqual(y3, [])


    def test_create_bipolar_signals(self):
        # case 1: bipolar
        bipolar = self.helper.create_bipolar_signals(
            torch.tensor(np.array(self.signals), dtype=torch.float32)
        )

        self.assertEqual(bipolar.shape[0], 20)


        # case 2: uni_bipolar
        self.helper.eeg_type = "uni_bipolar"
        uni_bipolar = self.helper.create_bipolar_signals(
            torch.tensor(np.array(self.signals), dtype=torch.float32)
        )

        self.assertEqual(uni_bipolar.shape[0], 20+len(self.signals))


if __name__ == "__main__":
    unittest.main()