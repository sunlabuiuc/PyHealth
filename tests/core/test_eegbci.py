import unittest

import numpy as np

from pyhealth.tasks.eegbci import (
    label_family_for_run,
    numeric_label_for_task,
    run_type_for_run,
    task_label_for_event,
)


class TestEEGBCIHelpers(unittest.TestCase):
    def test_run_type_for_run(self):
        self.assertEqual(run_type_for_run(3), "motor_execution_left_right")
        self.assertEqual(run_type_for_run(4), "motor_imagery_left_right")
        self.assertEqual(run_type_for_run(5), "motor_execution_fists_feet")
        self.assertEqual(run_type_for_run(6), "motor_imagery_fists_feet")
        self.assertEqual(run_type_for_run(14), "motor_imagery_fists_feet")

    def test_task_label_for_event_is_run_aware(self):
        self.assertEqual(task_label_for_event(3, "T0"), "rest")
        self.assertEqual(task_label_for_event(3, "T1"), "execute_left_fist")
        self.assertEqual(task_label_for_event(3, "T2"), "execute_right_fist")
        self.assertEqual(task_label_for_event(4, "T1"), "imagine_left_fist")
        self.assertEqual(task_label_for_event(4, "T2"), "imagine_right_fist")
        self.assertEqual(task_label_for_event(5, "T1"), "execute_both_fists")
        self.assertEqual(task_label_for_event(5, "T2"), "execute_both_feet")
        self.assertEqual(task_label_for_event(6, "T1"), "imagine_both_fists")
        self.assertEqual(task_label_for_event(6, "T2"), "imagine_both_feet")

    def test_label_family_and_numeric_labels(self):
        self.assertEqual(label_family_for_run(3), "motor_execution")
        self.assertEqual(label_family_for_run(4), "motor_imagery")
        self.assertEqual(numeric_label_for_task("rest"), 0)
        self.assertEqual(numeric_label_for_task("execute_left_fist"), 1)
        self.assertEqual(numeric_label_for_task("imagine_both_feet"), 8)

    def test_invalid_run_and_event_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "Unsupported EEGBCI run"):
            run_type_for_run(2)
        with self.assertRaisesRegex(ValueError, "Unsupported EEGBCI event"):
            task_label_for_event(3, "BAD")

    def test_select_eegbci_channels_compat16(self):
        from pyhealth.tasks.eegbci import EEGBCI_COMPAT_CHANNELS, select_eegbci_channels

        ch_names = list(EEGBCI_COMPAT_CHANNELS) + ["EXTRA"]
        data = np.arange(len(ch_names) * 100, dtype=float).reshape(len(ch_names), 100)
        selected, selected_names = select_eegbci_channels(data, ch_names, "compat16")
        self.assertEqual(selected.shape, (16, 100))
        self.assertEqual(selected_names, list(EEGBCI_COMPAT_CHANNELS))
        np.testing.assert_allclose(selected[0], data[0])

    def test_select_eegbci_channels_all(self):
        from pyhealth.tasks.eegbci import select_eegbci_channels

        data = np.ones((64, 50))
        ch_names = [f"CH{i}" for i in range(64)]
        selected, selected_names = select_eegbci_channels(data, ch_names, "all")
        self.assertEqual(selected.shape, (64, 50))
        self.assertEqual(selected_names, ch_names)

    def test_select_eegbci_channels_missing_channel_raises(self):
        from pyhealth.tasks.eegbci import select_eegbci_channels

        with self.assertRaisesRegex(ValueError, "Missing EEGBCI channels"):
            select_eegbci_channels(np.ones((2, 20)), ["C3", "C4"], "compat16")

    def test_normalize_signal_95th_percentile(self):
        from pyhealth.tasks.eegbci import normalize_signal

        signal = np.array([[0.0, 1.0, 2.0, 100.0], [0.0, -2.0, 2.0, 4.0]])
        normalized = normalize_signal(signal, "95th_percentile")
        self.assertEqual(normalized.shape, signal.shape)
        self.assertLess(np.max(np.abs(normalized[0])), 2.0)

    def test_compute_band_powers_detects_alpha_sinusoid(self):
        from pyhealth.tasks.eegbci import compute_band_powers

        sfreq = 200.0
        times = np.arange(0, 2, 1 / sfreq)
        alpha = np.sin(2 * np.pi * 10 * times)
        data = np.stack([alpha, alpha])
        features = compute_band_powers(data, sfreq)
        self.assertEqual(features["dominant_band"], "alpha")
        self.assertGreater(features["alpha_relative"], 0.5)
        self.assertGreater(features["alpha_beta_ratio"], 1.0)

    def test_compute_band_powers_detects_beta_sinusoid(self):
        from pyhealth.tasks.eegbci import compute_band_powers

        sfreq = 200.0
        times = np.arange(0, 2, 1 / sfreq)
        beta = np.sin(2 * np.pi * 20 * times)
        data = np.stack([beta, beta])
        features = compute_band_powers(data, sfreq)
        self.assertEqual(features["dominant_band"], "beta")
        self.assertGreater(features["beta_relative"], 0.5)

    def test_interpret_band_profile_returns_cautious_metadata(self):
        from pyhealth.tasks.eegbci import interpret_band_profile

        interpretation = interpret_band_profile(
            {
                "dominant_band": "alpha",
                "alpha_relative": 0.65,
                "beta_relative": 0.10,
                "theta_relative": 0.10,
                "gamma_relative": 0.05,
                "alpha_beta_ratio": 6.5,
                "theta_beta_ratio": 1.0,
            }
        )
        self.assertEqual(interpretation["brain_state_hypothesis"], "relaxed_or_idle")
        self.assertIn(interpretation["confidence"], {"low", "medium", "high"})
        self.assertIn("consistent with", interpretation["interpretation"])
