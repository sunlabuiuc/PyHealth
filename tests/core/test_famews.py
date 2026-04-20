"""
Unit tests for the HiRIDDataset and FAMEWSFairnessAudit task.

Author:
    John Doll
"""

import unittest
from datetime import datetime
from types import SimpleNamespace

import polars as pl

from pyhealth.tasks import FAMEWSFairnessAudit


class _MockPatient:
	def __init__(
		self,
		patient_id: str,
		general_events,
		stage_df: pl.DataFrame,
	):
		self.patient_id = patient_id
		self._general_events = general_events
		self._stage_df = stage_df

	def get_events(self, event_type: str, return_df: bool = False):
		if event_type == "general_table":
			return self._general_events
		if return_df:
			return self._stage_df
		return []


class TestFAMEWSFairnessAudit(unittest.TestCase):
	def test_invalid_stage_table_raises(self):
		with self.assertRaises(ValueError):
			FAMEWSFairnessAudit(stage_table="raw_stage")

	def test_task_generates_sample_with_datetime_axis(self):
		task = FAMEWSFairnessAudit(stage_table="imputed_stage")
		stage_df = pl.DataFrame(
			{
				"imputed_stage/reldatetime": ["0", "5", "10"],
				"imputed_stage/heart_rate": [80.0, 82.0, 84.0],
				"imputed_stage/spo2": [96.0, 97.0, 98.0],
			}
		)
		general_event = SimpleNamespace(
			sex="F",
			age=73,
			discharge_status="alive",
		)
		patient = _MockPatient("p1", [general_event], stage_df)

		samples = task(patient)

		self.assertEqual(len(samples), 1)
		sample = samples[0]
		self.assertEqual(sample["patient_id"], "p1")
		self.assertEqual(sample["age_group"], "65-75")
		self.assertListEqual(sample["feature_columns"], ["heart_rate", "spo2"])

		timestamps, values = sample["signals"]
		self.assertEqual(len(timestamps), 3)
		self.assertTrue(all(isinstance(ts, datetime) for ts in timestamps))
		self.assertEqual(values.shape, (3, 2))

	def test_returns_empty_when_general_table_missing(self):
		task = FAMEWSFairnessAudit(stage_table="imputed_stage")
		stage_df = pl.DataFrame(
			{
				"imputed_stage/reldatetime": [0, 1],
				"imputed_stage/heart_rate": [70.0, 75.0],
			}
		)
		patient = _MockPatient("p2", [], stage_df)

		self.assertEqual(task(patient), [])

	def test_returns_empty_when_no_feature_columns_present(self):
		task = FAMEWSFairnessAudit(stage_table="imputed_stage")
		stage_df = pl.DataFrame(
			{
				"imputed_stage/reldatetime": [0, 1, 2],
				"imputed_stage/non_matching_feature": [1.0, 2.0, 3.0],
			}
		)
		general_event = SimpleNamespace(
			sex="M",
			age=45,
			discharge_status="dead",
		)
		patient = _MockPatient("p3", [general_event], stage_df)

		self.assertEqual(task(patient), [])


if __name__ == "__main__":
    unittest.main()
