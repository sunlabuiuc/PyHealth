import unittest
import polars as pl
from pyhealth.tasks.cxr_bias_mitigation import CXRBiasMitigationSamplingTask


class DummyPatient:
    def __init__(self, patient_id, metadata=None, patients=None, admissions=None, chexpert=None):
        self.patient_id = patient_id
        self._data = {
            "metadata": metadata if metadata is not None else pl.DataFrame(),
            "patients": patients if patients is not None else pl.DataFrame(),
            "admissions": admissions if admissions is not None else pl.DataFrame(),
            "chexpert": chexpert if chexpert is not None else pl.DataFrame()
        }

    def get_events(self, event_type, return_df=False):
        return self._data.get(event_type, pl.DataFrame())


class TestCXRBiasMitigationSamplingTask(unittest.TestCase):
    def setUp(self):
        self.task = CXRBiasMitigationSamplingTask()

    def test_pre_filter(self):
        data = [
            # ✅ Patient A
            {"event_type": "patients", "patient_id": 101, "patients/gender": "M"},
            {"event_type": "admissions", "patient_id": 101, "admissions/race": "White"},
            {"event_type": "admissions", "patient_id": 101, "admissions/insurance": "Medicare"},

            # ❌ Patient B
            {"event_type": "patients", "patient_id": 102, "patients/gender": None},
            {"event_type": "admissions", "patient_id": 102, "admissions/race": "Black"},
            {"event_type": "admissions", "patient_id": 102, "admissions/insurance": "Medicaid"},

            # ❌ Patient C
            {"event_type": "patients", "patient_id": 103, "patients/gender": "F"},
            {"event_type": "admissions", "patient_id": 103, "admissions/insurance": "Private"},

            # ✅ Patient D
            {"event_type": "patients", "patient_id": 104, "patients/gender": "F"},
            {"event_type": "admissions", "patient_id": 104, "admissions/race": "Asian"},
            {"event_type": "admissions", "patient_id": 104, "admissions/insurance": "Private"}
        ]
        df = pl.DataFrame(data)
        filtered = self.task.pre_filter(df)
        patient_ids = set(filtered["patient_id"].to_list())

        expected_ids = {101, 104}
        self.assertEqual(patient_ids, expected_ids)

    def test_call_no_metadata(self):
        patient = DummyPatient(patient_id=1)
        result = self.task(patient)
        self.assertEqual(result, [])

    def test_call_filters_viewposition(self):
        metadata = pl.DataFrame({
            "metadata/viewposition": ["LL", "PA"],
            "metadata/dicom_id": ["x", "y"],
            "metadata/study_id": ["s1", "s2"],
            "metadata/image_path": ["p1", "p2"],
            "patient_id": [1, 1]
        })
        patient = DummyPatient(patient_id=1, metadata=metadata)
        result = self.task(patient)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ViewPosition"], "PA")

    def test_call_merges_metadata(self):
        metadata = pl.DataFrame({
            "metadata/viewposition": ["PA"],
            "metadata/dicom_id": ["d1"],
            "metadata/study_id": ["s1"],
            "metadata/image_path": ["p1"],
            "patient_id": [1]
        })

        patients = pl.DataFrame({
            "patients/gender": ["F"]
        })

        admissions = pl.DataFrame({
            "admissions/insurance": ["Private"],
            "admissions/race": ["Asian"],
            "admissions/marital_status": ["Married"]
        })

        patient = DummyPatient(patient_id=1, metadata=metadata, patients=patients, admissions=admissions)
        result = self.task(patient)
        self.assertEqual(result[0]["gender"], "F")
        self.assertEqual(result[0]["insurance"], "Private")
        self.assertEqual(result[0]["ethnicity"], "Asian")
        self.assertEqual(result[0]["marital_status"], "Married")

    def test_call_joins_chexpert(self):
        metadata = pl.DataFrame({
            "metadata/viewposition": ["PA"],
            "metadata/dicom_id": ["d1"],
            "metadata/study_id": ["s1"],
            "metadata/image_path": ["img"],
            "patient_id": [1]
        })

        chexpert = pl.DataFrame({
            "chexpert/dicom_id": ["d1"],
            "chexpert/atelectasis": [1],
            "chexpert/cardiomegaly": [0],
            "chexpert/consolidation": [1],
            "chexpert/edema": [0],
            "chexpert/enlarged cardiomediastinum": [0],
            "chexpert/fracture": [0],
            "chexpert/lung lesion": [0],
            "chexpert/lung opacity": [1],
            "chexpert/no finding": [0],
            "chexpert/pleural effusion": [1],
            "chexpert/pleural other": [0],
            "chexpert/pneumonia": [1],
            "chexpert/pneumothorax": [0],
            "chexpert/support devices": [1]
        })

        patient = DummyPatient(patient_id=1, metadata=metadata, chexpert=chexpert)
        result = self.task(patient)

        self.assertEqual(result[0]["Atelectasis"], 1)
        self.assertEqual(result[0]["Cardiomegaly"], 0)
        self.assertEqual(result[0]["Pneumonia"], 1)
        self.assertEqual(result[0]["Support Devices"], 1)


if __name__ == "__main__":
    unittest.main()
