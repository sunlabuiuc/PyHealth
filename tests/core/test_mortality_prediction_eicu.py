import unittest
import numpy as np
from pyhealth.data import Patient, Visit
from pyhealth.tasks import MortalityPredictionEICU

class TestEICUTask(unittest.TestCase):
    def setUp(self):
        self.p1 = Patient(patient_id="1", age=50, gender="Female")
        self.p1.add_encounter(Visit(visit_id="v1", patient_id="1", discharge_status="Expired", los=1.5))
        self.p2 = Patient(patient_id="2", age=30, gender="Male")
        self.p2.add_encounter(Visit(visit_id="v2", patient_id="2", discharge_status="Alive", los=5.0))

    def test_label_generation(self):
        task_m = MortalityPredictionEICU(task_type="mortality")
        self.assertEqual(task_m(self.p1)[0]["label"], 1)
        self.assertEqual(task_m(self.p2)[0]["label"], 0)
        task_l = MortalityPredictionEICU(task_type="los")
        self.assertEqual(task_l(self.p2)[0]["label"], 1)

    def test_reproduction_metrics(self):
        task = MortalityPredictionEICU()
        data = np.array([0.1, 0.2, 0.3])
        self.assertEqual(task.kl_divergence(data, data), 0.0)

if __name__ == "__main__":
    unittest.main()