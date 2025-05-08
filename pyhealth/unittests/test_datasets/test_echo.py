import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image
from pyhealth.datasets import EchoBagDataset
class TestEchoDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        self.study_ids = ["patient1_study1", "patient2_study1"]
        for study in self.study_ids:
            for i in range(3):
                img = Image.fromarray(np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8))
                img.save(os.path.join(self.temp_dir, f"{study}_{i}.png"))

        data = {
            "patient_study": self.study_ids,
            "diagnosis_label": ["no_AS", "severe_AS"],
        }
        self.summary_table = pd.DataFrame(data)

        self.summary_csv_path = os.path.join(self.temp_dir, "TMED2SummaryTable.csv")
        self.summary_table.to_csv(self.summary_csv_path, index=False)


    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_dataset_loading(self):
        dataset = EchoBagDataset(
            root_dir=self.temp_dir,
            summary_table=self.summary_table,
            transform_fn=None,  
            sampling_strategy="first_frame"
        )

        self.assertEqual(len(dataset), 2)

        images, label = dataset[0]
        self.assertEqual(images.shape[1:], (112, 112, 3))
        self.assertTrue(label in [0, 2])
        self.assertEqual(images.shape[0], 3)

    def test_get_item(self):
        dataset = EchoBagDataset(
        root_dir=self.temp_dir,
        summary_table=self.summary_table
    )

        images, label = dataset[0]

        self.assertIsInstance(images, np.ndarray)
        self.assertEqual(images.shape, (3, 112, 112, 3))
        self.assertIsInstance(label, int)
        self.assertIn(label, [0, 2])
        self.assertTrue(np.all(images >= 0) and np.all(images <= 255))

if __name__ == "__main__":
    unittest.main(verbosity=2)
