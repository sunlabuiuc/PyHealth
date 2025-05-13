import datetime
import unittest

import sys
#sys.path.append("/home/uroy/uroy/DLH598/BOLD-Data/PyHealth")

from pyhealth.datasets import BoldDataset


import os, sys

current = os.path.dirname(os.path.realpath(__file__))
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current)))
sys.path.append(repo_root)

#dataset : https://physionet.org/content/blood-gas-oximetry/1.0/

# this test suite verifies the BOLD dataset is consistently parsing the dataset.
# a dataset is qualified if it produces the correct statistics, and if a sample from the dataset
# matches the expected data.
# Synthetic_BOLD dataset provided in the root is a dependancy to the expected values
# used for testing correctness
# like the BOLD dataset, if this test suite fails, it may be due to a regression in the
# code, or due to the dataset at the root chaning.


class TestsBoldDataset(unittest.TestCase):
    #DATASET_NAME = "bold"
    #ROOT = "https://physionet.org/files/blood-gas-oximetry/1.0/"
    ROOT = "./"
    TABLES = ["patients","demographics","hospital","abgdata","vitalsdata","labdata","coagulationlabs","bmpdata","hfpdata","otherlabdata","sofascores"]

    bold_dataset = BoldDataset(
        #dataset_name=DATASET_NAME,
        root=ROOT,
        tables=TABLES,
    )

    def setUp(self):
        pass

    # tests that a single event is correctly parsed
    def test_patient(self):

        selected_patient_id = "16"
        selected_hospitalid="73"
        self.assertEqual(selected_patient_id , self.bold_dataset.get_patient(selected_patient_id).patient_id)
        self.assertEqual(selected_hospitalid, self.bold_dataset.get_patient(selected_patient_id).get_events("hospital")[0].attr_dict.get("hospitalid"))


    def test_statistics(self):
        self.bold_dataset.stats()
        print("dataset_name = ",self.bold_dataset.dataset_name)
        bold_data = self.bold_dataset
        print(" bold_data.patient_id(16) = ",bold_data.get_patient("16").patient_id)
        print(" bold_data.data_source = ",bold_data.get_patient("16").data_source)
        print(" bold_data.patient_id(16).get_events(patients) = ",bold_data.get_patient("16").get_events("patients"))
        print("\n")
        print(" bold_data.patient_id(16).get_events(demographics) = ", bold_data.get_patient("16").get_events("demographics"))
        print("\n")
        print(" bold_data.patient_id(16).get_events(hospital) = ",bold_data.get_patient("16").get_events("hospital"))
        print("\n")
        print(" bold_data.patient_id(16).get_events(abgdDta) = ",bold_data.get_patient("16").get_events("abgdata"))
        print("\n")
        print(" bold_data.patient_id(16).get_events(vitalsData) = ",bold_data.get_patient("16").get_events("vitalsdata"))
        print("\n")
        print(" bold_data.patient_id(16).get_events(labdata) = ",bold_data.get_patient("16").get_events("labdata"))
        print("\n")
        print(" bold_data.patient_id(16).get_events(coagulationlabs) = ",bold_data.get_patient("16").get_events("coagulationlabs"))
        print("\n")
        print(" bold_data.patient_id(16).get_events(bmpdata) = ",bold_data.get_patient("16").get_events("bmpdata"))
        print("\n")
        print(" bold_data.patient_id(16).get_events(hfpdata) = ",bold_data.get_patient("16").get_events("hfpdata"))
        print("\n")
        print(" bold_data.patient_id(16).get_events(otherlabdata) = ",bold_data.get_patient("16").get_events("otherlabdata"))
        print("\n")
        print(" bold_data.patient_id(16).get_events(sofascores) = ",bold_data.get_patient("16").get_events("sofascores"))

        self.assertEqual(sorted(self.TABLES), sorted(bold_data.tables))



if __name__ == "__main__":
    unittest.main(verbosity=2)
