"""Tests for the MIMIC-III tables added to the default config (part of #328).

All data here is synthetic: a few made-up rows per table, written to a temp
directory as gzipped CSVs with the column names of the MIMIC-III 1.4 schema
(https://mimic.mit.edu/docs/iii/tables/).
"""

import gzip
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from pyhealth.datasets import MIMIC3Dataset

FILES = {
    "PATIENTS.csv.gz": (
        "row_id,subject_id,gender,dob,dod,dod_hosp,dod_ssn,expire_flag\n"
        "1,100,F,2090-01-01 00:00:00,,,,0\n"
    ),
    "ADMISSIONS.csv.gz": (
        "row_id,subject_id,hadm_id,admittime,dischtime,deathtime,admission_type,"
        "admission_location,discharge_location,insurance,language,religion,"
        "marital_status,ethnicity,edregtime,edouttime,diagnosis,"
        "hospital_expire_flag,has_chartevents_data\n"
        "1,100,200,2150-01-01 08:00:00,2150-01-10 12:00:00,,EMERGENCY,"
        "EMERGENCY ROOM ADMIT,HOME,Medicare,ENGL,NOT SPECIFIED,SINGLE,WHITE,"
        ",,PNEUMONIA,0,1\n"
    ),
    "ICUSTAYS.csv.gz": (
        "row_id,subject_id,hadm_id,icustay_id,dbsource,first_careunit,"
        "last_careunit,first_wardid,last_wardid,intime,outtime,los\n"
        "1,100,200,300,metavision,MICU,MICU,50,50,"
        "2150-01-02 09:00:00,2150-01-04 09:00:00,2.0\n"
    ),
    "SERVICES.csv.gz": (
        "row_id,subject_id,hadm_id,transfertime,prev_service,curr_service\n"
        "1,100,200,2150-01-01 08:30:00,,MED\n"
        "2,100,200,2150-01-05 10:00:00,MED,CMED\n"
    ),
    "DRGCODES.csv.gz": (
        "row_id,subject_id,hadm_id,drg_type,drg_code,description,drg_severity,"
        "drg_mortality\n"
        "1,100,200,APR,139,OTHER PNEUMONIA,2,2\n"
    ),
    "CPTEVENTS.csv.gz": (
        "row_id,subject_id,hadm_id,costcenter,chartdate,cpt_cd,cpt_number,"
        "cpt_suffix,ticket_id_seq,sectionheader,subsectionheader,description\n"
        "1,100,200,ICU,2150-01-03 00:00:00,99291,99291,,1,Medicine,"
        "Critical care services,Critical care first hour\n"
    ),
}

NEW_TABLES = ["services", "drgcodes", "cptevents"]


class TestMIMIC3ExtraTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name) / "mimic3"
        root.mkdir()
        for name, text in FILES.items():
            with gzip.open(root / name, "wt") as f:
                f.write(text)
        cls.dataset = MIMIC3Dataset(
            root=str(root),
            tables=NEW_TABLES,
            cache_dir=str(Path(cls._tmp.name) / "cache"),
        )
        cls.patient = cls.dataset.get_patient("100")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def events(self, table):
        return self.patient.get_events(event_type=table)

    def test_every_new_table_loads(self):
        for table in NEW_TABLES:
            with self.subTest(table=table):
                self.assertGreater(len(self.events(table)), 0)

    def test_services(self):
        ev = self.events("services")
        self.assertEqual(len(ev), 2)
        self.assertEqual(ev[1].timestamp, datetime(2150, 1, 5, 10, 0))
        self.assertEqual(ev[1].curr_service, "CMED")

    def test_drgcodes_uses_discharge_time(self):
        (ev,) = self.events("drgcodes")
        self.assertEqual(ev.timestamp, datetime(2150, 1, 10, 12, 0))
        self.assertEqual(ev.drg_code, "139")

    def test_cptevents(self):
        (ev,) = self.events("cptevents")
        self.assertEqual(ev.timestamp, datetime(2150, 1, 3, 0, 0))
        self.assertEqual(ev.cpt_cd, "99291")


if __name__ == "__main__":
    unittest.main()
