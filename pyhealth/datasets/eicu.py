import os
from typing import Optional, List, Dict

import pandas as pd
from tqdm import tqdm

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseDataset


# TODO: add other tables


class eICUDataset(BaseDataset):
    """Base dataset for eICU dataset.

    The eICU dataset is a large dataset of de-identified health records of ICU
        patients. The dataset is available at https://eicu-crd.mit.edu/.

    The basic information is stored in the following tables:
        - patient: defines a patient (uniquepid), a hospital admission
            (patienthealthsystemstayid), and a ICU stay (patientunitstayid)
            in the database.
        - hospital: contains information about a hospital (e.g., region).

    Note that in eICU, a patient can have multiple hospital admissions and each
        hospital admission can have multiple ICU stays. The data in eICU is centered
        around the ICU stay and all timestamps are relative to the ICU admission time.
        Thus, we only know the order of ICU stays within a hospital admission, but not
        the order of hospital admissions within a patient. As a result, we use Patient
        object to represent a hospital admission of a patient, and use Visit object to
        store the ICU stays within that hospital admission.

    We further support the following tables:
        - diagnosis: contains ICD diagnoses (ICD9CM and ICD10CM code)
            for patients
        - treatment: contains treatment information (eICU_TREATMENTSTRING code)
            for patients.
        - medication: contains medication related order entries (eICU_DRUGNAME
            code) for patients.
        - lab: contains laboratory measurements (eICU_LABNAME code)
            for patients
        - physicalExam: contains all physical exam (eICU_PHYSICALEXAMPATH)
            conducted for patients.

    Args:
        dataset_name: str, name of the dataset.
        root: str, root directory of the raw data (should contain many csv files).
        tables: List[str], list of tables to be loaded. Must be a subset of the
            following tables: diagnosis, medication, lab, treatment, physicalExam.
        code_mapping: Optional[Dict[str, str]], key is the source code vocabulary and
            value is the target code vocabulary (e.g., {"ICD9CM": "CCSCM"}).
            Default is empty dict, which means the original code will be used.
        dev: bool, whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: bool, whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Attributes:
        visit_id_to_patient_id: Dict[str, str], a mapping from visit_id to patient_id.
        task: Optional[str], name of the task (e.g., "mortality prediction").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, visit_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.
    """

    def __init__(
            self,
            root: str,
            tables: List[str],
            code_mapping: Optional[Dict[str, str]] = None,
            dev=False,
            refresh_cache=False,
    ):
        # store a mapping from visit_id to patient_id
        # will be used to parse clinical tables as they only contain visit_id
        self.visit_id_to_patient_id: Dict[str, str] = {}

        super(eICUDataset, self).__init__(
            dataset_name="eICU",
            root=root,
            tables=tables,
            code_mapping=code_mapping,
            dev=dev,
            refresh_cache=refresh_cache,
        )

    def _parse_tables(self) -> Dict[str, Patient]:
        """This function overrides the _parse_tables() function in BaseDataset.

        It parses the corresponding tables and creates a dict of patients which
            will be cached later.

        Returns:
            patients: a dictionary of Patient objects indexed by patient_id.
        """
        # patients is a dict of Patient objects indexed by patient_id
        patients: Dict[str, Patient] = dict()
        # process patients and admissions tables
        patients = self._parse_basic_info(patients)
        # process clinical tables
        for table in self.tables:
            try:
                # use lower case for function name
                patients = getattr(self, f"_parse_{table.lower()}")(patients)
            except AttributeError:
                raise NotImplementedError(
                    f"Parser for table {table} is not implemented yet."
                )
        return patients

    def _parse_basic_info(self, patients) -> Dict[str, Patient]:
        """Helper functions which parses patient and hospital tables.

        Will be called in _parse_tables().

        Docs:
            - patient: https://eicu-crd.mit.edu/eicutables/patient/
            - hospital: https://eicu-crd.mit.edu/eicutables/hospital/

        Note that we use Patient object to represent a hospital admission of a
            patient, and use Visit object to store the ICU stays within that hospital
            admission.
        """
        # read patient table
        patient_df = pd.read_csv(
            os.path.join(self.root, "patient.csv"),
            dtype={"uniquepid": str,
                   "patienthealthsystemstayid": str,
                   "patientunitstayid": str},
            nrows=5000 if self.dev else None,
        )
        # read hospital table
        hospital_df = pd.read_csv(os.path.join(self.root, "hospital.csv"))
        hospital_df.region = hospital_df.region.fillna("Unknown").astype(str)
        # merge patient and hospital tables
        df = pd.merge(patient_df, hospital_df, on="hospitalid", how="left")
        # sort by ICU admission and discharge time
        df["neg_hospitaladmitoffset"] = -df["hospitaladmitoffset"]
        df = df.sort_values(
            [
                "uniquepid",
                "patienthealthsystemstayid",
                "neg_hospitaladmitoffset",
                "unitdischargeoffset",
            ],
            ascending=True,
        )
        # group by patient and hospital admission
        df_group = df.groupby(["uniquepid", "patienthealthsystemstayid"])
        # load patients
        for (p_id, ha_id), p_info in tqdm(df_group, desc="Parsing patients"):
            # each Patient object is a single hospital admission of a patient
            patient_id = f"{p_id}+{ha_id}"

            # hospital admission time (Jan 1 of hospitaldischargeyear, 00:00:00)
            ha_datetime = self._strptime(
                str(p_info["hospitaldischargeyear"].values[0]),
                "%Y"
            )

            # no exact birth datetime in eICU
            # use hospital admission time and age to approximate birth datetime
            age = p_info["age"].values[0]
            if pd.isna(age):
                birth_datetime = None
            elif age == "> 89":
                birth_datetime = ha_datetime - pd.DateOffset(years=89)
            else:
                birth_datetime = ha_datetime - pd.DateOffset(years=int(age))

            # no exact death datetime in eICU
            # use hospital discharge time to approximate death datetime
            death_datetime = None
            if p_info["hospitaldischargestatus"].values[0] == "Expired":
                ha_los_min = p_info["hospitaldischargeoffset"].values[0] \
                             - p_info["hospitaladmitoffset"].values[0]
                death_datetime = ha_datetime + pd.Timedelta(minutes=ha_los_min)

            patient = Patient(
                patient_id=patient_id,
                birth_datetime=birth_datetime,
                death_datetime=death_datetime,
                gender=p_info["gender"].values[0],
                ethnicity=p_info["ethnicity"].values[0]
            )

            # load visits
            for v_id, v_info in p_info.groupby("patientunitstayid"):
                # each Visit object is a single ICU stay within a hospital admission

                # base time is the hospital admission time
                unit_admit = v_info["neg_hospitaladmitoffset"].values[0]
                unit_discharge = unit_admit + v_info["unitdischargeoffset"].values[0]
                encounter_time = ha_datetime + pd.Timedelta(minutes=unit_admit)
                discharge_time = ha_datetime + pd.Timedelta(minutes=unit_discharge)

                visit = Visit(
                    visit_id=v_id,
                    patient_id=patient_id,
                    encounter_time=encounter_time,
                    discharge_time=discharge_time,
                    discharge_status=v_info["unitdischargestatus"].values[0],
                    hospital_id=v_info["hospitalid"].values[0],
                    region=v_info["region"].values[0],
                )

                # add visit
                patient.add_visit(visit)
                # add visit id to patient id mapping
                self.visit_id_to_patient_id[v_id] = patient_id
            # add patient
            patients[patient_id] = patient
        return patients

    def _parse_diagnosis(self, patients) -> Dict[str, Patient]:
        """Helper functions which parses diagnosis table.

        Will be called in _parse_tables().

        Docs:
            - diagnosis: https://eicu-crd.mit.edu/eicutables/diagnosis/

        Note that this table contains both ICD9CM and ICD10CM codes in one single
            cell. We need to use medcode to distinguish them.
        """

        # load ICD9CM and ICD10CM coding systems
        from pyhealth.medcode import ICD9CM, ICD10CM

        icd9cm = ICD9CM()
        icd10cm = ICD10CM()

        def icd9cm_or_icd10cm(code):
            if code in icd9cm:
                return "ICD9CM"
            elif code in icd10cm:
                return "ICD10CM"
            else:
                return "Unknown"

        table = "diagnosis"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patientunitstayid": str, "icd9code": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patientunitstayid", "icd9code"])
        # sort by diagnosisoffset
        df = df.sort_values(["patientunitstayid", "diagnosisoffset"], ascending=True)
        # group by visit
        df_group = df.groupby("patientunitstayid")
        # iterate over each visit
        for v_id, v_info in tqdm(df_group, desc=f"Parsing {table}"):
            if v_id not in self.visit_id_to_patient_id:
                continue
            patient_id = self.visit_id_to_patient_id[v_id]
            for offset, codes in zip(v_info["diagnosisoffset"], v_info["icd9code"]):
                timestamp = patients[patient_id].get_visit_by_id(v_id).encounter_time \
                            + pd.Timedelta(minutes=offset)
                codes = [c.strip() for c in codes.split(",")]
                # for each code in a single cell (mixed ICD9CM and ICD10CM)
                for code in codes:
                    vocab = icd9cm_or_icd10cm(code)
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary=vocab,
                        visit_id=v_id,
                        patient_id=patient_id,
                        timestamp=timestamp,
                    )
                    # update patients
                    patients = self._add_event_to_patient_dict(patients, event)
        return patients

    def _parse_treatment(self, patients) -> Dict[str, Patient]:
        """Helper functions which parses treatment table.

        Will be called in _parse_tables().

        Docs:
            - treatment: https://eicu-crd.mit.edu/eicutables/treatment/
        """
        table = "treatment"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patientunitstayid": str, "treatmentstring": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patientunitstayid", "treatmentstring"])
        # sort by treatmentoffset
        df = df.sort_values(["patientunitstayid", "treatmentoffset"], ascending=True)
        # group by visit
        df_group = df.groupby("patientunitstayid")
        # iterate over each visit
        for v_id, v_info in tqdm(df_group, desc=f"Parsing {table}"):
            if v_id not in self.visit_id_to_patient_id:
                continue
            patient_id = self.visit_id_to_patient_id[v_id]
            for offset, code in zip(v_info["treatmentoffset"],
                                    v_info["treatmentstring"]):
                timestamp = patients[patient_id].get_visit_by_id(v_id).encounter_time \
                            + pd.Timedelta(minutes=offset)
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_TREATMENTSTRING",
                    visit_id=v_id,
                    patient_id=patient_id,
                    timestamp=timestamp,
                )
                # update patients
                patients = self._add_event_to_patient_dict(patients, event)
        return patients

    def _parse_medication(self, patients) -> Dict[str, Patient]:
        """Helper functions which parses medication table.

        Will be called in _parse_tables().

        Docs:
            - medication: https://eicu-crd.mit.edu/eicutables/medication/
        """
        table = "medication"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"patientunitstayid": str, "drugname": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patientunitstayid", "drugname"])
        # sort by drugstartoffset
        df = df.sort_values(["patientunitstayid", "drugstartoffset"], ascending=True)
        # group by visit
        df_group = df.groupby("patientunitstayid")
        # iterate over each visit
        for v_id, v_info in tqdm(df_group, desc=f"Parsing {table}"):
            if v_id not in self.visit_id_to_patient_id:
                continue
            patient_id = self.visit_id_to_patient_id[v_id]
            for offset, code in zip(v_info["drugstartoffset"], v_info["drugname"]):
                timestamp = patients[patient_id].get_visit_by_id(v_id).encounter_time \
                            + pd.Timedelta(minutes=offset)
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_DRUGNAME",
                    visit_id=v_id,
                    patient_id=patient_id,
                    timestamp=timestamp,
                )
                # update patients
                patients = self._add_event_to_patient_dict(patients, event)
        return patients

    def _parse_lab(self, patients) -> Dict[str, Patient]:
        """Helper functions which parses lab table.

        Will be called in _parse_tables().

        Docs:
            - lab: https://eicu-crd.mit.edu/eicutables/lab/
        """
        table = "lab"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patientunitstayid": str, "labname": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patientunitstayid", "labname"])
        # sort by labresultoffset
        df = df.sort_values(["patientunitstayid", "labresultoffset"], ascending=True)
        # group by visit
        df_group = df.groupby("patientunitstayid")
        # iterate over each visit
        for v_id, v_info in tqdm(df_group, desc=f"Parsing {table}"):
            if v_id not in self.visit_id_to_patient_id:
                continue
            patient_id = self.visit_id_to_patient_id[v_id]
            for offset, code in zip(v_info["labresultoffset"], v_info["labname"]):
                timestamp = patients[patient_id].get_visit_by_id(v_id).encounter_time \
                            + pd.Timedelta(minutes=offset)
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_LABNAME",
                    visit_id=v_id,
                    patient_id=patient_id,
                    timestamp=timestamp,
                )
                # update patients
                patients = self._add_event_to_patient_dict(patients, event)
        return patients

    def _parse_physicalexam(self, patients) -> Dict[str, Patient]:
        """Helper functions which parses physicalExam table.

        Will be called in _parse_tables().

        Docs:
            - physicalExam: https://eicu-crd.mit.edu/eicutables/physicalexam/
        """
        table = "physicalExam"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patientunitstayid": str, "physicalexampath": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patientunitstayid", "physicalexampath"])
        # sort by treatmentoffset
        df = df.sort_values(["patientunitstayid", "physicalexamoffset"], ascending=True)
        # group by visit
        df_group = df.groupby("patientunitstayid")
        # iterate over each visit
        for v_id, v_info in tqdm(df_group, desc=f"Parsing {table}"):
            if v_id not in self.visit_id_to_patient_id:
                continue
            patient_id = self.visit_id_to_patient_id[v_id]
            for offset, code in zip(v_info["physicalexamoffset"],
                                    v_info["physicalexampath"]):
                timestamp = patients[patient_id].get_visit_by_id(v_id).encounter_time \
                            + pd.Timedelta(minutes=offset)
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_PHYSICALEXAMPATH",
                    visit_id=v_id,
                    patient_id=patient_id,
                    timestamp=timestamp,
                )
                # update patients
                patients = self._add_event_to_patient_dict(patients, event)
        return patients


if __name__ == "__main__":
    dataset = eICUDataset(
        root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        tables=["diagnosis", "medication", "lab", "treatment", "physicalExam"],
        dev=False,
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()
