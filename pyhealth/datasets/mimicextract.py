import os
from typing import Optional, List, Dict, Tuple, Union

import pandas as pd

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseEHRDataset
from pyhealth.datasets.utils import strptime

# TODO: add other tables

class MIMICExtractDataset(BaseEHRDataset):
    """Base dataset for MIMIC-Extract dataset.

    TODO: Dataset description

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain one or more HDF5 files).
        tables: list of tables to be loaded (e.g., ["DIAGNOSES_ICD", "NOTES"]). TODO: What here?
        code_mapping: a dictionary containing the code mapping information.
            The key is a str of the source code vocabulary and the value is of
            two formats:
                (1) a str of the target code vocabulary;
                (2) a tuple with two elements. The first element is a str of the
                    target code vocabulary and the second element is a dict with
                    keys "source_kwargs" or "target_kwargs" and values of the
                    corresponding kwargs for the `CrossMap.map()` method.
            Default is empty dict, which means the original code will be used.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.
        pop_size: If your MIMIC-Extract dataset was created with a pop_size parameter,
            include it here. This is used to find the correct filenames.

    Attributes:
        task: Optional[str], name of the task (e.g., "mortality prediction").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, visit_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.

    Examples:
        >>> from pyhealth.datasets import MIMICExtractDataset
        >>> dataset = MIMICExtractDataset(
        ...         root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        ...         tables=["DIAGNOSES_ICD", "NOTES"], TODO: What here?
        ...         code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: Optional[str] = None,
        code_mapping: Optional[Dict[str, Union[str, Tuple[str, Dict]]]] = None,
        dev: bool = False,
        refresh_cache: bool = False,
        pop_size: Optional[int] = None,
        itemid_to_variable_map: Optional[str] = None,
        is_icustay_visit: Optional[bool] = False
    ):
        if pop_size is not None:
            self._fname_suffix = f"_{pop_size}"
        self._ahd_filename = os.path.join(root, f"all_hourly_data{self._fname_suffix}.h5")
        self._c_filename = os.path.join(root, f"C{self._fname_suffix}.h5")
        self._notes_filename = os.path.join(root, f"all_hourly_data{self._fname_suffix}.hdf")
        self._v_id_column = 'icustay_id' if is_icustay_visit else 'hadm_id'

        # This could be implemented with MedCode.CrossMap, however part of the idea behind
        # MIMIC-Extract is that the user can customize this mapping--therefore we will
        # make a map specific to this dataset instance based on a possibly-customized CSV.
        self._vocab_map = { "chartevents": {}, "labevents": {}, "vitals_labs_mean": {} }
        if itemid_to_variable_map is not None:
            # We are just going to read some metadata here...
            df_ahd = pd.read_hdf(self._ahd_filename, 'vitals_labs_mean')
            grptype = "LEVEL1" if "LEVEL1" in df_ahd.columns.names else "LEVEL2"

            itemid_map = pd.read_csv(itemid_to_variable_map)
            for linksto, dict in self._vocab_map.items():
                df = itemid_map
                if linksto != 'vitals_labs_mean':
                    df = df[df["LINKSTO"] == linksto]
                # Pick the most common ITEMID to use for our vocabulary...
                df = df.sort_values(by="COUNT", ascending=False).groupby(grptype).head(1)
                df = df[[grptype,"ITEMID"]].set_index([grptype])
                #TODO: Probably a better way than iterrows? At least this is a small df.
                #self._vocab_map[linksto] = df[["ITEMID"]].to_dict(orient="index")
                for r in df.iterrows():
                    self._vocab_map[linksto][r[0].lower()] = r[1]["ITEMID"]

        # reverse engineered from mimic-code concepts SQL and MIMIC-Extractt SQL...
        self._vocab_map['interventions'] = { 
            'vent': 467,
            'adenosine': 4649,
            'dobutamine': 30042,
            'dopamine': 30043,
            'epinephrine': 30044,
            'isuprel': 30046,
            'milrinone': 30125,
            'norepinephrine': 30047,
            'phenylephrine': 30127,
            'vasopressin': 30051,
            'colloid_bolus': 46729, # "Dextran" Arbitrary! No general itemid!
            'crystalloid_bolus': 41491, # "fluid bolus"
            'nivdurations': 468
        }

                
        super().__init__(root=root, tables=tables,
            dataset_name=dataset_name, code_mapping=code_mapping,
            dev=dev, refresh_cache=refresh_cache)
        

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PATIENTS and ADMISSIONS tables.

        Will be called in `self.parse_tables()`

        Docs:
            - PATIENTS: https://mimic.mit.edu/docs/iii/tables/patients/
            - ADMISSIONS: https://mimic.mit.edu/docs/iii/tables/admissions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id which is updated with the mimic-3 table result.

        Returns:
            The updated patients dict.
        """
        # read patients table
        patients_df = pd.read_hdf(self._ahd_filename, 'patients')
        # sort by admission and discharge time
        df = patients_df.reset_index().sort_values(["subject_id", "admittime", "dischtime"], ascending=True)
        # group by patient
        #TODO: This can probably be simplified--MIMIC-Extract includes only the first ICU
        # visit for each patient (see paper)... it is unclear whether it might be easily
        # modified to include multiple visits however, so this may have value for customised
        # versions of the pipeline.
        df_group = df.groupby("subject_id")

        # parallel unit of basic information (per patient)
        def basic_unit(p_id, p_info):
            #FIXME: This is insanity.
            #tdelta = pd.Timedelta(days=365.2425*p_info["age"].values[0]) 
            # pd.Timedelta cannot handle 300-year deltas!
            tdeltahalf = pd.Timedelta(days=0.5*365.2425*p_info["age"].values[0]) 
            patient = Patient(
                patient_id=p_id,
                birth_datetime=pd.to_datetime(p_info["admittime"].values[0]-tdeltahalf-tdeltahalf), #see?
                death_datetime=p_info["deathtime"].values[0],
                gender=p_info["gender"].values[0],
                ethnicity=p_info["ethnicity"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("hadm_id"):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    encounter_time=pd.to_datetime(v_info["admittime"].values[0]),
                    discharge_time=pd.to_datetime(v_info["dischtime"].values[0]),
                    discharge_status=v_info["hospital_expire_flag"].values[0],
                )
                # add visit
                patient.add_visit(visit)
            return patient

        # parallel apply
        df_group = df_group.parallel_apply(
            lambda x: basic_unit(x.subject_id.unique()[0], x)
        )
        # summarize the results
        for pat_id, pat in df_group.items():
            patients[pat_id] = pat

        return patients

    def parse_diagnoses_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses the C (ICD9 diagnosis codes) table in a way compatible with MIMIC3Dataset.

        Will be called in `self.parse_tables()`

        Docs:
            - DIAGNOSES_ICD: https://mimic.mit.edu/docs/iii/tables/diagnoses_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-III does not provide specific timestamps in DIAGNOSES_ICD
                table, so we set it to None.
        """
        return self._parse_c(patients, table='DIAGNOSES_ICD')

    def parse_c(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses the C (ICD9 diagnosis codes) table.

        Will be called in `self.parse_tables()`

        Docs:
            - DIAGNOSES_ICD: https://mimic.mit.edu/docs/iii/tables/diagnoses_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-III does not provide specific timestamps in DIAGNOSES_ICD
                table, so we set it to None.
        """
        return self._parse_c(patients, table='C')

    def _parse_c(self, patients: Dict[str, Patient], table: str = 'C') -> Dict[str, Patient]:
        # read table
        df = pd.read_hdf(self._c_filename, 'C')
        # drop records of the other patients
        df = df.loc[(list(patients.keys()),slice(None),slice(None)),:]
        # drop rows with missing values
        #df = df.dropna(subset=["subject_id", "hadm_id", "icd9_codes"])
        dfgroup = df.reset_index().groupby("subject_id")

        #display(df)
        #df = df.reset_index(['icustay_id']) #drops this one only.. interesting
        #display(df)
        captured_v_id_column = self._v_id_column
        def diagnosis_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby(captured_v_id_column):
                codes = set(v_info['icd9_codes'].sum())
                for code in codes:
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="ICD9CM",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    events.append(event)
            return events

        # parallel apply
        dfgroup = dfgroup.parallel_apply(
        #dfgroup = dfgroup.apply(
            lambda x: diagnosis_unit(x.subject_id.unique()[0], x)
        )
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, dfgroup)
        return patients

    def parse_labevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses ...

        Will be called in `self.parse_tables()`

        Docs:
            - LABEVENTS: https://mimic.mit.edu/docs/iii/tables/labevents/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "LABEVENTS"
        return self._parse_vitals_labs_mean(patients=patients, table=table)

    def parse_chartevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses ...

        Will be called in `self.parse_tables()`

        Docs:
            - CHARTEVENTS: https://mimic.mit.edu/docs/iii/tables/chartevents/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "CHARTEVENTS"
        return self._parse_vitals_labs_mean(patients=patients, table=table)

    def parse_vitals_labs_mean(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses ...

        Will be called in `self.parse_tables()`

        Docs:
            - TODO

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "vitals_labs_mean"
        return self._parse_vitals_labs_mean(patients=patients, table=table)

    def _parse_vitals_labs_mean(self, patients: Dict[str, Patient], table: str = 'vitals_labs_mean') -> Dict[str, Patient]:
        linksto = table.lower()
        # read table
        df = pd.read_hdf(self._ahd_filename, 'vitals_labs_mean')
        # drop records of the other patients
        df = df.loc[(list(patients.keys()),slice(None),slice(None)),:]

        # parallel unit for lab (per patient)
        captured_v_id_column = self._v_id_column
        captured_vocab_map = self._vocab_map[linksto]
        def vlm_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby(captured_v_id_column):
                for e_id, e_info in v_info.iterrows():
                    #print(e_id)
                    #print(f"{e_info['variable']} -> {self._vocab_map[linksto][e_info['variable']]=}")
                    event = Event(
                        code=captured_vocab_map[e_info['variable']],
                        table=table,
                        vocabulary="MIMIC3_ITEMID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=pd.Timestamp.to_pydatetime(e_info['timestamp']),
                        hours_in=int(e_info['hours_in']),
                        #level_n=e_info['variable'], #this can be reverse-looked up, so save some mem here?
                        value=e_info['value'], # This is not stored in MIMIC3Dataset... why?
                        #TODO: Units, somewhere?
                    )
                    events.append(event)
            return events

        ahd_index = ["subject_id","hadm_id","icustay_id","hours_in"]

        if 'LEVEL1' in df.columns.names:
            df.columns = df.columns.get_level_values(2)
        else:
            df.columns = df.columns.get_level_values(0)

        # reconstruct nominal timestamps for hours_in values...
        df_p = pd.read_hdf(self._ahd_filename, 'patients')
        df_p = df_p.loc[(list(patients.keys()),slice(None),slice(None)),:][['intime']]
        df = df.merge(df_p, left_on=['subject_id','hadm_id','icustay_id'], right_index=True, how="left")
        df['timestamp'] = df['intime'].dt.ceil('H')+pd.to_timedelta(df.index.get_level_values(3), unit="H")

        df = df.drop(columns=[col for col in df.columns if col not in self._vocab_map[linksto] and col not in ['timestamp']])
        df = df.reset_index()
        df = df.melt(id_vars=ahd_index+['timestamp']).dropna()
        df = df.sort_values(ahd_index)
        group_df = df.groupby("subject_id")

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: vlm_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_interventions(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        table = 'interventions'
        linksto = table.lower() # we might put these in CHARTEVENTS also?

        # read table
        df = pd.read_hdf(self._ahd_filename, 'interventions')
        # drop records of the other patients
        df = df.loc[(list(patients.keys()),slice(None),slice(None)),:]

        # parallel unit for interventions (per patient)
        captured_v_id_column = self._v_id_column
        captured_vocab_map = self._vocab_map[linksto] 
        def interv_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby(captured_v_id_column):
                for e_id, e_info in v_info.iterrows():
                    event = Event(
                        code=captured_vocab_map[e_info['variable']],
                        table=table,
                        vocabulary="MIMIC3_ITEMID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=pd.Timestamp.to_pydatetime(e_info['timestamp']),
                        hours_in=int(e_info['hours_in']),
                        intervention=e_info['variable']
                    )
                    events.append(event)
            return events

        ahd_index = ["subject_id","hadm_id","icustay_id","hours_in"]

        #if 'LEVEL1' in df.columns.names:
        #    df.columns = df.columns.get_level_values(2)
        #else:
        #    df.columns = df.columns.get_level_values(0)

        # reconstruct nominal timestamps for hours_in values...
        df_p = pd.read_hdf(self._ahd_filename, 'patients')
        df_p = df_p.loc[(list(patients.keys()),slice(None),slice(None)),:][['intime']]
        df = df.merge(df_p, left_on=['subject_id','hadm_id','icustay_id'], right_index=True, how="left")
        df['timestamp'] = df['intime'].dt.ceil('H')+pd.to_timedelta(df.index.get_level_values(3), unit="H")

        df = df.drop(columns=[col for col in df.columns if col not in self._vocab_map[linksto] and col not in ['timestamp']])
        df = df.reset_index()
        df = df.melt(id_vars=ahd_index+['timestamp'])
        df = df[df['value'] > 0]
        df = df.sort_values(ahd_index)
        group_df = df.groupby("subject_id")

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: interv_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients


if __name__ == "__main__":
    dataset = MIMICExtractDataset(
        root="../data/baseline5000/grouping",
        tables=[
            "C",
            "vitals_labs_mean"
        ],
        #code_mapping={"NDC": "ATC"},
        dev=True,
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()
