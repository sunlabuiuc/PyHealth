import os
import time
import pandas as pd
from tqdm import tqdm
from typing import Optional, List, Dict, Union, Tuple
from pandarallel import pandarallel
from pyhealth.data.data_v2 import Event, Patient
from pyhealth.datasets.base_dataset_v2 import BaseDataset
from pyhealth.datasets.utils import strptime
from pyhealth.medcode import CrossMap
from copy import deepcopy
from pyhealth.datasets.utils import MODULE_CACHE_PATH, DATASET_BASIC_TABLES
from pyhealth.datasets.utils import hash_str
from pyhealth.tasks.medical_coding import MIMIC4ICD9Coding
# TODO: add other tables, pyspark or pyarrow for preprocessing.


class MIMIC4Dataset(BaseDataset):

    """Base dataset for MIMIC-IV dataset.

    The MIMIC-IV dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://mimic.physionet.org/.

    The basic information is stored in the following tables:
        - patients: defines a patient in the database, subject_id.
        - admission: define a patient's hospital admission, hadm_id.

    We further support the following tables:
        - diagnoses_icd: contains ICD diagnoses (ICD9CM and ICD10CM code)
            for patients.
        - procedures_icd: contains ICD procedures (ICD9PROC and ICD10PROC
            code) for patients.
        - prescriptions: contains medication related order entries (NDC code)
            for patients.
        - labevents: contains laboratory measurements (MIMIC4_ITEMID code)
            for patients

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain many csv files).
        tables: list of tables to be loaded (e.g., ["DIAGNOSES_ICD", "PROCEDURES_ICD"]).
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
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> dataset = MIMIC4Dataset(
        ...         root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        ...         tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
        ...         code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """
    def __init__(self, root: str, 
                 dataset_name: Optional[str] = "MIMIC-IV", 
                 tables : List[str] = None, 
                 additional_dirs : Optional[Dict[str, str]] = {}, 
                 code_mapping: Optional[Dict[str, Union[str, Tuple[str, Dict]]]] = None, 
                 **kwargs):
        self.code_mapping = code_mapping
        self.code_mapping_tools = self._load_code_mapping_tools()
        self.code_vocs = {}
        super().__init__(root, dataset_name, tables, additional_dirs, **kwargs)
        

    def get_cache_path(self):
        args_to_hash = (
            [self.dataset_name, self.root]
            + sorted(self.tables)
            + sorted(self.code_mapping.items())
            + ["dev" if self.dev else "prod"]
            + sorted([(k, v) for k, v in self.tables_dir.items()])
        )
        filename = hash_str("+".join([str(arg) for arg in args_to_hash])) + ".pkl"
        filepath = os.path.join(MODULE_CACHE_PATH, filename) # questions here (?)
        return filepath

    def process(self):
        # process the data
        patients = self.parse_tables()
        # convert codes
        patients = self._convert_code_in_patient_dict(patients)
        return patients
        

    def _load_code_mapping_tools(self) -> Dict[str, CrossMap]:
        """Helper function which loads code mapping tools CrossMap for code mapping.

        Will be called in `self.__init__()`.

        Returns:
            A dict whose key is the source and target code vocabulary and
                value is the `CrossMap` object.
        """
        code_mapping_tools = {}
        for s_vocab, target in self.code_mapping.items():
            if isinstance(target, tuple):
                assert len(target) == 2
                assert type(target[0]) == str
                assert type(target[1]) == dict
                assert target[1].keys() <= {"source_kwargs", "target_kwargs"}
                t_vocab = target[0]
            else:
                t_vocab = target
            # load code mapping from source to target
            code_mapping_tools[f"{s_vocab}_{t_vocab}"] = CrossMap(s_vocab, t_vocab)
        return code_mapping_tools
    
    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper functions which parses patients and admissions tables.

        Will be called in `self.parse_tables()`

        Docs:
            - patients:https://mimic.mit.edu/docs/iv/modules/hosp/patients/
            - admissions: https://mimic.mit.edu/docs/iv/modules/hosp/admissions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        # read patients table
        patients_df = pd.read_csv(
            os.path.join(self.root, "patients.csv"),
            dtype={"subject_id": str},
            nrows=1000 if self.dev else None,
        )
        # read admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "admissions.csv"),
            dtype={"subject_id": str, "hadm_id": str},
        )
        # merge patients and admissions tables
        df = pd.merge(patients_df, admissions_df, on="subject_id", how="inner")
        # sort by admission and discharge time
        df = df.sort_values(["subject_id", "admittime", "dischtime"], ascending=True)
        # group by patient
        df_group = df.groupby("subject_id")

        # parallel unit of basic information (per patient)
        def basic_unit(p_id, p_info):
            # no exact birth datetime in MIMIC-IV
            # use anchor_year and anchor_age to approximate birth datetime
            anchor_year = int(p_info["anchor_year"].values[0])
            anchor_age = int(p_info["anchor_age"].values[0])
            birth_year = anchor_year - anchor_age
            patient = Patient(
                patient_id=p_id,
                # no exact month, day, and time, use Jan 1st, 00:00:00
                birth_datetime=strptime(str(birth_year)),
                # no exact time, use 00:00:00
                death_datetime=strptime(p_info["dod"].values[0]),
                gender=p_info["gender"].values[0],
                ethnicity=p_info["race"].values[0],
                anchor_year_group=p_info["anchor_year_group"].values[0],
            )
            
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
        """Helper function which parses diagnosis_icd table.

        Will be called in `self.parse_tables()`

        Docs:
            - diagnosis_icd: https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-IV does not provide specific timestamps in diagnoses_icd
                table, so we set it to None.
        """
        table = "diagnoses_icd"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icd_code": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "icd_code", "icd_version"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of diagnosis (per patient)
        def diagnosis_unit(p_id, p_info):
            events = []
            # iterate over each patient and visit
            for v_id, v_info in p_info.groupby("hadm_id"):
                for code, version in zip(v_info["icd_code"], v_info["icd_version"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary=f"ICD{version}CM",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: diagnosis_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_procedures_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses procedures_icd table.

        Will be called in `self.parse_tables()`

        Docs:
            - procedures_icd: https://mimic.mit.edu/docs/iv/modules/hosp/procedures_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-IV does not provide specific timestamps in procedures_icd
                table, so we set it to None.
        """
        table = "procedures_icd"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icd_code": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "icd_code", "icd_version"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of procedure (per patient)
        def procedure_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for code, version in zip(v_info["icd_code"], v_info["icd_version"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary=f"ICD{version}PROC",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    # update patients
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: procedure_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

        return patients

    def parse_prescriptions(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses prescriptions table.

        Will be called in `self.parse_tables()`

        Docs:
            - prescriptions: https://mimic.mit.edu/docs/iv/modules/hosp/prescriptions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "prescriptions"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"subject_id": str, "hadm_id": str, "ndc": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "ndc"])
        # sort by start date and end date
        df = df.sort_values(
            ["subject_id", "hadm_id", "starttime", "stoptime"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of prescription (per patient)
        def prescription_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for timestamp, code in zip(v_info["starttime"], v_info["ndc"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="NDC",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    # update patients
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: prescription_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

        return patients

    def parse_labevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses labevents table.

        Will be called in `self.parse_tables()`

        Docs:
            - labevents: https://mimic.mit.edu/docs/iv/modules/hosp/labevents/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "labevents"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "itemid": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "itemid"])
        # sort by charttime
        df = df.sort_values(["subject_id", "hadm_id", "charttime"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of labevent (per patient)
        def lab_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for timestamp, code in zip(v_info["charttime"], v_info["itemid"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="MIMIC4_ITEMID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: lab_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_hcpcsevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses hcpcsevents table.

        Will be called in `self.parse_tables()`

        Docs:
            - hcpcsevents: https://mimic.mit.edu/docs/iv/modules/hosp/hcpcsevents/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-IV does not provide specific timestamps in hcpcsevents
                table, so we set it to None.
        """
        table = "hcpcsevents"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "hcpcs_cd": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "hcpcs_cd"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of hcpcsevents (per patient)
        def hcpcsevents_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for code in v_info["hcpcs_cd"]:
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="MIMIC4_HCPCS_CD",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    # update patients
                    events.append(event)
            return events
            
        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: hcpcsevents_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        
        return patients
    
    
    def parse_discharge(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        table = "discharge" # hardcoded
        df = pd.read_csv(os.path.join(self.tables_dir[table], f"{table}.csv"), 
                         dtype={"subject_id": str, "hadm_id": str})
        df = df.dropna(subset=["subject_id", "hadm_id", "text", "charttime"])
        df = df.sort_values(["subject_id", "hadm_id"], ascending=True)
        group_df = df.groupby("subject_id")
        def discharge_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for text in v_info["text"]:
                    event = Event(
                        code=text,
                        table=table,
                        vocabulary="text",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(v_info["charttime"].values[0])
                    )
                    events.append(event)
            return events
        group_df = group_df.parallel_apply(
            lambda x: discharge_unit(x.subject_id.unique()[0], x)
        )
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients
    

    def transform_study_datetime(self, date_str, time_str):
        # Extract year, month, and day from date_str
        year = date_str[:4]
        month = date_str[4:6]
        day = date_str[6:]

        # Extract hours, minutes, and seconds from time_str
        time_parts = time_str.split('.')
        time_main = time_parts[0].zfill(6)
        hours = time_main[:2]
        minutes = time_main[2:4]
        seconds = time_main[4:]

        # Combine into the desired format
        formatted_datetime = f"{year}-{month}-{day} {hours}:{minutes}:{seconds}"
        
        return formatted_datetime


    def parse_cxr(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        table = "cxr"
        cxr_file = "mimic-cxr-2.0.0-metadata"
        # hardcoded 
        df = pd.read_csv(os.path.join(self.tables_dir[table], f"{cxr_file}.csv"),
                           dtype={"subject_id": str, "hadm_id": str})
        
        # combine date and time to create timestamp 
        df = df.dropna(subset=["subject_id", "study_id", "dicom_id"])
        df.StudyDate = df.StudyDate.astype(str)
        df.StudyTime = df.StudyTime.astype(str)
        # process all the dates and times
        df['StudyDateTime'] = df.apply(lambda row: self.transform_study_datetime(str(row['StudyDate']), str(row['StudyTime'])), axis=1)
        df = df.sort_values(["subject_id", "study_id"], ascending=True)
        
        group_df = df.groupby("subject_id")
        
        def cxr_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("study_id"):
                for dicom_id, timestamp in zip(v_info["dicom_id"], v_info["StudyDateTime"]):
                    # print(timestamp)
                    event = Event(
                        code=dicom_id, # used for the dicom_id pathing
                        table=table,
                        vocabulary="dicom_id",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp)
                    )
                    events.append(event)
            return events
        
        group_df = group_df.parallel_apply(lambda x: cxr_unit(x.subject_id.unique()[0], x))
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_tables(self) -> Dict[str, Patient]:
        """Parses the tables in `self.tables` and return a dict of patients.

        Will be called in `self.__init__()` if cache file does not exist or
            refresh_cache is True.

        This function will first call `self.parse_basic_info()` to parse the
        basic patient information, and then call `self.parse_[table_name]()` to
        parse the table with name `table_name`. Both `self.parse_basic_info()` and
        `self.parse_[table_name]()` should be implemented in the subclass.

        Returns:
           A dict mapping patient_id to `Patient` object.
        """
        pandarallel.initialize(progress_bar=False)

        patients: Dict[str, Patient] = dict()
        tic = time.time()
        patients = self.parse_basic_info(patients)
        print(f"finish basic patient information parsing : {time.time() - tic}s")

        for table in self.tables:
            try:
                tic = time.time()
                parse_method = getattr(self, f"parse_{table.lower()}")
                patients = parse_method(patients)
                print(f"finish parsing {table} : {time.time() - tic}s")
            except AttributeError:
                raise NotImplementedError(f"Parser for table {table} is not implemented yet.")

        return patients
    
    def stat(self) -> str:
        """Returns some statistics of the base dataset."""
        lines = list()
        lines.append("")
        lines.append(f"Statistics of base dataset (dev={self.dev}):")
        lines.append(f"\t- Dataset: {self.dataset_name}")
        lines.append(f"\t- Number of patients: {len(self.patients)}")
        # num_visits = [len(p) for p in self.patients.values()] # ask Zhenbang if it even makes sense to writ ea function like this?
        # lines.append(f"\t- Number of visits: {sum(num_visits)}")
        # lines.append(
        #     f"\t- Number of visits per patient: {sum(num_visits) / len(num_visits):.4f}"
        # )
        for table in self.tables:
            num_events = [
                len(p.get_event_list(table)) for p in self.patients.values()
            ]
            lines.append(
                f"\t- Number of events in {table}: "
                f"{sum(num_events) :.4f}"
            )
        lines.append("")
        print("\n".join(lines))
        return "\n".join(lines)
    
    @property
    def available_tables(self) -> List[str]:
        """Returns a list of available tables for the dataset.

        Returns:
            List of available tables.
        """
        tables = []
        for patient in self.patients.values():
            tables.extend(patient.available_tables)
        return list(set(tables))
    
    # util funcs
    def _add_events_to_patient_dict(
        self,
        patient_dict: Dict[str, Patient],
        group_df: pd.DataFrame,
    ) -> Dict[str, Patient]:
        """Helper function which adds the events column of a df.groupby object to the patient dict.

        Will be called at the end of each `self.parse_[table_name]()` function.

        Args:
            patient_dict: a dict mapping patient_id to `Patient` object.
            group_df: a df.groupby object, having two columns: patient_id and events.
                - the patient_id column is the index of the patient
                - the events column is a list of <Event> objects

        Returns:
            The updated patient dict.
        """
        for _, events in group_df.items():
            for event in events:

                patient_dict = self._add_event_to_patient_dict(patient_dict, event)
        return patient_dict
    
    @staticmethod
    def _add_event_to_patient_dict(
        patient_dict: Dict[str, Patient],
        event: Event,
    ) -> Dict[str, Patient]:
        """Helper function which adds an event to the patient dict.

        Will be called in `self._add_events_to_patient_dict`.

        Note that if the patient of the event is not in the patient dict, or the
        visit of the event is not in the patient, this function will do nothing.

        Args:
            patient_dict: a dict mapping patient_id to `Patient` object.
            event: an event to be added to the patient dict.

        Returns:
            The updated patient dict.
        """
        patient_id = event.patient_id
        try:
            patient_dict[patient_id].add_event(event)
        except KeyError:
            pass
        return patient_dict
    
    def _convert_code_in_event(self, event: Event) -> List[Event]:
        """Helper function which converts the code for a single event.

        Note that an event may be mapped to multiple events after code conversion.

        Will be called in `self._convert_code_in_patient()`.

        Args:
            event: an `Event` object.

        Returns:
            A list of `Event` objects after code conversion.
        """
        src_vocab = event.vocabulary
        if src_vocab in self.code_mapping:
            target = self.code_mapping[src_vocab]
            if isinstance(target, tuple):
                tgt_vocab, kwargs = target
                source_kwargs = kwargs.get("source_kwargs", {})
                target_kwargs = kwargs.get("target_kwargs", {})
            else:
                tgt_vocab = self.code_mapping[src_vocab]
                source_kwargs = {}
                target_kwargs = {}
            code_mapping_tool = self.code_mapping_tools[f"{src_vocab}_{tgt_vocab}"]
            mapped_code_list = code_mapping_tool.map(
                event.code, source_kwargs=source_kwargs, target_kwargs=target_kwargs
            )
            mapped_event_list = [deepcopy(event) for _ in range(len(mapped_code_list))]
            for i, mapped_event in enumerate(mapped_event_list):
                mapped_event.code = mapped_code_list[i]
                mapped_event.vocabulary = tgt_vocab
            
            # update the code vocs
            for key, value in self.code_vocs.items():
                if value == src_vocab:
                    self.code_vocs[key] = tgt_vocab

            return mapped_event_list
        # TODO: should normalize the code here
        return [event]
    

    def _convert_code_in_patient(self, patient: Patient) -> Patient:
        """Helper function which converts the codes for a single patient.

        Will be called in `self._convert_code_in_patient_dict()`.

        Args:
            patient:a `Patient` object.

        Returns:
            The updated `Patient` object.
        """
        # for visit in patient:
        #     for table in visit.available_tables:
        #         all_mapped_events = []
        #         for event in visit.get_event_list(table):
        all_mapped_events = []
        for event in patient.events:
            # an event may be mapped to multiple events after code conversion
            mapped_events: List[Event]
            mapped_events = self._convert_code_in_event(event)
            all_mapped_events.extend(mapped_events)
            # visit.set_event_list(table, all_mapped_events)
        patient.events = all_mapped_events
        return patient
    
    def _convert_code_in_patient_dict(
        self,
        patients: Dict[str, Patient],
    ) -> Dict[str, Patient]:
        """Helper function which converts the codes for all patients.

        The codes to be converted are specified in `self.code_mapping`.

        Will be called in `self.__init__()` after `self.parse_tables()`.

        Args:
            patients: a dict mapping patient_id to `Patient` object.

        Returns:
            The updated patient dict.
        """
        for p_id, patient in tqdm(patients.items(), desc="Mapping codes"):
            patients[p_id] = self._convert_code_in_patient(patient)
        return patients
    


def main():
    # dataset = MIMIC4Dataset(
    #     root="/srv/local/data/jw3/physionet.org/files/MIMIC-IV/2.0/hosp",
    #     tables=["diagnoses_icd","procedures_icd"],
    #     code_mapping={"NDC": "ATC"},
    #     refresh_cache=False,
    #     dev=False,
    #     additional_dirs={"discharge" : "/srv/local/data/jw3/physionet.org/files/MIMIC-IV/2.0/note",
    #                      "cxr" : "/srv/local/data/jw3/physionet.org/files/MIMIC-CXR"}
    # )
    # dataset.stat()

    # print(dataset.available_tables)
    task = MIMIC4ICD9Coding(dataset=None, refresh_cache=False)
    print(len(task.samples))
    print(task.samples[0]["icd_codes"])
    # task.process()
    
    # sample_dataset = task.to_torch_dataset() 

if __name__ == "__main__":
    main()
