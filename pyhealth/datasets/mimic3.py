import os
from typing import Optional, List, Dict, Tuple, Union
import time
import pandas as pd
from pandarallel import pandarallel 
from tqdm import tqdm
from pyhealth.data.data_v2 import Event, Patient
# from pyhealth.datasets import BaseEHRDataset
from pyhealth.datasets.base_dataset_v2 import BaseDataset
from pyhealth.tasks.medical_coding import MIMIC3ICD9Coding
from pyhealth.datasets.utils import strptime
from pyhealth.medcode import CrossMap
from copy import deepcopy
from pyhealth.datasets.utils import MODULE_CACHE_PATH, DATASET_BASIC_TABLES
from pyhealth.datasets.utils import hash_str
# TODO: add other tables


class MIMIC3Dataset(BaseDataset):    
    """Base dataset for MIMIC-III dataset.

    The MIMIC-III dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://mimic.physionet.org/.

    The basic information is stored in the following tables:
        - PATIENTS: defines a patient in the database, SUBJECT_ID.
        - ADMISSIONS: defines a patient's hospital admission, HADM_ID.

    We further support the following tables:
        - DIAGNOSES_ICD: contains ICD-9 diagnoses (ICD9CM code) for patients.
        - PROCEDURES_ICD: contains ICD-9 procedures (ICD9PROC code) for patients.
        - PRESCRIPTIONS: contains medication related order entries (NDC code)
            for patients.
        - LABEVENTS: contains laboratory measurements (MIMIC3_ITEMID code)
            for patients
        - NOTEEVENTS: contains discharge summaries 

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
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> dataset = MIMIC3Dataset(
        ...         root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        ...         tables=["DIAGNOSES_ICD", "PRESCRIPTIONS"],
        ...         code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """
    def __init__(self, root: str, 
                 dataset_name: Optional[str] = [], 
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
    
    def process(self):
        # process the data
        patients = self.parse_tables()
        # convert codes
        patients = self._convert_code_in_patient_dict(patients)
        return patients

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
        patients_df = pd.read_csv(
            os.path.join(self.root, "PATIENTS.csv"),
            dtype={"SUBJECT_ID": str},
            nrows=1000 if self.dev else None,
        )
        # read admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "ADMISSIONS.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str},
        )
        # merge patient and admission tables
        df = pd.merge(patients_df, admissions_df, on="SUBJECT_ID", how="inner")
        # sort by admission and discharge time
        df = df.sort_values(["SUBJECT_ID", "ADMITTIME", "DISCHTIME"], ascending=True)
        # group by patient
        df_group = df.groupby("SUBJECT_ID")

        # parallel unit of basic information (per patient)
        def basic_unit(p_id, p_info):
            patient = Patient(
                patient_id=p_id,
                birth_datetime=strptime(p_info["DOB"].values[0]),
                death_datetime=strptime(p_info["DOD_HOSP"].values[0]),
                gender=p_info["GENDER"].values[0],
                ethnicity=p_info["ETHNICITY"].values[0],
            )
            # Remove the loop that created Visit objects
            return patient

        # parallel apply,
        df_group = df_group.parallel_apply(
            lambda x: basic_unit(x.SUBJECT_ID.unique()[0], x)
        )
        # summarize the results
        for pat_id, pat in df_group.items():
            patients[pat_id] = pat

        return patients

    def parse_diagnoses_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses DIAGNOSES_ICD table.

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
        table = "DIAGNOSES_ICD"
        self.code_vocs["conditions"] = "ICD9CM"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit of diagnosis (per patient)
        def diagnosis_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for code in v_info["ICD9_CODE"]:
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="ICD9CM",
                        visit_id=v_id,
                        patient_id=p_id
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: diagnosis_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_procedures_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PROCEDURES_ICD table.

        Will be called in `self.parse_tables()`

        Docs:
            - PROCEDURES_ICD: https://mimic.mit.edu/docs/iii/tables/procedures_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-III does not provide specific timestamps in PROCEDURES_ICD
                table, so we set it to None.
        """
        table = "PROCEDURES_ICD"
        self.code_vocs["procedures"] = "ICD9PROC"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit of procedure (per patient)
        def procedure_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for code in v_info["ICD9_CODE"]:
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="ICD9PROC",
                        visit_id=v_id,
                        patient_id=p_id
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: procedure_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_prescriptions(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PRESCRIPTIONS table.

        Will be called in `self.parse_tables()`

        Docs:
            - PRESCRIPTIONS: https://mimic.mit.edu/docs/iii/tables/prescriptions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "PRESCRIPTIONS"
        self.code_vocs["drugs"] = "NDC"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "NDC": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "NDC"])
        # sort by start date and end date
        df = df.sort_values(
            ["SUBJECT_ID", "HADM_ID", "STARTDATE", "ENDDATE"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit for prescription (per patient)
        def prescription_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for timestamp, code in zip(v_info["STARTDATE"], v_info["NDC"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="NDC",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp)
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: prescription_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_labevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses LABEVENTS table.

        Will be called in `self.parse_tables()`

        Docs:
            - LABEVENTS: https://mimic.mit.edu/docs/iii/tables/labevents/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "LABEVENTS"
        self.code_vocs["labs"] = "MIMIC3_ITEMID"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ITEMID": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "ITEMID"])
        # sort by charttime
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "CHARTTIME"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit for lab (per patient)
        def lab_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for timestamp, code in zip(v_info["CHARTTIME"], v_info["ITEMID"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="MIMIC3_ITEMID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp)
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: lab_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_noteevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses NOTEEVENTS table.

        Will be called in `self.parse_tables()`

        Docs:
            - NOTEEVENTS: https://mimic.mit.edu/docs/iii/tables/noteevents/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "NOTEEVENTS"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "TEXT"])
        # sort by charttime
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "CHARTTIME"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit for note (per patient)
        def note_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for _, row in v_info.iterrows():
                    event = Event(
                        code=row["TEXT"],
                        table=table,
                        vocabulary="note",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(row["CHARTTIME"])
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: note_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients


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


def main():
    root = "/srv/local/data/jw3/physionet.org/files/mimic-iii-clinical-database-1.4"
    # dataset = MIMIC3Dataset(
    #     root=root,
    #     dataset_name="mimic3",  
    #     tables=[
    #         "DIAGNOSES_ICD",
    #         "PROCEDURES_ICD",
    #         "PRESCRIPTIONS",
    #         "LABEVENTS",
    #         "NOTEEVENTS"
    #     ],
    #     code_mapping={"NDC": "ATC"},
    #     dev=False,
    #     refresh_cache=False,
    # )
    # dataset.stat()
    mimic3_coding = MIMIC3ICD9Coding(refresh_cache= False)
    print(len(mimic3_coding.samples))
    sample_dataset = mimic3_coding.to_torch_dataset()
    # print(sample_dataset[0])
    



if __name__ == "__main__":
    main()
