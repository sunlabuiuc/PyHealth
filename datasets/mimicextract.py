import os
from typing import Optional, List, Dict, Tuple, Union

import pandas as pd

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseEHRDataset
from pyhealth.datasets.utils import strptime

# TODO: add other tables

class MIMICExtractDataset(BaseEHRDataset):
    """Base dataset for MIMIC-Extract dataset.

    Reads the HDF5 data produced by 
    [MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract#step-4-set-cohort-selection-and-extraction-criteria).
    Works with files created with or without LEVEL2 grouping and with restricted cohort population
    sizes, other optional parameter values, and should work with many customized versions of the pipeline.

    You can create or obtain a MIMIC-Extract dataset in several ways:

    * The default chort dataset is [available on GCP](https://console.cloud.google.com/storage/browser/mimic_extract)
      (requires PhysioNet access provisioned in GCP).
    * Follow the [step-by-step instructions](https://github.com/MLforHealth/MIMIC_Extract#step-by-step-instructions)
      on the MIMIC_Extract github site, which includes setting up a PostgreSQL database and loading
      the MIMIC-III data files.
    * Use the instructions at [MIMICExtractEasy](https://github.com/SphtKr/MIMICExtractEasy) which uses DuckDB
      instead and should be a good bit simpler.
    
    Any of these methods will provide you with a set of HDF5 files containing a cleaned subset of the MIMIC-III dataset.
    This class can be used to read that dataset (mainly the `all_hourly_data.h5` file). Consult the MIMIC-Extract
    documentation for all the options available for dataset generation (cohort selection, aggregation level, etc.).

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain one or more HDF5 files).
        tables: list of tables to be loaded (e.g., ["vitals_labs", "interventions"]). 
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
        itemid_to_variable_map: Path to the CSV file used for aggregation mapping during
            your dataset's creation. Probably the one located in the MIMIC-Extract
            repo at `resources/itemid_to_variable_map.csv`, or your own version if you
            have customized it.

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
        #is_icustay_visit: Optional[bool] = False #TODO: implement fully
    ):
        if pop_size is not None:
            self._fname_suffix = f"_{pop_size}"
        else:
            self._fname_suffix = ""
        self._ahd_filename = os.path.join(root, f"all_hourly_data{self._fname_suffix}.h5")
        self._c_filename = os.path.join(root, f"C{self._fname_suffix}.h5")
        self._notes_filename = os.path.join(root, f"all_hourly_data{self._fname_suffix}.hdf")
        self._v_id_column = 'hadm_id' #'icustay_id' if is_icustay_visit else 'hadm_id'

        # This could be implemented with MedCode.CrossMap, however part of the idea behind
        # MIMIC-Extract is that the user can customize this mapping--therefore we will
        # make a map specific to this dataset instance based on a possibly-customized CSV.
        self._vocab_map = { "chartevents": {}, "labevents": {}, "vitals_labs": {} }
        if itemid_to_variable_map is not None:
            # We are just going to read some metadata here...
            df_ahd = pd.read_hdf(self._ahd_filename, 'vitals_labs')
            grptype = "LEVEL1" if "LEVEL1" in df_ahd.columns.names else "LEVEL2"

            itemid_map = pd.read_csv(itemid_to_variable_map)
            for linksto, dict in self._vocab_map.items():
                df = itemid_map
                if linksto != 'vitals_labs':
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
        """Helper function which parses `patients` dataset (within `all_hourly_data.h5`)

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
        """Helper function which parses the `C` (ICD9 diagnosis codes) dataset (within `C.h5`) in
          a way compatible with MIMIC3Dataset.

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
        """Helper function which parses the `C` (ICD9 diagnosis codes) dataset (within `C.h5`).

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
        """Helper function which parses the `vitals_labs` dataset (within `all_hourly_data.h5`)
        in a way compatible with MIMIC3Dataset.

        Features in `vitals_labs` are corellated with MIMIC-III ITEM_ID values, and those ITEM_IDs
        that correspond to LABEVENTS table items in raw MIMIC-III will be
        added as events. This corellation depends on the contents of the provided `itemid_to_variable_map.csv`
        file. Note that this will likely *not* match the raw MIMIC-III data because of the
        harmonization/aggregation done by MIMIC-Extract.

        See also `self.parse_vitals_labs()` 

        Will be called in `self.parse_tables()`

        Docs:
            - LABEVENTS: https://mimic.mit.edu/docs/iii/tables/labevents/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "LABEVENTS"
        return self._parse_vitals_labs(patients=patients, table=table)

    def parse_chartevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses the `vitals_labs` dataset (within `all_hourly_data.h5`)
        in a way compatible with MIMIC3Dataset.

        Features in `vitals_labs` are corellated with MIMIC-III ITEM_ID values, and those ITEM_IDs
        that correspond to CHARTEVENTS table items in raw MIMIC-III will be
        added as events. This corellation depends on the contents of the provided `itemid_to_variable_map.csv`
        file. Note that this will likely *not* match the raw MIMIC-III data because of the
        harmonization/aggregation done in MIMIC-Extract. 

        Will be called in `self.parse_tables()`

        Docs:
            - CHARTEVENTS: https://mimic.mit.edu/docs/iii/tables/chartevents/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "CHARTEVENTS"
        return self._parse_vitals_labs(patients=patients, table=table)

    def parse_vitals_labs(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses the `vitals_labs` dataset (within `all_hourly_data.h5`). 

        Events are added using the `MIMIC3_ITEMID` vocabulary, and the mapping is determined by the
        CSV file passed to the constructor in `itemid_to_variable_map`. Since MIMIC-Extract aggregates
        like events, only a single MIMIC-III ITEMID will be used to represent all like items in the
        MIMIC-Extract dataset--so the data here will likely *not* match raw MIMIC-III data. Which ITEMIDs are
        used depends on the aggregation level in your dataset (i.e. whether you used `--no_group_by_level2`).

        Will be called in `self.parse_tables()`

        See also `self.parse_chartevents()` and `self.parse_labevents()`

        Docs:
            - https://github.com/MLforHealth/MIMIC_Extract#step-4-set-cohort-selection-and-extraction-criteria

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "vitals_labs"
        return self._parse_vitals_labs(patients=patients, table=table)

    def _parse_vitals_labs(self, patients: Dict[str, Patient], table: str = 'vitals_labs') -> Dict[str, Patient]:
        linksto = table.lower()
        # read table
        df = pd.read_hdf(self._ahd_filename, 'vitals_labs')
        # drop records of the other patients
        df = df.loc[(list(patients.keys()),slice(None),slice(None)),:]

        # parallel unit for lab (per patient)
        captured_v_id_column = self._v_id_column
        captured_vocab_map = self._vocab_map[linksto]
        def vl_unit(p_id, p_info):
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
                        mean=e_info['mean'], # Value is not stored in MIMIC3Dataset... why? Unit uncertainty?
                        #TODO: Units, somewhere?
                        count=e_info['count'],
                        std=e_info['std']
                    )
                    events.append(event)
            return events

        ahd_index = ["subject_id","hadm_id","icustay_id","hours_in"]

        df.columns = df.columns.values # Collapse MultiIndex to tuples!
        is_level1 = True if(len(df.columns[0]) == 5) else False

        # drop columns not applicable to the wanted table...
        if is_level1:
            df = df.drop(columns=[col for col in df.columns if col[2] not in self._vocab_map[linksto]])
        else:
            df = df.drop(columns=[col for col in df.columns if col[0] not in self._vocab_map[linksto]])

        # "melt" down to a per-event representation...
        df = df.reset_index().melt(id_vars=ahd_index).dropna()
        if is_level1:
            _,_,df['variable'],_,df['Aggregation Function'] = zip(*df['variable'])
        else:
            df['variable'],df['Aggregation Function'] = zip(*df['variable'])

        # Discard count == 0.0 rows
        df = df.loc[(df['Aggregation Function'] != 'count') | (df['value'] != 0.0)]
        df = df.drop_duplicates()

        # Manual/brute force "pivot", as I can't get pivot functions to work right with the MultiIndex columns...
        df = df.reset_index().sort_values(ahd_index+['variable']).set_index(ahd_index+['variable'])
        df_mean = df.loc[df['Aggregation Function'] == 'mean'].rename(columns={"value":"mean"})['mean']
        df_count = df.loc[df['Aggregation Function'] == 'count'].rename(columns={"value":"count"})['count']
        df_std = df.loc[df['Aggregation Function'] == 'std'].rename(columns={"value":"std"})['std']
        if is_level1:
            #FIXME: Duplicates appear in the LEVEL1 representation... this is puzzling.
            # These should all be almost equal, or there is a significant problem.
            # For now, take some mean, and the highest count and std... though they 
            # may not match. LEVEL1 representation is usually not preferred anyway.
            # In theory, these should probably be aggregated??
            df_mean = df_mean[~df_mean.index.duplicated(keep='first')]
            df_count = df_count.sort_values(ascending=False)
            df_count = df_count[~df_count.index.duplicated(keep='first')]
            df_std = df_std.sort_values(ascending=False)
            df_std = df_std[~df_std.index.duplicated(keep='first')]
        df = pd.concat([df_mean, df_count, df_std], axis=1)
        df = df.reset_index().sort_values(ahd_index+['variable'])

        # reconstruct nominal timestamps for hours_in values...
        df_p = pd.read_hdf(self._ahd_filename, 'patients')
        df_p = df_p.loc[(list(patients.keys()),slice(None),slice(None)),:][['intime']]
        df = df.merge(df_p, on=['subject_id','hadm_id','icustay_id'], how="left")
        df['timestamp'] = df['intime'].dt.ceil('H')+pd.to_timedelta(df['hours_in'], unit="H")

        group_df = df.groupby("subject_id")

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: vl_unit(x.subject_id.unique()[0], x)
        )
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_interventions(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses the `interventions` dataset (within `all_hourly_data.h5`). 
        Events are added using the `MIMIC3_ITEMID` vocabulary, using a manually derived mapping corresponding to
        general items descriptive of the intervention. Since the raw MIMIC-III data had multiple codes, and 
        MIMIC-Extract aggregates like items, these will not match raw MIMIC-III data.
        
        In particular, note
        that ITEMID 41491 ("fluid bolus") is used for `crystalloid_bolus` and ITEMID 46729 ("Dextran") is used 
        for `colloid_bolus` because there is no existing general ITEMID for colloid boluses.

        Will be called in `self.parse_tables()`

        Docs:
            - https://github.com/MLforHealth/MIMIC_Extract#step-4-set-cohort-selection-and-extraction-criteria

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
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
        root="../mimic3demo/grouping",
        tables=[
            #"DIAGNOSES_ICD",
            "C",
            #"LABEVENTS",
            #"CHARTEVENTS",
            "vitals_labs",
            "interventions",
        ],
        dev=True,
        refresh_cache=True,
        itemid_to_variable_map='../MIMIC_Extract/resources/itemid_to_variable_map.csv'
    )
    dataset.stat()
    dataset.info()
