import os
from typing import Optional, List, Dict, Tuple, Union

import pandas as pd
import numpy as np

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseNoteDataset
from pyhealth.datasets.utils import strptime


class MIMIC3NoteDataset(BaseNoteDataset):
    """
    TODO: add docs
    """

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PATIENTS and ADMISSIONS tables.

        Will be called in `self.parse_tables()`

        Docs:
            - PATIENTS: https://mimic.mit.edu/docs/iii/tables/patients/
            - ADMISSIONS: https://mimic.mit.edu/docs/iii/tables/admissions/
            - NOTEEVENTS: https://mimic.mit.edu/docs/iii/tables/noteevents/

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
        # read noteevents table
        noteevents_df = pd.read_csv(
            os.path.join(self.root, "NOTEEVENTS.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str},
        )
        
        admissions_df.ADMITTIME = pd.to_datetime(admissions_df.ADMITTIME, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        admissions_df.DISCHTIME = pd.to_datetime(admissions_df.DISCHTIME, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        admissions_df.DEATHTIME = pd.to_datetime(admissions_df.DEATHTIME, format='%Y-%m-%d %H:%M:%S', errors='coerce') 
        
        admissions_df = admissions_df.sort_values(['SUBJECT_ID', 'ADMITTIME'])
        admissions_df = admissions_df.reset_index(drop=True)
        admissions_df['NEXT_ADMITTIME'] = admissions_df.groupby('SUBJECT_ID').ADMITTIME.shift(-1)
        admissions_df['NEXT_ADMISSION_TYPE'] = admissions_df.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)  
        
        rows = admissions_df.NEXT_ADMISSION_TYPE == 'ELECTIVE'
        admissions_df.loc[rows, 'NEXT_ADMITTIME'] = pd.NaT
        admissions_df.loc[rows, 'NEXT_ADMISSION_TYPE'] = np.NaN

        admissions_df = admissions_df.sort_values(['SUBJECT_ID', 'ADMITTIME'])
        admissions_df[['NEXT_ADMITTIME', 'NEXT_ADMISSION_TYPE']] = \
            admissions_df.groupby(['SUBJECT_ID'])[['NEXT_ADMITTIME', 'NEXT_ADMISSION_TYPE']].fillna(method='bfill')
        admissions_df['DAYS_NEXT_ADMIT'] = (admissions_df.NEXT_ADMITTIME - admissions_df.DISCHTIME).dt.total_seconds() / (24 * 60 * 60)
        admissions_df['OUTPUT_LABEL'] = (admissions_df.DAYS_NEXT_ADMIT < 30).astype('int')

        # filter out newborn and death
        admissions_df = admissions_df[admissions_df['ADMISSION_TYPE'] != 'NEWBORN']
        admissions_df = admissions_df[admissions_df.DEATHTIME.isnull()]
        admissions_df['DURATION'] = (admissions_df['DISCHTIME'] - admissions_df['ADMITTIME']).dt.total_seconds() / (24 * 60 * 60)
        
        noteevents_df = noteevents_df.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE'])
        
        # merge admission and noteevents tables
        admission_notes_df = pd.merge(
            admissions_df[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DAYS_NEXT_ADMIT', 'NEXT_ADMITTIME',
                    'ADMISSION_TYPE', 'DEATHTIME', 'OUTPUT_LABEL', 'DURATION']],
            noteevents_df[['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'TEXT', 'CATEGORY']],
            on=['SUBJECT_ID', 'HADM_ID'], how='left'
        )

        admission_notes_df['ADMITTIME_C'] = admission_notes_df.ADMITTIME.apply(lambda x: str(x).split(' ')[0])
        admission_notes_df['ADMITTIME_C'] = pd.to_datetime(admission_notes_df.ADMITTIME_C, format='%Y-%m-%d', errors='coerce')
        admission_notes_df['CHARTDATE'] = pd.to_datetime(admission_notes_df.CHARTDATE, format='%Y-%m-%d', errors='coerce')
        
        # merge patient and admission_noteevents tables
        df = pd.merge(patients_df, admission_notes_df, on="SUBJECT_ID", how="inner")
        
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
            )
            return patient

        # parallel apply
        df_group = df_group.parallel_apply(
            lambda x: basic_unit(x.SUBJECT_ID.unique()[0], x)
        )
        # summarize the results
        for pat_id, pat in df_group.items():
            patients[pat_id] = pat

        return patients