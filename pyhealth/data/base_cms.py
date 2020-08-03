# -*- coding: utf-8 -*-
"""Base class for CMS dataset
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause
import os

import pandas as pd
import numpy as np

from .base import Standard_Template

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class CMS_Data(Standard_Template):
    """The data template to store CMS data. Customized fields can be added
    in each parse_xxx methods.

    Parameters
        ----------
        patient_id : str
            Unique identifier for a patient.
    """

    def __init__(self, patient_id, procudure_cols, diagnosis_cols):
        super(CMS_Data, self).__init__(patient_id=patient_id)
        self.procudure_cols = procudure_cols
        self.diagnosis_cols = diagnosis_cols

    def parse_patient(self, pd_series):
        self.data['gender'] = pd_series['bene_sex_ident_cd'].values[0]
        self.data['dob'] = pd_series['dob'].values[0]
        # whether there is a death date associated with it
        self.data['death_indicator'] = int(
            ~pd.isna(pd_series['bene_death_dt']).values[0])

    def parse_admission(self, pd_df, mapping_dict=None):
        # TODO: implement the mapping dict
        for ind, row in pd_df.iterrows():
            # each admission is stored as a seperate dictionary and
            # added to admission_list
            admission_event = {}
            admission_event['admission_id'] = row['clm_id']
            admission_event['admission_date'] = row['clm_from_dt']
            admission_event['discharge_date'] = row['clm_thru_dt']
            # more elements can be added here
            self.data['admission_list'].append(admission_event)

    def generate_phenotyping(self, pd_df, diagnosis_mapping_df,
                             diagnosis_codes, diagnosis_dict):
        if len(self.data['admission_list']) == 0:
            raise ValueError(
                "No admission information found. Parse admission info first.")
        # fine the last row
        last_claim = pd_df.tail(1)
        # print(last_claim)

        phenotyping_list = np.zeros([len(diagnosis_codes), 1])
        # select the procedures
        diag_df = last_claim[self.diagnosis_cols].transpose()
        diag_df.columns = ['event diagnosis code']
        # print(diag_df.shape)

        # here we join with the event mapping to receive the short code
        diag_df = diag_df.merge(diagnosis_mapping_df,
                                left_on='event diagnosis code',
                                right_on='diagnosis code cleaned')
        # print(diag_df)

        # mark the specific entry as 1 if presented
        for idx, row in diag_df.iterrows():
            # print(row['diagnosis code short'])
            phenotyping_list[diagnosis_dict[row['diagnosis code short']]] = 1

        return phenotyping_list

    def parse_event(self, pd_df, event_mapping_df, procedure_codes,
                    procedure_dict, save_dir=''):
        if len(self.data['admission_list']) == 0:
            raise ValueError(
                "No admission information found. Parse admission info first.")

        # two claims are needed for generating valid sequence
        if len(self.data['admission_list']) < 2:
            print('Only one claim find')
            return

        # make saving directory if needed
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # calculate the first entry
        pd_df = pd_df.sort_values('clm_from_dt')
        first_index = pd_df['clm_from_dt'].index[0]

        pd_df['first_entry'] = pd.to_datetime(
            pd_df['clm_from_dt'][first_index])

        pd_df['days_since_entry'] = (pd_df['clm_from_dt'] - pd_df[
            'first_entry']).dt.days

        # initialize the matrix for storing the scores
        event_mat = np.zeros(
            [len(self.data['admission_list']), len(procedure_codes)])
        print(event_mat.shape)

        for i, admission_event in enumerate(self.data['admission_list']):
            print(self.data['patient_id'], admission_event['admission_id'])

            temp_df = pd_df.loc[
                pd_df['clm_id'] == admission_event['admission_id']]

            # select the procedures
            if temp_df[self.procudure_cols].shape[0] > 1:
                print('Duplicate claim ID')
                return

            proc_df = temp_df[self.procudure_cols].transpose()
            proc_df.columns = ['event procedure code']
            # print(proc_df.shape)

            # here we join with the event mapping to receive the short code
            proc_df = proc_df.merge(event_mapping_df,
                                    left_on='event procedure code',
                                    right_on='procedure code cleaned')

            # mark the specific entry as 1 if presented
            for idx, row in proc_df.iterrows():
                # print(row['procedure code short'])
                event_mat[i, procedure_dict[row['procedure code short']]] = 1

            self.data['admission_list'][i] = admission_event
        # print(event_mat, np.sum(event_mat))

        # remove all zero rows, we keep it advised by Zhi
        # event_mat = get_non_zeros_rows(event_mat)

        # skip if it is too sparse
        if np.sum(event_mat) < 2:
            print(
                'Too sparse. Fewer than two procedures in the predefined list')
            return
        # elif event_mat.shape[0] < 2:
        #     print('Only one claim left after pruning')
        #     return
        else:
            self.data['episode_csv'] = str(self.data['patient_id']) + '.csv'

            # add the timestamp
            event_mat = np.c_[pd_df['days_since_entry'], event_mat]

            pd.DataFrame(event_mat).to_csv(
                os.path.join(save_dir, self.data['episode_csv']),
                index=False, header=False)
        return
