# -*- coding: utf-8 -*-
"""CMS dataset handling
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause
import os

import pandas as pd
import numpy as np

from tqdm import tqdm

from pyhealth.data.base_cms import CMS_Data

from pyhealth.utils.utility import read_csv_to_df
from pyhealth.utils.utility import read_excel_to_df

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def get_non_zeros_rows(r):
    """Utility function to return non-zero rows
    """
    return r[~np.all(r == 0, axis=1)]


def padding_right(pd_series, n_digits):
    """Padding zeros to the right of a pandas series

    Parameters
    ----------
    pd_series
    n_digits

    Returns
    -------

    """
    str_format = '{:<0' + str(n_digits) + '}'
    return pd_series.apply(lambda x: str_format.format(x))


if __name__ == "__main__":
    n_jobs = 6  # number of parallel jobs
    save_dir = os.path.join('outputs', 'cms')

    # one may use fewer procedures if preferred
    procedure_cols = ['icd9_prcdr_cd_1', 'icd9_prcdr_cd_2',
                      'icd9_prcdr_cd_3', 'icd9_prcdr_cd_4',
                      'icd9_prcdr_cd_5', 'icd9_prcdr_cd_6', ]

    diagnosis_cols = ['icd9_dgns_cd_1', 'icd9_dgns_cd_2', 'icd9_dgns_cd_3',
                      'icd9_dgns_cd_4', 'icd9_dgns_cd_5', 'icd9_dgns_cd_6',
                      'icd9_dgns_cd_7', 'icd9_dgns_cd_8', 'icd9_dgns_cd_9',
                      'icd9_dgns_cd_10', ]

    patient_data_loc = 'cms_patient_data.json'
    patient_list_loc = 'cms_patient_list.json'

    valid_data_list = []  # keep tracking the stored data
    valid_id_list = []  # keep tracking a list of patient IDs#

    valid_sequence_list = []

    # read in tables
    patient_df = read_csv_to_df(
        os.path.join('data', 'cms-sample-1',
                     'DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv'))
    patient_id_list = patient_df['desynpuf_id'].tolist()
    # change the format of the date
    patient_df['dob'] = pd.to_datetime(patient_df['bene_birth_dt'],
                                       format='%Y%m%d')

    event_df = read_csv_to_df(
        os.path.join('data', 'cms-sample-1',
                     'DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv'))
    event_df['icd9_prcdr_cd_1'] = event_df['icd9_prcdr_cd_1'].astype('Int64')
    event_df['icd9_prcdr_cd_1'] = event_df['icd9_prcdr_cd_1'].astype(str)

    # change the format of the date
    event_df['clm_from_dt'] = pd.to_datetime(event_df['clm_from_dt'],
                                             format='%Y%m%d')
    event_df['clm_thru_dt'] = pd.to_datetime(event_df['clm_thru_dt'],
                                             format='%Y%m%d')
    ##########################################################################
    # read ICD-9 code
    # define datatype to reduce the memory cost
    dtypes_dict = {
        'PROCEDURE CODE': 'str',
        'LONG DESCRIPTION': 'str',
        'SHORT DESCRIPTION': 'str',
    }
    event_mapping_df = read_excel_to_df(
        os.path.join('resources', 'CMS28_DESC_LONG_SHORT_SG.xlsx'),
        dtype=dtypes_dict)

    # padding zeros to the right -> ICD 9 is 5 digit
    event_mapping_df['procedure code full'] = padding_right(
        event_mapping_df['procedure code'], 5)

    # remove the leading zeros for the join with event df
    event_mapping_df['procedure code cleaned'] = event_mapping_df[
        'procedure code'].str.lstrip('0')

    # use the first three bits for joining
    event_mapping_df['procedure code short'] = \
        event_mapping_df['procedure code full'].str[:3]

    # find the unique 3-bit codes
    unique_counts = event_mapping_df['procedure code short'].value_counts(
        normalize=True, dropna=True)
    unique_counts_df = pd.DataFrame(
        [unique_counts.index, unique_counts.values],
        index=['procedure', 'frequency']).transpose()
    unique_counts_df.sort_values(by='procedure', inplace=True)

    top_procedures = unique_counts_df['procedure'].tolist()
    actual_n_procedures = len(top_procedures)

    procedure_dict = {}
    for i in range(actual_n_procedures):
        procedure_dict[top_procedures[i]] = i
    ########################################################################
    # read ICD-9 diagnosis code
    # define datatype to reduce the memory cost
    dtypes_dict = {
        'DIAGNOSIS CODE': 'str',
        'LONG DESCRIPTION': 'str',
        'SHORT DESCRIPTION': 'str',
    }
    diagnosis_mapping_df = read_excel_to_df(
        os.path.join('resources', 'CMS28_DESC_LONG_SHORT_DX.xlsx'),
        dtype=dtypes_dict)

    # padding zeros to the right -> ICD 9 is 5 digit
    diagnosis_mapping_df['diagnosis code full'] = padding_right(
        diagnosis_mapping_df['diagnosis code'], 5)

    # remove the leading zeros for the join with event df
    diagnosis_mapping_df['diagnosis code cleaned'] = diagnosis_mapping_df[
        'diagnosis code'].str.lstrip('0')

    # use the first three bits for joining
    diagnosis_mapping_df['diagnosis code short'] = \
        diagnosis_mapping_df['diagnosis code full'].str[:3]

    # find the unique 3-bit codes
    unique_counts = diagnosis_mapping_df['diagnosis code short'].value_counts(
        normalize=True, dropna=True)
    unique_counts_df = pd.DataFrame(
        [unique_counts.index, unique_counts.values],
        index=['diagnosis', 'frequency']).transpose()
    unique_counts_df.sort_values(by='diagnosis', inplace=True)

    top_diagnosis = unique_counts_df['diagnosis'].tolist()
    actual_n_diagnosis = len(top_diagnosis)

    diagnosis_dict = {}
    for i in range(actual_n_diagnosis):
        diagnosis_dict[top_diagnosis[i]] = i

    ########################################################################

    y_phenotyping = []
    y_mortality = []
    y_files = []
    # patient_id_list = patient_id_list[0:1000]
    patient_id_list = patient_id_list[0:116352]
    for i in tqdm(range(len(patient_id_list))):
        p_id = patient_id_list[i]
        # print('Processing Patient', i + 1, p_id)
        # initialize the
        temp_data = CMS_Data(p_id, procedure_cols, diagnosis_cols)
        p_df = patient_df.loc[patient_df['desynpuf_id'] == p_id]
        e_df = event_df.loc[event_df['desynpuf_id'] == p_id]

        print(i, p_id)

        if not p_df.empty:
            if p_df.shape[0] > 1:
                raise ValueError("Patient ID cannot be repeated")
            temp_data.parse_patient(p_df)

        if not e_df.empty:
            temp_data.parse_admission(e_df)
            temp_data.parse_event(e_df, event_mapping_df, top_procedures,
                                  procedure_dict,
                                  save_dir=os.path.join(save_dir, 'raw'))
        else:
            print('no inpatient claim found')

        # only generating y if the csv file exists
        if 'episode_csv' in temp_data.data.keys():
            y_phenotyping.append(
                temp_data.generate_phenotyping(e_df, diagnosis_mapping_df,
                                               top_diagnosis,
                                               diagnosis_dict))
            y_mortality.append(temp_data.data['death_indicator'])
            y_files.append(temp_data.data['episode_csv'])

        valid_data_list.append(temp_data)
        valid_id_list.append(p_id)

    y_phenotyping = np.asarray(y_phenotyping).reshape(len(y_phenotyping),
                                                      len(top_diagnosis))

    # make saving directory if needed
    if not os.path.isdir(os.path.join(save_dir, 'phenotyping')):
        os.makedirs(os.path.join(save_dir, 'phenotyping'))

    # add the files
    y = np.c_[y_files, y_phenotyping]
    output_headers = ['episode_file']
    output_headers.extend(top_diagnosis)
    pd.DataFrame(y, columns=output_headers).to_csv(
        os.path.join(save_dir, 'phenotyping', 'y_phenotyping_cms.csv'),
        index=False)

    # make saving directory if needed
    if not os.path.isdir(os.path.join(save_dir, 'mortality')):
        os.makedirs(os.path.join(save_dir, 'mortality'))
    y_mortality = np.c_[y_files, y_mortality]
    output_headers = ['episode_file', 'death_indicator']
    pd.DataFrame(y_mortality, columns=output_headers).to_csv(
        os.path.join(save_dir, 'mortality', 'y_mortality_cms.csv'),
        index=False)
