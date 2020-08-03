# -*- coding: utf-8 -*-
"""Mortality Prediction Label Generation
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause
import os

import pandas as pd
import json
from tqdm import tqdm

import sys

# temporary solution for relative imports in case combo is not installed
# if combo is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyhealth.utils.utility import make_dirs_if_not_exists
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == "__main__":
    save_dir = os.path.join('outputs', 'mimic_demo', 'y_data')

    make_dirs_if_not_exists(save_dir)
    patient_data_loc = os.path.join('outputs', 'mimic_demo', 'raw',
                                    'patient_data_demo.json')

    with open(patient_data_loc) as f:
        patient_data_list = json.load(f)

    ##########################################################
    output_headers = ['episode_file', 'death_indicator']

    output_df = pd.DataFrame(columns=output_headers)

    # for i, p_id in tqdm(enumerate(patient_data_list), total=len(patient_data_list)):
    for i, p_id in enumerate(tqdm(patient_data_list)):
        for adm in p_id['admission_list']:
            # csv file does not exist
            if 'episode_csv' in adm.keys():
                temp_list = [adm['episode_csv'], adm['death_indicator']]

                # append to the major episode dataframe
                temp_df = pd.DataFrame(temp_list).transpose()
                temp_df.columns = output_headers
                output_df = pd.concat([output_df, temp_df], axis=0)

    # change file header to lower case
    output_df.columns = output_df.columns.str.lower()
    output_df.to_csv(
        os.path.join(save_dir, 'y_mortality_mimic_demo.csv'),
        index=False)
