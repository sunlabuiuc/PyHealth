# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys
import json

import pandas as pd
from joblib import Parallel, delayed
import unittest
# noinspection PyProtectedMember
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_array_less
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_greater_equal
from sklearn.utils.testing import assert_less_equal
from sklearn.utils.testing import assert_raises

from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyhealth.data.base_mimic import parallel_parse_tables
from pyhealth.utils.utility import read_csv_to_df

from pyhealth.utils.utility_parallel import unfold_parallel
from pyhealth.utils.utility_parallel import partition_estimators


class TestMIMIC(unittest.TestCase):
    def setUp(self):
        self.save_dir = os.path.join('outputs', 'mimic_demo')
        self.patient_data_loc = os.path.join(self.save_dir,
                                             'patient_data.json')

    def test_01_flow(self):

        n_jobs = 2  # number of parallel jobs
        n_samples = 10
        duration = 21600  # time window for episode generation
        selection_method = 'last'

        # make saving directory if needed
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        patient_data_loc = 'patient_data.json'
        patient_list_loc = 'patient_list.json'

        valid_data_list = []  # keep tracking the stored data
        valid_id_list = []  # keep tracking a list of patient IDs

        # key variables to track in the episode
        var_list = ['Capillary refill rate',
                    'Diastolic blood pressure',
                    'Fraction inspired oxygen',
                    'Glascow coma scale eye opening',
                    'Glascow coma scale motor response',
                    'Glascow coma scale total',
                    'Glascow coma scale verbal response',
                    'Glucose',
                    'Heart Rate',
                    'Height',
                    'Mean blood pressure',
                    'Oxygen saturation',
                    'Respiratory rate',
                    'Systolic blood pressure',
                    'Temperature',
                    'Weight',
                    'pH']

        # enforce and convert to lower case
        var_list = [item.lower() for item in var_list]

        event_mapping_df = read_csv_to_df(
            os.path.join('resources', 'itemid_to_variable_map.csv'))
        event_mapping_df['level2'] = event_mapping_df['level2'].str.lower()

        key_df = event_mapping_df[event_mapping_df['level2'].isin(var_list)]
        key_items = key_df['itemid'].tolist()

        #################################################################
        # read in tables
        patient_df = read_csv_to_df(
            os.path.join('data', 'mimic-iii-clinical-database-demo-1.4',
                         'PATIENTS.csv'))
        patient_id_list = patient_df['subject_id'].tolist()

        admission_df = read_csv_to_df(
            os.path.join('data', 'mimic-iii-clinical-database-demo-1.4',
                         'ADMISSIONS.csv'))

        icu_df = read_csv_to_df(
            os.path.join('data', 'mimic-iii-clinical-database-demo-1.4',
                         'ICUSTAYS.csv'))

        events_vars = ['subject_id',
                       'hadm_id',
                       'icustay_id',
                       'itemid',
                       'charttime',
                       'value',
                       'valueuom', ]
        # because MIMIC's header is in upper case
        # however, demo code does not
        # events_vars = [item.upper() for item in events_vars]

        # define datatype to reduce the memory cost
        dtypes_dict = {
            'subject_id': 'int32',
            'hadm_id': 'int32',
            'icustay_id': 'object',
            'itemid': 'int32',
            'charttime': 'object',
            'value': 'object',
            'valueuom': 'object',
        }

        event_df = read_csv_to_df(
            # os.path.join('data', 'mimic-iii-clinical-database-demo-1.4',
            os.path.join('data', 'mimic-iii-clinical-database-demo-1.4',
                         'CHARTEVENTS.csv'), usecols=events_vars,
            dtype=dtypes_dict,
            low_memory=True)

        # only keep the events we are interested in
        event_df = event_df[event_df['itemid'].isin(key_items)]

        oevent_df = read_csv_to_df(
            os.path.join('data', 'mimic-iii-clinical-database-demo-1.4',
                         'OUTPUTEVENTS.csv'), usecols=events_vars,
            dtype=dtypes_dict,
            low_memory=True)

        # only keep the events we are interested in
        oevent_df = oevent_df[oevent_df['itemid'].isin(key_items)]

        event_df = pd.concat([event_df, oevent_df])
        event_df['charttime'] = pd.to_datetime(event_df['charttime'])

        # Start data processing
        n_patients_list, starts, n_jobs = partition_estimators(
            n_samples, n_jobs=n_jobs)

        all_results = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=True)(
            delayed(parallel_parse_tables)(
                patient_id_list=patient_id_list[starts[i]:starts[i + 1]],
                patient_df=patient_df,
                admission_df=admission_df,
                icu_df=icu_df,
                event_df=event_df,
                event_mapping_df=event_mapping_df,
                duration=duration,
                selection_method=selection_method,
                var_list=var_list,
                save_dir=self.save_dir)
            for i in range(n_jobs))

        all_results = list(map(list, zip(*all_results)))
        valid_data_list = unfold_parallel(all_results[0], n_jobs)
        valid_id_list = unfold_parallel(all_results[1], n_jobs)

        patient_data_list = []
        for p in valid_data_list:
            patient_data_list.append(p.data)

        with open(self.patient_data_loc, 'w') as outfile:
            json.dump(patient_data_list, outfile)

        print(patient_data_list)

    def test_02_file_generation(self):
        assert (os.path.exists(self.patient_data_loc))
