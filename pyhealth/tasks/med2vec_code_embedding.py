import pandas as pd
from pyhealth.data import Visit, Patient, TaskDataset
from datetime import datetime
from tqdm import tqdm
import torch


class Med2VecDataset(TaskDataset):

    def __init__(self, base_dataset):
        super().__init__(base_dataset)
        self.base_dataset = base_dataset
        self.transform = None
        self.target_transform = None
        self.processed_data = None
        self.train = None
        self.num_codes = None
        self.train_data = None
        self.test_data = None
        self.preprocess_init()

    def preprocess_init(self):
        df_diagnosis = self.base_dataset.raw_diagnosis()
        df_admission = self.base_dataset.raw_admissions()
        REMOVE_DIAGNOSIS = ~((df_admission['DIAGNOSIS'] == 'ORGAN DONOR ACCOUNT') | (
                df_admission['DIAGNOSIS'] == 'ORGAN DONOR') | \
                             (df_admission['DIAGNOSIS'] == 'DONOR ACCOUNT'))
        df = df_admission[REMOVE_DIAGNOSIS]

        patient_data = {}
        patient_id = set(df['SUBJECT_ID'])

        data = df_diagnosis['ICD9_CODE'].values

        def code2idx(data):
            data_set = set()
            for i in range(len(data)):
                data_set.add(data[i])
            voc_size = len(data_set)
            data_map = {}
            for i in range(voc_size):
                data_map[data_set.pop()] = i

            return voc_size, data_map

        def get_idx(list_, datamap):
            res = []
            for i in range(len(list_)):
                res.append(datamap[list_[i]])
            return res

        def convert_to_med2vec(patient_data):
            data = []
            for k, vv in patient_data.items():
                for v in vv:
                    data.append(v[0])
                data.append([-1])
            return data

        self.num_codes, datamap = code2idx(data)

        for pid in tqdm(patient_id):
            pid_df = df[df['SUBJECT_ID'] == pid]
            if (len(pid_df) < 2):
                continue
            adm_list = pid_df[['HADM_ID', 'ADMITTIME', 'DEATHTIME']]  # add DISCHATIME ?
            patient_data[pid] = []
            for i, r in adm_list.iterrows():
                admid = r['HADM_ID']
                admitime = r['ADMITTIME']
                icd9_raw = df_diagnosis[df_diagnosis['HADM_ID'] == admid]['ICD9_CODE'].values
                icd9_raw = list(set(icd9_raw))
                icd9 = get_idx(icd9_raw, datamap)
                mortality = r['DEATHTIME'] == r['DEATHTIME']  # check not nan
                admtime = datetime.strptime(r['ADMITTIME'], '%Y-%m-%d %H:%M:%S')
                tup = (icd9, admtime, mortality)
                patient_data[pid].append(tup)

        self.processed_data = convert_to_med2vec(patient_data)
        return self.processed_data

    def preprocess_(self, seq):

        x = torch.zeros((self.num_codes,), dtype=torch.long)

        ivec = []
        jvec = []
        d = []
        if seq == [-1]:  # masked, separator between patient
            return x, torch.LongTensor(ivec), torch.LongTensor(jvec), d

        x[seq] = 1  # one-hot
        for i in seq:
            for j in seq:
                if i == j:
                    continue
                ivec.append(i)
                jvec.append(j)  # code to code coordination, code pairs in one visit
        return x, torch.LongTensor(ivec), torch.LongTensor(jvec), d

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.processed_data is None:
            self.preprocess_init()

        x, ivec, jvec, d = self.preprocess_(self.processed_data[index])
        return x, ivec, jvec, d


