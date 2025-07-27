import logging
import yaml
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class HALO_MIMIC3Dataset:
    """
    A dataset class for handling MIMIC-III data, specifically designed to be compatible with HALO.

    This class is responsible for loading and managing the MIMIC-III dataset,
    which includes tables such as patients, admissions, and icustays.

    Attributes:
        mimic3_dir (str): The root directory where the dataset is stored.
        pkl_data_dir (str): The directory in which .pkl files related to the dataset object will be stored.
        gzip (Optional[bool]): Determines whether the object will look for ".csv.gz" (True) or ".csv" (False) files in mimic3_dir.
    """

    def __init__(
        self,
        mimic3_dir: str = "./", 
        pkl_data_dir: str = "./",
        gzip: bool = False
    ) -> None:
        self.gzip = gzip
        self.mimic3_dir = mimic3_dir
        self.pkl_data_dir = pkl_data_dir
        self.build_dataset()

    def build_dataset(self) -> None:
        admissionFile = self.mimic3_dir + f"ADMISSIONS.csv{'.gz' if self.gzip else ''}"
        diagnosisFile = self.mimic3_dir + f"DIAGNOSES_ICD.csv{'.gz' if self.gzip else ''}"

        admissionDf = pd.read_csv(admissionFile, dtype=str)
        admissionDf['ADMITTIME'] = pd.to_datetime(admissionDf['ADMITTIME'])
        admissionDf = admissionDf.sort_values('ADMITTIME')
        admissionDf = admissionDf.reset_index(drop=True)
        diagnosisDf = pd.read_csv(diagnosisFile, dtype=str).set_index("HADM_ID")
        diagnosisDf = diagnosisDf[diagnosisDf['ICD9_CODE'].notnull()]
        diagnosisDf = diagnosisDf[['ICD9_CODE']]

        data = {}
        for row in tqdm(admissionDf.itertuples(), total=admissionDf.shape[0]):          
            #Extracting Admissions Table Info
            hadm_id = row.HADM_ID
            subject_id = row.SUBJECT_ID
                    
            # Extracting the Diagnoses
            if hadm_id in diagnosisDf.index: 
                diagnoses = list(set(diagnosisDf.loc[[hadm_id]]["ICD9_CODE"]))
            else:
                diagnoses = []
            
            # Building the hospital admission data point
            if subject_id not in data:
                data[subject_id] = {'visits': [diagnoses]}
            else:
                data[subject_id]['visits'].append(diagnoses)

        code_to_index = {}
        all_codes = list(set([c for p in data.values() for v in p['visits'] for c in v]))
        np.random.shuffle(all_codes)
        for c in all_codes:
            code_to_index[c] = len(code_to_index)
        print(f"VOCAB SIZE: {len(code_to_index)}")
        index_to_code = {v: k for k, v in code_to_index.items()}

        data = list(data.values())

        with open("./configs/hcup_ccs_2015_definitions_benchmark.yaml") as definitions_file:
            definitions = yaml.full_load(definitions_file)

        code_to_group = {}
        for group in definitions:
            if definitions[group]['use_in_benchmark'] == False:
                continue
            codes = definitions[group]['codes']
            for code in codes:
                if code not in code_to_group:
                    code_to_group[code] = group
                else:
                    assert code_to_group[code] == group

        id_to_group = sorted([k for k in definitions.keys() if definitions[k]['use_in_benchmark'] ==  True])
        group_to_id = dict((x, i) for (i, x) in enumerate(id_to_group))

        # Add Labels
        for p in data:
            label = np.zeros(len(group_to_id))
            for v in p['visits']:
                for c in v:
                    if c in code_to_group:
                        label[group_to_id[code_to_group[c]]] = 1
            
            p['labels'] = label

        for p in data:
            new_visits = []
            for v in p['visits']:
                new_visit = []
                for c in v:
                    new_visit.append(code_to_index[c])
                        
                new_visits.append((list(set(new_visit))))
                
            p['visits'] = new_visits    

        print(f"MAX LEN: {max([len(p['visits']) for p in data])}")
        print(f"AVG LEN: {np.mean([len(p['visits']) for p in data])}")
        print(f"MAX VISIT LEN: {max([len(v) for p in data for v in p['visits']])}")
        print(f"AVG VISIT LEN: {np.mean([len(v) for p in data for v in p['visits']])}")
        print(f"NUM RECORDS: {len(data)}")
        print(f"NUM LONGITUDINAL RECORDS: {len([p for p in data if len(p['visits']) > 1])}")

        # Train-Val-Test Split
        train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=4, shuffle=True)
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=4, shuffle=True)

        # Save Everything
        print("Saving Everything")
        print(len(index_to_code))
        pickle.dump(code_to_index, open(f"{self.pkl_data_dir}codeToIndex.pkl", "wb"))
        pickle.dump(index_to_code, open(f"{self.pkl_data_dir}indexToCode.pkl", "wb"))
        pickle.dump(id_to_group, open(f"{self.pkl_data_dir}idToLabel.pkl", "wb"))
        pickle.dump(train_dataset, open(f"{self.pkl_data_dir}trainDataset.pkl", "wb"))
        pickle.dump(val_dataset, open(f"{self.pkl_data_dir}valDataset.pkl", "wb"))
        pickle.dump(test_dataset, open(f"{self.pkl_data_dir}testDataset.pkl", "wb"))

