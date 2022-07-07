from urllib.robotparser import RequestRate
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from MedCode import CodeMapping
# check https://github.com/ycq091044/MedCode
# $ MedCode to get more instructions
from urllib import request
from collections import defaultdict

def collate_fn_RETAIN(cur_patient, voc_size):
    """ data is a list of sample from the dataset """
    max_len = max([len(visit[0]) + len(visit[1]) for visit in cur_patient])
    X = []
    y = torch.zeros((len(cur_patient), voc_size[2]))
    for idx, visit in enumerate(cur_patient):
        input_tmp = visit[0]
        input_tmp += [item + voc_size[0] for item in visit[1]]
        input_tmp += [voc_size[0] + voc_size[1]] * (max_len - len(input_tmp))
        X.append(input_tmp)
        y[idx, visit[2]] = 1
        
    X = torch.LongTensor(X)
    y = torch.FloatTensor(y)
    return X, y

def collate_fn_MICRON(cur_patient, voc_size):
    """ data is a list of sample from the dataset """
    diag = torch.zeros((len(cur_patient), voc_size[0]))
    prod = torch.zeros((len(cur_patient), voc_size[1]))
    y = torch.zeros((len(cur_patient), voc_size[2]))
    for idx, visit in enumerate(cur_patient):
        diag[idx, visit[0]] = 1
        prod[idx, visit[1]] = 1
        y[idx, visit[2]] = 1
        
    return diag.long(), prod.long(), y.float()

class CustomDataset(Dataset):
    def __init__(self, patients):
        self.patients = patients
        
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, index):
        return self.patients[index]

class CodetoIndex:
    """
    This Class is used to store the Code-to-index dict object
    For example: 
        - ['a', 'b', 'c', 'a'], ['b', 'c'], ... are the inputs
        - we transform them into [1,2,3,1], [2,3]
        - the mapping dict is stored in
            - self.code_to_idx = {'a': 1, 'b': 2, 'c': 3}
    """
    def __init__(self):
        self.code_to_idx = {}
    
    def build(self, code_list):
        """
        Build the code to index mapping by feeding on raw code list
        INPUT
            - code_list <list>: raw code list
        """
        if len(code_list) == 0: return
        for code in code_list:
            if code not in self.code_to_idx:
                self.code_to_idx[code] = len(self.code_to_idx)
            
    def _len(self):
        return len(self.code_to_idx)

    def encode(self, code):
        """
        encode single code
        INPUT
            - code <string> or <int>: based on the format of each code
        OUTPUT
            - idx <int>: look up the code_to_idx mapping and get the idx
        """
        idx = str(self.code_to_idx.get(code, -1))
        return idx
    
    def encodes(self, code_list):
        """
        encode a list of code
        INPUT
            - code_list <list>: a list of raw code
        OUTPUT
            - result <list>: a list of their indices
        """
        if len(code_list) == 0: return ''
        result = ','.join([self.encode(code) for code in code_list])
        return result

class MIMIC_III:
    """
    MIMIC-III datasets object
        - the original MIMIC-III medication is encoded by RxNorm (the column name uses "NDC")
        - when initialize, input the target_code and the according code_map
    For example, 
        - target_code = 'ATC4'
        - code_map = {RxNorm: ATC4} mapping dict
    """
    def __init__(self, table_names=['med', 'diag', 'prod']):
        # path to each single file
        root = '/srv/local/data/physionet.org/files/mimiciii/1.4'
        self.med_path = os.path.join(root, 'PRESCRIPTIONS.csv')
        self.diag_path = os.path.join(root, 'DIAGNOSES_ICD.csv')
        self.prod_path = os.path.join(root, 'PROCEDURES_ICD.csv')

        # table_names
        self.table_names = table_names
        # {table_name: df}
        self.tables = {}
        # {visit_id: {table_name: df}}
        self.visit_dict = {}
        # df with visit_id as index
        self.pat_to_visit = {}
        # {table_name: CodetoIndex Object}
        self.maps = {}
        # encoded {visit_id: {table_name: df}}
        self.encoded_visit_dict = {}
        # dataloader
        self.train_loader = None
        self.test_loader = None
        self.vocab_size = None
        self.ddi_adj = None

        self._get_data_tables()
        self._get_pat_and_visit_dicts()

        # map med coding to ATC3
        # First generate RxNorm (the raw drug coding) to ATC4, then use [:-1] to get ATC3
        target_code = 'ATC4'
        tool = CodeMapping('RxNorm', target_code)
        tool.load_mapping()
        self._encode_visit_info(tool.RxNorm_to_ATC4)
        self._generate_ddi_matrix_ATC3()
        # self._summarize()

    def _get_data_tables(self):
        """
        INPUT:
            - self.table_names <string list>: for example, ["med", "diag", "prod"]
        OUTPUT:
            - self.tables <dict>: key is the table name, value is the dataframe for each table
        """
        for name in self.table_names:
            cur_table = pd.read_csv(eval("self.{}_path".format(name)))
            cur_table.fillna(method='pad', inplace=True)
            cur_table.drop_duplicates(inplace=True)
            self.tables[name] = cur_table
            print ("loaded the {} table!".format(name))

    def _get_pat_and_visit_dicts(self):
        """
        INPUT:
            - self.tables <dict>: key is the table name, value is the dataframe for each table
        OUTPUT:
            - self.pat_to_visit <dict>: key is the pat id, value is a list of visits
            - self.visit_dict <dict>: key is the visit id, value is a dict of <table_name: df>
        """
        for name, df in self.tables.items():
            for pat_id, pat_info in df.groupby('SUBJECT_ID'):
                self.pat_to_visit[pat_id] = []
                for HAMD_id, HADM_info in pat_info.groupby('HADM_ID'):
                    self.pat_to_visit[pat_id].append(HAMD_id)
                    if HAMD_id not in self.visit_dict:
                        self.visit_dict[HAMD_id] = {}
                    self.visit_dict[HAMD_id][name] = HADM_info
        print ("generated .pat_to_visit!")
        print ("generated .visit_dict!")

    def _encode_visit_info(self, code_map):

        # initialize code-to-index map
        med_map = CodetoIndex()
        diag_map = CodetoIndex()
        prod_map = CodetoIndex()

        def get_atc3(x):
            # one rxnorm maps to one or more ATC3
            result = []
            for rxnorm in x:
                if rxnorm in code_map:
                    result += code_map[rxnorm]
            result = np.unique([item[:-1] for item in result]).tolist()
            return result

        encoded_visit_dict = {}
        for visit_id, value in self.visit_dict.items():

            # if one of them does not exist, then drop this visit
            if 'med' not in value or 'diag' not in value or 'prod' not in value: continue
            cur_med, cur_diag, cur_prod = value['med'], value['diag'], value['prod']

            # RxNorm->ATC3 coded med
            cur_med = get_atc3(["{:011}".format(med) for med in cur_med.NDC.unique().astype('int')])
            # ICD9 coded diag
            cur_diag = cur_diag.ICD9_CODE.unique()
            # ICD9 coded prod
            cur_prod = cur_prod.ICD9_CODE.unique()

            # if one of them does not exist, then drop this visit
            if len(cur_med) * len(cur_diag) * len(cur_prod) == 0: continue

            # build the maps
            med_map.build(cur_med)
            diag_map.build(cur_diag)
            prod_map.build(cur_prod)

            encoded_visit_dict[visit_id] = {}
            encoded_visit_dict[visit_id]['med'] = med_map.encodes(cur_med)
            encoded_visit_dict[visit_id]['diag'] = diag_map.encodes(cur_diag)
            encoded_visit_dict[visit_id]['prod'] = prod_map.encodes(cur_prod)

        self.encoded_visit_dict = encoded_visit_dict
        print ("generated .encoded_visit_dict!")

        self.maps = {
            'med': med_map,
            'diag': diag_map,
            'prod': prod_map,
        }
        self.voc_size = (len(self.maps['diag'].code_to_idx), len(self.maps['prod'].code_to_idx), len(self.maps['med'].code_to_idx))
        
        print ("generated .maps (for code to index mappings)!")

    # get ddi matrix based on ATC3 coding
    def _generate_ddi_matrix_ATC3(self):
        cid2atc_dic = defaultdict(set)
        med_voc_size = self.voc_size[2]

        cid_to_ATC6 = request.urlopen("https://drive.google.com/uc?id=1CVfa91nDu3S_NTxnn5GT93o-UfZGyewI").readlines()
        for line in cid_to_ATC6:
            line_ls = str(line[:-1]).split(',')
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if atc[:4] in self.maps['med'].code_to_idx:
                    cid2atc_dic[cid[2:]].add(atc[:4])

        # load ddi_df
        print ('load severe ddi pairs from https://drive.google.com/uc?id=1R88OIhn-DbOYmtmVYICmjBSOIsEljJMh!')
        print ('all ddi pairs can be extracted from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing!')
        ddi_df = pd.read_csv(request.urlopen('https://drive.google.com/uc?id=1R88OIhn-DbOYmtmVYICmjBSOIsEljJMh'))

        # ddi adj
        ddi_adj = np.zeros((med_voc_size, med_voc_size))
        for index, row in ddi_df.iterrows():
            # ddi
            cid1 = row['STITCH 1']
            cid2 = row['STITCH 2']

            # cid -> atc_level3
            for atc_i in cid2atc_dic[cid1]:
                for atc_j in cid2atc_dic[cid2]:
                    ddi_adj[self.maps['med'].code_to_idx[atc_i], self.maps['med'].code_to_idx[atc_j]] = 1
                    ddi_adj[self.maps['med'].code_to_idx[atc_j], self.maps['med'].code_to_idx[atc_i]] = 1

        self.ddi_adj = ddi_adj

    def _generate_ehr_matrix_for_GAMENet(self, data_train):
        """
        generate the ehr graph adj for GAMENet model input
        - loop over the training data to check whether any med pair appear
        """
        ehr_adj = np.zeros((self.voc_size[2], self.voc_size[2]))
        for patient in data_train:
            for visit in patient:
                for idx1, med1 in enumerate(visit[2]):
                    for idx2, med2 in enumerate(visit[2]):
                        if idx1 >= idx2: continue
                        ehr_adj[med1, med2] = 1
                        ehr_adj[med2, med1] = 1
        return ehr_adj

    def get_dataloader(self, MODEL):
        """
        get the dataloaders for MODEL, since different models has different datasets loader (input formats are different)
            - datasets <list>: each element is a patient record
                - datasets[0] <list>: each element is a visit
                    - datasets[0][0] <list>: diag encoded list for this visit
                    - datasets[0][1] <list>: prod encoded list for this visit
                    - datasets[0][2] <list>: med encoded list for this visit
        """
        data = []
        for _, visit_ls in self.pat_to_visit.items():
            visit_ls = sorted(visit_ls)
            cur_pat = []
            for visit_id in visit_ls:
                if visit_id not in self.encoded_visit_dict: continue
                value = self.encoded_visit_dict[visit_id]
                cur_med, cur_diag, cur_prod = value['med'], value['diag'], value['prod']
                cur_diag_info = list(map(int, cur_diag.split(',')))
                cur_prod_info = list(map(int, cur_prod.split(',')))
                cur_med_info = list(map(int, cur_med.split(',')))
                cur_pat.append([cur_diag_info, cur_prod_info, cur_med_info])
            if len(cur_pat) <= 1: continue
            data.append(cur_pat)

        # datasets split
        split_point = int(len(data) * 2 / 3)
        data_train = data[:split_point]
        eval_len = int(len(data[split_point:]) / 2)
        data_test = data[split_point:split_point + eval_len]
        data_val = data[split_point+eval_len:]

        if MODEL in ['RETAIN']:
            self.train_loader = DataLoader(CustomDataset(data_train), batch_size=1, shuffle=True, \
                collate_fn=lambda x: collate_fn_RETAIN(x[0], self.voc_size))
            self.val_loader = DataLoader(CustomDataset(data_val), batch_size=1, shuffle=False, \
                collate_fn=lambda x: collate_fn_RETAIN(x[0], self.voc_size))
            self.test_loader = DataLoader(CustomDataset(data_test), batch_size=1, shuffle=False, \
                collate_fn=lambda x: collate_fn_RETAIN(x[0], self.voc_size))
            print ("generated train/val/test dataloaders for RETAIN model!")

        elif MODEL in ['GAMENet']:
            self.train_loader = DataLoader(CustomDataset(data_train), batch_size=1, shuffle=True, \
                collate_fn=lambda x: collate_fn_MICRON(x[0], self.voc_size))
            self.val_loader = DataLoader(CustomDataset(data_val), batch_size=1, shuffle=False, \
                collate_fn=lambda x: collate_fn_MICRON(x[0], self.voc_size))
            self.test_loader = DataLoader(CustomDataset(data_test), batch_size=1, shuffle=False, \
                collate_fn=lambda x: collate_fn_MICRON(x[0], self.voc_size))
            self.ehr_adj = self._generate_ehr_matrix_for_GAMENet(data_train)
            print ("generated train/val/test dataloaders for GAMENet model!")

        elif MODEL in ['MICRON']:
            self.train_loader = DataLoader(CustomDataset(data_train), batch_size=1, shuffle=True, \
                collate_fn=lambda x: collate_fn_MICRON(x[0], self.voc_size))
            self.val_loader = DataLoader(CustomDataset(data_val), batch_size=1, shuffle=False, \
                collate_fn=lambda x: collate_fn_MICRON(x[0], self.voc_size))
            self.test_loader = DataLoader(CustomDataset(data_test), batch_size=1, shuffle=False, \
                collate_fn=lambda x: collate_fn_MICRON(x[0], self.voc_size))
            print ("generated train/val/test dataloaders for MICRON model!")



        




