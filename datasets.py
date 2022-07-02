import numpy as np
import pandas as pd
import os
from MedCode import CodeMapping
# check https://github.com/ycq091044/MedCode
# $ MedCode to get more instructions

class CodetoIndex:
    def __init__(self):
        self.code_to_idx = {}
    
    def build(self, code_list):
        if len(code_list) == 0: return
        for code in code_list:
            if code not in self.code_to_idx:
                self.code_to_idx[code] = len(self.code_to_idx)
            
    def _len(self):
        return len(self.code_to_idx)

    def encode(self, code):
        return str(self.code_to_idx.get(code, -1))
    
    def encodes(self, code_list):
        if len(code_list) == 0: return ''
        return ','.join([self.encode(code) for code in code_list])

class MIMIC_III:
    def __init__(self, table_names=['med', 'diag', 'prod'], code_map=None, target_code='ATC4'):
        root = '/srv/local/data/physionet.org/files/mimiciii/1.4'
        self.med_path = os.path.join(root, 'PRESCRIPTIONS.csv')
        self.diag_path = os.path.join(root, 'DIAGNOSES_ICD.csv')
        self.prod_path = os.path.join(root, 'PROCEDURES_ICD.csv')
        self.table_names = table_names
        self.tables = {}
        self.visit_dict = {}
        self.visit_df = {}
        self.pat_to_visit = {}
        self.maps = {}
        self.clean_data = {}

        self._get_data_tables()
        self._get_pat_and_visit_dicts()
        self._get_visit_df()
        self._encode_visit_df(code_map=code_map, target_code=target_code)
        self._get_clean_data()

    def _get_data_tables(self):
        """
        INPUT:
            - name_list <string list>: for example, ["med", "diag", "prod"]
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
            - self.visit_dict <dict>: key is the visit id, value is a dict of <table name: dataframe>
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

    def _get_visit_df(self):
        """
        clean out the med, diag, prod for each visit
        INPUT:
            - dataset <datasets Object>
        OUTPUT:
            - pd.DataFrame using visit_id as key
        """
        med_rxnorm_list = []
        medname_list = []
        diag_list = []
        prod_list = []
        visit_ids = []
        for visit_id, dict in self.visit_dict.items():
            visit_ids.append(visit_id)

            if 'med' in dict:
                cur_med_name = dict['med'].DRUG.unique()
                dict['med'].dropna(inplace=True)
                cur_med_rxnorm = dict['med'].NDC.unique().astype('int')
                cur_med_rxnorm = ["{:011}".format(med) for med in cur_med_rxnorm]
                med_rxnorm_list.append(cur_med_rxnorm)
                medname_list.append(cur_med_name)
            else:
                med_rxnorm_list.append([])
                medname_list.append([])

            if 'diag' in dict:
                cur_diag = dict['diag'].ICD9_CODE.unique()
                diag_list.append(cur_diag)
            else:
                diag_list.append([])

            if 'prod' in dict:
                cur_prod = dict['prod'].ICD9_CODE.unique()
                prod_list.append(cur_prod)
            else:
                prod_list.append([])

        # create the dataframe
        self.visit_df = pd.DataFrame(visit_ids, columns=['visit_id'])
        self.visit_df['medrxnorm'] = med_rxnorm_list
        self.visit_df['medname'] = medname_list
        self.visit_df['diag'] = diag_list
        self.visit_df['Prod'] = prod_list
        print ("generated .visit_df!")

    def _encode_visit_df(self, code_map, target_code='ATC4'):
        med_map = CodetoIndex()
        diag_map = CodetoIndex()
        prod_map = CodetoIndex()

        # change medrxnorm to ATC-3
        medcode_col = 'med{}'.format(target_code)
        def get_atc3(x):
            # one rxnorm maps to one or more ATC4
            result = []
            for rxnorm in x:
                if rxnorm in code_map:
                    result += code_map[rxnorm]
            result = np.unique([item[:-1] for item in result]).tolist()
            return result

        self.visit_df[medcode_col] = self.visit_df.medrxnorm.apply(lambda x: get_atc3(x))
        for item in eval("self.visit_df.{}".format(medcode_col)):
            med_map.build(item)
        # encode medATC3
        encoded_medcode_col = "{}_encoded".format(medcode_col)
        self.visit_df[encoded_medcode_col] = eval("self.visit_df.{}".format(medcode_col)).apply(lambda x: med_map.encodes(x))

        # encode diag
        for diags in self.visit_df.diag:
            diag_map.build(diags)
        self.visit_df['diag_encoded'] = self.visit_df.diag.apply(lambda x: diag_map.encodes(x))

        # encode prod, because self.visit_df.prod is a built in method, \
        # cannot use self.visit_df.prod, we use self.visit_df.Prod
        for prods in self.visit_df.Prod:
            prod_map.build(prods)
        self.visit_df['prod_encoded'] = self.visit_df.Prod.apply(lambda x: prod_map.encodes(x))

        print ("generated encoded .visit_df!")

        self.maps = {
            'med': med_map,
            'diag': diag_map,
            'prod': prod_map,
        }
        print ("generated code mappings: .maps!")

    def _get_clean_data(self):
        clean_data = self.visit_df[(self.visit_df.medATC4_encoded != "") & \
            ((self.visit_df.diag_encoded != "") | (self.visit_df.prod_encoded != ""))]
        self.clean_data = clean_data[["medATC4_encoded", "diag_encoded", "prod_encoded"]]
        print ("generated cleaned data for ML models .clean_data!")

    def help(self):
        help_message = """
        We provide
            - .tables
                - key: {}
                - value: data tables
            - .visit_info
                - key: visit_id
                - value: a dict of <table name: dataframe>
            - .pat_to_visit
                - key: patient_id
                - value: a list of visit_ids.
        """.format(self.table_names)
        print (help_message)
