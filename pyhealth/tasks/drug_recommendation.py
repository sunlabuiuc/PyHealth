from lib2to3.pgen2.pgen import generate_grammar
from tokenize import Token
from urllib.parse import _NetlocResultMixinBytes
import numpy as np
from collections import defaultdict
from urllib import request
import os
import pandas as pd
from pathlib import Path
from pyhealth.data import Visit, Patient, TaskDataset
from pyhealth.models.tokenizer import Tokenizer
from rdkit import Chem


class DrugRecDataset(TaskDataset):
    """ Dataset for drug recommendation task """

    def preprocess(self):
        # from MedCode import CodeMapping
        # tool = CodeMapping('RxNorm', 'ATC4')
        # tool.load()

        # def get_atc3(x):
        #     # one rxnorm maps to one or more ATC3
        #     result = []
        #     for rxnorm in x:
        #         if rxnorm in tool.RxNorm_to_ATC4:
        #             result += tool.RxNorm_to_ATC4[rxnorm]
        #     result = np.unique([item[:-1] for item in result]).tolist()
        #     return result

        def remove_nan_from_list(lst):
            if (type(lst) != type([0,1,2])) and (type(lst) != type(np.array([0,1,2]))):
                return []
            return [i for i in lst if not i != i]

        processed_patients = []
        for patient in self.base_dataset.patients:
            processed_visits = []
            for visit in patient.visits:
                conditions = list(set(remove_nan_from_list(visit.conditions)))
                procedures = list(set(remove_nan_from_list(visit.procedures)))
                drugs = list(set(remove_nan_from_list(visit.drugs))) #get_atc3(["{:011}".format(int(med)) for med in set(remove_nan_from_list(visit.drugs))])
                # exclude: visits without condition, procedure, or drug code
                if len(conditions) * len(procedures) * len(drugs) == 0:
                    continue
                processed_visit = Visit(visit_id=visit.visit_id,
                                        patient_id=visit.patient_id,
                                        conditions=conditions,
                                        procedures=procedures,
                                        drugs=drugs)
                processed_visits.append(processed_visit)
            # exclude: patients with less than 2 visit
            if len(processed_visits) < 2:
                continue
            processed_patient = Patient(patient_id=patient.patient_id,
                                        visits=processed_visits)
            processed_patients.append(processed_patient)

        self.processed_patients = processed_patients
        self.params = None

    def set_all_tokens(self):
        conditions = []
        procedures = []
        drugs = []
        for patient in self.processed_patients:
            for visit in patient.visits:
                conditions.extend(visit.conditions)
                procedures.extend(visit.procedures)
                drugs.extend(visit.drugs)
        conditions = list(set(conditions))
        procedures = list(set(procedures))
        drugs = list(set(drugs))
        self.all_tokens = {'conditions': conditions, 'procedures': procedures, 'drugs': drugs}
        self.condition_tokenizer = Tokenizer(conditions)
        self.procedures_tokenizer = Tokenizer(procedures)
        self.drugs_tokenizer = Tokenizer(drugs)
        self.voc_size = (
            self.condition_tokenizer.get_vocabulary_size(),
            self.procedures_tokenizer.get_vocabulary_size(),
            self.drugs_tokenizer.get_vocabulary_size()
        )

    def __len__(self):
        return len(self.processed_patients)

    def __getitem__(self, index):
        conditions = []
        procedures = []
        drugs = []
        patient = self.processed_patients[index]
        for visit in patient.visits:
            conditions.append(visit.conditions)
            procedures.append(visit.procedures)
            drugs.append(visit.drugs)
        return {"conditions": self.condition_tokenizer(conditions),
                "procedures": self.procedures_tokenizer(procedures),
                "drugs": self.drugs_tokenizer(drugs)}

    def info(self):
        info = """
        ----- Output Data Structure -----
        >> drug_rec_dataloader[0]
        >> {
            "conditions": List[tensor],
            "procedures": List[tensor],
            "drugs": List[tensor]
        }
        """
        print (info)

    def task(self):
        return "DrugRec"

    def generate_ddi_adj(self):
        cid2atc_dic = defaultdict(set)
        med_voc_size = self.voc_size[2]

        if not os.path.exists(os.path.join(str(Path.home()), ".cache/pyhealth/cid_to_ATC6.csv")):
            cid_to_ATC6 = request.urlopen("https://drive.google.com/uc?id=1CVfa91nDu3S_NTxnn5GT93o-UfZGyewI").readlines()
            with open(os.path.join(str(Path.home()), ".cache/pyhealth/cid_to_ATC6.csv"), 'w') as outfile:
                for line in cid_to_ATC6:
                    print (str(line[:-1]), file=outfile)
        else:
            cid_to_ATC6 = open(os.path.join(str(Path.home()), ".cache/pyhealth/cid_to_ATC6.csv"), 'r').readlines()

        for line in cid_to_ATC6:
            line_ls = str(line[:-1]).split(',')
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if atc[:4] in self.maps['med'].code_to_idx:
                    cid2atc_dic[cid[2:]].add(atc[:4])

        # load ddi_df
        if not os.path.exists(os.path.join(str(Path.home()), ".cache/pyhealth/drug-DDI-TOP40.csv")):
            ddi_df = pd.read_csv(request.urlopen('https://drive.google.com/uc?id=1R88OIhn-DbOYmtmVYICmjBSOIsEljJMh'))
            ddi_df.to_csv(os.path.join(str(Path.home()), ".cache/pyhealth/drug-DDI-TOP40.csv"), index=False)
        else:
            ddi_df = pd.read_csv(os.path.join(str(Path.home()), ".cache/pyhealth/drug-DDI-TOP40.csv"))
            
        # ddi adj
        ddi_adj = np.zeros((med_voc_size, med_voc_size))
        for index, row in ddi_df.iterrows():
            # ddi
            cid1 = row['STITCH 1']
            cid2 = row['STITCH 2']

            # cid -> atc_level3
            for atc_i in cid2atc_dic[cid1]:
                for atc_j in cid2atc_dic[cid2]:
                    ddi_adj[self.maps['med'].encode(atc_i), self.maps['med'].encode(atc_j)] = 1
                    ddi_adj[self.maps['med'].encode(atc_j), self.maps['med'].encode(atc_i)] = 1
        return ddi_adj

    def generate_ehr_adj_for_GAMENet(self, data_train):
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

    def generate_ddi_mask_H_for_SafeDrug(self, ATC4_to_SMILES):
        # idx_to_SMILES
        SMILES = [[] for _ in range(self.voc_size[2])]
        # each idx contains what segments
        fraction = [[] for _ in range(self.voc_size[2])]
        
        for atc4, smiles_ls in ATC4_to_SMILES.items():
            if atc4[:-1] in self.maps['med'].code_to_idx:
                pos = self.maps['med'].encode(atc4[:-1])
                SMILES[pos] += smiles_ls
                for smiles in smiles_ls:
                    if smiles != 'nan':
                        try:
                            m = Chem.BRICS.BRICSDecompose(Chem.MolFromSmiles(smiles))
                            for frac in m:
                                fraction[pos].add(frac)
                        except:
                            pass
        # all segment set
        fraction_set = []
        for i in fraction:
            fraction_set += i
        fraction_set = list(set(fraction_set)) # set of all segments

        # ddi_mask
        ddi_mask_H = np.zeros((self.voc_size[2], len(fraction_set)))
        for idx, cur_fraction in enumerate(fraction):
            for frac in cur_fraction:
                ddi_mask_H[idx, fraction_set.index(frac)] = 1
        return ddi_mask_H, SMILES

    def generate_med_molecule_info_for_SafeDrug(self, SMILES, radius=1):

        atom_dict = defaultdict(lambda: len(atom_dict))
        bond_dict = defaultdict(lambda: len(bond_dict))
        fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
        edge_dict = defaultdict(lambda: len(edge_dict))
        MPNNSet, average_index = [], []

        for smilesList in SMILES:
            """Create each data with the above defined functions."""
            counter = 0 # counter how many drugs are under that ATC-3
            for smiles in smilesList:
                try:
                    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                    atoms = create_atoms(mol, atom_dict)
                    molecular_size = len(atoms)
                    i_jbond_dict = create_ijbonddict(mol, bond_dict)
                    fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                        fingerprint_dict, edge_dict)
                    adjacency = Chem.GetAdjacencyMatrix(mol)
                    # if fingerprints.shape[0] == adjacency.shape[0]:
                    for _ in range(adjacency.shape[0] - fingerprints.shape[0]):
                        fingerprints = np.append(fingerprints, 1)
                    
                    fingerprints = torch.LongTensor(fingerprints)
                    adjacency = torch.FloatTensor(adjacency)
                    MPNNSet.append((fingerprints, adjacency, molecular_size))
                    counter += 1
                except:
                    continue
            
            average_index.append(counter)

            """Transform the above each data of numpy
            to pytorch tensor on a device (i.e., CPU or GPU).
            """

        N_fingerprint = len(fingerprint_dict)
        # transform into projection matrix
        n_col = sum(average_index)
        n_row = len(average_index)

        average_projection = np.zeros((n_row, n_col))
        col_counter = 0
        for i, item in enumerate(average_index):
            if item > 0:
                average_projection[i, col_counter : col_counter + item] = 1 / item
            col_counter += item

        return [MPNNSet, N_fingerprint, torch.FloatTensor(average_projection)]

    def info(self):
        info = """
        ----- Output Data Structure -----
        TaskDataset.patients dict[str, Patient]
            - key: patient_id
            - value: <Patient> object
        
        <Patient>
            - patient_id: str
            - visits: dict[str, Visit]
                - key: visit_id
                - value: <DrugRecVisit> object
        
        <DrugRecVisit>
            - visit_id: str
            - patient_id: str
            - conditions: List = [],
            - procedures: List = [],
            - drugs: List = [],
            - labs: List = [],
            - physicalExams: List = []
        """
        print (info)

if __name__ == "__main__":
    from pyhealth.datasets import MIMIC3BaseDataset

    base_dataset = MIMIC3BaseDataset(root="/srv/local/data/physionet.org/files/mimiciii/1.4")
    drug_rec_dataloader = DrugRecDataset(base_dataset)
    print(len(drug_rec_dataloader))
    print(drug_rec_dataloader[0])
