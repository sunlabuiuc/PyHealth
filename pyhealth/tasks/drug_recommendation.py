from typing import List
import numpy as np
from collections import defaultdict
from urllib import request
import os
from pathlib import Path
import pandas as pd
from pyhealth import CACHE_PATH
from pyhealth.data import Patient, TaskDataset
from pyhealth.models.tokenizer import Tokenizer
import torch
from tqdm import tqdm
import pickle


class DrugRecVisit:
    """Contains information about a single visit (for drug recommendation task)"""

    def __init__(
        self,
        visit_id: str,
        patient_id: str,
        conditions: List[str] = [],
        procedures: List[str] = [],
        drugs: List[str] = [],
        labs: List[str] = [],
        physicalExams: List[str] = [],
        admission_time: float = 0.0,
    ):
        self.visit_id = visit_id
        self.patient_id = patient_id
        self.conditions = conditions
        self.procedures = procedures
        self.drugs = drugs
        self.labs = labs
        self.physicalExams = physicalExams
        self.admission_time = admission_time

    def __str__(self):
        return f"Visit {self.visit_id} of patient {self.patient_id}"


class DrugRecDataset(TaskDataset):
    """
    Dataset for drug recommendation task
    Transform the <BaseDataset> object to <TaskDataset> object
    """

    @staticmethod
    def remove_nan_from_list(list_with_nan):
        """
        e.g., [1, 2, nan, 3] -> [1, 2, 3]
        e.g., [1, 2, 3] -> [1, 2, 3]
        e.g., np.array([1, 2, nan, 3]) -> [1, 2, 3]
        """
        if (type(list_with_nan) != type([0, 1, 2])) and (
            type(list_with_nan) != type(np.array([0, 1, 2]))
        ):
            return []
        return [i for i in list_with_nan if not i != i]

    def get_code_from_list_of_Event(self, list_of_Event):
        """
        INPUT
            - list_of_Event: List[Event]
        OUTPUT
            - list_of_code: List[str]
        """
        list_of_code = [event.code for event in list_of_Event]
        list_of_code = np.unique(list_of_code)
        list_of_code = self.remove_nan_from_list(list_of_code)
        return list_of_code

    def preprocess(self):
        """clean the data for drug recommendation task"""

        # ---------- for drug coding ------
        from MedCode import CodeMapping

        tool = CodeMapping("RxNorm", "ATC4")
        tool.load()

        def get_atc3(x):
            # one rxnorm maps to one or more ATC3
            result = []
            for rxnorm in x:
                if rxnorm in tool.RxNorm_to_ATC4:
                    result += tool.RxNorm_to_ATC4[rxnorm]
            result = np.unique([item[:-1] for item in result]).tolist()
            return result

        # ---------------------

        processed_patients = {}
        for patient_id, patient_obj in tqdm(self.base_dataset.patients.items()):
            processed_visits = {}
            for visit_id, visit_obj in patient_obj.visits.items():
                conditions = self.get_code_from_list_of_Event(visit_obj.conditions)
                procedures = self.get_code_from_list_of_Event(visit_obj.procedures)
                drugs = self.get_code_from_list_of_Event(visit_obj.drugs)
                drugs = get_atc3(
                    ["{:011}".format(int(med)) for med in drugs]
                )  # drug coding
                # exclude: visits without condition, procedure, or drug code
                if (len(conditions) + len(procedures)) * len(drugs) == 0:
                    continue
                cur_visit = DrugRecVisit(
                    visit_id=visit_id,
                    patient_id=patient_id,
                    conditions=conditions,
                    procedures=procedures,
                    drugs=drugs,
                    admission_time=visit_obj.encounter_time,
                )

                processed_visits[visit_id] = cur_visit

            # exclude: patients with less than 2 visit
            if len(processed_visits) < 2:
                continue

            cur_pat = Patient(
                patient_id=patient_id,
                visits=[
                    v
                    for _, v in sorted(
                        processed_visits.items(),
                        key=lambda item: item[1].admission_time,
                    )
                ],  # sort the visits and change into a list
            )
            processed_patients[patient_id] = cur_pat

        print("1. finish cleaning the dataset for drug recommendation task")
        self.patients = processed_patients

        # get (0, N-1) to (patients, visit_pos) map
        self.index_map = {}
        self.index_group = []
        t = 0
        for patient_id, patient_obj in self.patients.items():
            group = []
            for pos in range(len(patient_obj.visits)):
                self.index_map[t] = (patient_id, pos)
                group.append(t)
                t += 1
            self.index_group.append(group)
        self.params = None

    def set_all_tokens(self):
        """tokenize by medical codes"""
        conditions = []
        procedures = []
        drugs = []
        for patient_id, patient_obj in self.patients.items():
            for visit_obj in patient_obj.visits:
                conditions.extend(visit_obj.conditions)
                procedures.extend(visit_obj.procedures)
                drugs.extend(visit_obj.drugs)
        conditions = list(set(conditions))
        procedures = list(set(procedures))
        drugs = list(set(drugs))
        self.all_tokens = {
            "conditions": conditions,
            "procedures": procedures,
            "drugs": drugs,
        }

        # store the tokenizer
        condition_tokenizer = Tokenizer(conditions)
        procedures_tokenizer = Tokenizer(procedures)
        drugs_tokenizer = Tokenizer(drugs)
        self.tokenizers = (
            condition_tokenizer,
            procedures_tokenizer,
            drugs_tokenizer,
        )
        self.voc_size = [item.get_vocabulary_size() for item in self.tokenizers]
        print("2. tokenized the medical codes")

    def get_ddi_matrix(self):
        """get drug-drug interaction (DDI)"""
        cid2atc_dic = defaultdict(set)
        med_voc_size = self.voc_size[2]

        vocab_to_index = self.tokenizers[2].vocabulary.word2idx

        # load cid2atc
        if not os.path.exists(
            os.path.join(str(Path.home()), ".cache/pyhealth/cid_to_ATC6.csv")
        ):
            cid_to_ATC6 = request.urlopen(
                "https://drive.google.com/uc?id=1CVfa91nDu3S_NTxnn5GT93o-UfZGyewI"
            ).readlines()
            with open(
                os.path.join(str(Path.home()), ".cache/pyhealth/cid_to_ATC6.csv"), "w"
            ) as outfile:
                for line in cid_to_ATC6:
                    print(str(line[:-1]), file=outfile)
        else:
            cid_to_ATC6 = open(
                os.path.join(str(Path.home()), ".cache/pyhealth/cid_to_ATC6.csv"), "r"
            ).readlines()

        # map cid to atc
        for line in cid_to_ATC6:
            line_ls = str(line[:-1]).split(",")
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if atc[:4] in vocab_to_index:
                    cid2atc_dic[cid[2:]].add(atc[:4])

        # ddi on (cid, cid)
        if not os.path.exists(
            os.path.join(str(Path.home()), ".cache/pyhealth/drug-DDI-TOP40.csv")
        ):
            ddi_df = pd.read_csv(
                request.urlopen(
                    "https://drive.google.com/uc?id=1R88OIhn-DbOYmtmVYICmjBSOIsEljJMh"
                )
            )
            ddi_df.to_csv(
                os.path.join(str(Path.home()), ".cache/pyhealth/drug-DDI-TOP40.csv"),
                index=False,
            )
        else:
            ddi_df = pd.read_csv(
                os.path.join(str(Path.home()), ".cache/pyhealth/drug-DDI-TOP40.csv")
            )

        # map to ddi on (atc, atc)
        # remove the padding and invalid drugs
        ddi_adj = np.zeros((med_voc_size - 2, med_voc_size - 2))
        for index, row in ddi_df.iterrows():
            # ddi
            cid1 = row["STITCH 1"]
            cid2 = row["STITCH 2"]

            # cid -> atc_level3
            for atc_i in cid2atc_dic[cid1]:
                for atc_j in cid2atc_dic[cid2]:
                    i = vocab_to_index.get(atc_i, 0)
                    j = vocab_to_index.get(atc_j, 0)
                    if (i > 1) and (j > 1):
                        ddi_adj[i - 2, j - 2] = 1
                        ddi_adj[j - 2, i - 2] = 1
        self.ddi_adj = ddi_adj
        return ddi_adj

    def generate_ehr_adj_for_GAMENet(self, visit_ls):
        """
        generate the ehr graph adj for GAMENet model input
        - loop over the training data to check whether any med pair appear
        """
        ehr_adj = np.zeros((self.voc_size[2], self.voc_size[2]))
        for visit_index in visit_ls:
            patient_id, visit_pos = self.index_map[visit_index]
            patient = self.patients[patient_id]
            visit = patient.visits[visit_pos]
            encoded_drugs = self.tokenizers[2]([visit.drugs])[0]
            for idx1, med1 in enumerate(encoded_drugs):
                for idx2, med2 in enumerate(encoded_drugs):
                    if idx1 >= idx2:
                        continue
                    ehr_adj[med1, med2] = 1
                    ehr_adj[med2, med1] = 1
        return ehr_adj

    def generate_ddi_mask_H_for_SafeDrug(self):
        # TODO: update on this based on rdkit version
        from rdkit import Chem
        import rdkit.Chem.BRICS as BRICS

        # idx_to_SMILES
        SMILES = [[] for _ in range(self.voc_size[2])]
        # each idx contains what segments
        fraction = [[] for _ in range(self.voc_size[2])]

        ATC4_to_SMILES = pickle.load(
            open(os.path.join(CACHE_PATH, "atc4toSMILES.pkl"), "rb")
        )
        vocab_to_index = self.tokenizers[2].vocabulary.word2idx

        for atc4, smiles_ls in ATC4_to_SMILES.items():
            if atc4 in vocab_to_index:
                pos = vocab_to_index[atc4]
                SMILES[pos] += smiles_ls
                for smiles in smiles_ls:
                    try:
                        m = BRICS.BRICSDecompose(Chem.MolFromSmiles(smiles))
                        for frac in m:
                            fraction[pos].append(frac)
                    except:
                        pass
        # all segment set
        fraction_set = []
        for i in fraction:
            fraction_set += i
        fraction_set = list(set(fraction_set))  # set of all segments

        # ddi_mask
        ddi_mask_H = np.zeros((self.voc_size[2], len(fraction_set)))
        for idx, cur_fraction in enumerate(fraction):
            for frac in cur_fraction:
                ddi_mask_H[idx, fraction_set.index(frac)] = 1
        self.SMILES = SMILES

        return ddi_mask_H

    def generate_med_molecule_info_for_SafeDrug(self, radius=1):
        from rdkit import Chem

        atom_dict = defaultdict(lambda: len(atom_dict))
        bond_dict = defaultdict(lambda: len(bond_dict))
        fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
        edge_dict = defaultdict(lambda: len(edge_dict))
        MPNNSet, average_index = [], []

        def create_atoms(mol, atom_dict):
            """Transform the atom types in a molecule (e.g., H, C, and O)
            into the indices (e.g., H=0, C=1, and O=2).
            Note that each atom index considers the aromaticity.
            """
            atoms = [a.GetSymbol() for a in mol.GetAtoms()]
            for a in mol.GetAromaticAtoms():
                i = a.GetIdx()
                atoms[i] = (atoms[i], "aromatic")
            atoms = [atom_dict[a] for a in atoms]
            return np.array(atoms)

        def create_ijbonddict(mol, bond_dict):
            """Create a dictionary, in which each key is a node ID
            and each value is the tuples of its neighboring node
            and chemical bond (e.g., single and double) IDs.
            """
            i_jbond_dict = defaultdict(lambda: [])
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                bond = bond_dict[str(b.GetBondType())]
                i_jbond_dict[i].append((j, bond))
                i_jbond_dict[j].append((i, bond))
            return i_jbond_dict

        def extract_fingerprints(
            radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict
        ):
            """Extract the fingerprints from a molecular graph
            based on Weisfeiler-Lehman algorithm.
            """

            if (len(atoms) == 1) or (radius == 0):
                nodes = [fingerprint_dict[a] for a in atoms]

            else:
                nodes = atoms
                i_jedge_dict = i_jbond_dict

                for _ in range(radius):

                    """Update each node ID considering its neighboring nodes and edges.
                    The updated node IDs are the fingerprint IDs.
                    """
                    nodes_ = []
                    for i, j_edge in i_jedge_dict.items():
                        neighbors = [(nodes[j], edge) for j, edge in j_edge]
                        fingerprint = (nodes[i], tuple(sorted(neighbors)))
                        nodes_.append(fingerprint_dict[fingerprint])

                    """Also update each edge ID considering
                    its two nodes on both sides.
                    """
                    i_jedge_dict_ = defaultdict(lambda: [])
                    for i, j_edge in i_jedge_dict.items():
                        for j, edge in j_edge:
                            both_side = tuple(sorted((nodes[i], nodes[j])))
                            edge = edge_dict[(both_side, edge)]
                            i_jedge_dict_[i].append((j, edge))

                    nodes = nodes_
                    i_jedge_dict = i_jedge_dict_

            return np.array(nodes)

        for smilesList in self.SMILES:
            """Create each data with the above defined functions."""
            counter = 0  # counter how many drugs are under that ATC-3
            for smiles in smilesList:
                try:
                    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                    atoms = create_atoms(mol, atom_dict)
                    molecular_size = len(atoms)
                    i_jbond_dict = create_ijbonddict(mol, bond_dict)
                    fingerprints = extract_fingerprints(
                        radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict
                    )
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

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        patient_id, visit_pos = self.index_map[index]
        patient = self.patients[patient_id]

        conditions, procedures, drugs = [], [], []
        # locate all previous visits
        for visit in patient.visits[: visit_pos + 1]:
            conditions.append(visit.conditions)
            procedures.append(visit.procedures)
            drugs.append(visit.drugs)
        return {"conditions": conditions, "procedures": procedures, "drugs": drugs}

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
        print(info)


if __name__ == "__main__":
    from pyhealth.datasets import MIMIC3BaseDataset

    base_dataset = MIMIC3BaseDataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4"
    )
    drug_rec_dataloader = DrugRecDataset(base_dataset)
    print(len(drug_rec_dataloader))
    print(drug_rec_dataloader[0])
