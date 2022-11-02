from collections import defaultdict
from copy import deepcopy
from typing import List

import numpy as np
import rdkit.Chem.BRICS as BRICS
import torch
import torch.nn as nn
# import model specific modules here
from rdkit import Chem

from pyhealth.datasets import BaseDataset
from pyhealth.medcode import ATC
from pyhealth.metrics import ddi_rate_score
from pyhealth.models import BaseModel
from pyhealth.models.utils import batch_to_multihot
from pyhealth.models.utils import get_last_visit


class MaskLinear(nn.Module):
    """The MaskLinear layer.
    We customize the linear layer and add a hard mask for the parameter matrix.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    """

    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / self.weight.size(1) ** 0.5
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        """
        Args:
            input: [batch size, ..., input_size]
            mask: [input_size, output_size], the same size as the weight matrix
        Returns:
            outputs [batch size, ..., output_size]
        """
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
                self.__class__.__name__
                + " ("
                + str(self.in_features)
                + " -> "
                + str(self.out_features)
                + ")"
        )


class MolecularGraphNeuralNetwork(nn.Module):
    """refer to https://github.com/masashitsubaki/molecularGNN_smiles
    Args:
        num_fingerprints: the number of fingerprints
        dim: the dimension of the fingerprint vectors
        layer_hidden: the number of hidden layers
    """

    def __init__(self, num_fingerprints, dim, layer_hidden):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(num_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(layer_hidden)]
        )
        self.layer_hidden = layer_hidden

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, fingerprints, adjacencies, molecular_sizes):
        """
        Args:
            inputs: [fingerprints, adjacencies, molecular_sizes]
                - fingerprints: a list of fingerprints
                - adjacencies: a list of adjacency matrices
                - molecular_sizes: a list of the number of atoms in each molecule
        """
        """MPNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for layer in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, layer)
            # fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.
            fingerprint_vectors = hs

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors


class SafeDrugLayer(nn.Module):
    """The SafeDrug layer.
    Args:
        input_size: the size of the input vector
        hidden_size: the size of the hidden vector
        mask_H: the mask matrix for MaskLinear layer
        tabels: a list of table names
        num_fingerprints: the number of fingerprints
        molecular_set: a list of molecule tuples (A, B, C)
            - A <matrix>: fingerprints of atoms in the molecule
            - B <matrix>: adjacency matrix of the molecule
            - C <int>: molecular_size
        average: the average projection for aggregating multiple molecules of the same drug into one vector
        dropout: the dropout rate
        num_layers: the number of layers
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            mask_H: torch.Tensor,
            ddi_adj,
            num_fingerprints: int,
            molecule_set: List[tuple],
            average_projection: torch.Tensor,
            dropout: float = 0.5,
            num_layers: int = 1,
            kp: float = 0.05,
            target_ddi: float = 0.08,
    ):
        super(SafeDrugLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.kp = kp
        self.target_ddi = target_ddi
        self.ddi_adj = nn.Parameter(ddi_adj, requires_grad=False)

        self.mask_H = nn.Parameter(mask_H, requires_grad=False)
        # med space size
        self.label_size = mask_H.shape[0]

        # local bipartite encoder
        self.bipartite_transform = nn.Linear(hidden_size, mask_H.shape[1])
        self.bipartite_output = MaskLinear(mask_H.shape[1], self.label_size, False)
        # self.bipartite_output = nn.Linear(ddi_mask_H.shape[1], hidden_size)

        # global MPNN encoder (add fingerprints and adjacency matrix to parameter list)
        MPNN_molecule_set = list(zip(*molecule_set))

        # process three parts of information
        fingerprints = torch.cat(MPNN_molecule_set[0])
        self.fingerprints = nn.Parameter(fingerprints, requires_grad=False)
        adjacencies = self.pad(MPNN_molecule_set[1], 0)
        self.adjacencies = nn.Parameter(adjacencies, requires_grad=False)
        self.molecule_sizes = MPNN_molecule_set[2]
        self.average_projection = nn.Parameter(average_projection, requires_grad=False)

        self.MPNN = MolecularGraphNeuralNetwork(num_fingerprints, hidden_size, layer_hidden=2)
        self.MPNN_output = nn.Linear(self.label_size, self.label_size)
        self.MPNN_layernorm = nn.LayerNorm(self.label_size)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N)))
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i: i + m, j: j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def calculate_loss(self, logits, y_prob, labels):
        mul_pred_prob = y_prob.T @ y_prob  # (voc_size, voc_size)
        batch_ddi_loss = (
                torch.sum(mul_pred_prob.mul(self.ddi_adj)) / self.ddi_adj.shape[0] ** 2
        )

        y_pred = y_prob.detach().cpu().numpy()
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y_pred = [np.where(sample == 1)[0] for sample in y_pred]

        cur_ddi_rate = ddi_rate_score(y_pred, self.ddi_adj.cpu().numpy())
        if cur_ddi_rate > self.target_ddi:
            beta = max(0, 1 + (self.target_ddi - cur_ddi_rate) / self.kp)
            add_loss, beta = batch_ddi_loss, beta
        else:
            add_loss, beta = 0, 1

        # obtain target, loss, prob, pred
        bce_loss = self.loss_fn(logits, labels)

        loss = beta * bce_loss + (1 - beta) * add_loss
        return loss

    def forward(self, queries, mask: torch.tensor, drugs):
        """
        Args:
            X: a dict with <str, [batch size, seq len, input_size]>
            mask: [batch size, seq len]
        Returns:
            outputs [batch size, seq len, hidden_size]
        """
        query = get_last_visit(queries, mask)  # (batch, dim)

        # MPNN Encoder
        MPNN_emb = self.MPNN(self.fingerprints, self.adjacencies, self.molecule_sizes)  # (#molecule, hidden_size)
        MPNN_emb = torch.mm(self.average_projection, MPNN_emb)  # (#med, hidden_size)
        MPNN_match = torch.sigmoid(torch.mm(query, MPNN_emb.T))  # (patient, #med)
        MPNN_att = self.MPNN_layernorm(
            MPNN_match + self.MPNN_output(MPNN_match)
        )  # (batch, #med)

        # Bipartite Encoder (use the bipartite encoder only for now)
        bipartite_emb = self.bipartite_transform(query)  # (batch, dim)
        bipartite_att = self.bipartite_output(
            bipartite_emb, self.mask_H.T
        )  # (batch, hidden_size)

        # combine
        logits = bipartite_att * MPNN_att

        # calculate the ddi_loss by PID stragegy and add to final loss
        y_prob = torch.sigmoid(logits)

        loss = self.calculate_loss(logits, y_prob, drugs)

        return loss, y_prob


class SafeDrug(BaseModel):
    """SafeDrug Class, use "task" as key to identify specific SafeDrug model and route there
    Args:
        dataset: the dataset object
        feature_keys: the list of table names to use
        label_key: the target table name
        mode: the mode of the model, "multilabel", "multiclass" or "binary"
        embedding_dim: the embedding dimension
        hidden_dim: the hidden dimension
        kp: the keep probability in PID strategy
        target_ddi: the target ddi value
    """

    def __init__(
            self,
            dataset: BaseDataset,
            embedding_dim: int = 128,
            hidden_dim: int = 128,
            num_layers: int = 1,
            dropout: float = 0.5,
            kp: float = 0.05,
            target_ddi: float = 0.08,
            **kwargs
    ):
        super(SafeDrug, self).__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key="label",
            mode="multilabel",
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.target_ddi = target_ddi
        self.kp = kp

        self.feat_tokenizers = self._get_feature_tokenizers()
        self.label_tokenizer = self._get_label_tokenizer()
        self.embeddings = self._get_embeddings(self.feat_tokenizers, embedding_dim)

        # med space size
        self.label_size = self.label_tokenizer.get_vocabulary_size()

        self.ddi_adj = self.generate_ddi_adj()
        self.all_smiles_list = self.generate_smiles_list()
        self.mask_H = self.generate_mask_H()
        molecule_set, num_fingerprints, average_projection = self.generate_molecule_info()

        # define the rnn layers
        self.cond_rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.proc_rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.safedrug = SafeDrugLayer(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_fingerprints=num_fingerprints,
            molecule_set=molecule_set,
            average_projection=average_projection,
            mask_H=self.mask_H,
            ddi_adj=self.ddi_adj,
            kp=kp,
            target_ddi=target_ddi,
            **kwargs
        )

    def generate_ddi_adj(self):
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True)
        label_size = self.label_tokenizer.get_vocabulary_size()
        vocab_to_index = self.label_tokenizer.vocabulary
        ddi_adj = np.zeros((label_size, label_size))
        ddi_atc3 = [[l[0][:4], l[1][:4]] for l in ddi]
        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        ddi_adj = torch.FloatTensor(ddi_adj)
        return ddi_adj

    def generate_smiles_list(self):
        atc3_to_smiles = {}
        atc = ATC()
        for code in atc.graph.nodes:
            if len(code) != 7:
                continue
            code_atc3 = code[:4]
            smiles = atc.graph.nodes[code]["smiles"]
            if smiles != smiles:
                continue
            atc3_to_smiles[code_atc3] = atc3_to_smiles.get(code_atc3, []) + [smiles]
        # just take first one for computational efficiency
        atc3_to_smiles = {k: v[:1] for k, v in atc3_to_smiles.items()}
        all_smiles_list = [[] for _ in range(self.label_size)]
        vocab_to_index = self.label_tokenizer.vocabulary
        for atc3, smiles_list in atc3_to_smiles.items():
            if atc3 in vocab_to_index:
                index = vocab_to_index(atc3)
                all_smiles_list[index] += smiles_list
        return all_smiles_list

    def generate_mask_H(self):
        all_substructures_list = [[] for _ in range(self.label_size)]
        for index, smiles_list in enumerate(self.all_smiles_list):
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                substructures = BRICS.BRICSDecompose(mol)
                all_substructures_list[index] += substructures
        # all segment set
        substructures_set = list(set(sum(all_substructures_list, [])))
        # mask_H
        mask_H = np.zeros((self.label_size, len(substructures_set)))
        for index, substructures in enumerate(all_substructures_list):
            for s in substructures:
                mask_H[index, substructures_set.index(s)] = 1
        mask_H = torch.FloatTensor(mask_H)
        return mask_H

    def generate_molecule_info(self, radius=1):
        def create_atoms(mol, atom2idx):
            """Transform the atom types in a molecule (e.g., H, C, and O)
            into the indices (e.g., H=0, C=1, and O=2).
            Note that each atom index considers the aromaticity.
            """
            atoms = [a.GetSymbol() for a in mol.GetAtoms()]
            for a in mol.GetAromaticAtoms():
                i = a.GetIdx()
                atoms[i] = (atoms[i], "aromatic")
            atoms = [atom2idx[a] for a in atoms]
            return np.array(atoms)

        def create_ijbonddict(mol, bond2idx):
            """Create a dictionary, in which each key is a node ID
            and each value is the tuples of its neighboring node
            and chemical bond (e.g., single and double) IDs.
            """
            i_jbond_dict = defaultdict(lambda: [])
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                bond = bond2idx[str(b.GetBondType())]
                i_jbond_dict[i].append((j, bond))
                i_jbond_dict[j].append((i, bond))
            return i_jbond_dict

        def extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint2idx, edge2idx):
            """Extract the fingerprints from a molecular graph
            based on Weisfeiler-Lehman algorithm.
            """
            nodes = [fingerprint2idx[a] for a in atoms]
            i_jedge_dict = i_jbond_dict

            for _ in range(radius):

                """Update each node ID considering its neighboring nodes and edges.
                The updated node IDs are the fingerprint IDs.
                """
                nodes_ = deepcopy(nodes)
                for i, j_edge in i_jedge_dict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    nodes_[i] = fingerprint2idx[fingerprint]

                """Also update each edge ID considering
                its two nodes on both sides.
                """
                i_jedge_dict_ = defaultdict(list)
                for i, j_edge in i_jedge_dict.items():
                    for j, edge in j_edge:
                        both_side = tuple(sorted((nodes[i], nodes[j])))
                        edge = edge2idx[(both_side, edge)]
                        i_jedge_dict_[i].append((j, edge))

                nodes = deepcopy(nodes_)
                i_jedge_dict = deepcopy(i_jedge_dict_)
                del nodes_, i_jedge_dict_

            return np.array(nodes)

        atom2idx = defaultdict(lambda: len(atom2idx))
        bond2idx = defaultdict(lambda: len(bond2idx))
        fingerprint2idx = defaultdict(lambda: len(fingerprint2idx))
        edge2idx = defaultdict(lambda: len(edge2idx))
        molecule_set, average_index = [], []

        for smiles_list in self.all_smiles_list:
            """Create each data with the above defined functions."""
            counter = 0  # counter how many drugs are under that ATC-3
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                mol = Chem.AddHs(mol)
                atoms = create_atoms(mol, atom2idx)
                molecular_size = len(atoms)
                i_jbond_dict = create_ijbonddict(mol, bond2idx)
                fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint2idx, edge2idx)
                adjacency = Chem.GetAdjacencyMatrix(mol)
                """Transform the above each data of numpy to pytorch tensor."""
                fingerprints = torch.LongTensor(fingerprints)
                adjacency = torch.FloatTensor(adjacency)
                molecule_set.append((fingerprints, adjacency, molecular_size))
                counter += 1
            average_index.append(counter)

        num_fingerprints = len(fingerprint2idx)
        # transform into projection matrix
        n_col = sum(average_index)
        n_row = len(average_index)
        average_projection = np.zeros((n_row, n_col))
        col_counter = 0
        for i, item in enumerate(average_index):
            if item > 0:
                average_projection[i, col_counter: col_counter + item] = 1 / item
            col_counter += item
        average_projection = torch.FloatTensor(average_projection)
        return molecule_set, num_fingerprints, average_projection

    def forward(self, device, conditions, procedures, label, **kwargs):
        conditions = self.feat_tokenizers["conditions"].batch_encode_3d(conditions)
        conditions = torch.tensor(conditions, dtype=torch.long, device=device)
        # (patient, visit, code, embedding_dim)
        conditions = self.embeddings["conditions"](conditions)
        # (patient, visit, embedding_dim)
        conditions = torch.sum(conditions, dim=2)
        # (batch, visit, hidden_size)
        conditions, _ = self.cond_rnn(conditions)

        procedures = self.feat_tokenizers["procedures"].batch_encode_3d(procedures)
        procedures = torch.tensor(procedures, dtype=torch.long, device=device)
        # (patient, visit, code, embedding_dim)
        procedures = self.embeddings["procedures"](procedures)
        # (patient, visit, embedding_dim)
        procedures = torch.sum(procedures, dim=2)
        # (batch, visit, hidden_size)
        procedures, _ = self.proc_rnn(procedures)

        # (batch, visit, 2 * hidden_size)
        patient_representations = torch.cat([conditions, procedures], dim=-1)
        # (batch, visit, hidden_size)
        queries = self.query(patient_representations)

        # get mask
        mask = torch.sum(conditions, dim=2) != 0

        label = self.label_tokenizer.batch_encode_2d(label, padding=False, truncation=False)
        label = batch_to_multihot(label, self.label_tokenizer.get_vocabulary_size())
        label = label.to(device)

        loss, y_prob = self.safedrug(queries, mask, label)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": label,
        }


if __name__ == '__main__':
    from pyhealth.datasets import MIMIC3Dataset
    from torch.utils.data import DataLoader
    from pyhealth.utils import collate_fn_dict
    from pyhealth.tasks import drug_recommendation_mimic3_fn

    dataset = MIMIC3Dataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=True,
        code_mapping={"NDC": "ATC"},
        refresh_cache=False,
    )

    # visit level + multilabel
    dataset.set_task(drug_recommendation_mimic3_fn)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
    )
    model = SafeDrug(
        dataset=dataset,
    )
    model.to("cuda")
    batch = iter(dataloader).next()
    output = model(**batch, device="cuda")
    print(output["loss"])
