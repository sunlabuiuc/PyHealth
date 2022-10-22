from typing import List, Tuple, Union
import torch
import torch.nn as nn
from pyhealth.datasets import BaseDataset
from pyhealth.models import BaseModel
from pyhealth.tokenizer import Tokenizer
import numpy as np
from pyhealth.models.utils import get_last_visit
from pyhealth.metrics import ddi_rate_score
from collections import defaultdict
from pathlib import Path
import os
from urllib import request
import pandas as pd
import pickle

# import model specific modules here
from rdkit import Chem
import rdkit.Chem.BRICS as BRICS


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
        N_fingerprint: the number of fingerprints
        dim: the dimension of the fingerprint vectors
        layer_hidden: the number of hidden layers
    """

    def __init__(self, N_fingerprint, dim, layer_hidden):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprint, dim)
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

    def forward(self, inputs):
        """
        Args:
            inputs: [fingerprints, adjacencies, molecular_sizes]
                - fingerprints: a list of fingerprints
                - adjacencies: a list of adjacency matrices
                - molecular_sizes: a list of the number of atoms in each molecule
        """
        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs

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
        ddi_mask_H: the mask matrix for MaskLinear layer
        tabels: a list of table names
        N_fingerprint: the number of fingerprints
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
        ddi_mask_H: np.ndarray,
        tables: List[str],
        N_fingerprints: int,
        molecule_set: List[tuple],
        average_projection: torch.Tensor,
        dropout: float = 0.5,
        num_layers: int = 1,
    ):
        super(SafeDrugLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        ddi_mask_H = torch.FloatTensor(ddi_mask_H)
        self.ddi_mask_H = nn.Parameter(ddi_mask_H, requires_grad=False)
        # med space size
        self.label_size = ddi_mask_H.shape[0]

        self.tables = tables
        self.rnn = nn.ModuleDict()
        for domain in self.tables:
            self.rnn[domain] = nn.GRU(
                input_size,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
            )

        self.query = nn.Sequential(
            nn.ReLU(), nn.Linear(len(tables) * hidden_size, hidden_size)
        )

        # local bipartite encoder
        self.bipartite_transform = nn.Linear(hidden_size, ddi_mask_H.shape[1])
        self.bipartite_output = MaskLinear(ddi_mask_H.shape[1], self.label_size, False)
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

        self.MPNN = MolecularGraphNeuralNetwork(
            N_fingerprints, hidden_size, layer_hidden=2
        )
        self.MPNN_output = nn.Linear(self.label_size, self.label_size)
        self.MPNN_layernorm = nn.LayerNorm(self.label_size)

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch proc essing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = nn.Parameter(torch.FloatTensor(np.zeros((M, N))), requires_grad=False)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i : i + m, j : j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def forward(self, X: torch.tensor, mask: torch.tensor):
        """
        Args:
            X: a dict with <str, [batch size, seq len, input_size]>
            mask: [batch size, seq len]
        Returns:
            outputs [batch size, seq len, hidden_size]
        """
        # get patient embedding by RNN
        patient_emb = []
        for domain in self.tables:
            domain_emb, _ = self.rnn[domain](X[domain])
            patient_emb.append(domain_emb)
        patient_representations = torch.cat(
            patient_emb, dim=-1
        )  # (batch, visit, dim * len(tables))
        queries = self.query(patient_representations)  # (batch, visit, hidden_size)
        query = get_last_visit(queries, mask)  # (batch, dim)

        # MPNN Encoder
        MPNN_emb = self.MPNN(
            [self.fingerprints, self.adjacencies, self.molecule_sizes]
        )  # (#molecule, hidden_size)
        MPNN_emb = torch.mm(self.average_projection, MPNN_emb)  # (#med, hidden_size)
        MPNN_match = torch.sigmoid(torch.mm(query, MPNN_emb.T))  # (patient, #med)
        MPNN_att = self.MPNN_layernorm(
            MPNN_match + self.MPNN_output(MPNN_match)
        )  # (batch, hidden_size)

        # Bipartite Encoder (use the bipartite encoder only for now)
        bipartite_emb = self.bipartite_transform(query)  # (batch, dim)
        bipartite_att = self.bipartite_output(
            bipartite_emb, self.ddi_mask_H.T
        )  # (batch, hidden_size)

        # combine
        result = bipartite_att * MPNN_att

        return result


class SafeDrug(BaseModel):
    """SafeDrug Class, use "task" as key to identify specific SafeDrug model and route there
    Args:
        dataset: the dataset object
        tables: the list of table names to use
        target: the target table name
        mode: the mode of the model, "multilabel", "multiclass" or "binary"
        embedding_dim: the embedding dimension
        hidden_dim: the hidden dimension
        kp: the keep probability in PID strategy
        target_ddi: the target ddi value
    """

    def __init__(
        self,
        dataset: BaseDataset,
        tables: List[str],
        target: str,
        mode: str,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        kp: float = 0.05,
        target_ddi: float = 0.08,
        **kwargs
    ):
        super(SafeDrug, self).__init__(
            dataset=dataset,
            tables=tables,
            target=target,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.target_ddi = target_ddi
        self.kp = kp

        self.tokenizers = {}
        for domain in tables:
            self.tokenizers[domain] = Tokenizer(
                dataset.get_all_tokens(key=domain), special_tokens=["<pad>", "<unk>"]
            )
        self.label_tokenizer = Tokenizer(dataset.get_all_tokens(key=target))

        self.embeddings = nn.ModuleDict()
        for domain in tables:
            # TODO: use get_pad_token_id() instead of hard code
            self.embeddings[domain] = nn.Embedding(
                self.tokenizers[domain].get_vocabulary_size(),
                embedding_dim,
                padding_idx=0,
            )

        # med space size
        self.label_size = self.label_tokenizer.get_vocabulary_size()

        ddi_adj = self.generate_ddi_adj()
        self.cal_ddi_adj = ddi_adj.copy()  # this one is for calculating the ddi_rate
        self.ddi_adj = nn.Parameter(
            torch.Tensor(ddi_adj), requires_grad=False
        )  # this one join the gradient descent
        ddi_mask_H = self.generate_ddi_mask_H()
        (
            molecule_set,
            N_fingerprints,
            average_projection,
        ) = self.generate_med_molecule_info()

        self.safedrug = SafeDrugLayer(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            tables=tables,
            N_fingerprints=N_fingerprints,
            molecule_set=molecule_set,
            average_projection=average_projection,
            ddi_mask_H=ddi_mask_H,
            **kwargs
        )

    def generate_ddi_adj(self):
        """get drug-drug interaction (DDI)"""
        cid2atc_dic = defaultdict(set)
        label_size = self.label_tokenizer.get_vocabulary_size()
        vocab_to_index = self.label_tokenizer.vocabulary

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
                if atc[:4] in vocab_to_index.token2idx:
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
        ddi_adj = np.zeros((label_size, label_size))
        for index, row in ddi_df.iterrows():
            # ddi
            cid1 = row["STITCH 1"]
            cid2 = row["STITCH 2"]
            # cid -> atc_level3
            for atc_i in cid2atc_dic[cid1]:
                for atc_j in cid2atc_dic[cid2]:
                    ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                    ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        self.ddi_adj = ddi_adj
        return ddi_adj

    def generate_ddi_mask_H(self):
        # idx_to_SMILES
        SMILES = [[] for _ in range(self.label_size)]
        # each idx contains what segments
        fraction = [[] for _ in range(self.label_size)]

        if self.dataset.dataset_name not in ["MIMIC-III", "MIMIC-IV"]:
            raise ValueError(
                "SafeDrug currently only supports mimic3 and mimic4 dataset, not {}.\nUser need to implement their own drug -> SMILES string mapping for other datasets.".format(
                    self.dataset.dataset_name
                )
            )

        atc3toSMILES = {}
        from pyhealth.medcode import ATC

        atc = ATC()
        for code in atc.graph.nodes:
            if len(code) != 7:
                continue
            code_atc3 = code[:4]
            smiles = atc.graph.nodes[code]["smiles"]
            if smiles != smiles:
                continue
            atc3toSMILES[code_atc3] = atc3toSMILES.get(code_atc3, []) + [smiles]
        # take first three
        atc3toSMILES = {k: v[:1] for k, v in atc3toSMILES.items()}

        vocab_to_index = self.label_tokenizer.vocabulary.token2idx

        for atc4, smiles_ls in atc3toSMILES.items():
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
        ddi_mask_H = np.zeros((self.label_size, len(fraction_set)))
        for idx, cur_fraction in enumerate(fraction):
            for frac in cur_fraction:
                ddi_mask_H[idx, fraction_set.index(frac)] = 1
        self.SMILES = SMILES
        return ddi_mask_H

    def generate_med_molecule_info(self, radius=1):
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

    def forward(self, device, **kwargs):
        """
        if "kwargs[domain][0][0] is list" means "use history", then run visit level RNN
        elif "kwargs[domain][0][0] is not list" means not "use history", then run code level RNN
        """
        for domain in self.tables:
            if type(kwargs[domain][0][0]) == list:
                kwargs[domain] = self.tokenizers[domain].batch_encode_3d(kwargs[domain])
                kwargs[domain] = torch.tensor(
                    kwargs[domain], dtype=torch.long, device=device
                )
                # (patient, visit, code, embedding_dim)
                kwargs[domain] = self.embeddings[domain](kwargs[domain])
                # (patient, visit, embedding_dim)
                kwargs[domain] = torch.sum(kwargs[domain], dim=2)
            else:
                raise ValueError("Sample data format is not correct")

        # get mask
        mask = torch.sum(kwargs[domain], dim=2) != 0
        mask[:, 0] = 1

        logits = self.safedrug(kwargs, mask)

        # calculate the ddi_loss by PID stragegy and add to final loss
        pred_prob = torch.sigmoid(logits)
        mul_pred_prob = pred_prob.T @ pred_prob  # (voc_size, voc_size)
        batch_ddi_loss = (
            torch.sum(mul_pred_prob.mul(self.ddi_adj)) / self.ddi_adj.shape[0] ** 2
        )

        pred_prob = pred_prob.detach().cpu().numpy()
        pred_prob[pred_prob >= 0.5] = 1
        pred_prob[pred_prob < 0.5] = 0
        pred_prob = [np.where(sample == 1)[0] for sample in pred_prob]

        cur_ddi_rate = ddi_rate_score(pred_prob, self.cal_ddi_adj)
        if cur_ddi_rate > self.target_ddi:
            beta = max(0, 1 + (self.target_ddi - cur_ddi_rate) / self.kp)
            add_loss, beta = batch_ddi_loss, beta
        else:
            add_loss, beta = 0, 1

        # obtain target, loss, prob, pred
        loss, y_true, y_prod, y_pred = self.cal_loss_and_output(
            logits, device, **kwargs
        )

        return {
            "loss": beta * loss + (1 - beta) * add_loss,
            "y_prob": y_prod,
            "y_pred": y_pred,
            "y_true": y_true,
        }
