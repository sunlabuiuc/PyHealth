from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple, Dict, Optional
import os

import numpy as np
import rdkit.Chem.BRICS as BRICS
import torch
import torch.nn as nn
from rdkit import Chem

from pyhealth.datasets import SampleEHRDataset
from pyhealth.medcode import ATC
from pyhealth.metrics import ddi_rate_score
from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit
from pyhealth import BASE_CACHE_PATH as CACHE_PATH

class MaskLinear(nn.Module):
    """MaskLinear layer.

    This layer wraps the PyTorch linear layer and adds a hard mask for
    the parameter matrix. It is used in the SafeDrug model.

    Args:
        in_features: input feature size.
        out_features: output feature size.
        bias: whether to use bias. Default is True.
    """

    def __init__(self, in_features: int, out_features: int, bias=True):
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

    def forward(self, input: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """
        Args:
            input: input feature tensor of shape [batch size, ..., input_size].
            mask: mask tensor of shape [input_size, output_size], i.e., the same
                size as the weight matrix.

        Returns:
            Output tensor of shape [batch size, ..., output_size].
        """
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class MolecularGraphNeuralNetwork(nn.Module):
    """Molecular Graph Neural Network.

    Paper: Masashi Tsubaki et al. Compound-protein interaction
    prediction with end-to-end learning of neural networks for
    graphs and sequences. Bioinformatics, 2019.

    Args:
        num_fingerprints: total number of fingerprints.
        dim: embedding dimension of the fingerprint vectors.
        layer_hidden: number of hidden layers.
    """

    def __init__(self, num_fingerprints, dim, layer_hidden):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.layer_hidden = layer_hidden
        self.embed_fingerprint = nn.Embedding(num_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(layer_hidden)]
        )

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
            fingerprints: a list of fingerprints
            adjacencies: a list of adjacency matrices
            molecular_sizes: a list of the number of atoms in each molecule
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
    """SafeDrug model.

    Paper: Chaoqi Yang et al. SafeDrug: Dual Molecular Graph Encoders for
    Recommending Effective and Safe Drug Combinations. IJCAI 2021.

    This layer is used in the SafeDrug model. But it can also be used as a
    standalone layer. Note that we improve the layer a little bit to make it
    compatible with the package. Original code can be found at 
    https://github.com/ycq091044/SafeDrug/blob/main/src/models.py.

    Args:
        hidden_size: hidden feature size.
        mask_H: the mask matrix H of shape [num_drugs, num_substructures].
        ddi_adj: an adjacency tensor of shape [num_drugs, num_drugs].
        num_fingerprints: total number of different fingerprints.
        molecule_set: a list of molecule tuples (A, B, C) of length num_molecules.
            - A <torch.tensor>: fingerprints of atoms in the molecule
            - B <torch.tensor>: adjacency matrix of the molecule
            - C <int>: molecular_size
        average_projection: a tensor of shape [num_drugs, num_molecules] representing
            the average projection for aggregating multiple molecules of the
            same drug into one vector.
        kp: correcting factor for the proportional signal. Default is 0.5.
        target_ddi: DDI acceptance rate. Default is 0.08.
    """

    def __init__(
        self,
        hidden_size: int,
        mask_H: torch.Tensor,
        ddi_adj: torch.Tensor,
        num_fingerprints: int,
        molecule_set: List[Tuple],
        average_projection: torch.Tensor,
        kp: float = 0.05,
        target_ddi: float = 0.08,
    ):
        super(SafeDrugLayer, self).__init__()
        self.hidden_size = hidden_size
        self.kp = kp
        self.target_ddi = target_ddi

        self.mask_H = nn.Parameter(mask_H, requires_grad=False)
        self.ddi_adj = nn.Parameter(ddi_adj, requires_grad=False)

        # medication space size
        label_size = mask_H.shape[0]

        # local bipartite encoder
        self.bipartite_transform = nn.Linear(hidden_size, mask_H.shape[1])
        # self.bipartite_output = MaskLinear(mask_H.shape[1], label_size, False)
        self.bipartite_output = nn.Linear(mask_H.shape[1], label_size)

        # global MPNN encoder (add fingerprints and adjacency matrix to parameter list)
        mpnn_molecule_set = list(zip(*molecule_set))

        # process three parts of information
        fingerprints = torch.cat(mpnn_molecule_set[0])
        self.fingerprints = nn.Parameter(fingerprints, requires_grad=False)
        adjacencies = self.pad(mpnn_molecule_set[1], 0)
        self.adjacencies = nn.Parameter(adjacencies, requires_grad=False)
        self.molecule_sizes = mpnn_molecule_set[2]
        self.average_projection = nn.Parameter(average_projection, requires_grad=False)

        self.mpnn = MolecularGraphNeuralNetwork(
            num_fingerprints, hidden_size, layer_hidden=2
        )
        self.mpnn_output = nn.Linear(label_size, label_size)
        self.mpnn_layernorm = nn.LayerNorm(label_size)

        self.test = nn.Linear(hidden_size, label_size)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def pad(self, matrices, pad_value):
        """Pads the list of matrices.

        Padding with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C], we obtain a new
        matrix [A00, 0B0, 00C], where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N)))
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i : i + m, j : j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def calculate_loss(
        self, logits: torch.Tensor, y_prob: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        mul_pred_prob = y_prob.T @ y_prob  # (voc_size, voc_size)
        batch_ddi_loss = (
            torch.sum(mul_pred_prob.mul(self.ddi_adj)) / self.ddi_adj.shape[0] ** 2
        )

        y_pred = y_prob.clone().detach().cpu().numpy()
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y_pred = [np.where(sample == 1)[0] for sample in y_pred]

        cur_ddi_rate = ddi_rate_score(y_pred, self.ddi_adj.cpu().numpy())
        if cur_ddi_rate > self.target_ddi:
            beta = max(0.0, 1 + (self.target_ddi - cur_ddi_rate) / self.kp)
            add_loss, beta = batch_ddi_loss, beta
        else:
            add_loss, beta = 0, 1

        # obtain target, loss, prob, pred
        bce_loss = self.loss_fn(logits, labels)

        loss = beta * bce_loss + (1 - beta) * add_loss
        # loss = bce_loss
        return loss

    def forward(
        self,
        patient_emb: torch.tensor,
        drugs: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            patient_emb: a tensor of shape [patient, visit, input_size].
            drugs: a multihot tensor of shape [patient, num_labels].
            mask: an optional tensor of shape [patient, visit] where 1
                indicates valid visits and 0 indicates invalid visits.

        Returns:
            loss: a scalar tensor representing the loss.
            y_prob: a tensor of shape [patient, num_labels] representing
                the probability of each drug.
        """
        if mask is None:
            mask = torch.ones_like(patient_emb[:, :, 0])

        query = get_last_visit(patient_emb, mask)  # (batch, dim)

        # MPNN Encoder
        MPNN_emb = self.mpnn(
            self.fingerprints, self.adjacencies, self.molecule_sizes
        )  # (#molecule, hidden_size)
        MPNN_emb = torch.mm(self.average_projection, MPNN_emb)  # (#med, hidden_size)
        MPNN_match = torch.sigmoid(torch.mm(query, MPNN_emb.T))  # (patient, #med)
        MPNN_att = self.mpnn_layernorm(
            MPNN_match + self.mpnn_output(MPNN_match)
        )  # (batch, #med)

        # Bipartite Encoder (use the bipartite encoder only for now)
        bipartite_emb = torch.sigmoid(self.bipartite_transform(query))  # (batch, dim)
        bipartite_att = self.bipartite_output(
            bipartite_emb
        )  # (batch, #med)

        # combine
        logits = bipartite_att * MPNN_att

        # calculate the ddi_loss by PID stragegy and add to final loss
        y_prob = torch.sigmoid(logits)
        loss = self.calculate_loss(logits, y_prob, drugs)

        return loss, y_prob


class SafeDrug(BaseModel):
    """SafeDrug model.

    Paper: Chaoqi Yang et al. SafeDrug: Dual Molecular Graph Encoders for
    Recommending Effective and Safe Drug Combinations. IJCAI 2021.

    Note:
        This model is only for medication prediction which takes conditions
        and procedures as feature_keys, and drugs as label_key. It only operates
        on the visit level.

    Note:
        This model only accepts ATC level 3 as medication codes.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        num_layers: the number of layers used in RNN. Default is 1.
        dropout: the dropout rate. Default is 0.5.
        **kwargs: other parameters for the SafeDrug layer.
    """

    def __init__(
        self,
        dataset: SampleEHRDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.5,
        **kwargs,
    ):
        super(SafeDrug, self).__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            mode="multilabel",
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)

        # drug space size
        self.label_size = self.label_tokenizer.get_vocabulary_size()

        self.all_smiles_list = self.generate_smiles_list()
        mask_H = self.generate_mask_H()
        (
            molecule_set,
            num_fingerprints,
            average_projection,
        ) = self.generate_molecule_info()
        ddi_adj = self.generate_ddi_adj()

        self.cond_rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.proc_rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # validate kwargs for GAMENet layer
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")
        if "mask_H" in kwargs:
            raise ValueError("mask_H is determined by the dataset")
        if "ddi_adj" in kwargs:
            raise ValueError("ddi_adj is determined by the dataset")
        if "num_fingerprints" in kwargs:
            raise ValueError("num_fingerprints is determined by the dataset")
        if "molecule_set" in kwargs:
            raise ValueError("molecule_set is determined by the dataset")
        if "average_projection" in kwargs:
            raise ValueError("average_projection is determined by the dataset")
        self.safedrug = SafeDrugLayer(
            hidden_size=hidden_dim,
            mask_H=mask_H,
            ddi_adj=ddi_adj,
            num_fingerprints=num_fingerprints,
            molecule_set=molecule_set,
            average_projection=average_projection,
            **kwargs,
        )
        
        # save ddi adj
        ddi_adj = self.generate_ddi_adj()
        np.save(os.path.join(CACHE_PATH, "ddi_adj.npy"), ddi_adj.numpy())

    def generate_ddi_adj(self) -> torch.tensor:
        """Generates the DDI graph adjacency matrix."""
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True)
        label_size = self.label_tokenizer.get_vocabulary_size()
        vocab_to_index = self.label_tokenizer.vocabulary
        ddi_adj = np.zeros((label_size, label_size))
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi
        ]
        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        ddi_adj = torch.FloatTensor(ddi_adj)
        return ddi_adj

    def generate_smiles_list(self) -> List[List[str]]:
        """Generates the list of SMILES strings."""
        atc3_to_smiles = {}
        atc = ATC()
        for code in atc.graph.nodes:
            if len(code) != 7:
                continue
            code_atc3 = ATC.convert(code, level=3)
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

    def generate_mask_H(self) -> torch.tensor:
        """Generates the molecular segmentation mask H."""
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

    def generate_molecule_info(self, radius: int = 1):
        """Generates the molecule information."""

        def create_atoms(mol, atom2idx):
            """Transform the atom types in a molecule (e.g., H, C, and O)
            into the indices (e.g., H=0, C=1, and O=2). Note that each atom
            index considers the aromaticity.
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

        def extract_fingerprints(r, atoms, i_jbond_dict, fingerprint2idx, edge2idx):
            """Extract the fingerprints from a molecular graph
            based on Weisfeiler-Lehman algorithm.
            """
            nodes = [fingerprint2idx[a] for a in atoms]
            i_jedge_dict = i_jbond_dict

            for _ in range(r):

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
                fingerprints = extract_fingerprints(
                    radius, atoms, i_jbond_dict, fingerprint2idx, edge2idx
                )
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
                average_projection[i, col_counter : col_counter + item] = 1 / item
            col_counter += item
        average_projection = torch.FloatTensor(average_projection)
        return molecule_set, num_fingerprints, average_projection

    def forward(
        self,
        conditions: List[List[List[str]]],
        procedures: List[List[List[str]]],
        drugs: List[List[str]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            conditions: a nested list in three levels [patient, visit, condition].
            procedures: a nested list in three levels [patient, visit, procedure].
            drugs: a nested list in two levels [patient, drug].

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor of shape [patient, visit, num_labels] representing
                    the probability of each drug.
                y_true: a tensor of shape [patient, visit, num_labels] representing
                    the ground truth of each drug.
        """
        conditions = self.feat_tokenizers["conditions"].batch_encode_3d(conditions)
        # (patient, visit, code)
        conditions = torch.tensor(conditions, dtype=torch.long, device=self.device)
        # (patient, visit, code, embedding_dim)
        conditions = self.embeddings["conditions"](conditions)
        # (patient, visit, embedding_dim)
        conditions = torch.sum(conditions, dim=2)
        # (batch, visit, hidden_size)
        conditions, _ = self.cond_rnn(conditions)

        procedures = self.feat_tokenizers["procedures"].batch_encode_3d(procedures)
        # (patient, visit, code)
        procedures = torch.tensor(procedures, dtype=torch.long, device=self.device)
        # (patient, visit, code, embedding_dim)
        procedures = self.embeddings["procedures"](procedures)
        # (patient, visit, embedding_dim)
        procedures = torch.sum(procedures, dim=2)
        # (batch, visit, hidden_size)
        procedures, _ = self.proc_rnn(procedures)

        # (batch, visit, 2 * hidden_size)
        patient_emb = torch.cat([conditions, procedures], dim=-1)
        # (batch, visit, hidden_size)
        patient_emb = self.query(patient_emb)

        # get mask
        mask = torch.sum(conditions, dim=2) != 0

        drugs = self.prepare_labels(drugs, self.label_tokenizer)

        loss, y_prob = self.safedrug(patient_emb, drugs, mask)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": drugs,
        }
