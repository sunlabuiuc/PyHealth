import torch
import os
import math
import pkg_resources
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union
from rdkit import Chem
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import multilabel_margin_loss

from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit
from pyhealth.models.utils import batch_to_multihot
from pyhealth.metrics import ddi_rate_score
from pyhealth.medcode import ATC
from pyhealth.datasets import SampleEHRDataset

from pyhealth import BASE_CACHE_PATH as CACHE_PATH

def graph_batch_from_smiles(smiles_list, device=torch.device("cpu")):
    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    graphs = [smiles2graph(x) for x in smiles_list]

    for idx, graph in enumerate(graphs):
        edge_idxes.append(graph["edge_index"] + lstnode)
        edge_feats.append(graph["edge_feat"])
        node_feats.append(graph["node_feat"])
        lstnode += graph["num_nodes"]
        batch.append(np.ones(graph["num_nodes"], dtype=np.int64) * idx)

    result = {
        "edge_index": np.concatenate(edge_idxes, axis=-1),
        "edge_attr": np.concatenate(edge_feats, axis=0),
        "batch": np.concatenate(batch, axis=0),
        "x": np.concatenate(node_feats, axis=0),
    }
    result = {k: torch.from_numpy(v).to(device) for k, v in result.items()}
    result["num_nodes"] = lstnode
    result["num_edges"] = result["edge_index"].shape[1]
    return result


class StaticParaDict(torch.nn.Module):
    def __init__(self, **kwargs):
        super(StaticParaDict, self).__init__()
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, torch.nn.Parameter(v, requires_grad=False))
            elif isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                setattr(self, k, torch.nn.Parameter(v, requires_grad=False))
            else:
                setattr(self, k, v)

    def forward(self, key: str) -> Any:
        return getattr(self, key)

    def __getitem__(self, key: str) -> Any:
        return self(key)

    def __setitem__(self, key: str, value: Any):
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if isinstance(value, torch.Tensor):
            value = torch.nn.Parameter(value, requires_grad=False)
        setattr(self, key, value)


class GINConv(torch.nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super(GINConv, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 2 * embedding_dim),
            torch.nn.BatchNorm1d(2 * embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embedding_dim, embedding_dim),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=embedding_dim)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
    ) -> torch.Tensor:
        edge_feats = self.bond_encoder(edge_feats)
        message_node = torch.index_select(input=node_feats, dim=0, index=edge_index[1])
        message = torch.relu(message_node + edge_feats)
        dim = message.shape[-1]

        message_reduce = torch.zeros(num_nodes, dim).to(message)
        index = edge_index[0].unsqueeze(-1).repeat(1, dim)
        message_reduce.scatter_add_(dim=0, index=index, src=message)

        return self.mlp((1 + self.eps) * node_feats + message_reduce)


class GINGraph(torch.nn.Module):
    def __init__(
        self, num_layers: int = 4, embedding_dim: int = 64, dropout: float = 0.7
    ):
        super(GINGraph, self).__init__()
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim=embedding_dim)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout_fun = torch.nn.Dropout(dropout)
        for layer in range(self.num_layers):
            self.convs.append(GINConv(embedding_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(embedding_dim))

    def forward(self, graph: Dict[str, Union[int, torch.Tensor]]) -> torch.Tensor:
        h_list = [self.atom_encoder(graph["x"])]
        for layer in range(self.num_layers):
            h = self.batch_norms[layer](
                self.convs[layer](
                    node_feats=h_list[layer],
                    edge_feats=graph["edge_attr"],
                    edge_index=graph["edge_index"],
                    num_nodes=graph["num_nodes"],
                    num_edges=graph["num_edges"],
                )
            )
            if layer != self.num_layers - 1:
                h = self.dropout_fun(torch.relu(h))
            else:
                h = self.dropout_fun(h)
            h_list.append(h)

        batch_size, dim = graph["batch"].max().item() + 1, h_list[-1].shape[-1]
        out_feat = torch.zeros(batch_size, dim).to(h_list[-1])
        cnt = torch.zeros_like(out_feat).to(out_feat)
        index = graph["batch"].unsqueeze(-1).repeat(1, dim)

        out_feat.scatter_add_(dim=0, index=index, src=h_list[-1])
        cnt.scatter_add_(
            dim=0, index=index, src=torch.ones_like(h_list[-1]).to(h_list[-1])
        )

        return out_feat / (cnt + 1e-9)


class MAB(torch.nn.Module):
    def __init__(
        self, Qdim: int, Kdim: int, Vdim: int, number_heads: int, use_ln: bool = False
    ):
        super(MAB, self).__init__()
        self.Vdim = Vdim
        self.number_heads = number_heads

        assert (
            self.Vdim % self.number_heads == 0
        ), "the dim of features should be divisible by number_heads"

        self.Qdense = torch.nn.Linear(Qdim, self.Vdim)
        self.Kdense = torch.nn.Linear(Kdim, self.Vdim)
        self.Vdense = torch.nn.Linear(Kdim, self.Vdim)
        self.Odense = torch.nn.Linear(self.Vdim, self.Vdim)

        self.use_ln = use_ln
        if self.use_ln:
            self.ln1 = torch.nn.LayerNorm(self.Vdim)
            self.ln2 = torch.nn.LayerNorm(self.Vdim)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        Q, K, V = self.Qdense(X), self.Kdense(Y), self.Vdense(Y)
        batch_size, dim_split = Q.shape[0], self.Vdim // self.number_heads

        Q_split = torch.cat(Q.split(dim_split, 2), 0)
        K_split = torch.cat(K.split(dim_split, 2), 0)
        V_split = torch.cat(V.split(dim_split, 2), 0)

        Attn = torch.matmul(Q_split, K_split.transpose(1, 2))
        Attn = torch.softmax(Attn / math.sqrt(dim_split), dim=-1)
        O = Q_split + torch.matmul(Attn, V_split)
        O = torch.cat(O.split(batch_size, 0), 2)

        O = O if not self.use_ln else self.ln1(O)
        O = self.Odense(O)
        O = O if not self.use_ln else self.ln2(O)

        return O


class SAB(torch.nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, number_heads: int, use_ln: bool = False
    ):
        super(SAB, self).__init__()
        self.net = MAB(in_dim, in_dim, out_dim, number_heads, use_ln)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X, X)


class AttnAgg(torch.nn.Module):
    def __init__(self, Qdim: int, Kdim: int, mid_dim: int):
        super(AttnAgg, self).__init__()
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)
        # self.use_ln = use_ln

    def forward(
        self,
        main_feat: torch.Tensor,
        other_feat: torch.Tensor,
        fix_feat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward propagation.

        Adjusted Attention Aggregator

        Args:
            main_feat (torch.Tensor): shape of [main_num, Q_dim]
            other_feat (torch.Tensor): shape of [other_num, K_dim]
            fix_feat (torch.Tensor): shape of [batch, other_num],
                adjust parameter for attention weight
            mask (torch.Tensor): shape of [main_num, other_num] a mask
                representing where main object should have attention
                with other obj 1 means no attention should be done.
                (default: `None`)

        Returns:
            torch.Tensor: aggregated features, shape of
                [batch, main_num, K_dim]
        """
        Q = self.Qdense(main_feat)
        K = self.Kdense(other_feat)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)

        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))
        Attn = torch.softmax(Attn, dim=-1)

        batch_size = fix_feat.shape[0]
        # [batch_size, other_num, other_num]
        fix_feat = torch.diag_embed(fix_feat)
        # [batch_size, other_num, K_dim]
        other_feat = other_feat.repeat(batch_size, 1, 1)
        other_feat = torch.matmul(fix_feat, other_feat)
        Attn = Attn.repeat(batch_size, 1, 1)

        return torch.matmul(Attn, other_feat)


class MoleRecLayer(torch.nn.Module):
    """MoleRec model.

    Paper: Nianzu Yang et al. MoleRec: Combinatorial Drug Recommendation
    with Substructure-Aware Molecular Representation Learning. WWW 2023.

    This layer is used in the MoleRec model. But it can also be used as a
    standalone layer.

    Args:
        hidden_size: hidden feature size.
        coef: coefficient of ddi loss weight annealing. larger coefficient
            means higher penalty to the drug-drug-interaction. Default is 2.5.
        target_ddi: DDI acceptance rate. Default is 0.06.
        GNN_layers: the number of layers of GNNs encoding molecule and
            substructures. Default is 4.
        dropout: the dropout ratio of model. Default is 0.7.
        multiloss_weight: the weight of multilabel_margin_loss for
            multilabel classification. Value should be set between [0, 1].
            Default is 0.05
    """

    def __init__(
        self,
        hidden_size: int,
        coef: float = 2.5,
        target_ddi: float = 0.08,
        GNN_layers: int = 4,
        dropout: float = 0.5,
        multiloss_weight: float = 0.05,
        **kwargs,
    ):
        super(MoleRecLayer, self).__init__()

        dependencies = ["ogb>=1.3.5"]

        # test whether the ogb and torch_scatter packages are ready
        try:
            pkg_resources.require(dependencies)
            global smiles2graph, AtomEncoder, BondEncoder
            from ogb.utils import smiles2graph
            from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
        except Exception as e:
            print(
                "Please follow the error message and install the [ogb>=1.3.5] packages first."
            )
            print(e)

        self.hidden_size = hidden_size
        self.coef, self.target_ddi = coef, target_ddi
        GNN_para = {
            "num_layers": GNN_layers,
            "dropout": dropout,
            "embedding_dim": hidden_size,
        }
        self.substructure_encoder = GINGraph(**GNN_para)
        self.molecule_encoder = GINGraph(**GNN_para)
        self.substructure_interaction_module = SAB(
            hidden_size, hidden_size, 2, use_ln=True
        )
        self.combination_feature_aggregator = AttnAgg(
            hidden_size, hidden_size, hidden_size
        )
        score_extractor = [
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, 1),
        ]
        self.score_extractor = torch.nn.Sequential(*score_extractor)
        self.multiloss_weight = multiloss_weight

    def calc_loss(
        self,
        logits: torch.Tensor,
        y_prob: torch.Tensor,
        ddi_adj: torch.Tensor,
        labels: torch.Tensor,
        label_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mul_pred_prob = y_prob.T @ y_prob  # (voc_size, voc_size)
        ddi_loss = (mul_pred_prob * ddi_adj).sum() / (ddi_adj.shape[0] ** 2)

        y_pred = y_prob.detach().cpu().numpy()
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y_pred = [np.where(sample == 1)[0] for sample in y_pred]

        loss_cls = binary_cross_entropy_with_logits(logits, labels)
        if self.multiloss_weight > 0 and label_index is not None:
            loss_multi = multilabel_margin_loss(y_prob, label_index)
            loss_cls = (
                self.multiloss_weight * loss_multi
                + (1 - self.multiloss_weight) * loss_cls
            )

        cur_ddi_rate = ddi_rate_score(y_pred, ddi_adj.cpu().numpy())
        if cur_ddi_rate > self.target_ddi:
            beta = self.coef * (1 - (cur_ddi_rate / self.target_ddi))
            beta = min(math.exp(beta), 1)
            loss = beta * loss_cls + (1 - beta) * ddi_loss
        else:
            loss = loss_cls
        return loss

    def forward(
        self,
        patient_emb: torch.Tensor,
        drugs: torch.Tensor,
        average_projection: torch.Tensor,
        ddi_adj: torch.Tensor,
        substructure_mask: torch.Tensor,
        substructure_graph: Union[StaticParaDict, Dict[str, Union[int, torch.Tensor]]],
        molecule_graph: Union[StaticParaDict, Dict[str, Union[int, torch.Tensor]]],
        mask: Optional[torch.tensor] = None,
        drug_indexes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation.

        Args:
            patient_emb: a tensor of shape [patient, visit, num_substructures],
                representating the relation between each patient visit and
                each substructures.
            drugs: a multihot tensor of shape [patient, num_labels].
            mask: an optional tensor of shape [patient, visit] where 1
                indicates valid visits and 0 indicates invalid visits.
            substructure_mask: tensor of shape [num_drugs, num_substructures],
                representing whether a substructure shows up in one of the
                molecule of each drug.
            average_projection: a tensor of shape [num_drugs, num_molecules]
                representing the average projection for aggregating multiple
                molecules of the same drug into one vector.
            substructure_graph: a dictionary representating a graph batch
                of all substructures, where each graph is extracted via
                'smiles2graph' api of ogb library.
            molecule_graph: dictionary with same form of substructure_graph,
                representing the graph batch of all molecules.
            ddi_adj: an adjacency tensor for drug drug interaction
                of shape [num_drugs, num_drugs].
            drug_indexes: the index version of drugs (ground truth) of shape
                [patient, num_labels], padded with -1
        Returns:
            loss: a scalar tensor representing the loss.
            y_prob: a tensor of shape [patient, num_labels] representing
                the probability of each drug.
        """
        if mask is None:
            mask = torch.ones_like(patient_emb[:, :, 0])
        substructure_relation = get_last_visit(patient_emb, mask)
        # [patient, num_substructures]

        substructure_embedding = self.substructure_interaction_module(
            self.substructure_encoder(substructure_graph).unsqueeze(0)
        ).squeeze(0)

        if substructure_relation.shape[-1] != substructure_embedding.shape[0]:
            raise RuntimeError(
                "the substructure relation vector of each patient should have "
                "the same dimension as the number of substructure"
            )

        molecule_embedding = self.molecule_encoder(molecule_graph)
        molecule_embedding = torch.mm(average_projection, molecule_embedding)

        combination_embedding = self.combination_feature_aggregator(
            molecule_embedding,
            substructure_embedding,
            substructure_relation,
            torch.logical_not(substructure_mask > 0),
        )
        # [patient, num_drugs, hidden]
        logits = self.score_extractor(combination_embedding).squeeze(-1)

        y_prob = torch.sigmoid(logits)

        loss = self.calc_loss(logits, y_prob, ddi_adj, drugs, drug_indexes)

        return loss, y_prob


class MoleRec(BaseModel):
    """MoleRec model.

    Paper: Nianzu Yang et al. MoleRec: Combinatorial Drug Recommendation
    with Substructure-Aware Molecular Representation Learning. WWW 2023.

    Note:
        This model is only for medication prediction which takes conditions
        and procedures as feature_keys, and drugs as label_key. It only
        operates on the visit level.

    Note:
        This model only accepts ATC level 3 as medication codes.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        num_rnn_layers: the number of layers used in RNN. Default is 1.
        num_gnn_layers: the number of layers used in GNN. Default is 4.
        dropout: the dropout rate. Default is 0.7.
        **kwargs: other parameters for the MoleRec layer.
    """

    def __init__(
        self,
        dataset: SampleEHRDataset,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_rnn_layers: int = 1,
        num_gnn_layers: int = 4,
        dropout: float = 0.5,
        **kwargs,
    ):
        super(MoleRec, self).__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            mode="multilabel",
        )

        dependencies = ["ogb>=1.3.5"]

        # test whether the ogb and torch_scatter packages are ready
        try:
            pkg_resources.require(dependencies)
            global smiles2graph, AtomEncoder, BondEncoder
            from ogb.utils import smiles2graph
            from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
        except Exception as e:
            print(
                "Please follow the error message and install the [ogb>=1.3.5] packages first."
            )
            print(e)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_rnn_layers = num_rnn_layers
        self.num_gnn_layers = num_gnn_layers
        self.dropout = dropout
        self.dropout_fn = torch.nn.Dropout(dropout)

        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)

        self.label_size = self.label_tokenizer.get_vocabulary_size()

        self.ddi_adj = torch.nn.Parameter(self.generate_ddi_adj(), requires_grad=False)
        self.all_smiles_list = self.generate_smiles_list()

        substructure_mask, self.substructure_smiles = self.generate_substructure_mask()

        self.substructure_mask = torch.nn.Parameter(
            substructure_mask, requires_grad=False
        )

        average_projection, self.all_smiles_flatten = self.generate_average_projection()

        self.average_projection = torch.nn.Parameter(
            average_projection, requires_grad=False
        )
        self.substructure_graphs = StaticParaDict(
            **graph_batch_from_smiles(self.substructure_smiles)
        )
        self.molecule_graphs = StaticParaDict(
            **graph_batch_from_smiles(self.all_smiles_flatten)
        )

        self.rnns = torch.nn.ModuleDict(
            {
                x: torch.nn.GRU(
                    embedding_dim,
                    hidden_dim,
                    num_layers=num_rnn_layers,
                    dropout=dropout if num_rnn_layers > 1 else 0,
                    batch_first=True,
                )
                for x in ["conditions", "procedures"]
            }
        )
        num_substructures = substructure_mask.shape[1]
        self.substructure_relation = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_substructures),
        )
        self.layer = MoleRecLayer(
            hidden_size=hidden_dim, dropout=dropout, GNN_layers=num_gnn_layers, **kwargs
        )

        if "GNN_layers" in kwargs:
            raise ValueError("number of GNN layers is determined by num_gnn_layers")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")
    
            # save ddi adj
        ddi_adj = self.generate_ddi_adj()
        np.save(os.path.join(CACHE_PATH, "ddi_adj.npy"), ddi_adj.numpy())

    def generate_ddi_adj(self) -> torch.FloatTensor:
        """Generates the DDI graph adjacency matrix."""
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True)
        vocab_to_index = self.label_tokenizer.vocabulary
        ddi_adj = np.zeros((self.label_size, self.label_size))
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi
        ]
        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        ddi_adj = torch.FloatTensor(ddi_adj)
        return ddi_adj

    def generate_substructure_mask(self) -> Tuple[torch.Tensor, List[str]]:
        # Generates the molecular segmentation mask H and substructure smiles.
        all_substructures_list = [[] for _ in range(self.label_size)]
        for index, smiles_list in enumerate(self.all_smiles_list):
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                substructures = Chem.BRICS.BRICSDecompose(mol)
                all_substructures_list[index] += substructures
        # all segment set
        substructures_set = list(set(sum(all_substructures_list, [])))
        # mask_H
        mask_H = np.zeros((self.label_size, len(substructures_set)))
        for index, substructures in enumerate(all_substructures_list):
            for s in substructures:
                mask_H[index, substructures_set.index(s)] = 1
        mask_H = torch.from_numpy(mask_H)
        return mask_H, substructures_set

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

    def generate_average_projection(self) -> Tuple[torch.Tensor, List[str]]:
        molecule_set, average_index = [], []
        for smiles_list in self.all_smiles_list:
            """Create each data with the above defined functions."""
            counter = 0  # counter how many drugs are under that ATC-3
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                molecule_set.append(smiles)
                counter += 1
            average_index.append(counter)
        average_projection = np.zeros((len(average_index), sum(average_index)))
        col_counter = 0
        for i, item in enumerate(average_index):
            if item <= 0:
                continue
            average_projection[i, col_counter : col_counter + item] = 1 / item
            col_counter += item
        average_projection = torch.FloatTensor(average_projection)
        return average_projection, molecule_set

    def encode_patient(
        self, feature_key: str, raw_values: List[List[List[str]]]
    ) -> torch.Tensor:
        codes = self.feat_tokenizers[feature_key].batch_encode_3d(raw_values)
        codes = torch.tensor(codes, dtype=torch.long, device=self.device)
        embeddings = self.embeddings[feature_key](codes)
        embeddings = torch.sum(self.dropout_fn(embeddings), dim=2)
        outputs, _ = self.rnns[feature_key](embeddings)
        return outputs

    def forward(
        self,
        conditions: List[List[List[str]]],
        procedures: List[List[List[str]]],
        drugs: List[List[str]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            conditions: a nested list in three levels with
                shape [patient, visit, condition].
            procedures: a nested list in three levels with
                shape [patient, visit, procedure].
            drugs: a nested list in two levels [patient, drug].

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor of shape [patient, visit, num_labels]
                    representing the probability of each drug.
                y_true: a tensor of shape [patient, visit, num_labels]
                    representing the ground truth of each drug.
        """

        # prepare labels
        labels_index = self.label_tokenizer.batch_encode_2d(
            drugs, padding=False, truncation=False
        )
        # convert to multihot
        labels = batch_to_multihot(labels_index, self.label_size)
        index_labels = -np.ones((len(labels), self.label_size), dtype=np.int64)
        for idx, cont in enumerate(labels_index):
            # remove redundant labels
            cont = list(set(cont))
            index_labels[idx, : len(cont)] = cont
        index_labels = torch.from_numpy(index_labels)

        labels = labels.to(self.device)
        index_labels = index_labels.to(self.device)

        # encoding procs and diags
        condition_emb = self.encode_patient("conditions", conditions)
        procedure_emb = self.encode_patient("procedures", procedures)
        mask = torch.sum(condition_emb, dim=2) != 0

        patient_emb = torch.cat([condition_emb, procedure_emb], dim=-1)
        substruct_rela = self.substructure_relation(patient_emb)

        # calculate loss
        loss, y_prob = self.layer(
            patient_emb=substruct_rela,
            drugs=labels,
            ddi_adj=self.ddi_adj,
            average_projection=self.average_projection,
            substructure_mask=self.substructure_mask,
            substructure_graph=self.substructure_graphs,
            molecule_graph=self.molecule_graphs,
            mask=mask,
            drug_indexes=index_labels,
        )

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": labels,
        }
