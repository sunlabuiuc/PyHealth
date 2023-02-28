import torch
import math
import torch_scatter
import numpy as np
from ogb.utils import smiles2graph
from typing import Any, Dict, List, Tuple, Optional, Union
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from rdkit import Chem

# from .model import BaseModel
# from .utils import get_last_visit


def graph_batch_from_smile(smiles_list, device=torch.device('cpu')):
    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    graphs = [smiles2graph(x) for x in smiles_list]
    for idx, graph in enumerate(graphs):
        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])
        lstnode += graph['num_nodes']
        batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0)
    }
    result = {k: torch.from_numpy(v).to(device) for k, v in result.items()}
    result['num_nodes'] = lstnode
    result['num_edges'] = result['edge_index'].shape[1]
    return result


class GINConv(torch.nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super(GINConv, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 2 * embedding_dim),
            torch.nn.BatchNorm1d(2 * embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embedding_dim, embedding_dim)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=embedding_dim)

    def forward(
        self,
        node_feats: torch.Tensor, edge_feats: torch.Tensor,
        edge_index: torch.Tensor, num_nodes: int, num_edges: int
    ) -> torch.Tensor:
        edge_feats = self.bond_encoder(edge_feats)
        message_node = torch.index_select(
            input=node_feats, dim=0, index=edge_index[1]
        )
        message = torch.relu(message_node + edge_feats)
        message_reduce = torch_scatter.scatter(
            message, index=edge_index[0], dim=0,
            dim_size=num_nodes, reduce='sum'
        )

        return self.mlp((1 + self.eps) * node_feats + message_reduce)


class GINGraph(torch.nn.Module):
    def __init__(
        self,  num_layers: int = 4,
        embedding_dim: int = 64,
        dropout: float = 0.7
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

    def forward(
        self,
        graph: Dict[str, Union[int, torch.Tensor]]
    ) -> torch.Tensor:
        h_list = [self.atom_encoder(graph['x'])]
        for layer in range(self.num_layers):
            h = self.batch_norms[layer](self.convs[layer](
                node_feats=h_list[layer], edge_feats=graph['edge_attr'],
                edge_index=graph['edge_index'], num_nodes=graph['num_nodes'],
                num_edges=graph['num_edges']
            ))
            if layer != self.num_layers - 1:
                h = self.dropout_fun(torch.relu(h))
            else:
                h = self.dropout_fun(h)
            h_list.append(h)
        return torch_scatter.scatter(
            h_list[-1], dim=0, index=graph['batch'], reduce='mean'
        )


class MAB(torch.nn.Module):
    def __init__(
        self, Qdim: int, Kdim: int, Vdim: int,
        number_heads: int, use_ln: bool = False
    ):
        super(MAB, self).__init__()
        self.Vdim = Vdim
        self.number_heads = number_heads

        assert self.Vdim % self.number_heads == 0, \
            'the dim of features should be divisible by number_heads'

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
        self, in_dim: int, out_dim: int,
        number_heads: int, use_ln: bool = False
    ):
        super(SAB, self).__init__()
        self.net = MAB(in_dim, in_dim, out_dim, number_heads, use_ln)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X, X)


class AttenAgger(torch.nn.Module):
    def __init__(self, Qdim: int, Kdim: int, mid_dim: int):
        super(AdjAttenAgger, self).__init__()
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)
        # self.use_ln = use_ln

    def forward(
        self, main_feat: torch.Tensor, other_feat: torch.Tensor,
        fix_feat: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        Q = self.Qdense(main_feat)
        K = self.Kdense(other_feat)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)

        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))

        Attn = torch.softmax(Attn, dim=-1)
        fix_feat = torch.diag(fix_feat)
        other_feat = torch.matmul(fix_feat, other_feat)
        O = torch.matmul(Attn, other_feat)

        return O


class MoleRecLayer(torch.nn.Module):
    """MoleRec model.

    Paper: Nianzu Yang et al. MoleRec: Combinatorial Drug Recommendation 
    with Substructure-Aware Molecular Representation Learning. WWW 2023.

    This layer is used in the MoleRec model. But it can also be used as a
    standalone layer.

    Args:
        hidden_size: hidden feature size.
        coef: coefficient of ddi loss weight annealing. Default is 2.5.
        target_ddi: DDI acceptance rate. Default is 0.06.
        GNN_layers: the number of layers of GNNs encoding molecule and
            substructures. Default is 4.
        dropout: the dropout ratio of model. Default is 0.7.
        multi_loss_weight: the weight for multilabel_margin_loss for 
            prediction, the weight should between 0 and 1. Default is 0.05.

    """

    def __init__(
        self,
        hidden_size: int,
        coef: float = 2.5,
        target_ddi: float = 0.06,
        GNN_layers: int = 4,
        dropout: float = 0.7,
        multi_loss_weight: float = 0.05
    ):
        super(MoleRecLayer, self).__init__()
        self.hidden_size = hidden_size
        self.coef, self.target_ddi = coef, target_ddi
        GNN_para = {
            'num_layers': GNN_layers, 'dropout': dropout,
            'embedding_dim': hidden_size
        }
        self.substructure_encoder = GINGraph(**GNN_para)
        self.molecule_encoder = GINGraph(**GNN_para)
        self.substructure_relation = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size * 4, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size)
        )
        self.substructure_interaction_module = SAB(
            hidden_size, hidden_size, 2, use_ln=True
        )
        self.combination_feature_aggregator = AttenAgger(
            hidden_size, hidden_size, hidden_size
        )
        self.multi_weight = multi_loss_weight

    def calc_loss(
        self, logits: torch.Tensor, y_prob: torch.Tensor,
        ddi_adj: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        mul_pred_prob = y_prob.T @ y_prob  # (voc_size, voc_size)
        batch_ddi_loss = torch.sum(mul_pred_prob.mul(ddi_adj))\
            / (self.ddi_adj.shape[0] ** 2)

        y_pred = y_prob.detach().cpu().numpy()
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y_pred = [np.where(sample == 1)[0] for sample in y_pred]


        # TODO: modify loss

        cur_ddi_rate = ddi_rate_score(y_pred, ddi_adj.cpu().numpy())
        if cur_ddi_rate > self.target_ddi:
            beta = max(0.0, 1 + (self.target_ddi - cur_ddi_rate) / self.kp)
            add_loss, beta = batch_ddi_loss, beta
        else:
            add_loss, beta = 0, 1

        # obtain target, loss, prob, pred
        bce_loss = self.loss_fn(logits, labels)

        loss = beta * bce_loss + (1 - beta) * add_loss
        return loss


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    smiles_list = ['c1=cc=cc=c1-c2=cc=cc=c2'.upper(), 'c1=cc=nc=c1'.upper()]
    graph_batch = graph_batch_from_smile(smiles_list, device)
    print(graph_batch)
    Net = GINGraph().to(device)
    result = Net(graph_batch)
    print(result)
