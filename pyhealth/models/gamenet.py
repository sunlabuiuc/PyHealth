import math
from typing import Tuple, List, Dict, Optional
import os

import torch
import torch.nn as nn
import numpy as np
from pyhealth.datasets import SampleEHRDataset
from pyhealth.medcode import ATC
from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit, batch_to_multihot
from pyhealth import BASE_CACHE_PATH as CACHE_PATH

class GCNLayer(nn.Module):
    """GCN layer.

    Paper: Thomas N. Kipf et al. Semi-Supervised Classification with Graph
    Convolutional Networks. ICLR 2017.

    This layer is used in the GCN model.

    Args:
        in_features: input feature size.
        out_features: output feature size.
        bias: whether to use bias. Default is True.
    """

    def __init__(self, in_features: int, out_features: int, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.tensor, adj: torch.tensor) -> torch.tensor:
        """
        Args:
            input: input feature tensor of shape [num_nodes, in_features].
            adj: adjacency tensor of shape [num_nodes, num_nodes].

        Returns:
            Output tensor of shape [num_nodes, out_features].
        """
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    """GCN model.

    Paper: Thomas N. Kipf et al. Semi-Supervised Classification with Graph
    Convolutional Networks. ICLR 2017.

    This model is used in the GAMENet layer.

    Args:
        hidden_size: hidden feature size.
        adj: adjacency tensor of shape [num_nodes, num_nodes].
        dropout: dropout rate. Default is 0.5.
    """

    def __init__(self, adj: torch.tensor, hidden_size: int, dropout: float = 0.5):
        super(GCN, self).__init__()
        self.emb_dim = hidden_size
        self.dropout = dropout

        voc_size = adj.shape[0]
        adj = adj + torch.eye(adj.shape[0])
        adj = self.normalize(adj)
        self.adj = torch.nn.Parameter(adj, requires_grad=False)
        self.x = torch.nn.Parameter(torch.eye(voc_size), requires_grad=False)

        self.gcn1 = GCNLayer(voc_size, hidden_size)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.gcn2 = GCNLayer(hidden_size, hidden_size)

    def normalize(self, mx: torch.tensor) -> torch.tensor:
        """Normalizes the matrix row-wise."""
        rowsum = mx.sum(1)
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diagflat(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

    def forward(self) -> torch.tensor:
        """Forward propagation.

        Returns:
            Output tensor of shape [num_nodes, hidden_size].
        """
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = torch.relu(node_embedding)
        node_embedding = self.dropout_layer(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding


class GAMENetLayer(nn.Module):
    """GAMENet layer.

    Paper: Junyuan Shang et al. GAMENet: Graph Augmented MEmory Networks for
    Recommending Medication Combination AAAI 2019.

    This layer is used in the GAMENet model. But it can also be used as a
    standalone layer.

    Args:
        hidden_size: hidden feature size.
        ehr_adj: an adjacency tensor of shape [num_drugs, num_drugs].
        ddi_adj: an adjacency tensor of shape [num_drugs, num_drugs].
        dropout : the dropout rate. Default is 0.5.

    Examples:
        >>> from pyhealth.models import GAMENetLayer
        >>> queries = torch.randn(3, 5, 32) # [patient, visit, hidden_size]
        >>> prev_drugs = torch.randint(0, 2, (3, 4, 50)).float()
        >>> curr_drugs = torch.randint(0, 2, (3, 50)).float()
        >>> ehr_adj = torch.randint(0, 2, (50, 50)).float()
        >>> ddi_adj = torch.randint(0, 2, (50, 50)).float()
        >>> layer = GAMENetLayer(32, ehr_adj, ddi_adj)
        >>> loss, y_prob = layer(queries, prev_drugs, curr_drugs)
        >>> loss.shape
        torch.Size([])
        >>> y_prob.shape
        torch.Size([3, 50])
    """

    def __init__(
        self,
        hidden_size: int,
        ehr_adj: torch.tensor,
        ddi_adj: torch.tensor,
        dropout: float = 0.5,
    ):
        super(GAMENetLayer, self).__init__()
        self.hidden_size = hidden_size
        self.ehr_adj = ehr_adj
        self.ddi_adj = ddi_adj

        num_labels = ehr_adj.shape[0]
        self.ehr_gcn = GCN(adj=ehr_adj, hidden_size=hidden_size, dropout=dropout)
        self.ddi_gcn = GCN(adj=ddi_adj, hidden_size=hidden_size, dropout=dropout)
        self.beta = nn.Parameter(torch.FloatTensor(1))
        self.fc = nn.Linear(3 * hidden_size, num_labels)

        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        queries: torch.tensor,
        prev_drugs: torch.tensor,
        curr_drugs: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            queries: query tensor of shape [patient, visit, hidden_size].
            prev_drugs: multihot tensor indicating drug usage in all previous
                visits of shape [patient, visit - 1, num_drugs].
            curr_drugs: multihot tensor indicating drug usage in the current
                visit of shape [patient, num_drugs].
            mask: an optional mask tensor of shape [patient, visit] where 1
                indicates valid visits and 0 indicates invalid visits.

        Returns:
            loss: a scalar tensor representing the loss.
            y_prob: a tensor of shape [patient, num_labels] representing
                the probability of each drug.
        """
        if mask is None:
            mask = torch.ones_like(queries[:, :, 0])

        """I: Input memory representation"""
        query = get_last_visit(queries, mask)

        """G: Generalization"""
        # memory bank
        MB = self.ehr_gcn() - self.ddi_gcn() * torch.sigmoid(self.beta)

        # dynamic memory
        DM_keys = queries[:, :-1, :]
        DM_values = prev_drugs[:, :-1, :]

        """O: Output memory representation"""
        a_c = torch.softmax(torch.mm(query, MB.t()), dim=-1)
        o_b = torch.mm(a_c, MB)

        a_s = torch.softmax(torch.einsum("bd,bvd->bv", query, DM_keys), dim=1)
        a_m = torch.einsum("bv,bvz->bz", a_s, DM_values.float())
        o_d = torch.mm(a_m, MB)

        """R: Response"""
        memory_output = torch.cat([query, o_b, o_d], dim=-1)
        logits = self.fc(memory_output)

        loss = self.bce_loss_fn(logits, curr_drugs)
        y_prob = torch.sigmoid(logits)

        return loss, y_prob


class GAMENet(BaseModel):
    """GAMENet model.

    Paper: Junyuan Shang et al. GAMENet: Graph Augmented MEmory Networks for
    Recommending Medication Combination AAAI 2019.

    Note:
        This model is only for medication prediction which takes conditions
        and procedures as feature_keys, and drugs as label_key.
        It only operates on the visit level. Thus, we have disable the 
        feature_keys, label_key, and mode arguments.

    Note:
        This model only accepts ATC level 3 as medication codes.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        num_layers: the number of layers used in RNN. Default is 1.
        dropout: the dropout rate. Default is 0.5.
        **kwargs: other parameters for the GAMENet layer.
    """

    def __init__(
        self,
        dataset: SampleEHRDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.5,
        **kwargs
    ):
        super(GAMENet, self).__init__(
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

        ehr_adj = self.generate_ehr_adj()
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
        if "ehr_adj" in kwargs:
            raise ValueError("ehr_adj is determined by the dataset")
        if "ddi_adj" in kwargs:
            raise ValueError("ddi_adj is determined by the dataset")
        self.gamenet = GAMENetLayer(
            hidden_size=hidden_dim,
            ehr_adj=ehr_adj,
            ddi_adj=ddi_adj,
            dropout=dropout,
            **kwargs,
        )
        
        # save ddi adj
        ddi_adj = self.generate_ddi_adj()
        np.save(os.path.join(CACHE_PATH, "ddi_adj.npy"), ddi_adj)
        
    def generate_ddi_adj():
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
        return ddi_adj

    def generate_ehr_adj(self) -> torch.tensor:
        """Generates the EHR graph adjacency matrix."""
        label_size = self.label_tokenizer.get_vocabulary_size()
        ehr_adj = torch.zeros((label_size, label_size))
        for sample in self.dataset:
            curr_drugs = sample["drugs"]
            encoded_drugs = self.label_tokenizer.convert_tokens_to_indices(curr_drugs)
            for idx1, med1 in enumerate(encoded_drugs):
                for idx2, med2 in enumerate(encoded_drugs):
                    if idx1 >= idx2:
                        continue
                    ehr_adj[med1, med2] = 1
                    ehr_adj[med2, med1] = 1
        return ehr_adj

    def generate_ddi_adj(self) -> torch.tensor:
        """Generates the DDI graph adjacency matrix."""
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True)
        label_size = self.label_tokenizer.get_vocabulary_size()
        vocab_to_index = self.label_tokenizer.vocabulary
        ddi_adj = torch.zeros((label_size, label_size))
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi
        ]
        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        return ddi_adj

    def forward(
        self,
        conditions: List[List[List[str]]],
        procedures: List[List[List[str]]],
        drugs_hist: List[List[List[str]]],
        drugs: List[List[str]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            conditions: a nested list in three levels [patient, visit, condition].
            procedures: a nested list in three levels [patient, visit, procedure].
            drugs_hist: a nested list in three levels [patient, visit, drug], up to visit (N-1)
            drugs: a nested list in two levels [patient, drug], at visit N

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
        patient_representations = torch.cat([conditions, procedures], dim=-1)
        # (batch, visit, hidden_size)
        queries = self.query(patient_representations)

        label_size = self.label_tokenizer.get_vocabulary_size()
        drugs_hist = self.label_tokenizer.batch_encode_3d(
            drugs_hist, padding=(False, False), truncation=(True, False)
        )

        curr_drugs = self.prepare_labels(drugs, self.label_tokenizer)

        prev_drugs = drugs_hist
        max_num_visit = max([len(p) for p in prev_drugs])
        prev_drugs = [p + [[]] * (max_num_visit - len(p)) for p in prev_drugs]
        prev_drugs = [batch_to_multihot(p, label_size) for p in prev_drugs]
        prev_drugs = torch.stack(prev_drugs, dim=0)
        prev_drugs = prev_drugs.to(self.device)

        # get mask
        mask = torch.sum(conditions, dim=2) != 0

        # process drugs
        loss, y_prob = self.gamenet(queries, prev_drugs, curr_drugs, mask)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": curr_drugs,
        }
