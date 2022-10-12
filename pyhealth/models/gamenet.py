from typing import List, Tuple, Union
from urllib import request

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from pyhealth.datasets import BaseDataset
from pyhealth.models import BaseModel
from pyhealth.tokenizer import Tokenizer
import os
from pathlib import Path
from urllib import request
import pandas as pd
from pyhealth.models.utils import get_last_visit


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
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


class GCN(nn.Module):
    """The GCN model as in https://arxiv.org/abs/1609.02907
    Args:
        voc_size (int): the size of the (med) vocabulary
        emb_dim (int): the dimension of the embedding
        adj (np.array): the adjacency matrix
    """

    def __init__(self, voc_size, emb_dim, adj):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim

        adj = self.normalize(adj + np.eye(adj.shape[0]))
        self.adj = torch.nn.Parameter(torch.FloatTensor(adj), requires_grad=False)
        self.x = torch.nn.Parameter(torch.eye(voc_size), requires_grad=False)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = nn.functional.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class GAMENetLayer(nn.Module):
    """We separate the GAMENet layer from the model for flexible usage.
    Args:
        input (int): the input embedding size
        hidden (int): the hidden embedding size
        tables  (list): the list of table names
        ehr_adj (np.array): the adjacency matrix of EHR
        ddi_adj (np.array): the adjacency matrix of DDI
        num_layers (int): the number of layers used in RNN
        dropout (float): the dropout rate
        ddi_in_memory (bool): whether to use DDI GCN in forward function
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        tables: List[str],
        ehr_adj: np.ndarray,
        ddi_adj: np.ndarray,
        num_layers: int = 1,
        dropout: float = 0.5,
        ddi_in_memory: bool = True,
    ):
        super(GAMENetLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tables = tables
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.ddi_in_memory = ddi_in_memory

        # med space size
        self.label_size = ehr_adj.shape[0]

        # define the rnn layers
        self.rnn = nn.ModuleDict()
        for domain in tables:
            self.rnn[domain] = nn.GRU(
                input_size,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
            )

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size * len(tables), hidden_size),
        )
        self.ehr_gcn = GCN(voc_size=self.label_size, emb_dim=hidden_size, adj=ehr_adj)
        self.ddi_gcn = GCN(voc_size=self.label_size, emb_dim=hidden_size, adj=ddi_adj)
        self.inter = nn.Parameter(torch.FloatTensor(1))

    def forward(self, X: dict, mask: torch.tensor):
        """
        Args:
            X: a dict with <str, [batch size, seq len, input_size]>
            mask: [batch size, seq len]
        Returns:
            outputs [batch size, seq len, hidden_size]
        """
        patient_emb = []
        for domain in self.tables:
            domain_emb, _ = self.rnn[domain](X[domain])
            patient_emb.append(domain_emb)

        patient_representations = torch.cat(
            patient_emb, dim=-1
        )  # (batch, visit, dim * len(tables))
        queries = self.query(patient_representations)  # (batch, visit, hidden_size)

        # graph memory module
        """I:generate current input"""
        query = get_last_visit(queries, mask)  # (batch, hidden_size)

        """G:generate graph memory bank and insert history information"""
        if self.ddi_in_memory:
            drug_memory = self.ehr_gcn() - self.ddi_gcn() * torch.sigmoid(
                self.inter
            )  # (med_size, dim)
        else:
            drug_memory = self.ehr_gcn()

        history_keys = queries  # (batch, visit, dim)
        # remove the current visit (with is the gt information)

        med_tensor = X["drugs"]
        history_values = (
            torch.nn.functional.one_hot(med_tensor, num_classes=self.label_size)
            .sum(-2)
            .bool()
        )  # (batch, visit, med_size)

        """O:read from global memory bank and dynamic memory bank"""
        key_weights1 = torch.softmax(
            torch.mm(query, drug_memory.t()), dim=-1
        )  # (batch, med_size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (batch, dim)

        # remove the last visit from mask
        visit_weight = torch.softmax(
            torch.einsum("bd,bvd->bv", query, history_keys), dim=1
        )  # (batch, visit)

        # use masked attention (empirically it is better to not use mask)
        # visit_weight = torch.softmax(
        #     torch.einsum("bd,bvd->bv", query, history_keys)
        #     - (1 - diag_mask[:, :, 0].float()) * 1e10,
        #     dim=1,
        # )  # (batch, visit)

        weighted_values = torch.einsum(
            "bv,bvz->bz", visit_weight, history_values.float()
        )  # (batch, med_size)
        fact2 = torch.mm(weighted_values, drug_memory)  # (batch, dim)

        """R:convert O and predict"""
        return torch.cat([query, fact1, fact2], dim=-1)


class GAMENet(BaseModel):
    """GAMENet Class, use "task" as key to identify specific GAMENet model and route there
    Args:
        dataset: the dataset object
        tables: the list of table names to use
        target: the target table name
        mode: the mode of the model, "multilabel", "multiclass" or "binary"
        embedding_dim: the embedding dimension
        hidden_dim: the hidden dimension
    """

    def __init__(
        self,
        dataset: BaseDataset,
        tables: List[str],
        target: str,
        mode: str,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs
    ):
        super(GAMENet, self).__init__(
            dataset=dataset,
            tables=tables,
            target=target,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # define tokenizers
        self.tokenizers = {}
        for domain in tables:
            self.tokenizers[domain] = Tokenizer(
                dataset.get_all_tokens(key=domain),
                special_tokens=["<pad>", "<unk>"],
            )
        self.drug_tokenizer = Tokenizer(
            dataset.get_all_tokens(key="drugs"),
            special_tokens=["<pad>", "<unk>"],
        )
        self.label_tokenizer = Tokenizer(dataset.get_all_tokens(key=target))

        # embedding tables for each domain
        self.embeddings = nn.ModuleDict()
        for domain in tables:
            # TODO: use get_pad_token_id() instead of hard code
            self.embeddings[domain] = nn.Embedding(
                self.tokenizers[domain].get_vocabulary_size(),
                embedding_dim,
                padding_idx=0,
            )

        ehr_adj = self.generate_ehr_adj()
        ddi_adj = self.generate_ddi_adj()

        self.gamenet = GAMENetLayer(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            tables=tables,
            ehr_adj=ehr_adj,
            ddi_adj=ddi_adj,
        )
        self.fc = nn.Linear(3 * hidden_dim, self.label_tokenizer.get_vocabulary_size())

    def generate_ehr_adj(self):
        """
        generate the ehr graph adj for GAMENet model input
        - loop over the training data to check whether any med pair appear
        """
        label_size = self.label_tokenizer.get_vocabulary_size()
        ehr_adj = np.zeros((label_size, label_size))
        for sample in self.dataset.samples:
            encoded_drugs = self.label_tokenizer.convert_tokens_to_indices(
                sample["label"]
            )
            for idx1, med1 in enumerate(encoded_drugs):
                for idx2, med2 in enumerate(encoded_drugs):
                    if idx1 >= idx2:
                        continue
                    ehr_adj[med1, med2] = 1
                    ehr_adj[med2, med1] = 1
        return ehr_adj

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
                os.path.join(str(Path.home()), ".cache/pyhealth/cid_to_ATC6.csv"),
                "w",
            ) as outfile:
                for line in cid_to_ATC6:
                    print(str(line[:-1]), file=outfile)
        else:
            cid_to_ATC6 = open(
                os.path.join(str(Path.home()), ".cache/pyhealth/cid_to_ATC6.csv"),
                "r",
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

        # process drugs
        kwargs["drugs"] = self.drug_tokenizer.batch_encode_3d(kwargs["drugs"])
        kwargs["drugs"] = torch.tensor(kwargs["drugs"], dtype=torch.long, device=device)
        patient_emb = self.gamenet(kwargs, mask)
        logits = self.fc(patient_emb)

        # obtain target, loss, prob, pred
        loss, y_true, y_prod, y_pred = self.cal_loss_and_output(
            logits, device, **kwargs
        )

        return {
            "loss": loss,
            "y_prob": y_prod,
            "y_pred": y_pred,
            "y_true": y_true,
        }
