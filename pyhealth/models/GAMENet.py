from gettext import npgettext
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pyhealth.models.tokenizer import Tokenizer


def get_last_visit(hidden_states, mask):
    last_visit = torch.sum(mask, 1) - 1
    last_visit = last_visit.unsqueeze(-1)
    last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
    last_visit = torch.reshape(last_visit, hidden_states.shape)
    last_hidden_states = torch.gather(hidden_states, 1, last_visit)
    last_hidden_state = last_hidden_states[:, 0, :]
    return last_hidden_state


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
        node_embedding = F.relu(node_embedding)
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
    def __init__(
        self, voc_size, ehr_adj, ddi_adj, emb_dim=64, ddi_in_memory=True, **kwargs
    ):
        super(GAMENetLayer, self).__init__()

        ddi_adj = torch.FloatTensor(ddi_adj)
        ehr_adj = torch.FloatTensor(ehr_adj)
        self.ddi_in_memory = ddi_in_memory
        self.dropout = nn.Dropout(p=0.5)
        self.voc_size = voc_size

        # parameters
        self.embedding = nn.ModuleList([nn.Embedding(s + 1, emb_dim) for s in voc_size])
        # GRU encoder for conditions and procedures
        self.encoder = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)]
        )

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.ehr_gcn = GCN(voc_size=voc_size[2], emb_dim=emb_dim, adj=ehr_adj)
        self.ddi_gcn = GCN(voc_size=voc_size[2], emb_dim=emb_dim, adj=ddi_adj)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim * 3, emb_dim))

    def forward(self, tensors, masks=None):
        """
        Args:
            tensors: list of input tensors, each tensor is of shape (batch, visit, code_len)
            masks: list of input masks, each mask is of shape (batch, visit)
        """
        diag_tensor, proc_tensor, med_tensor = tensors
        diag_mask, proc_mask, med_mask = masks
        diag_emb = self.embedding[0](diag_tensor)  # (batch, visit, code_len, dim)
        proc_emb = self.embedding[1](proc_tensor)  # (batch, visit, code_len, dim)

        # sum over code_len
        diag_emb = (diag_emb * diag_mask.unsqueeze(-1).float()).sum(
            dim=2
        )  # (batch, visit, dim)
        proc_emb = (proc_emb * proc_mask.unsqueeze(-1).float()).sum(
            dim=2
        )  # (batch, visit, dim)

        # use RNN encoder
        diag_emb, _ = self.encoder[0](diag_emb)  # (batch, visit, dim)
        proc_emb, _ = self.encoder[1](proc_emb)  # (batch, visit, dim)

        patient_representations = torch.cat(
            [diag_emb, proc_emb], dim=-1
        )  # (batch, visit, dim * 2)
        queries = self.query(patient_representations)  # (batch, visit, dim)

        # graph memory module
        """I:generate current input"""
        query = get_last_visit(queries, diag_mask[:, :, 0])  # (batch, dim)

        """G:generate graph memory bank and insert history information"""
        if self.ddi_in_memory:
            drug_memory = self.ehr_gcn() - self.ddi_gcn() * torch.sigmoid(
                self.inter
            )  # (med_size, dim)
        else:
            drug_memory = self.ehr_gcn()

        history_keys = queries  # (batch, visit, dim)
        # remove the current visit (with is the gt information)
        history_values = (
            torch.nn.functional.one_hot(
                med_tensor[:, :-1, :], num_classes=self.voc_size[2]
            )
            .sum(-2)
            .bool()
        ) * 0  # (batch, visit-1, med_size)

        """O:read from global memory bank and dynamic memory bank"""
        key_weights1 = torch.softmax(
            torch.mm(query, drug_memory.t()), dim=-1
        )  # (batch, med_size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (batch, dim)

        # # remove the last visit from mask
        visit_weight = torch.softmax(
            torch.einsum("bd,bvd->bv", query, history_keys), dim=1
        )  # (batch, visit)
        # # visit_weight = torch.softmax(torch.einsum("bd,bvd->bv", query, history_keys) - (1-diag_mask[:, :, 0].float()) * 1e10, dim=1) # (batch, visit)
        weighted_values = torch.einsum(
            "bv,bvz->bz", visit_weight[:, :-1], history_values.float()
        )  # (batch, med_size)
        fact2 = torch.mm(weighted_values, drug_memory)  # (batch, dim)

        """R:convert O and predict"""
        output = self.output(torch.cat([query, fact1, fact2], dim=-1))  # (batch, dim)

        return output


class GAMENet(nn.Module):
    def __init__(
        self,
        voc_size,
        ehr_adj,
        ddi_adj,
        tokenizers,
        emb_dim=64,
        ddi_in_memory=True,
        **kwargs
    ):
        super(GAMENet, self).__init__()

        self.condition_tokenizer = tokenizers[0]
        self.procedure_tokenizer = tokenizers[1]
        self.drug_tokenizer = tokenizers[2]

        self.gamenet_layer = GAMENetLayer(
            voc_size, ehr_adj, ddi_adj, emb_dim, ddi_in_memory, **kwargs
        )
        self.drug_fc = nn.Linear(emb_dim, voc_size[2])

    def forward(self, conditions, procedures, drugs, device=None, **kwargs):
        diag_tensor, diag_mask = [
            item.to(device)
            for item in self.condition_tokenizer.batch_tokenize(conditions)
        ]
        proc_tensor, proc_mask = [
            item.to(device)
            for item in self.procedure_tokenizer.batch_tokenize(procedures)
        ]
        drugs_tensor, drugs_mask = [
            item.to(device) for item in self.drug_tokenizer.batch_tokenize(drugs)
        ]
        tensors = [diag_tensor, proc_tensor, drugs_tensor]
        masks = [diag_mask, proc_mask, drugs_mask]

        embedding = self.gamenet_layer(tensors, masks)
        logits = self.drug_fc(embedding)
        y_prob = torch.sigmoid(logits)

        # target
        y = torch.zeros(diag_tensor.shape[0], self.drug_tokenizer.get_vocabulary_size())
        for idx, sample in enumerate(drugs):
            y[idx, self.drug_tokenizer(sample[-1:])[0]] = 1

        # loss
        loss = F.binary_cross_entropy_with_logits(logits, y.to(device))
        return {"loss": loss, "y_prob": y_prob, "y_true": y}
