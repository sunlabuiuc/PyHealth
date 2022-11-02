import numpy as np
import torch
import torch.nn as nn

from pyhealth.datasets import BaseDataset
from pyhealth.medcode import ATC
from pyhealth.models import BaseModel
from pyhealth.models.utils import batch_to_multihot
from pyhealth.models.utils import get_last_visit


class GCNLayer(nn.Module):
    """Simple GCN layer.

    Reference: https://arxiv.org/abs/1609.02907.
    """

    def __init__(self, in_features, out_features, bias=True):
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
    """Simple GCN model.

    Reference: https://arxiv.org/abs/1609.02907.

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

        self.gcn1 = GCNLayer(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GCNLayer(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = torch.relu(node_embedding)
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
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.ddi_in_memory = ddi_in_memory

        # med space size
        self.label_size = ehr_adj.shape[0]

        self.ehr_gcn = GCN(voc_size=self.label_size, emb_dim=hidden_size, adj=ehr_adj)
        self.ddi_gcn = GCN(voc_size=self.label_size, emb_dim=hidden_size, adj=ddi_adj)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.fc = nn.Linear(3 * hidden_size, self.label_size)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, queries, masks, prev_drugs, curr_drugs):
        """I:generate current input"""
        query = get_last_visit(queries, masks)

        """G:generate graph memory bank and insert history information"""
        drug_memory = self.ehr_gcn() - self.ddi_gcn() * torch.sigmoid(self.inter)

        # (batch, visit-1, )
        history_keys = queries[:, :-1, :]
        history_values = prev_drugs

        """O:read from global memory bank and dynamic memory bank"""
        key_weights1 = torch.softmax(
            torch.mm(query, drug_memory.t()), dim=-1
        )
        fact1 = torch.mm(key_weights1, drug_memory)  # (batch, dim)

        # remove the last visit from mask
        visit_weight = torch.softmax(
            torch.einsum("bd,bvd->bv", query, history_keys), dim=1
        )

        weighted_values = torch.einsum(
            "bv,bvz->bz", visit_weight, history_values.float()
        )
        fact2 = torch.mm(weighted_values, drug_memory)  # (batch, dim)

        """R:convert O and predict"""
        memory_output = torch.cat([query, fact1, fact2], dim=-1)

        logits = self.fc(memory_output)

        loss = self.loss_fn(logits, curr_drugs)
        y_prob = torch.sigmoid(logits)

        return loss, y_prob


class GAMENet(BaseModel):
    """GAMENet Class, use "task" as key to identify specific GAMENet model and route there
    Args:
        dataset: the dataset object
        feature_keys: the list of table names to use
        label_key: the target table name
        mode: the mode of the model, "multilabel", "multiclass" or "binary"
        embedding_dim: the embedding dimension
        hidden_dim: the hidden dimension
    """

    def __init__(
            self,
            dataset: BaseDataset,
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

        self.feat_tokenizers = self._get_feature_tokenizers()
        self.label_tokenizer = self._get_label_tokenizer()
        self.embeddings = self._get_embeddings(self.feat_tokenizers, embedding_dim)

        ehr_adj = self.generate_ehr_adj()
        ddi_adj = self.generate_ddi_adj()

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

        self.gamenet = GAMENetLayer(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            ehr_adj=ehr_adj,
            ddi_adj=ddi_adj,
            **kwargs,
        )

    def generate_ehr_adj(self):
        """
        generate the ehr graph adj for GAMENet model input
        - loop over the training data to check whether any med pair appear
        """
        label_size = self.label_tokenizer.get_vocabulary_size()
        ehr_adj = np.zeros((label_size, label_size))
        for sample in self.dataset:
            curr_drugs = sample["drugs"][-1]
            encoded_drugs = self.label_tokenizer.convert_tokens_to_indices(curr_drugs)
            for idx1, med1 in enumerate(encoded_drugs):
                for idx2, med2 in enumerate(encoded_drugs):
                    if idx1 >= idx2:
                        continue
                    ehr_adj[med1, med2] = 1
                    ehr_adj[med2, med1] = 1
        return ehr_adj

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
        return ddi_adj

    def forward(self, device, conditions, procedures, drugs, **kwargs):
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

        label_size = self.label_tokenizer.get_vocabulary_size()
        drugs = self.label_tokenizer.batch_encode_3d(
            drugs, padding=(False, False), truncation=(True, False)
        )

        curr_drugs = [p[-1] for p in drugs]
        curr_drugs = batch_to_multihot(curr_drugs, label_size)
        curr_drugs = curr_drugs.to(device)

        prev_drugs = [p[:-1] for p in drugs]
        max_num_visit = max([len(p) for p in prev_drugs])
        prev_drugs = [p + [[]] * (max_num_visit - len(p)) for p in prev_drugs]
        prev_drugs = [batch_to_multihot(p, label_size) for p in prev_drugs]
        prev_drugs = torch.stack(prev_drugs, dim=0)
        prev_drugs = prev_drugs.to(device)

        # get mask
        mask = torch.sum(conditions, dim=2) != 0

        # process drugs
        loss, y_prob = self.gamenet(queries, mask, prev_drugs, curr_drugs)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": curr_drugs,
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
    model = GAMENet(
        dataset=dataset,
    )
    model.to("cuda")
    batch = iter(dataloader).next()
    output = model(**batch, device="cuda")
    print(output["loss"])
