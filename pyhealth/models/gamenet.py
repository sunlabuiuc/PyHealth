import math
from typing import Tuple, List, Dict, Optional
import os

import torch
import torch.nn as nn
import numpy as np
from pyhealth.datasets import SampleDataset
from pyhealth.medcode import ATC
from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit, batch_to_multihot
from pyhealth import BASE_CACHE_PATH as CACHE_PATH
from pyhealth.models.embedding import EmbeddingModel

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

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.tasks import drug_recommendation_mimic3_fn
        >>> from pyhealth.models import GAMENet
        >>> from pyhealth.datasets import split_by_patient, get_dataloader
        >>> from pyhealth.trainer import Trainer
        >>>
        >>> # Load MIMIC-III dataset
        >>> dataset = create_sample_dataset(
        ...     samples=[
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-0",
        ...             "conditions": [["cond-33", "cond-86"], ["cond-80"]],
        ...             "procedures": [["proc-12", "proc-45"], ["proc-23"]],
        ...             "drugs": ["drug-1", "drug-2", "drug-3"],
        ...         }
        ...     ],
        ...     input_schema={
        ...         "conditions": "nested_sequence",
        ...         "procedures": "nested_sequence",
        ...     },
        ...     output_schema={"drugs": "multilabel"},
        ...     dataset_name="test",
        ... )
        >>>
        >>> # Initialize model
        >>> model = GAMENet(
        ...     dataset=dataset,
        ...     embedding_dim=128,
        ...     hidden_dim=128,
        ... )
        >>>
        >>> # Create dataloaders
        >>> train_loader = get_dataloader(dataset, batch_size=32, shuffle=True)
        >>>
        >>> # Train model
        >>> trainer = Trainer(model=model, metrics=["jaccard_samples", "ddi"])
        >>> trainer.train(
        ...     train_dataloader=train_loader,
        ...     epochs=10,
        ... )
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.5,
        **kwargs
    ):
        super(GAMENet, self).__init__(
            dataset=dataset,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        assert "conditions" in self.dataset.input_schema, "conditions must be in input_schema"
        assert "procedures" in self.dataset.input_schema, "procedures must be in input_schema"
        assert "drugs" in self.dataset.output_schema, "drugs must be in output_schema"

        # feature_keys and label_key for GAMENet
        self.feature_keys = ["conditions", "procedures"]
        assert len(self.label_keys) == 1, "Only one label key is supported for GAMENet"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        self.label_size = len(self.dataset.output_processors[self.label_key].label_vocab)

        # adj matrix
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
        np.save(os.path.join(CACHE_PATH, "ddi_adj.npy"), ddi_adj.numpy())

    def generate_ehr_adj(self) -> torch.tensor:
        """Generates the EHR graph adjacency matrix."""
        label_vocab = self.dataset.output_processors[self.label_key].label_vocab
        label_size = len(label_vocab)
        ehr_adj = torch.zeros((label_size, label_size))
        for sample in self.dataset:
            curr_drugs = sample["drugs"]
            if isinstance(curr_drugs, torch.Tensor):
                continue
            encoded_drugs = []
            for drug in curr_drugs:
                if drug in label_vocab:
                    encoded_drugs.append(label_vocab[drug])
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
        label_vocab = self.dataset.output_processors[self.label_key].label_vocab
        label_size = len(label_vocab)
        ddi_adj = torch.zeros((label_size, label_size))
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi
        ]
        for atc_i, atc_j in ddi_atc3:
            if atc_i in label_vocab and atc_j in label_vocab:
                ddi_adj[label_vocab[atc_i], label_vocab[atc_j]] = 1
                ddi_adj[label_vocab[atc_j], label_vocab[atc_i]] = 1
        return ddi_adj

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: Keyword arguments that include feature keys and label key.
                The dataloader provides already-processed tensors.
                Expected keys:
                - conditions: tensor of shape [batch, visits, codes_per_visit]
                - procedures: tensor of shape [batch, visits, codes_per_visit]
                - drugs: tensor of shape [batch, num_drugs] (multilabel)

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor of shape [batch, num_labels] representing
                    the probability of each drug.
                y_true: a tensor of shape [batch, num_labels] representing
                    the ground truth of each drug.

        """
        embedded = self.embedding_model(kwargs)

        # embedded["conditions"] shape: [batch, visits, codes_per_visit, embedding_dim]
        # embedded["procedures"] shape: [batch, visits, codes_per_visit, embedding_dim]
        conditions = embedded["conditions"]
        procedures = embedded["procedures"]

        # [batch, visits, embedding_dim]
        conditions = conditions.sum(dim=2)
        procedures = procedures.sum(dim=2)

        # [batch, visits, hidden_dim]
        conditions, _ = self.cond_rnn(conditions)
        procedures, _ = self.proc_rnn(procedures)

        # [batch, visits, 2*hidden_dim] -> [batch, visits, hidden_dim]
        patient_representations = torch.cat([conditions, procedures], dim=-1)
        queries = self.query(patient_representations)

        # get current drugs (labels): [batch, num_drugs]
        y_true = kwargs[self.label_key].to(self.device)

        batch_size = queries.size(0)
        num_visits = queries.size(1)

        prev_drugs = torch.zeros(batch_size, num_visits, self.label_size, device=self.device)

        # [batch, visits]
        mask = (embedded["conditions"].sum(dim=-1) != 0).any(dim=-1)

        loss, y_prob = self.gamenet(queries, prev_drugs, y_true, mask)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }
