from typing import List, Tuple, Dict, Optional
import os

import torch
import torch.nn as nn
import numpy as np
from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit
from pyhealth import BASE_CACHE_PATH as CACHE_PATH
from pyhealth.medcode import ATC

class MICRONLayer(nn.Module):
    """MICRON layer.

    Paper: Chaoqi Yang et al. Change Matters: Medication Change Prediction
    with Recurrent Residual Networks. IJCAI 2021.

    This layer is used in the MICRON model. But it can also be used as a
    standalone layer.

    Args:
        input_size: input feature size.
        hidden_size: hidden feature size.
        num_drugs: total number of drugs to recommend.
        lam: regularization parameter for the reconstruction loss. Default is 0.1.

    Examples:
        >>> from pyhealth.models import MICRONLayer
        >>> patient_emb = torch.randn(3, 5, 32) # [patient, visit, input_size]
        >>> drugs = torch.randint(0, 2, (3, 50)).float()
        >>> layer = MICRONLayer(32, 64, 50)
        >>> loss, y_prob = layer(patient_emb, drugs)
        >>> loss.shape
        torch.Size([])
        >>> y_prob.shape
        torch.Size([3, 50])
    """

    def __init__(
        self, input_size: int, hidden_size: int, num_drugs: int, lam: float = 0.1
    ):
        super(MICRONLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_labels = num_drugs
        self.lam = lam

        self.health_net = nn.Linear(input_size, hidden_size)
        self.prescription_net = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_drugs)

        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    @staticmethod
    def compute_reconstruction_loss(
        logits: torch.tensor, logits_residual: torch.tensor, mask: torch.tensor
    ) -> torch.tensor:
        rec_loss = torch.mean(
            torch.square(
                torch.sigmoid(logits[:, 1:, :])
                - torch.sigmoid(logits[:, :-1, :] + logits_residual)
            )
            * mask[:, 1:].unsqueeze(2)
        )
        return rec_loss

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
            mask: an optional tensor of shape [patient, visit] where
                1 indicates valid visits and 0 indicates invalid visits.

        Returns:
            loss: a scalar tensor representing the loss.
            y_prob: a tensor of shape [patient, num_labels] representing
                the probability of each drug.
        """
        if mask is None:
            mask = torch.ones_like(patient_emb[:, :, 0])

        # (patient, visit, hidden_size)
        health_rep = self.health_net(patient_emb)
        drug_rep = self.prescription_net(health_rep)
        logits = self.fc(drug_rep)
        logits_last_visit = get_last_visit(logits, mask)
        bce_loss = self.bce_loss_fn(logits_last_visit, drugs)

        # (batch, visit-1, input_size)
        health_rep_last = health_rep[:, :-1, :]
        # (batch, visit-1, input_size)
        health_rep_cur = health_rep[:, 1:, :]
        # (batch, visit-1, input_size)
        health_rep_residual = health_rep_cur - health_rep_last
        drug_rep_residual = self.prescription_net(health_rep_residual)
        logits_residual = self.fc(drug_rep_residual)
        rec_loss = self.compute_reconstruction_loss(logits, logits_residual, mask)

        loss = bce_loss + self.lam * rec_loss
        y_prob = torch.sigmoid(logits_last_visit)

        return loss, y_prob


class MICRON(BaseModel):
    """MICRON model.

    Paper: Chaoqi Yang et al. Change Matters: Medication Change Prediction
    with Recurrent Residual Networks. IJCAI 2021.

    Note:
        This model is only for medication prediction which takes conditions
        and procedures as feature_keys, and drugs as label_key. It only operates
        on the visit level.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        **kwargs: other parameters for the MICRON layer.
    """

    def __init__(
        self,
        dataset: SampleEHRDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs
    ):
        super(MICRON, self).__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            mode="multilabel",
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)

        # validate kwargs for MICRON layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")
        if "num_drugs" in kwargs:
            raise ValueError("num_drugs is determined by the dataset")
        self.micron = MICRONLayer(
            input_size=embedding_dim * 2,
            hidden_size=hidden_dim,
            num_drugs=self.label_tokenizer.get_vocabulary_size(),
            **kwargs
        )
        
        # save ddi adj
        ddi_adj = self.generate_ddi_adj()
        np.save(os.path.join(CACHE_PATH, "ddi_adj.npy"), ddi_adj)

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
        return ddi_adj

    def forward(
        self,
        conditions: List[List[List[str]]],
        procedures: List[List[List[str]]],
        drugs: List[List[str]],
        **kwargs
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

        procedures = self.feat_tokenizers["procedures"].batch_encode_3d(procedures)
        # (patient, visit, code)
        procedures = torch.tensor(procedures, dtype=torch.long, device=self.device)
        # (patient, visit, code, embedding_dim)
        procedures = self.embeddings["procedures"](procedures)
        # (patient, visit, embedding_dim)
        procedures = torch.sum(procedures, dim=2)

        # (patient, visit, embedding_dim * 2)
        patient_emb = torch.cat([conditions, procedures], dim=2)
        # (patient, visit)
        mask = torch.sum(patient_emb, dim=2) != 0
        # (patient, num_labels)
        drugs = self.prepare_labels(drugs, self.label_tokenizer)

        loss, y_prob = self.micron(patient_emb, drugs, mask)

        return {"loss": loss, "y_prob": y_prob, "y_true": drugs}
