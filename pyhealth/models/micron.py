
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel
from pyhealth.processors.base_processor import FeatureProcessor
from pyhealth.processors import (
    SequenceProcessor, StageNetProcessor, StageNetTensorProcessor, TimeseriesProcessor, TensorProcessor, MultiHotProcessor
)
from pyhealth.processors.base_processor import FeatureProcessor
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
        >>> y_prob.shape  # Probabilities for each drug
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
        """Compute reconstruction loss between predicted and actual medication changes.

        The reconstruction loss measures how well the model captures medication changes
        between consecutive visits by comparing the predicted changes (through residual
        connections) with actual changes in prescriptions.

        Args:
            logits (torch.tensor): Raw logits for medication predictions across all visits.
            logits_residual (torch.tensor): Residual logits representing predicted changes.
            mask (torch.tensor): Boolean mask indicating valid visits.

        Returns:
            torch.tensor: Mean squared reconstruction loss value.
        """
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
    """MICRON model (PyHealth 2.0 compatible).

    Paper: Chaoqi Yang et al. Change Matters: Medication Change Prediction
    with Recurrent Residual Networks. IJCAI 2021.

    This model is for medication prediction using PyHealth 2.0 SampleDataset and processors.
    It expects input_schema to include 'conditions' and 'procedures' as sequence features,
    and output_schema to include 'drugs' as a multilabel/multihot feature.

    Args:
        dataset (SampleDataset): Dataset object containing patient records and schema information.
        embedding_dim (int, optional): Dimension for feature embeddings. Defaults to 128.
        hidden_dim (int, optional): Dimension for hidden layers. Defaults to 128.
        **kwargs: Additional parameters passed to the MICRON layer (e.g., lam for loss weighting).

    Attributes:
        embedding_model (EmbeddingModel): Handles embedding of input features.
        feature_processors (dict): Maps feature keys to their respective processors.
        micron (MICRONLayer): Core MICRON layer for medication prediction.

    Note:
        The model expects specific schema configurations:
        - input_schema should include 'conditions' and 'procedures' as sequence features
        - output_schema should include 'drugs' as a multilabel/multihot feature

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "conditions": ["E11.9", "I10"],
        ...         "procedures": ["0DJD8ZZ"],
        ...         "drugs": ["metformin", "lisinopril"]
        ...     }
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"conditions": "sequence", "procedures": "sequence"},
        ...     output_schema={"drugs": "multilabel"},
        ...     dataset_name="test",
        ... )
        >>> model = MICRON(dataset=dataset)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs
    ):
        super().__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        assert len(self.label_keys) == 1, "Only one label key is supported."
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        self.feature_processors = {
            feature_key: self.dataset.input_processors[feature_key]
            for feature_key in self.feature_keys
        }

        # validate kwargs for MICRON layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim and number of features")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")
        if "num_drugs" in kwargs:
            raise ValueError("num_drugs is determined by the dataset")

        # Get label processor
        label_processor = self.dataset.output_processors[self.label_key]
        
        # Get vocabulary size using the standard size() method
        if not hasattr(label_processor, "size"):
            raise ValueError(
                "Label processor must implement size() method. "
                "The processor type is: " + type(label_processor).__name__
            )
        
        num_drugs = label_processor.size()
        if num_drugs == 0:
            raise ValueError("Label processor returned 0 size")
        
        self.micron = MICRONLayer(
            input_size=embedding_dim * len(self.feature_keys),
            hidden_size=hidden_dim,
            num_drugs=num_drugs,
            **kwargs
        )

        # save ddi adjacency matrix for later use
        ddi_adj = self.generate_ddi_adj()
        np.save(os.path.join(CACHE_PATH, "ddi_adj.npy"), ddi_adj)

    @staticmethod
    def _split_temporal(feature):
        if isinstance(feature, tuple) and len(feature) == 2:
            return feature
        return None, feature

    def _ensure_tensor(self, feature_key: str, value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        processor = self.feature_processors[feature_key]
        if isinstance(processor, (SequenceProcessor, StageNetProcessor)):
            return torch.tensor(value, dtype=torch.long)
        return torch.tensor(value, dtype=torch.float)

    def _pool_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.sum(dim=2)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Make sure temporal dimension (dim=1) matches the longest sequence
        if x.size(1) == 1:
            # Repeat to handle shorter sequences
            x = x.repeat(1, 2, 1)
        return x

    def _create_mask(self, feature_key: str, value: torch.Tensor) -> torch.Tensor:
        processor = self.feature_processors[feature_key]
        if isinstance(processor, SequenceProcessor):
            mask = value != 0
        elif isinstance(processor, StageNetProcessor):
            if value.dim() >= 3:
                mask = torch.any(value != 0, dim=-1)
            else:
                mask = value != 0
        elif isinstance(processor, (TimeseriesProcessor, StageNetTensorProcessor)):
            if value.dim() >= 3:
                mask = torch.any(torch.abs(value) > 0, dim=-1)
            elif value.dim() == 2:
                mask = torch.any(torch.abs(value) > 0, dim=-1, keepdim=True)
            else:
                mask = torch.ones(
                    value.size(0),
                    1,
                    dtype=torch.bool,
                    device=value.device,
                )
        elif isinstance(processor, (TensorProcessor, MultiHotProcessor)):
            mask = torch.ones(
                value.size(0),
                1,
                dtype=torch.bool,
                device=value.device,
            )
        else:
            if value.dim() >= 2:
                mask = torch.any(value != 0, dim=-1)
            else:
                mask = torch.ones(
                    value.size(0),
                    1,
                    dtype=torch.bool,
                    device=value.device,
                )
        if mask.dim() == 1:
            mask = mask.unsqueeze(1)
        mask = mask.bool()
        if mask.dim() == 2:
            invalid_rows = ~mask.any(dim=1)
            if invalid_rows.any():
                mask[invalid_rows, 0] = True
        return mask

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation with PyHealth 2.0 inputs.

        Args:
            **kwargs: Keyword arguments that include every feature key defined in
                the dataset schema plus the label key. Additional arguments:
                - register_hook (bool): whether to register attention hooks
                - embed (bool): whether to return embeddings in output

        Returns:
            Dict[str, torch.Tensor]: Prediction dictionary containing the loss,
            probabilities, labels, and optionally embeddings.
        """
        patient_emb = []
        embedding_inputs: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}

        for feature_key in self.feature_keys:
            _, value = self._split_temporal(kwargs[feature_key])
            value_tensor = self._ensure_tensor(feature_key, value).to(self.device)
            embedding_inputs[feature_key] = value_tensor
            masks[feature_key] = self._create_mask(feature_key, value_tensor).to(self.device)

        embedded = self.embedding_model(embedding_inputs)

        for feature_key in self.feature_keys:
            x = embedded[feature_key]
            mask = masks[feature_key]
            x = self._pool_embedding(x)
            patient_emb.append(x)

        # Concatenate along last dim: [batch, seq_len, embedding_dim * n_features]
        patient_emb = torch.cat(patient_emb, dim=2)
        # Use visit-level mask from first feature (or combine as needed)
        mask = masks[self.feature_keys[0]]

        # Labels: expects multihot [batch, num_labels]
        y_true = kwargs[self.label_key].to(self.device)

        loss, y_prob = self.micron(patient_emb, y_true, mask)
        
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results

    def generate_ddi_adj(self) -> torch.Tensor:
        """Generates the drug-drug interaction (DDI) graph adjacency matrix using PyHealth 2.0 label processor."""
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True)
        label_processor = self.dataset.output_processors[self.label_key]
        label_size = label_processor.size()
        vocab_to_index = label_processor.label_vocab
        ddi_adj = np.zeros((label_size, label_size))
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi
        ]
        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index[atc_i], vocab_to_index[atc_j]] = 1
                ddi_adj[vocab_to_index[atc_j], vocab_to_index[atc_i]] = 1
        return torch.tensor(ddi_adj, dtype=torch.float)


