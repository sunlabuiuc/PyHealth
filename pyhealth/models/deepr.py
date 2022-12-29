from typing import Tuple, List, Dict, Optional
import functools

import torch
import torch.nn as nn

from pyhealth.datasets import BaseDataset
from pyhealth.models import BaseModel

VALID_OPERATION_LEVEL = ["visit"]

class DeeprLayer(nn.Module):
    """Deepr layer.

    Paper: P. Nguyen, T. Tran, N. Wickramasinghe and S. Venkatesh, 
    " Deepr : A Convolutional Net for Medical Records," in IEEE Journal 
    of Biomedical and Health Informatics, vol. 21, no. 1, pp. 22-30, 
    Jan. 2017, doi: 10.1109/JBHI.2016.2633963.
    
    This layer is used in the Deepr model. 

    Args:
        feature_size: embedding dim of codes (m in the original paper).
        window: sliding window (d in the original paper)
        hidden_size: number of conv filters (motif size, p, in the original paper)
    """

    def __init__(
        self,
        feature_size: int = 100, 
        window: int = 1,
        hidden_size: int = 3, 
    ):
        super(DeeprLayer, self).__init__()

        self.conv = torch.nn.Conv1d(feature_size, hidden_size, kernel_size=2 * window + 1)

    def forward(
        self,
        x: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input size].
            
        Returns:
            c: a tensor of shape [batch size, hidden_size] representing the
                summarized vector.
        """
        x = x.permute(0, 2, 1) # [batch size, input size, sequence len]
        x = torch.relu(self.conv(x))
        x = x.max(-1)[0]
        return x

class Deepr(BaseModel):
    """Deepr model.

    Paper: P. Nguyen, T. Tran, N. Wickramasinghe and S. Venkatesh, 
    " Deepr : A Convolutional Net for Medical Records," in IEEE Journal 
    of Biomedical and Health Informatics, vol. 21, no. 1, pp. 22-30, 
    Jan. 2017, doi: 10.1109/JBHI.2016.2633963.

    Note:
        We use separate Deepr layers for different feature_keys.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        **kwargs: other parameters for the RNN layer.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        operation_level: str = "visit",
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super(Deepr, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        assert (
            operation_level in VALID_OPERATION_LEVEL
        ), f"operation_level must be one of {VALID_OPERATION_LEVEL}"
        self.operation_level = operation_level
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        #TODO: Use more tokens for <gap> for different lengths once the input has such information
        self.feat_tokenizers = self.get_feature_tokenizers(
            special_tokens=["<pad>", "<unk>", "<gap>"]
        )
        self.label_tokenizer = self.get_label_tokenizer()

        # TODO: Pretrain this embeddings with word2vec?
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)

        # validate kwargs for CNN layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")
        self.cnn = nn.ModuleDict()
        for feature_key in feature_keys:
            self.cnn[feature_key] = DeeprLayer(
                feature_size=embedding_dim, hidden_size=hidden_dim, **kwargs
            )
        output_size = self.get_output_size(self.label_tokenizer)
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def visit_level_forward(self, **kwargs):
        """Visit-level CNN forward."""
        patient_emb = []
        for feature_key in self.feature_keys:
            assert type(kwargs[feature_key][0][0]) == list
            feature_vals = [functools.reduce(lambda a, b: a + ['<gap>'] + b, _) for _ in kwargs[feature_key]]
            x = self.feat_tokenizers[feature_key].batch_encode_2d(feature_vals, padding=True, truncation=False)
            # (patient, visit, code)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, code, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, hidden_dim)
            x = self.cnn[feature_key](x)
            patient_emb.append(x)
        # (patient, features * hidden_dim)
        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation."""
        if self.operation_level == "visit":
            return self.visit_level_forward(**kwargs)
        else:
            raise NotImplementedError()