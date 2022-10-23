from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch
import torch.nn as nn

from pyhealth.datasets import BaseDataset
from pyhealth.models.utils import get_default_loss_function, batch_to_multihot
from pyhealth.tokenizer import Tokenizer

VALID_MODE = ["binary", "multiclass", "multilabel"]


class BaseModel(ABC, nn.Module):
    """Abstract base model for all tasks.

    Args:
        dataset: BaseDataset object
        feature_keys: list of keys in samples to use as features, e.g. ["conditions",
            "procedures"]
        label_key: key in samples to use as label (e.g., "drugs")
        mode: "binary", "multiclass", or "multilabel"
    """

    def __init__(
            self,
            dataset: BaseDataset,
            feature_keys: Union[List[str], Tuple[str]],
            label_key: str,
            mode: str,
    ):
        super(BaseModel, self).__init__()
        assert mode in VALID_MODE, f"mode must be one of {VALID_MODE}"
        self.dataset = dataset
        self.feature_keys = feature_keys
        self.label_key = label_key
        self.mode = mode
        return

    @abstractmethod
    def forward(self, device, **kwargs):
        raise NotImplementedError

    def _get_feature_tokenizers(self):
        feature_tokenizers = {}
        for feature_key in self.feature_keys:
            feature_tokenizers[feature_key] = Tokenizer(
                tokens=self.dataset.get_all_tokens(key=feature_key),
                special_tokens=["<pad>", "<unk>"]
            )
        return feature_tokenizers

    def _get_label_tokenizer(self):
        label_tokenizer = Tokenizer(
            self.dataset.get_all_tokens(key=self.label_key),
            special_tokens=[]
        )
        return label_tokenizer

    def _get_embeddings(self, feature_tokenizers, embedding_dim):
        embeddings = nn.ModuleDict()
        for feature_key in self.feature_keys:
            embeddings[feature_key] = nn.Embedding(
                feature_tokenizers[feature_key].get_vocabulary_size(),
                embedding_dim,
                padding_idx=feature_tokenizers[feature_key].get_padding_index(),
            )
        return embeddings

    def _get_output_size(self, label_tokenizer):
        output_size = self.label_tokenizer.get_vocabulary_size()
        if self.mode == "binary":
            assert output_size == 2
            output_size = 1
        return output_size

    def _calculate_output(self, logits, labels):
        """Calculates the output.

        We support binary, multiclass, and multilabel classification.

        Args:
            logits: the logits of the model.
            labels: the ground truth labels.
        Returns:
            loss: the loss of the model
            y: ground truth
            y_prob: the probability of the output of the model
        """

        if self.mode in ["binary"]:
            y = self.label_tokenizer.convert_tokens_to_indices(labels)
            y = torch.FloatTensor(y).unsqueeze(-1)
            y = y.to(logits.device)
            loss = get_default_loss_function(self.mode)(logits, y)
            y_prob = torch.sigmoid(logits)

        elif self.mode in ["multiclass"]:
            y = self.label_tokenizer.convert_tokens_to_indices(labels)
            y = torch.LongTensor(y)
            y = y.to(logits.device)
            loss = get_default_loss_function(self.mode)(logits, y)
            y_prob = torch.softmax(logits, dim=-1)

        elif self.mode in ["multilabel"]:
            y = self.label_tokenizer.batch_encode_2d(
                labels, padding=False, truncation=False
            )
            y = batch_to_multihot(y, self.label_tokenizer.get_vocabulary_size())
            y = y.to(logits.device)
            loss = get_default_loss_function(self.mode)(logits, y)
            y_prob = torch.sigmoid(logits)

        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

        return loss, y, y_prob
