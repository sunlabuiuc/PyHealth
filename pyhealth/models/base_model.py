from abc import ABC
from typing import List, Dict, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import BaseDataset
from pyhealth.models.utils import batch_to_multihot
from pyhealth.tokenizer import Tokenizer

# TODO: add support for regression
VALID_MODE = ["binary", "multiclass", "multilabel"]


class BaseModel(ABC, nn.Module):
    """Abstract class for PyTorch models.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys: list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
    """

    def __init__(
            self,
            dataset: BaseDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
    ):
        super(BaseModel, self).__init__()
        assert mode in VALID_MODE, f"mode must be one of {VALID_MODE}"
        self.dataset = dataset
        self.feature_keys = feature_keys
        self.label_key = label_key
        self.mode = mode
        # used to query the device of the model
        self._dummy_param = nn.Parameter(torch.empty(0))
        return

    @property
    def device(self):
        """Gets the device of the model."""
        return self._dummy_param.device

    @staticmethod
    def obtain_element_type(obj):
        """
        Args:
            obj: a list or list of list obj (must be a homogeneous list)
        Returns:
            the type of the element in the list
            e.g., 
                obj = [1, 2, 3] -> int
                obj = [[1, 2], [3, 4]] -> int
                obj = [['a', 'b'], ['a', 'b']] -> str
                obj = [[1.3. 2], [5.5]] -> float
        """ 
        if type(obj[0]) == list:
            obj = sum(obj, [])
        return type(obj[0])

    @staticmethod
    def padding3d(batch):
        """
        Args:
            batch: a list of list of list obj
                - 1-level: number of samples/patients
                - 2-level: number of visit, length maybe not equal
                - 3-level: number of features per visit, length must be equal
        Returns:
            padded_batch: a padded list of list of list obj
            e.g., 
                batch = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] -> [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
                batch = [[[1, 2], [3, 4]], [[5, 6]]] -> [[[1, 2], [3, 4]], [[5, 6], [0, 0]]]
        """
        # level-2 padding
        batch_max_length = max([len(x) for x in batch])
        mask = torch.zeros(len(batch), batch_max_length, dtype=torch.bool)
        for i, x in enumerate(batch):
            mask[i, :len(x)] = 1
        padded_batch = [x + [[0] * len(x[0])] * (batch_max_length - len(x)) for x in batch]
        
        return padded_batch, mask
        
    def get_feature_tokenizers(self, special_tokens=None) -> Dict[str, Tokenizer]:
        """Gets the default feature tokenizers using `self.feature_keys`.

        Args:
            special_tokens: a list of special tokens to add to the tokenizer.
                Default is ["<pad>", "<unk>"].

        Returns:
            feature_tokenizers: a dictionary of feature tokenizers with keys
                corresponding to self.feature_keys.
        """
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>"]
            
        # obtain a sample and get the data type, only str needs tokenizer
        sample = self.dataset.samples[0]
        feature_tokenizers = {}
        for feature_key in self.feature_keys:
            if self.obtain_element_type(sample[feature_key]) == str:
                feature_tokenizers[feature_key] = Tokenizer(
                    tokens=self.dataset.get_all_tokens(key=feature_key),
                    special_tokens=special_tokens
                )
        return feature_tokenizers

    def get_label_tokenizer(self, special_tokens=None) -> Tokenizer:
        """Gets the default label tokenizers using `self.label_key`.

        Args:
            special_tokens: a list of special tokens to add to the tokenizer.
                Default is empty list.

        Returns:
            label_tokenizer: the label tokenizer.
        """
        if special_tokens is None:
            special_tokens = []
        label_tokenizer = Tokenizer(
            self.dataset.get_all_tokens(key=self.label_key),
            special_tokens=special_tokens
        )
        return label_tokenizer

    @staticmethod
    def get_embedding_layers(
            feature_tokenizers: Dict[str, Tokenizer],
            embedding_dim: int,
    ) -> nn.ModuleDict:
        """Gets the default embedding layers using the feature tokenizers.

        Args:
            feature_tokenizers: a dictionary of feature tokenizers with keys
                corresponding to `self.feature_keys`.
            embedding_dim: the dimension of the embedding.

        Returns:
            embedding_layers: a module dictionary of embedding layers with keys
                corresponding to `self.feature_keys`.
        """
        embedding_layers = nn.ModuleDict()
        for key, tokenizer in feature_tokenizers.items():
            embedding_layers[key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )
        return embedding_layers

    def get_output_size(self, label_tokenizer: Tokenizer) -> int:
        """Gets the default output size using the label tokenizer and `self.mode`.

        If the mode is "binary", the output size is 1. If the mode is "multiclass"
        or "multilabel", the output size is the number of classes or labels.

        Args:
            label_tokenizer: the label tokenizer.

        Returns:
            output_size: the output size of the model.
        """
        output_size = label_tokenizer.get_vocabulary_size()
        if self.mode == "binary":
            assert output_size == 2
            output_size = 1
        return output_size

    def get_loss_function(self) -> Callable:
        """Gets the default loss function using `self.mode`.

        The default loss functions are:
            - binary: `F.binary_cross_entropy_with_logits`
            - multiclass: `F.cross_entropy`
            - multilabel: `F.binary_cross_entropy_with_logits`

        Returns:
            The default loss function.
        """
        if self.mode == "binary":
            return F.binary_cross_entropy_with_logits
        elif self.mode == "multiclass":
            return F.cross_entropy
        elif self.mode == "multilabel":
            return F.binary_cross_entropy_with_logits
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def prepare_labels(
            self,
            labels: Union[List[str], List[List[str]]],
            label_tokenizer: Tokenizer,
    ) -> torch.Tensor:
        """Prepares the labels for model training and evaluation.

        This function converts the labels to different formats depending on the
        mode. The default formats are:
            - binary: a tensor of shape (batch_size, 1)
            - multiclass: a tensor of shape (batch_size,)
            - multilabel: a tensor of shape (batch_size, num_labels)

        Args:
            labels: the raw labels from the samples. It should be a list of
                str for binary and multiclass classification and a list of
                list of str for multilabel classification.
            label_tokenizer: the label tokenizer.

        Returns:
            labels: the processed labels.
        """
        if self.mode in ["binary"]:
            labels = label_tokenizer.convert_tokens_to_indices(labels)
            labels = torch.FloatTensor(labels).unsqueeze(-1)
        elif self.mode in ["multiclass"]:
            labels = label_tokenizer.convert_tokens_to_indices(labels)
            labels = torch.LongTensor(labels)
        elif self.mode in ["multilabel"]:
            # convert to indices
            labels_index = label_tokenizer.batch_encode_2d(
                labels, padding=False, truncation=False
            )
            # convert to multihot
            num_labels = label_tokenizer.get_vocabulary_size()
            labels = batch_to_multihot(labels_index, num_labels)
        else:
            raise NotImplementedError
        labels = labels.to(self.device)
        return labels

    def prepare_y_prob(self, logits: torch.Tensor) -> torch.Tensor:
        """Prepares the predicted probabilities for model evaluation.

        This function converts the predicted logits to predicted probabilities
        depending on the mode. The default formats are:
            - binary: a tensor of shape (batch_size, 1) with values in [0, 1],
                which is obtained with `torch.sigmoid()`
            - multiclass: a tensor of shape (batch_size, num_classes) with
                values in [0, 1] and sum to 1, which is obtained with
                `torch.softmax()`
            - multilabel: a tensor of shape (batch_size, num_labels) with values
                in [0, 1], which is obtained with `torch.sigmoid()`

        Args:
            logits: the predicted logit tensor.

        Returns:
            y_prob: the predicted probability tensor.
        """
        if self.mode in ["binary"]:
            y_prob = torch.sigmoid(logits)
        elif self.mode in ["multiclass"]:
            y_prob = F.softmax(logits, dim=-1)
        elif self.mode in ["multilabel"]:
            y_prob = torch.sigmoid(logits)
        else:
            raise NotImplementedError
        return y_prob
