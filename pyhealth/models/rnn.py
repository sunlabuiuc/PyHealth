from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.processors import (
    DeepNestedFloatsProcessor,
    DeepNestedSequenceProcessor,
    MultiHotProcessor,
    NestedFloatsProcessor,
    NestedSequenceProcessor,
    SequenceProcessor,
    StageNetProcessor,
    StageNetTensorProcessor,
    TensorProcessor,
    TimeseriesProcessor,
)

from .embedding import EmbeddingModel


class RNNLayer(nn.Module):
    """Recurrent neural network layer.

    This layer wraps the PyTorch RNN layer with masking and dropout support. It is
    used in the RNN model. But it can also be used as a standalone layer.

    Args:
        input_size: input feature size.
        hidden_size: hidden feature size.
        rnn_type: type of rnn, one of "RNN", "LSTM", "GRU". Default is "GRU".
        num_layers: number of recurrent layers. Default is 1.
        dropout: dropout rate. If non-zero, introduces a Dropout layer before each
            RNN layer. Default is 0.5.
        bidirectional: whether to use bidirectional recurrent layers. If True,
            a fully-connected layer is applied to the concatenation of the forward
            and backward hidden states to reduce the dimension to hidden_size.
            Default is False.

    Examples:
        >>> from pyhealth.models import RNNLayer
        >>> input = torch.randn(3, 128, 5)  # [batch size, sequence len, input_size]
        >>> layer = RNNLayer(5, 64)
        >>> outputs, last_outputs = layer(input)
        >>> outputs.shape
        torch.Size([3, 128, 64])
        >>> last_outputs.shape
        torch.Size([3, 64])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: str = "GRU",
        num_layers: int = 1,
        dropout: float = 0.5,
        bidirectional: bool = False,
    ):
        super(RNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.dropout_layer = nn.Dropout(dropout)
        self.num_directions = 2 if bidirectional else 1
        rnn_module = getattr(nn, rnn_type)
        self.rnn = rnn_module(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        if bidirectional:
            self.down_projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input size].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            outputs: a tensor of shape [batch size, sequence len, hidden size],
                containing the output features for each time step.
            last_outputs: a tensor of shape [batch size, hidden size], containing
                the output features for the last time step.
        """
        # pytorch's rnn will only apply dropout between layers
        x = self.dropout_layer(x)
        batch_size = x.size(0)
        if mask is None:
            lengths = torch.full(
                size=(batch_size,), fill_value=x.size(1), dtype=torch.int64
            )
        else:
            lengths = torch.sum(mask.int(), dim=-1).cpu()
        x = rnn_utils.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.rnn(x)
        outputs, _ = rnn_utils.pad_packed_sequence(outputs, batch_first=True)
        if not self.bidirectional:
            last_outputs = outputs[torch.arange(batch_size), (lengths - 1), :]
            return outputs, last_outputs
        else:
            outputs = outputs.view(batch_size, outputs.shape[1], 2, -1)
            f_last_outputs = outputs[torch.arange(batch_size), (lengths - 1), 0, :]
            b_last_outputs = outputs[:, 0, 1, :]
            last_outputs = torch.cat([f_last_outputs, b_last_outputs], dim=-1)
            outputs = outputs.view(batch_size, outputs.shape[1], -1)
            last_outputs = self.down_projection(last_outputs)
            outputs = self.down_projection(outputs)
            return outputs, last_outputs


class RNN(BaseModel):
    """Recurrent neural network model.

    This model applies a separate RNN layer for each feature, and then concatenates
    the final hidden states of each RNN layer. The concatenated hidden states are
    then fed into a fully connected layer to make predictions.

    Note:
        We use separate RNN layers for different feature_keys.
        Currently, we support two types of input formats:
            - Sequence of codes (e.g., diagnosis codes, procedure codes)
                - Input format: (batch_size, sequence_length)
                - Each code is embedded into a vector and RNN is applied on the sequence
            - Timeseries values (e.g., lab tests, vital signs)
                - Input format: (batch_size, sequence_length, num_features)
                - Each timestep contains a fixed number of measurements
                - RNN is applied directly on the timeseries data

    Args:
        dataset (SampleDataset): the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim (int): the embedding dimension. Default is 128.
        hidden_dim (int): the hidden dimension. Default is 128.
        **kwargs: other parameters for the RNN layer (e.g., rnn_type, num_layers, dropout, bidirectional).

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "conditions": ["cond-33", "cond-86", "cond-80"],
        ...         "procedures": ["proc-12", "proc-45"],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "patient-1",
        ...         "visit_id": "visit-1",
        ...         "conditions": ["cond-12", "cond-52"],
        ...         "procedures": ["proc-23"],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"conditions": "sequence", "procedures": "sequence"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="test"
        ... )
        >>>
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>>
        >>> model = RNN(dataset=dataset, embedding_dim=128, hidden_dim=64)
        >>>
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(...),
            'y_prob': tensor(...),
            'y_true': tensor(...),
            'logit': tensor(...)
        }
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs
    ):
        super(RNN, self).__init__(
            dataset=dataset,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # validate kwargs for RNN layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")
        assert len(self.label_keys) == 1, "Only one label key is supported if RNN is initialized"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        self.rnn = nn.ModuleDict()
        for feature_key in self.dataset.input_processors.keys():
            self.rnn[feature_key] = RNNLayer(
                input_size=embedding_dim, hidden_size=hidden_dim, **kwargs
            )
        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The label `kwargs[self.label_key]` is a list of labels for each patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with the following keys:
                - loss: a scalar tensor representing the loss.
                - y_prob: a tensor representing the predicted probabilities.
                - y_true: a tensor representing the true labels.
                - logit: a tensor representing the logits.
                - embed (optional): a tensor representing the patient embeddings if requested.
        """
        patient_emb = []
        
        # We need to preprocess kwargs to extract values and masks for EmbeddingModel
        # because EmbeddingModel expects dict of tensors
        inputs = {}
        masks = {}
        
        for feature_key in self.feature_keys:
            feature = kwargs[feature_key]
            if isinstance(feature, torch.Tensor):
                feature = (feature,)
            
            schema = self.dataset.input_processors[feature_key].schema()
            value = feature[schema.index("value")] if "value" in schema else None
            mask = feature[schema.index("mask")] if "mask" in schema else None
            
            if value is None:
                raise ValueError(f"Feature '{feature_key}' must contain 'value' in the schema.")
            
            inputs[feature_key] = value
            if mask is not None:
                masks[feature_key] = mask

        embedded = self.embedding_model(inputs, masks=masks)
        
        for feature_key in self.feature_keys:
            x = embedded[feature_key]
            # Use abs() before sum to catch edge cases where embeddings sum to 0
            # @TODO bug with 0 embedding sum can still persist if the embedding is all 0s but the mask is not all 0s. 
            # despite being valid values (e.g., [1.0, -1.0])
            
            # If we have an explicit mask, use it
            if feature_key in masks:
                mask = masks[feature_key].to(self.device).int()
                # Token-level mask (B, N_notes, L): reduce to note-level (B, N_notes)
                # by checking whether each note has at least one valid token.
                # This is needed when TupleTimeTextProcessor returns 3D token masks that
                # EmbeddingModel has already pooled down to (B, N_notes, H).
                if mask.dim() == 3:
                    mask = (mask.sum(dim=-1) > 0).int()   # (B, N_notes)
            else:
                mask = (torch.abs(x).sum(dim=-1) != 0).int()
            
            _, x = self.rnn[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


class MultimodalRNN(BaseModel):
    """Multimodal RNN model that handles both sequential and non-sequential features.

    This model extends the vanilla RNN to support mixed input modalities:
    - Sequential features (sequences, timeseries) go through RNN layers
    - Non-sequential features (multi-hot, tensor) bypass RNN and use embeddings directly

    The model automatically classifies input features based on their processor types:
    - Sequential processors (apply RNN): SequenceProcessor, NestedSequenceProcessor,
        DeepNestedSequenceProcessor, NestedFloatsProcessor, DeepNestedFloatsProcessor,
        TimeseriesProcessor
    - Non-sequential processors (embeddings only): MultiHotProcessor, TensorProcessor,
        StageNetProcessor, StageNetTensorProcessor

    For sequential features, the model:
    1. Embeds the input using EmbeddingModel
    2. Applies RNNLayer to get sequential representations
    3. Extracts the last hidden state

    For non-sequential features, the model:
    1. Embeds the input using EmbeddingModel
    2. Applies mean pooling if needed to reduce to 2D
    3. Uses the embedding directly

    All feature representations are concatenated and passed through a final
    fully connected layer for predictions.

    Args:
        dataset (SampleDataset): the dataset to train the model. It is used to query
            certain information such as the set of all tokens and processor types.
        embedding_dim (int): the embedding dimension. Default is 128.
        hidden_dim (int): the hidden dimension for RNN layers. Default is 128.
        **kwargs: other parameters for the RNN layer (e.g., rnn_type, num_layers,
            dropout, bidirectional).

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "conditions": ["cond-33", "cond-86"],  # sequential
        ...         "demographics": ["asian", "male"],      # multi-hot
        ...         "vitals": [120.0, 80.0, 98.6],        # tensor
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "patient-1",
        ...         "visit_id": "visit-1",
        ...         "conditions": ["cond-12", "cond-52"],  # sequential
        ...         "demographics": ["white", "female"],    # multi-hot
        ...         "vitals": [110.0, 75.0, 98.2],        # tensor
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "conditions": "sequence",
        ...         "demographics": "multi_hot",
        ...         "vitals": "tensor",
        ...     },
        ...     output_schema={"label": "binary"},
        ...     dataset_name="test"
        ... )
        >>>
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>>
        >>> model = MultimodalRNN(dataset=dataset, embedding_dim=128, hidden_dim=64)
        >>>
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(...),
            'y_prob': tensor(...),
            'y_true': tensor(...),
            'logit': tensor(...)
        }
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs
    ):
        super(MultimodalRNN, self).__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # validate kwargs for RNN layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # Classify features as sequential or non-sequential
        self.sequential_features = []
        self.non_sequential_features = []

        self.rnn = nn.ModuleDict()
        for feature_key in self.feature_keys:
            processor = dataset.input_processors[feature_key]
            if self._is_sequential_processor(processor):
                self.sequential_features.append(feature_key)
                # Create RNN for this feature
                self.rnn[feature_key] = RNNLayer(
                    input_size=embedding_dim,
                    hidden_size=hidden_dim,
                    **kwargs
                )
            else:
                self.non_sequential_features.append(feature_key)

        # Calculate final concatenated dimension
        final_dim = (len(self.sequential_features) * hidden_dim +
                     len(self.non_sequential_features) * embedding_dim)
        output_size = self.get_output_size()
        self.fc = nn.Linear(final_dim, output_size)

    def _is_sequential_processor(self, processor) -> bool:
        """Check if processor represents sequential data.

        Sequential processors are those that benefit from RNN processing,
        including sequences of codes and timeseries data.

        Note:
            StageNetProcessor and StageNetTensorProcessor are excluded as they
            are specialized for the StageNet model architecture and should be
            treated as non-sequential for standard RNN processing.

        Args:
            processor: The processor instance to check.

        Returns:
            bool: True if processor is sequential, False otherwise.
        """
        return isinstance(processor, (
            SequenceProcessor,
            NestedSequenceProcessor,
            DeepNestedSequenceProcessor,
            NestedFloatsProcessor,
            DeepNestedFloatsProcessor,
            TimeseriesProcessor,
        ))

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation handling mixed modalities.

        The label `kwargs[self.label_key]` is a list of labels for each patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with the following keys:
                - loss: a scalar tensor representing the loss.
                - y_prob: a tensor representing the predicted probabilities.
                - y_true: a tensor representing the true labels.
                - logit: a tensor representing the logits.
                - embed (optional): a tensor representing the patient embeddings if requested.
        """
        # Preprocess features
        inputs = {}
        masks = {}
        
        for feature_key in self.feature_keys:
            feature = kwargs[feature_key]
            if isinstance(feature, torch.Tensor):
                feature = (feature,)
            
            schema = self.dataset.input_processors[feature_key].schema()
            value = feature[schema.index("value")] if "value" in schema else None
            mask = feature[schema.index("mask")] if "mask" in schema else None
            
            if value is None:
                raise ValueError(f"Feature '{feature_key}' must contain 'value' in the schema.")
            
            inputs[feature_key] = value
            if mask is not None:
                masks[feature_key] = mask

        patient_emb = []
        embedded, mask = self.embedding_model(inputs, masks=masks, output_mask=True)

        # Process sequential features through RNN
        for feature_key in self.sequential_features:
            x = embedded[feature_key]
            m = mask[feature_key]
            _, last_hidden = self.rnn[feature_key](x, m)
            patient_emb.append(last_hidden)

        # Process non-sequential features (use embeddings directly)
        for feature_key in self.non_sequential_features:
            x = embedded[feature_key]
            # If multi-dimensional, aggregate (mean pooling)
            while x.dim() > 2:
                x = x.mean(dim=1)
            patient_emb.append(x)

        # Concatenate all representations
        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)

        # Calculate loss and predictions
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits
        }
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results
