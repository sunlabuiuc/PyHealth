from typing import Any, Dict, Optional, cast

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.interpret.api import Interpretable

from .embedding import EmbeddingModel
from .embedding.unified import UnifiedMultimodalEmbeddingModel


class MLP(BaseModel, Interpretable):
    """Multi-layer perceptron model.

    This model applies a separate MLP layer for each feature, and then
    concatenates the final hidden states of each MLP layer. The concatenated
    hidden states are then fed to a classifier layer.

    Note:
        We use separate MLP layers for different feature_keys.
        Currently, we automatically support different input formats:
            - code based input (need to use the embedding table later)
            - float/int based value input
        We follow the current convention for the rnn model:
            - case 1. [code1, code2, code3, ...]
                - we will assume the code follows the order; our model will
                  encode them to a hidden representation using the
                  embedding table
            - case 2. [[code1, code2]] or [[code1, code2],
                     [code3, code4, code5], ...]
                - we first use the embedding table to encode each code into a
                  vector and then use mean/sum pooling to get one vector for
                  each sample; we then apply the MLP on these pooled vectors
            - case 3. [1.5, 2.0, 0.0] or [1.5, 2.0, 0.0, ...]
                - This case applies MLP on the input vectors directly
            - case 4. [[1.5, 2.0, 0.0]] or [[1.5, 2.0, 0.0],
                      [8, 1.2, 4.5], ...]
                - This case only makes sense when each inner bracket has the
                  same length; we assume each dimension has the same meaning;
                  we use mean/sum pooling within each outer bracket and use MLP,
                  similar to case 1 after embedding table
            - case 5. [[[1.5, 2.0, 0.0]]] or [[[1.5, 2.0, 0.0], [8, 1.2, 4.5]], ...]
                - This case only makes sense when each inner bracket has the
                  same length; we assume each dimension has the same meaning;
                  we use mean/sum pooling within each outer bracket and use MLP,
                  similar to case 2 after embedding table

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        n_layers: the number of layers. Default is 2.
        activation: the activation function. Default is "relu".
        **kwargs: other parameters for the MLP layer.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-0",
        ...             "conditions": ["cond-33", "cond-86", "cond-80"],
        ...             "procedures": [1.0, 2.0, 3.5, 4],
        ...             "label": 0,
        ...         },
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-0",
        ...             "conditions": ["cond-33", "cond-86", "cond-80"],
        ...             "procedures": [5.0, 2.0, 3.5, 4],
        ...             "label": 1,
        ...         },
        ...     ]
        >>> input_schema = {"conditions": "sequence",
        ...                 "procedures": "timeseries"}
        >>> output_schema = {"label": "binary"}
        >>> dataset = create_sample_dataset(samples=samples,
        ...                        input_schema=input_schema,
        ...                        output_schema=output_schema,
        ...                        dataset_name="test")
        >>>
        >>> from pyhealth.models import MLP
        >>> model = MLP(dataset=dataset)
        >>>
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(0.6659, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>),
            'y_prob': tensor([[0.5680],
                            [0.5352]], grad_fn=<SigmoidBackward0>),
            'y_true': tensor([[1.],
                            [0.]]),
            'logit': tensor([[0.2736],
                            [0.1411]], grad_fn=<AddmmBackward0>)
        }
        >>>

    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 2,
        activation: str = "relu",
        unified_embedding: Optional[UnifiedMultimodalEmbeddingModel] = None,
        **kwargs,
    ):
        super(MLP, self).__init__(dataset)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self._use_unified = unified_embedding is not None

        # validate kwargs for MLP layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]

        # Set up activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function {activation}")

        output_size = self.get_output_size()

        if self._use_unified:
            self.embedding_model = unified_embedding
            modules = [nn.Linear(embedding_dim, hidden_dim)]
            for _ in range(n_layers - 1):
                modules.extend([self.activation, nn.Linear(hidden_dim, hidden_dim)])
            self.mlp = nn.ModuleDict({"unified": nn.Sequential(*modules)})
            self.fc = nn.Linear(hidden_dim, output_size)
        else:
            # Use the EmbeddingModel to handle embedding logic
            self.embedding_model = EmbeddingModel(dataset, embedding_dim)
            # Create MLP layers for each feature
            self.mlp = nn.ModuleDict()
            for feature_key in self.feature_keys:
                modules = [nn.Linear(self.embedding_dim, self.hidden_dim)]
                for _ in range(self.n_layers - 1):
                    modules.extend([self.activation, nn.Linear(self.hidden_dim, self.hidden_dim)])
                self.mlp[feature_key] = nn.Sequential(*modules)
            self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def _build_unified_inputs(
        self, kwargs: Dict[str, Any]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Build the inputs dict required by UnifiedMultimodalEmbeddingModel."""
        inputs: Dict[str, Dict[str, torch.Tensor]] = {}
        for field_name in self.feature_keys:
            feature = kwargs[field_name]
            if isinstance(feature, torch.Tensor):
                feature = (feature,)
            schema = self.dataset.input_processors[field_name].schema()
            field_dict: Dict[str, torch.Tensor] = {}
            if "value" in schema:
                field_dict["value"] = feature[schema.index("value")].to(self.device)
            if "time" in schema:
                field_dict["time"] = feature[schema.index("time")].to(self.device)
            if "mask" in schema:
                field_dict["mask"] = feature[schema.index("mask")].to(self.device)
            inputs[field_name] = field_dict
        return inputs

    def _forward_unified(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Forward pass in unified-embedding mode.

        Embeds all temporal fields jointly, mean-pools the event sequence,
        applies a single MLP, and projects to label space.
        """
        inputs = self._build_unified_inputs(kwargs)
        out = self.embedding_model(inputs)
        sequence = out["sequence"]          # (B, S, E)
        mask = out["mask"].float()          # (B, S)

        # Masked mean-pool over the event sequence
        x = (sequence * mask.unsqueeze(-1)).sum(dim=1)
        x = x / mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, E)

        x = self.mlp["unified"](x)          # (B, hidden_dim)
        logits = self.fc(x)
        y_prob = self.prepare_y_prob(logits)

        results: Dict[str, torch.Tensor] = {"logit": logits, "y_prob": y_prob}
        if self.label_key in kwargs:
            y_true = cast(torch.Tensor, kwargs[self.label_key]).to(self.device)
            results["loss"] = self.get_loss_function()(logits, y_true)
            results["y_true"] = y_true
        if kwargs.get("embed", False):
            results["embed"] = x
        return results

    @staticmethod
    def mean_pooling(x, mask):
        """Mean pooling over the middle dimension of the tensor.

        Args:
            x: tensor of shape (batch_size, seq_len, embedding_dim)
            mask: tensor of shape (batch_size, seq_len)

        Returns:
            x: tensor of shape (batch_size, embedding_dim)

        Examples:
            >>> x.shape
            [128, 5, 32]
            >>> mean_pooling(x).shape
            [128, 32]
        """
        return x.sum(dim=1) / mask.sum(dim=1, keepdim=True)

    @staticmethod
    def sum_pooling(x):
        """Sum pooling over the middle dimension of the tensor.

        Args:
            x: tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            x: tensor of shape (batch_size, embedding_dim)

        Examples:
            >>> x.shape
            [128, 5, 32]
            >>> sum_pooling(x).shape
            [128, 32]
        """
        return x.sum(dim=1)

    def forward_from_embedding(
        self,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass starting from feature embeddings.

        This method bypasses the embedding layers and processes
        pre-embedded features. This is useful for interpretability
        methods like Integrated Gradients that need to interpolate
        in embedding space.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

                It is expected to contain the following semantic tensors:
                    - "value": the embedded feature tensor of shape
                      [batch, seq_len, embedding_dim] or
                      [batch, embedding_dim].
                    - "mask" (optional): the mask tensor of shape
                      [batch, seq_len]. If not in the processor schema,
                      it can be provided as the last element of the
                      feature tuple. If not provided, masks will be
                      generated from the embedded values (non-zero
                      entries are treated as valid).

                The label key should contain the true labels for loss
                computation.

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the final loss.
                y_prob: a tensor of predicted probabilities.
                y_true: a tensor representing the true labels.
                logit: the raw logits before activation.
                embed: (if embed=True in kwargs) the patient embedding.
        """
        patient_emb = []

        for feature_key in self.feature_keys:
            processor = self.dataset.input_processors[feature_key]
            schema = processor.schema()
            feature = kwargs[feature_key]

            if isinstance(feature, torch.Tensor):
                # Backward compatibility: if feature is a tensor, treat it
                # as values without mask
                feature = (feature,)

            value = feature[schema.index("value")] if "value" in schema else None
            mask = feature[schema.index("mask")] if "mask" in schema else None

            if len(feature) == len(schema) + 1 and mask is None:
                # An optional mask can be provided as the last element
                # if not included in the schema
                mask = feature[-1]

            if value is None:
                raise ValueError(
                    f"Feature '{feature_key}' must contain 'value' "
                    f"in the schema."
                )
            else:
                value = value.to(self.device)

            # Handle different tensor dimensions for pooling
            if value.dim() == 3:
                # Case: (batch, seq_len, embedding_dim) - apply mean pooling
                if mask is None:
                    mask = (value.abs().sum(dim=-1) != 0).float()
                else:
                    mask = mask.to(self.device).float()
                    if mask.dim() == value.dim():
                        # Collapse feature dim from mask
                        mask = mask.any(dim=-1).float()
                x = self.mean_pooling(value, mask)
            elif value.dim() == 2:
                # Case: (batch, embedding_dim) - already pooled, use as is
                x = value
            else:
                raise ValueError(
                    f"Unsupported tensor dimension: {value.dim()}"
                )

            # Apply MLP
            x = self.mlp[feature_key](x)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)

        # (patient, label_size)
        logits = self.fc(patient_emb)
        y_prob = self.prepare_y_prob(logits)

        results = {
            "logit": logits,
            "y_prob": y_prob,
        }

        # obtain y_true, loss, y_prob
        if self.label_key in kwargs:
            y_true = cast(torch.Tensor, kwargs[self.label_key]).to(self.device)
            loss = self.get_loss_function()(logits, y_true)
            results["loss"] = loss
            results["y_true"] = y_true

        # Optionally return embeddings
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results

    def forward(
        self, **kwargs: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        In **unified mode** (when ``unified_embedding`` was supplied at init)
        the model jointly embeds all temporal fields, mean-pools the event
        sequence, and processes it with a single MLP.  Otherwise each field
        is embedded and encoded independently.

        Args:
            **kwargs: keyword arguments for the model.

                The keys must contain all the feature keys and the label key.

                Feature keys should contain tensors or tuples of tensors
                following the processor schema. The label key should
                contain the true labels for loss computation.

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the final loss.
                y_prob: a tensor of predicted probabilities.
                y_true: a tensor representing the true labels.
                logit: the raw logits before activation.
                embed: (if embed=True in kwargs) the patient embedding.
        """
        if self._use_unified:
            return self._forward_unified(**kwargs)

        for feature_key in self.feature_keys:
            feature = kwargs[feature_key]

            if isinstance(feature, torch.Tensor):
                # Backward compatibility: if feature is a tensor, treat it
                # as values without mask
                feature = (feature,)

            schema = self.dataset.input_processors[feature_key].schema()

            value = (
                feature[schema.index("value")] if "value" in schema else None
            )
            mask = feature[schema.index("mask")] if "mask" in schema else None

            if value is None:
                raise ValueError(
                    f"Feature '{feature_key}' must contain 'value' "
                    f"in the schema."
                )
            else:
                value = value.to(self.device)

            # Handle 3D input: (batch, event, # of codes) -> flatten to 2D
            # before embedding to treat all codes as a single flat sequence
            if value.dim() == 3:
                batch_size, seq_len, inner_len = value.shape
                value = value.view(batch_size, seq_len * inner_len)
                if mask is not None:
                    mask = mask.to(self.device)
                    if mask.dim() == 3:
                        mask = mask.view(batch_size, seq_len * inner_len)

            if mask is not None:
                mask = mask.to(self.device)
                value = self.embedding_model({feature_key: value}, masks={feature_key: mask})[feature_key]
            else:
                value = self.embedding_model({feature_key: value})[feature_key]

            i = schema.index("value")
            # We replace 'value' in the tuple with the embedded value
            # But we must ensure other elements are preserved.
            
            # If mask was present, we keep it? 
            # forward_from_embedding expects (value, mask) or (value,)
            # The embedded value is now the "value".
            # If we pass mask to forward_from_embedding, it will be used for pooling.
            
            args = list(feature)
            args[i] = value
            kwargs[feature_key] = tuple(args)

        return self.forward_from_embedding(**kwargs)

    def get_embedding_model(self) -> nn.Module | None:
        """Get the embedding model.

        Returns:
            nn.Module: The embedding model used to embed raw features.
        """
        return self.embedding_model


if __name__ == "__main__":
    from pyhealth.datasets import create_sample_dataset

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "conditions": ["cond-33", "cond-86", "cond-80"],
            "procedures": [1.0, 2.0, 3.5, 4],
            "label": 0,
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "conditions": ["cond-33", "cond-86", "cond-80"],
            "procedures": [5.0, 2.0, 3.5, 4],
            "label": 1,
        },
    ]

    # Define input and output schemas
    input_schema = {
        "conditions": "sequence",  # sequence of condition codes
        "procedures": "timeseries",  # timeseries of procedure values
    }
    output_schema = {"label": "binary"}  # binary classification

    # dataset
    dataset = create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="test",
    )

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    # model
    model = MLP(dataset=dataset)

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()
