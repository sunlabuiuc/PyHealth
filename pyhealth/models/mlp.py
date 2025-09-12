from typing import Dict

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class MLP(BaseModel):
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
        >>> from pyhealth.datasets import SampleDataset
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
        >>> dataset = SampleDataset(samples=samples,
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
        **kwargs,
    ):
        super(MLP, self).__init__(dataset)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # validate kwargs for MLP layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]

        # Create embedding and linear layers (will be populated dynamically)
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()

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

        # Create MLP layers for each feature
        self.mlp = nn.ModuleDict()
        for feature_key in self.feature_keys:
            Modules = []
            Modules.append(nn.Linear(self.embedding_dim, self.hidden_dim))
            for _ in range(self.n_layers - 1):
                Modules.append(self.activation)
                Modules.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.mlp[feature_key] = nn.Sequential(*Modules)

        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

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

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with the following keys:
                - loss: a scalar tensor representing the loss.
                - y_prob: a tensor representing the predicted probabilities.
                - y_true: a tensor representing the true labels.
                - logit: a tensor representing the logits.
                - embed (optional): a tensor representing the patient
                    embeddings if requested.
        """
        patient_emb = []

        for feature_key in self.feature_keys:
            x = kwargs[feature_key]

            # Convert to tensor if not already
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, device=self.device)
            else:
                x = x.to(self.device)

            # Handle different tensor dimensions
            if x.dim() == 1:
                # Case: Single sample with 1D data [val1, val2, ...]
                # Need to add batch dimension
                x = x.unsqueeze(0)  # (1, features)

            if x.dim() == 2:
                # Case 1: Sequence of tokens (batch, seq_len) - for codes
                # Case 3: Numerical features (batch, features) - for values

                # Check if sequence data (int) or numerical data (float)
                if x.dtype in [torch.int32, torch.int64]:
                    # This is sequence data (codes) - need embedding
                    if feature_key not in self.embeddings:
                        # Create embedding layer with max token + 1 vocab size
                        max_token_id = x.max().item()
                        vocab_size = max_token_id + 1
                        self.embeddings[feature_key] = nn.Embedding(
                            vocab_size, self.embedding_dim
                        ).to(self.device)

                    x = self.embeddings[feature_key](x)  # (batch, seq, embed)
                    # Apply mean pooling
                    mask = torch.any(x != 0, dim=2)
                    x = self.mean_pooling(x, mask)
                else:
                    # This is numerical data - apply linear layer
                    if feature_key not in self.linear_layers:
                        input_size = x.shape[-1]
                        self.linear_layers[feature_key] = nn.Linear(
                            input_size, self.embedding_dim
                        ).to(self.device)
                    # Ensure x is float for linear layer
                    x = x.float()
                    x = self.linear_layers[feature_key](x)

            elif x.dim() == 3:
                # Case 2: Nested sequences [[code1, code2], [code3, ...], ...]
                # Case 4: Time series [[val1, val2], [val3, val4], ...]

                if x.dtype in [torch.int32, torch.int64]:
                    # Sequence data with embeddings
                    if feature_key not in self.embeddings:
                        max_token_id = x.max().item()
                        vocab_size = max_token_id + 1
                        self.embeddings[feature_key] = nn.Embedding(
                            vocab_size, self.embedding_dim
                        ).to(self.device)

                    batch_size, seq_len, inner_len = x.shape
                    x = x.view(-1, inner_len)  # Flatten for embedding
                    x = self.embeddings[feature_key](x)  # Embed
                    x = x.view(batch_size, seq_len, -1)  # Reshape back
                    # Apply mean pooling over sequence dimension
                    mask = torch.any(x != 0, dim=2)
                    x = self.mean_pooling(x, mask)
                else:
                    # Numerical data - apply linear layer then pool
                    if feature_key not in self.linear_layers:
                        input_size = x.shape[-1]
                        self.linear_layers[feature_key] = nn.Linear(
                            input_size, self.embedding_dim
                        ).to(self.device)
                    # Ensure x is float for linear layer
                    x = x.float()
                    # Apply linear layer to get embeddings
                    x = self.linear_layers[feature_key](x)
                    # Apply mean pooling
                    mask = torch.ones(x.shape[:2], device=self.device)
                    x = self.mean_pooling(x, mask)

            else:
                raise ValueError(f"Unsupported tensor dimension: {x.dim()}")

            # Apply MLP
            x = self.mlp[feature_key](x)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        print("debug:", patient_emb.shape)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


if __name__ == "__main__":
    from pyhealth.datasets import SampleDataset

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
    dataset = SampleDataset(
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
