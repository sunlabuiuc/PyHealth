# Author: Michael Dankanich
# NetID: Mdanka2
# Paper title: Barttender
# Paper link:  https://arxiv.org/abs/2411.12707
# Description: Lasso regression model with L1 regularization for feature selection.

from typing import Dict, List

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

from .embedding import EmbeddingModel


class Lasso(BaseModel):
    """Lasso regression model with L1 regularization for feature selection.

    This model uses embeddings from different input features and applies a single
    linear transformation with L1 regularization (Lasso penalty) to produce
    predictions. The L1 penalty encourages sparsity in the learned weights,
    effectively performing feature selection.

    - For classification tasks: acts as logistic regression with L1 penalty
    - For regression tasks: acts as linear regression with L1 penalty (Lasso)

    The model automatically handles different input types through the EmbeddingModel,
    pools sequence dimensions, concatenates all feature embeddings, and applies a
    final linear layer with L1 regularization added to the loss.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim: the embedding dimension. Default is 128.
        alpha: L1 regularization strength. Higher values result in more
            sparse weights (more aggressive feature selection). Default is 0.01.
        **kwargs: other parameters (for compatibility).

    Note:
        The alpha parameter controls regularization strength. Common values:

        - 0.001: weak regularization (keeps more features)
        - 0.01: moderate (default, good starting point)
        - 0.1: strong (aggressive feature selection)

        For optimal alpha selection, use cross-validation with different
        alpha values and select based on validation performance.
        sklearn's LassoCV typically finds optimal alpha around 0.01-0.1
        for clinical data.

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
        ...             "patient_id": "patient-1",
        ...             "visit_id": "visit-1",
        ...             "conditions": ["cond-33", "cond-86", "cond-80"],
        ...             "procedures": [5.0, 2.0, 3.5, 4],
        ...             "label": 1,
        ...         },
        ...     ]
        >>> input_schema = {"conditions": "sequence",
        ...                 "procedures": "tensor"}
        >>> output_schema = {"label": "binary"}
        >>> dataset = SampleDataset(samples=samples,
        ...                        input_schema=input_schema,
        ...                        output_schema=output_schema,
        ...                        dataset_name="test")
        >>>
        >>> from pyhealth.models import Lasso
        >>> model = Lasso(dataset=dataset, alpha=0.01)
        >>>
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(0.6931, grad_fn=<AddBackward0>),
            'y_prob': tensor([[0.5123],
                            [0.4987]], grad_fn=<SigmoidBackward0>),
            'y_true': tensor([[1.],
                            [0.]]),
            'logit': tensor([[0.0492],
                            [-0.0052]], grad_fn=<AddmmBackward0>)
        }
        >>>
        >>> # Get feature importance (absolute weight magnitudes)
        >>> importance = model.get_feature_importance()
        >>> print(importance.shape)
        torch.Size([256])
        >>>
        >>> # Get indices of selected features (nonzero weights)
        >>> selected = model.get_selected_features(threshold=0.01)
        >>> print(len(selected))
        128

    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        alpha: float = 0.01,
        **kwargs,
    ):
        super(Lasso, self).__init__(dataset)
        self.embedding_dim = embedding_dim
        self.alpha = alpha

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]

        # Use the EmbeddingModel to handle embedding logic
        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # Single linear layer (no hidden layers, no activation)
        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)

    @staticmethod
    def mean_pooling(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling over the middle dimension of the tensor.

        Args:
            x: tensor of shape (batch_size, seq_len, embedding_dim)
            mask: tensor of shape (batch_size, seq_len)

        Returns:
            x: tensor of shape (batch_size, embedding_dim)

        Examples:
            >>> x.shape
            [128, 5, 32]
            >>> mean_pooling(x, mask).shape
            [128, 32]
        """
        return x.sum(dim=1) / mask.sum(dim=1, keepdim=True)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with the following keys:
                - loss: a scalar tensor representing the loss (includes L1 penalty).
                - y_prob: a tensor representing the predicted probabilities.
                - y_true: a tensor representing the true labels.
                - logit: a tensor representing the logits.
                - embed (optional): a tensor representing the patient
                    embeddings if requested.
        """
        patient_emb = []

        # Preprocess inputs for EmbeddingModel
        processed_inputs = {}
        reshape_info = {}  # Track which inputs were reshaped

        for feature_key in self.feature_keys:
            x = kwargs[feature_key]

            # Convert to tensor if not already
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, device=self.device)
            else:
                x = x.to(self.device)

            # Handle 3D input: (patient, event, # of codes) -> flatten to 2D
            if x.dim() == 3:
                batch_size, seq_len, inner_len = x.shape
                x = x.view(batch_size, seq_len * inner_len)
                reshape_info[feature_key] = {
                    "original_shape": (batch_size, seq_len, inner_len),
                    "was_3d": True,
                    "expanded": False,
                }
            elif x.dim() == 1:
                x = x.unsqueeze(0)
                reshape_info[feature_key] = {"was_3d": False, "expanded": True}
            else:
                reshape_info[feature_key] = {"was_3d": False, "expanded": False}

            processed_inputs[feature_key] = x

        # Pass through EmbeddingModel
        embedded = self.embedding_model(processed_inputs)

        for feature_key in self.feature_keys:
            x = embedded[feature_key]

            info = reshape_info[feature_key]
            if info.get("expanded") and x.dim() > 1:
                x = x.squeeze(0)

            # Handle different tensor dimensions for pooling
            if x.dim() == 3:
                # Case: (batch, seq_len, embedding_dim) - apply mean pooling
                mask = (x.sum(dim=-1) != 0).float()
                if mask.sum(dim=-1, keepdim=True).any():
                    x = self.mean_pooling(x, mask)
                else:
                    x = x.mean(dim=1)
            elif x.dim() == 2:
                # Case: (batch, embedding_dim) - already pooled, use as is
                pass
            else:
                raise ValueError(f"Unsupported tensor dimension: {x.dim()}")

            patient_emb.append(x)

        # Concatenate all feature embeddings
        patient_emb = torch.cat(patient_emb, dim=1)

        # Apply single linear layer (no activation)
        logits = self.fc(patient_emb)

        # Obtain y_true, base loss, y_prob
        y_true = kwargs[self.label_key].to(self.device)
        base_loss = self.get_loss_function()(logits, y_true)

        # Add L1 regularization penalty
        l1_penalty = self.alpha * torch.norm(self.fc.weight, p=1)
        loss = base_loss + l1_penalty

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

    def get_feature_importance(self) -> torch.Tensor:
        """Get feature importance based on absolute weight magnitudes.

        Returns the absolute values of the weights in the final linear layer,
        which indicates the importance of each feature dimension. Higher values
        indicate more important features.

        Returns:
            torch.Tensor: A 1D tensor of shape (input_dim,) containing the
                absolute weight magnitudes for each input feature dimension.

        Examples:
            >>> model = Lasso(dataset=dataset)
            >>> importance = model.get_feature_importance()
            >>> print(importance.shape)
            torch.Size([256])
        """
        # Get absolute values of weights, averaged across output dimensions
        weights = self.fc.weight.detach()
        importance = torch.abs(weights).mean(dim=0)
        return importance

    def get_selected_features(self, threshold: float = 0.0) -> List[int]:
        """Get indices of features with importance above threshold.

        This method identifies which feature dimensions have weights with
        absolute magnitude above the specified threshold, effectively
        performing feature selection based on the learned Lasso weights.

        Args:
            threshold: Minimum absolute weight value for a feature to be
                considered selected. Default is 0.0 (all nonzero weights).

        Returns:
            List[int]: List of indices of selected features (those with
                importance above the threshold).

        Examples:
            >>> model = Lasso(dataset=dataset, alpha=0.1)
            >>> # After training...
            >>> selected = model.get_selected_features(threshold=0.01)
            >>> print(f"Selected {len(selected)} features")
            Selected 64 features
        """
        importance = self.get_feature_importance()
        selected_indices = torch.where(importance > threshold)[0].tolist()
        return selected_indices


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
            "patient_id": "patient-1",
            "visit_id": "visit-1",
            "conditions": ["cond-33", "cond-86", "cond-80"],
            "procedures": [5.0, 2.0, 3.5, 4],
            "label": 1,
        },
    ]

    # Define input and output schemas
    input_schema = {
        "conditions": "sequence",  # sequence of condition codes
        "procedures": "tensor",  # tensor of procedure values
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
    model = Lasso(dataset=dataset, alpha=0.01)

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print("Forward pass results:")
    print(ret)

    # try loss backward
    ret["loss"].backward()
    print("\nBackward pass completed successfully!")

    # test feature importance
    importance = model.get_feature_importance()
    print(f"\nFeature importance shape: {importance.shape}")

    # test feature selection
    selected = model.get_selected_features(threshold=0.0)
    print(f"Number of selected features (threshold=0.0): {len(selected)}")

