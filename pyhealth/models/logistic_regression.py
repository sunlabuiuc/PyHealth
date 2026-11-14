from typing import Dict

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

from .embedding import EmbeddingModel
from .utils import get_step_mask, masked_mean_pooling


class LogisticRegression(BaseModel):
    """Logistic/Linear regression baseline model.

    This model uses embeddings from different input features and applies a single
    linear transformation (no hidden layers or non-linearity) to produce predictions.
    
    - For classification tasks: acts as logistic regression
    - For regression tasks: acts as linear regression
    
    The model automatically handles different input types through the EmbeddingModel,
    pools sequence dimensions, concatenates all feature embeddings, and applies a
    final linear layer.

    A numeric series (e.g. ``timeseries``) is embedded step by step and
    mean-pooled over the steps that hold data. A batch pads shorter series
    with all-zero steps, so a real step whose features are all zero (e.g.
    the zeros forward fill puts before the first measurement) also counts
    as padding. A series with no nonzero step is averaged over all its
    steps. An input whose processor gives a mask (e.g. a tokenized
    ``tuple_time_text``'s attention mask) is pooled over the steps that
    mask keeps.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim: the embedding dimension. Default is 128.
        **kwargs: other parameters (for compatibility).

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
        >>> dataset = create_sample_dataset(samples=samples,
        ...                        input_schema=input_schema,
        ...                        output_schema=output_schema,
        ...                        dataset_name="test")
        >>>
        >>> from pyhealth.models import LogisticRegression
        >>> model = LogisticRegression(dataset=dataset)
        >>>
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(0.6931, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>),
            'y_prob': tensor([[0.5123],
                            [0.4987]], grad_fn=<SigmoidBackward0>),
            'y_true': tensor([[1.],
                            [0.]]),
            'logit': tensor([[0.0492],
                            [-0.0052]], grad_fn=<AddmmBackward0>)
        }
        >>>

    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        **kwargs,
    ):
        super(LogisticRegression, self).__init__(dataset)
        self.embedding_dim = embedding_dim

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]

        # Use the EmbeddingModel to handle embedding logic
        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # Single linear layer (no hidden layers, no activation)
        output_size = self.label_processor.output_size()
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)

    @staticmethod
    def mean_pooling(x, mask):
        """Mean pooling over the middle dimension of the tensor.

        Thin wrapper around :func:`pyhealth.models.utils.masked_mean_pooling`:
        steps where ``mask`` is 0 are left out of the mean, and a row with no
        valid step is averaged over all its steps.

        Args:
            x: tensor of shape (batch_size, seq_len, embedding_dim)
            mask: tensor of shape (batch_size, seq_len)

        Returns:
            x: tensor of shape (batch_size, embedding_dim)

        Examples:
            >>> x = torch.ones(128, 5, 32)
            >>> mask = torch.ones(128, 5)
            >>> LogisticRegression.mean_pooling(x, mask).shape
            torch.Size([128, 32])
        """
        return masked_mean_pooling(x, mask)

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

        # Preprocess inputs for EmbeddingModel
        processed_inputs = {}
        reshape_info = {}  # Track which inputs were reshaped
        # Pooling masks: a processor's own mask (e.g. a tokenizer's
        # attention mask), or for a numeric series one from the raw steps
        pool_masks = {}

        for feature_key in self.feature_keys:
            x = kwargs[feature_key]
            # A processor with several outputs, such as stagenet_tensor's
            # (time, value) or a tokenized tuple_time_text's (value, mask,
            # ...), gives a tuple; the model uses the value and the mask.
            schema = self.dataset.input_processors[feature_key].schema()
            mask = None
            if isinstance(x, tuple) and "value" in schema:
                if "mask" in schema:
                    mask = x[schema.index("mask")]
                x = x[schema.index("value")]

            # Convert to tensor if not already
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, device=self.device)
            else:
                x = x.to(self.device)
            if mask is not None:
                mask = torch.as_tensor(mask, device=self.device)

            # Handle 3D input: (patient, event, # of codes) -> flatten to 2D.
            # A numeric series (patient, time, features) embedded by an
            # nn.Linear over its features keeps its shape instead: the
            # embedding projects each step, then pooling. Its padded steps
            # embed to the Linear bias, so the pooling mask comes from the
            # raw steps.
            layers = self.embedding_model.embedding_layers
            per_step = (x.dim() == 3 and feature_key in layers
                        and isinstance(layers[feature_key], nn.Linear))
            if per_step and mask is None:
                mask = get_step_mask(x)
            if x.dim() == 3 and not per_step:
                batch_size, seq_len, inner_len = x.shape
                x = x.view(batch_size, seq_len * inner_len)
                if mask is not None and mask.dim() == 3:
                    # Flattened with the codes, so it lines up with them
                    mask = mask.reshape(batch_size, seq_len * inner_len)
                reshape_info[feature_key] = {
                    "original_shape": (batch_size, seq_len, inner_len),
                    "was_3d": True,
                    "expanded": False,
                }
            elif x.dim() == 1:
                x = x.unsqueeze(0)
                if mask is not None and mask.dim() == 1:
                    mask = mask.unsqueeze(0)
                reshape_info[feature_key] = {"was_3d": False, "expanded": True}
            else:
                reshape_info[feature_key] = {"was_3d": False, "expanded": False}

            processed_inputs[feature_key] = x
            if mask is not None:
                pool_masks[feature_key] = mask

        # Pass through EmbeddingModel; a transformer layer uses the masks as
        # its attention mask, other layers ignore them
        embedded = self.embedding_model(processed_inputs, masks=pool_masks)

        for feature_key in self.feature_keys:
            x = embedded[feature_key]

            info = reshape_info[feature_key]
            if info.get("expanded") and x.dim() > 1:
                x = x.squeeze(0)

            # Handle different tensor dimensions for pooling
            if x.dim() == 3:
                # Case: (batch, seq_len, embedding_dim) - apply mean pooling
                mask = pool_masks.get(feature_key)
                if mask is None:
                    mask = (x.sum(dim=-1) != 0).float()
                elif mask.dim() == x.dim():
                    # A per-feature mask: a step is valid if any feature is
                    mask = mask.any(dim=-1)
                x = self.mean_pooling(x, mask)
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
        
        # Obtain y_true, loss, y_prob
        y_prob = self.label_processor.activation(logits)
        results = {"y_prob": y_prob, "logit": logits}
        # The label is optional, so the model can predict without it.
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            results["loss"] = self.label_processor.loss(logits, y_true)
            results["y_true"] = y_true
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


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
    model = LogisticRegression(dataset=dataset)

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()
