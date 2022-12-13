from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

# VALID_OPERATION_LEVEL = ["visit", "event"]


class RETAINLayer(nn.Module):
    """RETAIN layer.

    Paper: Edward Choi et al. RETAIN: An Interpretable Predictive Model for
    Healthcare using Reverse Time Attention Mechanism. NIPS 2016.

    This layer is used in the RETAIN model. But it can also be used as a
    standalone layer.

    Args:
        feature_size: the hidden feature size.
        dropout: dropout rate. Default is 0.5.

    Examples:
        >>> from pyhealth.models import RETAINLayer
        >>> input = torch.randn(3, 128, 64)  # [batch size, sequence len, feature_size]
        >>> layer = RETAINLayer(64)
        >>> c = layer(input)
        >>> c.shape
        torch.Size([3, 64])
    """

    def __init__(
        self,
        feature_size: int,
        dropout: float = 0.5,
    ):
        super(RETAINLayer, self).__init__()
        self.feature_size = feature_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.alpha_gru = nn.GRU(feature_size, feature_size, batch_first=True)
        self.beta_gru = nn.GRU(feature_size, feature_size, batch_first=True)

        self.alpha_li = nn.Linear(feature_size, 1)
        self.beta_li = nn.Linear(feature_size, feature_size)

    @staticmethod
    def reverse_x(input, lengths):
        """Reverses the input."""
        reversed_input = input.new(input.size())
        for i, length in enumerate(lengths):
            reversed_input[i, :length] = input[i, :length].flip(dims=[0])
        return reversed_input

    def compute_alpha(self, rx, lengths):
        """Computes alpha attention."""
        rx = rnn_utils.pack_padded_sequence(
            rx, lengths, batch_first=True, enforce_sorted=False
        )
        g, _ = self.alpha_gru(rx)
        g, _ = rnn_utils.pad_packed_sequence(g, batch_first=True)
        attn_alpha = torch.softmax(self.alpha_li(g), dim=1)
        return attn_alpha

    def compute_beta(self, rx, lengths):
        """Computes beta attention."""
        rx = rnn_utils.pack_padded_sequence(
            rx, lengths, batch_first=True, enforce_sorted=False
        )
        h, _ = self.beta_gru(rx)
        h, _ = rnn_utils.pad_packed_sequence(h, batch_first=True)
        attn_beta = torch.tanh(self.beta_li(h))
        return attn_beta

    def forward(
        self,
        x: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, feature_size].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            c: a tensor of shape [batch size, feature_size] representing the
                context vector.
        """
        # rnn will only apply dropout between layers
        x = self.dropout_layer(x)
        batch_size = x.size(0)
        if mask is None:
            lengths = torch.full(
                size=(batch_size,), fill_value=x.size(1), dtype=torch.int64
            )
        else:
            lengths = torch.sum(mask.int(), dim=-1).cpu()
        rx = self.reverse_x(x, lengths)
        attn_alpha = self.compute_alpha(rx, lengths)
        attn_beta = self.compute_beta(rx, lengths)
        c = attn_alpha * attn_beta * x  # (patient, sequence len, feature_size)
        c = torch.sum(c, dim=1)  # (patient, feature_size)
        return c


class RETAIN(BaseModel):
    """RETAIN model.

    Paper: Edward Choi et al. RETAIN: An Interpretable Predictive Model for
    Healthcare using Reverse Time Attention Mechanism. NIPS 2016.

    Note:
        We use separate Retain layers for different feature_keys.
        Currentluy, we automatically support different input formats:
            - code based input (need to use the embedding table later)
            - float/int based value input
        We follow the current convention for the Retain model:
            - case 1. [code1, code2, code3, ...]
                - we will assume the code follows the order; our model will encode
                each code into a vector and apply Retain on the code level
            - case 2. [[code1, code2]] or [[code1, code2], [code3, code4, code5], ...]
                - we will assume the inner bracket follows the order; our model first
                use the embedding table to encode each code into a vector and then use
                average/mean pooling to get one vector for one inner bracket; then use
                Retain one the braket level
            - case 3. [[1.5, 2.0, 0.0]] or [[1.5, 2.0, 0.0], [8, 1.2, 4.5], ...]
                - this case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we run Retain directly
                on the inner bracket level, similar to case 1 after embedding table
            - case 4. [[[1.5, 2.0, 0.0]]] or [[[1.5, 2.0, 0.0], [8, 1.2, 4.5]], ...]
                - this case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we run Retain directly
                on the inner bracket level, similar to case 2 after embedding table

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        embedding_dim: the embedding dimension. Default is 128.
        **kwargs: other parameters for the RETAIN layer.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        embedding_dim: int = 128,
        **kwargs,
    ):
        super(RETAIN, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim

        # validate kwargs for RETAIN layer
        if "feature_size" in kwargs:
            raise ValueError("feature_size is determined by embedding_dim")

        # the key of self.feat_tokenizers only contains the code based inputs
        self.feat_tokenizers = {}
        self.label_tokenizer = self.get_label_tokenizer()
        # the key of self.embeddings only contains the code based inputs
        self.embeddings = nn.ModuleDict()
        # the key of self.linear_layers only contains the float/int based inputs
        self.linear_layers = nn.ModuleDict()

        # add feature RETAIN layers
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            # sanity check
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "RETAIN only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [2, 3]):
                raise ValueError(
                    "RETAIN only supports 1-level or 2-level str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                input_info["dim"] not in [2, 3]
            ):
                raise ValueError(
                    "RETAIN only supports 2-level or 3-level float and int as input types"
                )
            # for code based input, we need Type
            # for float/int based input, we need Type, input_dim
            self.add_feature_transform_layer(feature_key, input_info)

        self.retain = nn.ModuleDict()
        for feature_key in feature_keys:
            self.retain[feature_key] = RETAINLayer(feature_size=embedding_dim, **kwargs)

        output_size = self.get_output_size(self.label_tokenizer)
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The label `kwargs[self.label_key]` is a list of labels for each patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor representing the predicted probabilities.
                y_true: a tensor representing the true labels.
        """
        patient_emb = []
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            level, Type = input_info["dim"], input_info["type"]

            # for case 1: [code1, code2, code3, ...]
            if (level == 2) and (Type == str):
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    kwargs[feature_key]
                )
                # (patient, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, event)
                mask = torch.sum(x, dim=2) != 0

            # for case 2: [[code1, code2], [code3, ...], ...]
            elif (level == 3) and (Type == str):
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    kwargs[feature_key]
                )
                # (patient, visit, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, visit, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)
                # (patient, visit)
                mask = torch.sum(x, dim=2) != 0

            # for case 3: [[1.5, 2.0, 0.0], ...]
            elif (level == 2) and (Type in [float, int]):
                x, mask = self.padding2d(kwargs[feature_key])
                # (patient, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, event, embedding_dim)
                x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = torch.tensor(mask, dtype=torch.bool, device=self.device)

            # for case 4: [[[1.5, 2.0, 0.0], [1.8, 2.4, 6.0]], ...]
            elif (level == 3) and (Type in [float, int]):
                x, mask = self.padding3d(kwargs[feature_key])
                # (patient, visit, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, visit, embedding_dim)
                x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = torch.tensor(mask, dtype=torch.bool, device=self.device)

            else:
                raise NotImplementedError

            x = self.retain[feature_key](x, mask)
            patient_emb.append(x)

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


if __name__ == "__main__":
    from pyhealth.datasets import SampleDataset

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "conditions": ["cond-33", "cond-86", "cond-80"],
            "procedures": [[1.0, 2.0, 3.5, 4]],
            "label": 0,
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "conditions": ["cond-33", "cond-86", "cond-80"],
            "procedures": [[5.0, 2.0, 3.5, 4]],
            "label": 1,
        },
    ]

    input_info = {
        "conditions": {"dim": 1, "type": str},
        "procedures": {"dim": 2, "type": float, "input_dim": 4},
    }

    # dataset
    dataset = SampleDataset(samples=samples, dataset_name="test")
    dataset.input_info = input_info

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    # model
    model = RETAIN(
        dataset=dataset,
        feature_keys=["conditions", "procedures"],
        label_key="label",
        mode="binary",
    )

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()
