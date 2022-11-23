from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
from pyhealth.datasets import BaseDataset
from pyhealth.models import BaseModel


class MLP(BaseModel):
    """Multi-layer perceptron model.

    This model applies a separate MLP layer for each feature, and then concatenates
    the final hidden states of each MLP layer. The concatenated hidden states are
    then fed into a fully connected layer to make predictions.

    Note:
        We use separate MLP layers for different feature_keys.
        Currentluy, we automatically support different input formats:
            - code based input (need to use the embedding table later)
            - float/int based value input
        We follow the current convention for the rnn model:
            - case 1. [code1, code2, code3, ...]
                - we will assume the code follows the order; our model will encode
                each code into a vector; we use mean/sum pooling and then MLP
            - case 2. [[code1, code2]] or [[code1, code2], [code3, code4, code5], ...]
                - we first use the embedding table to encode each code into a vector
                and then use mean/sum pooling to get one vector for each sample; we then
                use MLP layers
            - case 3. [1.5, 2.0, 0.0]
                - we run MLP directly
            - case 4. [[1.5, 2.0, 0.0]] or [[1.5, 2.0, 0.0], [8, 1.2, 4.5], ...]
                - This case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we use mean/sum pooling
                within each outer bracket and use MLP, similar to case 1 after embedding table
            - case 5. [[[1.5, 2.0, 0.0]]] or [[[1.5, 2.0, 0.0], [8, 1.2, 4.5]], ...]
                - This case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we use mean/sum pooling
                within each outer bracket and use MLP, similar to case 2 after embedding table

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        n_layers: the number of layers. Default is 2.
        activation: the activation function. Default is "relu".
        **kwargs: other parameters for the RNN layer.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 2,
        activation: str = "relu",
        **kwargs,
    ):
        super(MLP, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # validate kwargs for RNN layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")

        # the key of self.feat_tokenizers only contains the code based inputs
        self.feat_tokenizers = {}
        self.label_tokenizer = self.get_label_tokenizer()
        # the key of self.embeddings only contains the code based inputs
        self.embeddings = nn.ModuleDict()
        # the key of self.linear_layers only contains the float/int based inputs
        self.linear_layers = nn.ModuleDict()

        # add feature MLP layers
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            # sanity check
            if input_info["Type"] not in [str, float, int]:
                raise ValueError(
                    "MLP only supports str code, float and int as input types"
                )
            elif (input_info["Type"] == str) and (input_info["level"] not in [1, 2]):
                raise ValueError(
                    "MLP only supports 1-level or 2-level str code as input types"
                )
            elif (input_info["Type"] in [float, int]) and (
                input_info["level"] not in [1, 2, 3]
            ):
                raise ValueError(
                    "MLP only supports 1-level, 2-level or 3-level float and int as input types"
                )
            # for code based input, we need Type
            # for float/int based input, we need Type, input_dim
            self.add_feature_transform_layer(feature_key=feature_key, **input_info)

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

        self.mlp = nn.ModuleDict()
        for feature_key in feature_keys:
            Modules = []
            Modules.append(nn.Linear(self.embedding_dim, self.hidden_dim))
            for _ in range(self.n_layers - 1):
                Modules.append(self.activation)
                Modules.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.mlp[feature_key] = nn.Sequential(*Modules)

        output_size = self.get_output_size(self.label_tokenizer)
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
        """Mean pooling over the middle dimension of the tensor.
        Args:
            x: tensor of shape (batch_size, seq_len, embedding_dim)
            mask: tensor of shape (batch_size, seq_len)
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
            level, Type = input_info["level"], input_info["Type"]

            # for case 1: [code1, code2, code3, ...]
            if (level == 1) and (Type == str):
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    kwargs[feature_key]
                )
                # (patient, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, event)
                mask = torch.sum(x, dim=2) != 0
                # (patient, embedding_dim)
                x = self.mean_pooling(x, mask)

            # for case 2: [[code1, code2], [code3, ...], ...]
            elif (level == 2) and (Type == str):
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
                # (patient, embedding_dim)
                x = self.mean_pooling(x, mask)

            # for case 3: [1.5, 2.0, 0.0]
            elif (level == 1) and (Type in [float, int]):
                # (patient, values)
                x = torch.tensor(
                    kwargs[feature_key], dtype=torch.float, device=self.device
                )
                # (patient, embedding_dim)
                x = self.linear_layers[feature_key](x)

            # for case 4: [[1.5, 2.0, 0.0], ...]
            elif (level == 2) and (Type in [float, int]):
                x, mask = self.padding2d(kwargs[feature_key])
                # (patient, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, event, embedding_dim)
                x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
                # (patient, embedding_dim)
                x = self.mean_pooling(x, mask)

            # for case 5: [[[1.5, 2.0, 0.0], [1.8, 2.4, 6.0]], ...]
            elif (level == 3) and (Type in [float, int]):
                x, mask = self.padding3d(kwargs[feature_key])
                # (patient, visit, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, visit, embedding_dim)
                x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
                # (patient, embedding_dim)
                x = self.mean_pooling(x, mask)

            else:
                raise NotImplementedError

            x = self.mlp[feature_key](x)
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

    input_info = {
        "conditions": {"level": 1, "Type": str},
        "procedures": {"level": 1, "Type": float, "input_dim": 4},
    }

    # dataset
    dataset = SampleDataset(samples=samples, dataset_name="test")
    dataset.input_info = input_info

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    # model
    model = MLP(
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

    # TODO: the loss back propagation step seems slow.
    # try loss backward
    ret["loss"].backward()
