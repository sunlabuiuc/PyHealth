import functools
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from pyhealth.datasets import BaseEHRDataset
from pyhealth.models import BaseModel


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
    Examples:
        >>> from pyhealth.models import DeeprLayer
        >>> input = torch.randn(3, 128, 5)  # [batch size, sequence len, input_size]
        >>> layer = DeeprLayer(5, window=4, hidden_size=7) # window does not impact the output shape
        >>> outputs = layer(input)
        >>> outputs.shape
        torch.Size([3, 7])
    """

    def __init__(
        self,
        feature_size: int = 100,
        window: int = 1,
        hidden_size: int = 3,
    ):
        super(DeeprLayer, self).__init__()

        self.conv = torch.nn.Conv1d(
            feature_size, hidden_size, kernel_size=2 * window + 1
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward propagation.

        Args:
            x: a Tensor of shape [batch size, sequence len, input size].
            mask: an optional tensor of shape [batch size, sequence len], where
                            1 indicates valid and 0 indicates invalid.
        Returns:
            c: a Tensor of shape [batch size, hidden_size] representing the
                summarized vector.
        """
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        x = x.permute(0, 2, 1)  # [batch size, input size, sequence len]
        x = torch.relu(self.conv(x))
        x = x.max(-1)[0]
        return x


def _flatten_and_fill_gap(gap_embedding, batch, device):
    """Helper function to fill <gap> embedding into a batch of data."""
    embed_dim = gap_embedding.shape[-1]
    batch = [
        [
            [torch.tensor(_, device=device, dtype=torch.float) for _ in _visit_x]
            for _visit_x in _pat_x
        ]
        for _pat_x in batch
    ]
    batch = [
        torch.stack(functools.reduce(lambda a, b: a + [gap_embedding] + b, _), 0)
        for _ in batch
    ]
    batch_max_length = max(map(len, batch))
    mask = torch.tensor(
        [[1] * len(x) + [0] * (batch_max_length - len(x)) for x in batch],
        dtype=torch.long,
        device=device,
    )
    out = torch.zeros(
        [len(batch), batch_max_length, embed_dim], device=device, dtype=torch.float
    )
    for i, x in enumerate(batch):
        out[i, : len(x)] = x
    return out, mask


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
        **kwargs: other parameters for the Deepr layer.

    Examples:
        >>> from pyhealth.datasets import SampleEHRDataset
        >>> samples = [
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-0",
        ...             "list_codes": ["505800458", "50580045810", "50580045811"],  # NDC
        ...             "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
        ...             "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],  # ATC-4
        ...             "list_list_vectors": [
        ...                 [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
        ...                 [[7.7, 8.5, 9.4]],
        ...             ],
        ...             "label": 1,
        ...         },
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-1",
        ...             "list_codes": [
        ...                 "55154191800",
        ...                 "551541928",
        ...                 "55154192800",
        ...                 "705182798",
        ...                 "70518279800",
        ...             ],
        ...             "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7], [4.5, 5.9, 1.7]],
        ...             "list_list_codes": [["A04A", "B035", "C129"]],
        ...             "list_list_vectors": [
        ...                 [[1.0, 2.8, 3.3], [4.9, 5.0, 6.6], [7.7, 8.4, 1.3], [7.7, 8.4, 1.3]],
        ...             ],
        ...             "label": 0,
        ...         },
        ...     ]
        >>> dataset = SampleEHRDataset(samples=samples, dataset_name="test")
        >>>
        >>> from pyhealth.models import Deepr
        >>> model = Deepr(
        ...         dataset=dataset,
        ...         feature_keys=[
        ...             "list_list_codes",
        ...             "list_list_vectors",
        ...         ],
        ...         label_key="label",
        ...         mode="binary",
        ...     )
        >>>
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(0.8908, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward0>),
            'y_prob': tensor([[0.2295],
                        [0.2665]], device='cuda:0', grad_fn=<SigmoidBackward0>),
            'y_true': tensor([[1.],
                        [0.]], device='cuda:0'),
            'logit': tensor([[-1.2110],
                        [-1.0126]], device='cuda:0', grad_fn=<AddmmBackward0>)
        }
    """

    def __init__(
        self,
        dataset: BaseEHRDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
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
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # TODO: Use more tokens for <gap> for different lengths once the input has such information
        self.feat_tokenizers = {}
        self.label_tokenizer = self.get_label_tokenizer()
        # TODO: Pretrain this embeddings with word2vec?
        self.embeddings = nn.ModuleDict()
        # the key of self.linear_layers only contains the float/int based inputs
        self.linear_layers = nn.ModuleDict()

        # add feature Deepr layers
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            # sanity check
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "Deepr only supports str code, float and int as input types"
                )
            if (input_info["type"] == str) and (input_info["dim"] != 3):
                raise ValueError("Deepr only supports 2-level str code as input types")
            if (input_info["type"] in [float, int]) and (input_info["dim"] != 3):
                raise ValueError(
                    "Deepr only supports 3-level float and int as input types"
                )
            # for code based input, we need Type
            # for float/int based input, we need Type, input_dim
            self.add_feature_transform_layer(
                feature_key, input_info, special_tokens=["<pad>", "<unk>", "<gap>"]
            )
            if input_info["type"] != str:
                self.embeddings[feature_key] = torch.nn.Embedding(1, input_info["len"])

        self.cnn = nn.ModuleDict()
        for feature_key in feature_keys:
            self.cnn[feature_key] = DeeprLayer(
                feature_size=embedding_dim, hidden_size=hidden_dim, **kwargs
            )
        output_size = self.get_output_size(self.label_tokenizer)
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:

        """Forward propagation."""
        patient_emb = []
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

            # for case 2: [[code1, code2], [code3, ...], ...]
            if (dim_ == 3) and (type_ == str):
                feature_vals = [
                    functools.reduce(lambda a, b: a + ["<gap>"] + b, _)
                    for _ in kwargs[feature_key]
                ]
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    feature_vals, padding=True, truncation=False
                )
                pad_idx = self.feat_tokenizers[feature_key].vocabulary("<pad>")
                mask = torch.tensor(
                    [[_code != pad_idx for _code in _pat] for _pat in x],
                    dtype=torch.long,
                    device=self.device,
                )
                # (patient, code)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, event, embedding_dim)
                x = self.embeddings[feature_key](x)
            # for case 4: [[[1.5, 2.0, 0.0], [1.8, 2.4, 6.0]], ...]
            elif (dim_ == 3) and (type_ in [float, int]):
                gap_embedding = self.embeddings[feature_key](
                    torch.zeros(1, dtype=torch.long, device=self.device)
                )[0]
                x, mask = _flatten_and_fill_gap(
                    gap_embedding, kwargs[feature_key], self.device
                )
                # (patient, event, embedding_dim)
                x = self.linear_layers[feature_key](x)
            else:
                raise NotImplementedError(
                    f"Deepr does not support this input format (dim={dim_}, type={type_})."
                )
            # (patient, hidden_dim)
            x = self.cnn[feature_key](x, mask)
            patient_emb.append(x)

        # (patient, features * hidden_dim)
        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
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
    from pyhealth.datasets import SampleEHRDataset

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "single_vector": [1, 2, 3],
            "list_codes": ["505800458", "50580045810", "50580045811"],  # NDC
            "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
            "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],  # ATC-4
            "list_list_vectors": [
                [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
                [[7.7, 8.5, 9.4]],
            ],
            "label": 1,
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-1",
            "single_vector": [1, 5, 8],
            "list_codes": [
                "55154191800",
                "551541928",
                "55154192800",
                "705182798",
                "70518279800",
            ],
            "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7], [4.5, 5.9, 1.7]],
            "list_list_codes": [["A04A", "B035", "C129"]],
            "list_list_vectors": [
                [[1.0, 2.8, 3.3], [4.9, 5.0, 6.6], [7.7, 8.4, 1.3], [7.7, 8.4, 1.3]],
            ],
            "label": 0,
        },
    ]

    # dataset
    dataset = SampleEHRDataset(samples=samples, dataset_name="test")

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    # model
    model = Deepr(
        dataset=dataset,
        # feature_keys=["procedures"],
        feature_keys=["list_list_codes", "list_list_vectors"],
        label_key="label",
        mode="binary",
    ).to("cuda:0")

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()
