from typing import Tuple, List, Dict, Optional
import functools

import torch
import torch.nn as nn
import ipdb

from pyhealth.datasets import BaseDataset
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

    def forward(self, x: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input size].

        Returns:
            c: a tensor of shape [batch size, hidden_size] representing the
                summarized vector.
        """
        x = x.permute(0, 2, 1)  # [batch size, input size, sequence len]
        x = torch.relu(self.conv(x))
        x = x.max(-1)[0]
        return x


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
    """

    def __init__(
        self,
        dataset: BaseDataset,
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
        # TODO: find a way to add <gap> to self.add_feature_transform_layer
        self.feat_tokenizers = self.get_feature_tokenizers(
            special_tokens=["<pad>", "<unk>", "<gap>"]
        )
        self.label_tokenizer = self.get_label_tokenizer()

        # TODO: Pretrain this embeddings with word2vec?
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)

        # validate kwargs for CNN layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")
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

            # for case 1: [code1, code2, code3, ...]
            if (dim_ == 2) and (type_ == str):
                raise NotImplementedError(
                    f"Deepr does not support this input format (dim={dim_}, type={type_})."
                )
            # for case 2: [[code1, code2], [code3, ...], ...]
            elif (dim_ == 3) and (type_ == str):
                feature_vals = [
                    functools.reduce(lambda a, b: a + ["<gap>"] + b, _)
                    for _ in kwargs[feature_key]
                ]
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    feature_vals, padding=True, truncation=False
                )

                # (patient, code)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, event, embedding_dim)
                x = self.embeddings[feature_key](x)

            # for case 3: [[1.5, 2.0, 0.0], ...]
            elif (dim_ == 2) and (type_ in [float, int]):
                raise NotImplementedError(
                    f"Deepr does not support this input format (dim={dim_}, type={type_})."
                )
            # for case 4: [[[1.5, 2.0, 0.0], [1.8, 2.4, 6.0]], ...]
            elif (dim_ == 3) and (type_ in [float, int]):
                raise NotImplementedError(
                    f"Deepr does not support this input format (dim={dim_}, type={type_})."
                )
                x, mask = self.padding3d(kwargs[feature_key])
                ipdb.set_trace()
                # (patient, visit, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, visit, embedding_dim)
                x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = torch.tensor(mask, dtype=torch.bool, device=self.device)

            else:
                raise NotImplementedError
            # (patient, hidden_dim)
            x = self.cnn[feature_key](x)  # , mask)
            patient_emb.append(x)

        # (patient, features * hidden_dim)
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
            "conditions": [["cond-33", "cond-86", "cond-80"]],
            "procedures": [[[5.0, 2.0, 3.5, 4.0]], [[5.0, 2.0, 3.5, 4.0]]],
            "drugs": [["drug-1", "drug-2"], ["drug-3"]],
            "label": 0,
        },
        {
            "patient_id": "patient-1",
            "visit_id": "visit-0",
            "conditions": [["cond-33", "cond-86", "cond-80"]],
            "procedures": [[[5.0, 2.0, 3.5, 4.0]], [[5.0, 2.0, 3.5, 4.0]]],
            "drugs": [["drug-1", "drug-2"], ["drug-3"]],
            "label": 1,
        },
    ]

    input_info = {
        "conditions": {"dim": 3, "type": str},
        "procedures": {"dim": 2, "type": float, "len": 4},
        "label": {"dim": 0, "type": int},
        "drugs": {"dim": 3, "type": str},
    }

    # dataset
    dataset = SampleDataset(samples=samples, dataset_name="test")
    dataset.input_info = input_info

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    # model
    model = Deepr(
        dataset=dataset,
        # feature_keys=["procedures"],
        feature_keys=["drugs"],
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
