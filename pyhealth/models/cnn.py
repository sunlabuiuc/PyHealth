from typing import List

import torch
import torch.nn as nn

from pyhealth.datasets import BaseDataset
from pyhealth.models import BaseModel


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()

        self.conv1 = nn.Sequential(
            # stride=1 by default
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            # stride=1 by default
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                # stride=1, padding=0 by default
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class CNNLayer(nn.Module):
    """separate callable CNN layer

    Args:
        input_size: input size of rnn
        hidden_size: hidden size of rnn
        num_layers: number of rnn layers
        dropout: dropout rate

    **Examples:**
        >>> from pyhealth.models import CNNLayer
        >>> input = torch.randn(3, 128, 5) # [batch size, seq len, input_size]
        >>> model = CNNLayer(5, 64, 2, 0.5)
        >>> model(input, mask=None).shape
        torch.Size([3, 64]) # [batch size, hidden_size]

    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
    ):
        """separate callable CNN layer
        Args:
            input_size: input size of rnn
            hidden_size: hidden size of rnn
            num_layers: number of rnn layers
        """
        super(CNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cnn = nn.ModuleDict()
        for i in range(num_layers):
            in_channels = input_size if i == 0 else hidden_size
            self.cnn[f"CNN-{i}"] = CNNBlock(in_channels, hidden_size)

        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.tensor):
        """
        Args:
            x: [batch size, seq len, input_size]
        Returns:
            outputs [batch size, seq len, hidden_size]
        """
        # [batch size, seq len, emb size] -> [batch size, emb size, seq len]
        x = x.permute(0, 2, 1)
        for idx in range(len(self.cnn)):
            x = self.cnn[f"CNN-{idx}"](x)
        # pooling
        x = self.pooling(x)
        x = x.squeeze(-1)
        return x


class CNN(BaseModel):
    """CNN Class, use "task" as key to identify specific CNN model and route there

    Args:
        dataset: the dataset object
        tables: the list of table names to use
        target: the target table name
        mode: the mode of the model, "multilabel", "multiclass" or "binary"
        embedding_dim: the embedding dimension
        hidden_dim: the hidden dimension

    **Examples:**
        >>> from pyhealth.datasets import OMOPDataset
        >>> dataset = OMOPDataset(
        ...     root="https://storage.googleapis.com/pyhealth/synpuf1k_omop_cdm_5.2.2",
        ...     tables=["condition_occurrence", "procedure_occurrence"],
        ... ) # load dataset
        >>> from pyhealth.tasks import mortality_prediction_omop_fn
        >>> dataset.set_task(mortality_prediction_omop_fn) # set task

        >>> from pyhealth.models import CNN
        >>> model = CNN(
        ...     dataset=dataset,
        ...     tables=["conditions", "procedures"],
        ...     target="label",
        ...     mode="binary",
        ... )
    """

    def __init__(
            self,
            dataset: BaseDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            operation_level: str,
            embedding_dim: int = 128,
            hidden_dim: int = 128,
            **kwargs,
    ):
        super(CNN, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        assert operation_level in ["visit", "event"]
        self.operation_level = operation_level
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.feat_tokenizers = self._get_feature_tokenizers()
        self.label_tokenizer = self._get_label_tokenizer()
        self.embeddings = self._get_embeddings(self.feat_tokenizers, embedding_dim)

        self.cnn = nn.ModuleDict()
        for feature_key in feature_keys:
            self.cnn[feature_key] = CNNLayer(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                **kwargs
            )

        output_size = self._get_output_size(self.label_tokenizer)
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def _visit_level_forward(self, device, **kwargs):
        """Visit level CNN forward."""
        patient_emb = []
        for feature_key in self.feature_keys:
            assert type(kwargs[feature_key][0][0]) == list
            x = self.feat_tokenizers[feature_key].batch_encode_3d(kwargs[feature_key])
            # (patient, visit, code)
            x = torch.tensor(x, dtype=torch.long, device=device)
            # (patient, visit, code, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, visit, embedding_dim)
            x = torch.sum(x, dim=2)
            # (patient, hidden_dim)
            x = self.cnn[feature_key](x)
            patient_emb.append(x)
        # (patient, features * hidden_dim)
        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain loss, y_true, t_prob
        loss, y_true, y_prob = self._calculate_output(logits, kwargs[self.label_key])
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }

    def _event_level_forward(self, device, **kwargs):
        """Event level CNN forward."""
        patient_emb = []
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_2d(kwargs[feature_key])
            x = torch.tensor(x, dtype=torch.long, device=device)
            # (patient, code, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, hidden_dim)
            x = self.cnn[feature_key](x)
            patient_emb.append(x)
        # (patient, features * hidden_dim)
        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain loss, y_true, t_prob
        loss, y_true, y_prob = self._calculate_output(logits, kwargs[self.label_key])
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }

    def forward(self, device, **kwargs):
        if self.operation_level == "visit":
            return self._visit_level_forward(device, **kwargs)
        elif self.operation_level == "event":
            return self._event_level_forward(device, **kwargs)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    from pyhealth.datasets import MIMIC3Dataset
    from torch.utils.data import DataLoader
    from pyhealth.utils import collate_fn_dict


    def task_event(patient):
        samples = []
        for visit in patient:
            conditions = visit.get_code_list(table="DIAGNOSES_ICD")
            procedures = visit.get_code_list(table="PROCEDURES_ICD")
            drugs = visit.get_code_list(table="PRESCRIPTIONS")
            mortality_label = int(visit.discharge_status)
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue
            samples.append(
                {
                    "visit_id": visit.visit_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "list_label": drugs,
                    "value_label": mortality_label,
                }
            )
        return samples


    def task_visit(patient):
        samples = []
        for visit in patient:
            conditions = visit.get_code_list(table="DIAGNOSES_ICD")
            procedures = visit.get_code_list(table="PROCEDURES_ICD")
            drugs = visit.get_code_list(table="PRESCRIPTIONS")
            mortality_label = int(visit.discharge_status)
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue
            samples.append(
                {
                    "visit_id": visit.visit_id,
                    "patient_id": patient.patient_id,
                    "conditions": [conditions],
                    "procedures": [procedures],
                    "list_label": drugs,
                    "value_label": mortality_label,
                }
            )
        return samples


    dataset = MIMIC3Dataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=True,
        code_mapping={"NDC": "ATC"},
        refresh_cache=False,
    )

    # event level + binary
    dataset.set_task(task_event)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
    )
    model = CNN(
        dataset=dataset,
        feature_keys=["conditions", "procedures"],
        label_key="value_label",
        mode="binary",
        operation_level="event",
    )
    model.to("cuda")
    batch = iter(dataloader).next()
    output = model(**batch, device="cuda")
    print(output["loss"])

    # visit level + binary
    dataset.set_task(task_visit)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
    )
    model = CNN(
        dataset=dataset,
        feature_keys=["conditions", "procedures"],
        label_key="value_label",
        mode="binary",
        operation_level="visit",
    )
    model.to("cuda")
    batch = iter(dataloader).next()
    output = model(**batch, device="cuda")
    print(output["loss"])

    # event level + multiclass
    dataset.set_task(task_event)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
    )
    model = CNN(
        dataset=dataset,
        feature_keys=["conditions", "procedures"],
        label_key="value_label",
        mode="multiclass",
        operation_level="event",
    )
    model.to("cuda")
    batch = iter(dataloader).next()
    output = model(**batch, device="cuda")
    print(output["loss"])

    # visit level + multiclass
    dataset.set_task(task_visit)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
    )
    model = CNN(
        dataset=dataset,
        feature_keys=["conditions", "procedures"],
        label_key="value_label",
        mode="multiclass",
        operation_level="visit",
    )
    model.to("cuda")
    batch = iter(dataloader).next()
    output = model(**batch, device="cuda")
    print(output["loss"])

    # event level + multilabel
    dataset.set_task(task_event)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
    )
    model = CNN(
        dataset=dataset,
        feature_keys=["conditions", "procedures"],
        label_key="list_label",
        mode="multilabel",
        operation_level="event",
    )
    model.to("cuda")
    batch = iter(dataloader).next()
    output = model(**batch, device="cuda")
    print(output["loss"])

    # visit level + multilabel
    dataset.set_task(task_visit)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
    )
    model = CNN(
        dataset=dataset,
        feature_keys=["conditions", "procedures"],
        label_key="list_label",
        mode="multilabel",
        operation_level="visit",
    )
    model.to("cuda")
    batch = iter(dataloader).next()
    output = model(**batch, device="cuda")
    print(output["loss"])
