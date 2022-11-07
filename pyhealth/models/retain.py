from typing import List

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import BaseDataset
from pyhealth.models import BaseModel

VALID_OPERATION_LEVEL = ["visit", "event"]


class RETAINLayer(nn.Module):
    """RETAIN layer.

    Paper: Edward Choi et al. RETAIN: An Interpretable Predictive Model for
    Healthcare using Reverse Time Attention Mechanism. NIPS 2016.

    This layer is used in the RETAIN model. But it can also be used as a
    standalone layer.

    Args:
        input_size: input feature size.
        hidden_size: hidden feature size.
        dropout: dropout rate. Default is 0.5.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            dropout: float = 0.5,
    ):
        super(RETAINLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.alpha_gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.beta_gru = nn.GRU(input_size, hidden_size, batch_first=True)

        self.alpha_li = nn.Linear(input_size, 1)
        self.beta_li = nn.Linear(input_size, hidden_size)

    @staticmethod
    def reverse_x(input, lengths):
        """Reverses the input.
        """
        reversed_input = input.new(input.size())
        for i, length in enumerate(lengths):
            reversed_input[i, :length] = input[i, :length].flip(dims=[0])
        return reversed_input

    def compute_alpha(self, rx, lengths):
        rx = rnn_utils.pack_padded_sequence(
            rx, lengths, batch_first=True, enforce_sorted=False
        )
        g, _ = self.alpha_gru(rx)
        g, _ = rnn_utils.pad_packed_sequence(g, batch_first=True)
        attn_alpha = torch.softmax(self.alpha_li(g), dim=1)
        return attn_alpha

    def compute_beta(self, rx, lengths):
        rx = rnn_utils.pack_padded_sequence(
            rx, lengths, batch_first=True, enforce_sorted=False
        )
        h, _ = self.beta_gru(rx)
        h, _ = rnn_utils.pad_packed_sequence(h, batch_first=True)
        attn_beta = torch.tanh(self.beta_li(h))
        return attn_beta

    def forward(self, x: torch.tensor, mask: torch.tensor):
        """
        Args:
            x: a tensor of shape [batch size, sequence len, input size].
            mask: a tensor of shape [batch size, sequence len] where 1 indicates
                valid and 0 indicates invalid.

        Returns:
            outputs: a tensor of shape [batch size, sequence len, hidden size],
                containing the output features for each time step.
            last_outputs: a tensor of shape [batch size, hidden size], containing
                the output features for the last time step.
        """
        # rnn will only apply dropout between layers
        x = self.dropout_layer(x)
        lengths = torch.sum(mask.int(), dim=-1).cpu()
        rx = self.reverse_x(x, lengths)
        attn_alpha = self.compute_alpha(rx, lengths)
        attn_beta = self.compute_beta(rx, lengths)
        c = attn_alpha * attn_beta * x  # (patient, seq_len, hidden_size)
        c = torch.sum(c, dim=1)  # (patient, hidden_size)
        return c


class RETAIN(BaseModel):
    """RETAIN model.

    Paper: Edward Choi et al. RETAIN: An Interpretable Predictive Model for
    Healthcare using Reverse Time Attention Mechanism. NIPS 2016.

    Note:
        This model can operate on both visit and event level, as designated by
            the operation_level parameter.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        operation_level: one of "visit", "event".
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        **kwargs: other parameters for the RETAIN layer.
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
            **kwargs
    ):
        super(RETAIN, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        assert operation_level in VALID_OPERATION_LEVEL, \
            f"operation_level must be one of {VALID_OPERATION_LEVEL}"
        self.operation_level = operation_level
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)
        self.retain = nn.ModuleDict()
        for feature_key in feature_keys:
            self.retain[feature_key] = RETAINLayer(
                input_size=embedding_dim, hidden_size=hidden_dim, **kwargs
            )

        output_size = self.get_output_size(self.label_tokenizer)
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def _visit_level_forward(self, **kwargs):
        """Visit-level RETAIN forward."""
        patient_emb = []
        for feature_key in self.feature_keys:
            assert type(kwargs[feature_key][0][0]) == list
            x = self.feat_tokenizers[feature_key].batch_encode_3d(kwargs[feature_key])
            # (patient, visit, code)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, code, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, visit, embedding_dim)
            x = torch.sum(x, dim=2)
            # (patient, visit)
            mask = torch.sum(x, dim=2) != 0
            # (patient, hidden_dim)
            x = self.retain[feature_key](x, mask)
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

    def _event_level_forward(self, **kwargs):
        """Event-level RETAIN forward."""
        patient_emb = []
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_2d(kwargs[feature_key])
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, code, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, code)
            mask = torch.sum(x, dim=2) != 0
            # (patient, hidden_dim)
            x = self.retain[feature_key](x, mask)
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

    def forward(self, **kwargs):
        if self.operation_level == "visit":
            return self._visit_level_forward(**kwargs)
        elif self.operation_level == "event":
            return self._event_level_forward(**kwargs)
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
    model = RETAIN(
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
    model = RETAIN(
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
    model = RETAIN(
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
    model = RETAIN(
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
    model = RETAIN(
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
    model = RETAIN(
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
