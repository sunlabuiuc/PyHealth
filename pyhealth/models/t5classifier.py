"""T5 encoder backbone with a linear head for PyHealth classification tasks."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel

from pyhealth.models import BaseModel


class T5Classifier(BaseModel):
    """Encoder-only T5 (Hugging Face) with pooling and a linear classification head.

    Text inputs are built by concatenating string values for each ``feature_key``
    in the batch (see :class:`~pyhealth.models.BaseModel`). This fits
    single-field text tasks (e.g. sentence classification) and simple multi-field
    concatenation. Uses :class:`transformers.T5EncoderModel` only (not full
    sequence-to-sequence T5).

    Args:
        dataset: A :class:`~pyhealth.datasets.SampleDataset` (output of
            ``dataset.set_task(...)``) providing ``input_schema`` / ``output_schema``.
        pretrained_model_name: Hugging Face model id (e.g. ``"t5-small"``,
            ``"google/flan-t5-base"``).
        max_length: Maximum tokenizer sequence length.
        dropout: Dropout before the linear head.
        pooling: ``"mean"`` (masked mean over tokens) or ``"first"`` (first token).
    """

    def __init__(
        self,
        dataset,
        pretrained_model_name="t5-base",
        max_length=256,
        dropout=0.1,
        pooling="mean",
    ):
        super().__init__(dataset=dataset)

        self.label_key = self.label_keys[0]

        self.max_length = max_length
        self.pooling = pooling

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.encoder = T5EncoderModel.from_pretrained(pretrained_model_name)

        hidden_size = self.encoder.config.d_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, self.get_output_size())

    def forward(self, **samples):
        """Encode batched text features and return logits (and loss if labels present).

        Args:
            **samples: Batch dict from the PyHealth collate function. Must include
                keys in ``feature_keys``; optionally includes ``label_key`` for training.

        Returns:
            dict: At minimum ``logit`` and ``y_prob``; with labels, also ``loss`` and
            ``y_true``.
        """

        # -----------------------------
        # 1. Build text input (simple)
        # -----------------------------
        texts = []
        batch_size = len(next(iter(samples.values())))

        for i in range(batch_size):
            parts = []

            for key in self.feature_keys:
                if key not in samples:
                    continue

                val = samples[key][i]

                # basic flattening
                if isinstance(val, list):
                    val = " ".join(map(str, val))
                else:
                    val = str(val)

                parts.append(f"{key}: {val}")

            texts.append(" ".join(parts))

        # -----------------------------
        # 2. Tokenize
        # -----------------------------
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # -----------------------------
        # 3. Encoder
        # -----------------------------
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        hidden = outputs.last_hidden_state  # [B, L, H]

        # -----------------------------
        # 4. Pooling
        # -----------------------------
        if self.pooling == "first":
            pooled = hidden[:, 0, :]
        else:
            mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        # -----------------------------
        # 5. Head
        # -----------------------------
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        # -----------------------------
        # 6. Output
        # -----------------------------
        result = {
            "logit": logits,
            "y_prob": self.prepare_y_prob(logits),
        }

        # -----------------------------
        # 7. Loss
        # -----------------------------
        if self.label_key in samples:
            y_true = samples[self.label_key].to(self.device)
            loss = self.get_loss_function()(logits, y_true)

            result["loss"] = loss
            result["y_true"] = y_true

        return result