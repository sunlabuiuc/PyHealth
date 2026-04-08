import torch
import torch.nn as nn
from transformers import T5EncoderModel, AutoTokenizer
from pyhealth.models import BaseModel


class T5Classifier(BaseModel):
    def __init__(
        self,
        dataset,
        pretrained_model_name="t5-base",
        max_length=256,
        dropout=0.1,
        pooling="mean",   # "mean" or "first"
    ):
        super().__init__(dataset=dataset)

        # label setup from BaseModel
        self.label_key = self.label_keys[0]

        # hyperparameters
        self.max_length = max_length
        self.pooling = pooling

        # backbone
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.encoder = T5EncoderModel.from_pretrained(pretrained_model_name)

        # head
        hidden_size = self.encoder.config.d_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, self.get_output_size())

    def forward(self, **samples):
        """
        samples = batch from PyHealth (MIMIC-style)

        Example:
        {
            "conditions": [...],
            "procedures": [...],
            "note": [...],
            "label": tensor([...])
        }
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
            pooled = hidden.mean(dim=1)

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