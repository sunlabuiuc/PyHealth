import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pyhealth.models import BaseModel

class BioClinicalBERTAlcoholUse(BaseModel):
    """PyHealth wrapper around Bio-ClinicalBERT for alcohol-use labels."""
    def __init__(
        self,
        dataset,
        feature_key: str,
        label_key: str,
        mode: str,
        pretrained_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_len: int = 128,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            feature_keys=[feature_key],
            label_key=label_key,
            mode=mode,
        )
        self.feature_key = feature_key
        self.label_key = label_key
        self.max_len = max_len
        self.label_tokenizer = self.get_label_tokenizer()
        num_labels = self.get_output_size(self.label_tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
        )

    def forward(self, **kwargs):
        texts = kwargs[self.feature_key]
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        y_true = y_true.view(-1).long().to(self.device)
        outputs = self.model(**enc)
        logits = outputs.logits
        loss_fn = self.get_loss_function()
        loss_task = loss_fn(logits, y_true)
        loss = loss_task
        y_prob = self.prepare_y_prob(logits)
        result = {
            "loss": loss,
            "loss_task": loss_task,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            encoder = getattr(self.model, "bert", None)
            if encoder is not None:
                with torch.no_grad():
                    hidden = encoder(**enc).last_hidden_state
                    cls_embed = hidden[:, 0, :]
                result["embed"] = cls_embed
        return result
