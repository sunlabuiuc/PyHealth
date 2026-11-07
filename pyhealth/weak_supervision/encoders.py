"""Encoders implement encode(texts) -> a finite [N, D] numpy array."""

import numpy as np


class ClinicalTextEncoder:
    """Frozen Hugging Face encoder with masked mean pooling, no random projection.

    Checkpoint is explicit because the clinical reproduction checkpoint is not
    yet fixed. Notes are truncated to max_length tokens (including specials).
    This is a configurable starting baseline, not the paper's TF-IDF selection.
    Downloads occur only when this class is constructed.
    """

    def __init__(self, model_name, batch_size=16, max_length=512, device="cpu"):
        from transformers import AutoModel, AutoTokenizer

        if batch_size < 1 or max_length < 2:
            raise ValueError("batch_size must be positive and max_length >= 2")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.model.requires_grad_(False)
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device

    def encode(self, texts):
        import torch

        texts = list(texts)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                tokens = self.tokenizer(
                    texts[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                chunks.append(pooled.cpu().numpy())
        return (
            np.concatenate(chunks)
            if chunks
            else np.empty((0, self.model.config.hidden_size), dtype=np.float32)
        )
