"""Local seq2seq T5 model for PyHealth text-to-text tasks."""

from __future__ import annotations

from typing import Iterable, List

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from pyhealth.models import BaseModel


class T5(BaseModel):
    """Sequence-to-sequence T5 model for PyHealth text-to-text tasks.

    The dataset is expected to expose one or more text input fields and a text
    output field. The canonical shape used by the examples in this repository is
    ``source_text -> target_text``.

    Args:
        dataset: A :class:`~pyhealth.datasets.SampleDataset` with text inputs and
            a text output field.
        pretrained_model_name: Hugging Face checkpoint name.
        max_source_length: Maximum encoder sequence length.
        max_target_length: Maximum decoder target sequence length.
        generation_max_length: Maximum generated output length for inference.
        target_key: Output key containing the target text.
    """

    def __init__(
        self,
        dataset,
        pretrained_model_name: str = "t5-base",
        max_source_length: int = 256,
        max_target_length: int = 128,
        generation_max_length: int = 128,
        target_key: str = "target_text",
    ) -> None:
        super().__init__(dataset=dataset)

        self.target_key = target_key
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.generation_max_length = generation_max_length
        self.mode = None

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name
        )

    @staticmethod
    def _normalize_text(value) -> str:
        if isinstance(value, list):
            return " ".join(map(str, value))
        return str(value)

    def _build_source_texts(self, samples) -> List[str]:
        if "source_text" in samples:
            return [self._normalize_text(text) for text in samples["source_text"]]

        texts: List[str] = []
        batch_size = len(next(iter(samples.values())))
        for i in range(batch_size):
            parts = []
            for key in self.feature_keys:
                if key not in samples:
                    continue
                parts.append(f"{key}: {self._normalize_text(samples[key][i])}")
            texts.append(" ".join(parts))
        return texts

    def _tokenize_sources(self, source_texts: List[str]):
        return self.tokenizer(
            source_texts,
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )

    def _tokenize_targets(self, target_texts: List[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            text_target=target_texts,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        labels = encoded["input_ids"]
        labels = labels.masked_fill(
            labels == self.tokenizer.pad_token_id,
            -100,
        )
        return labels

    def forward(self, **samples):
        """Compute the seq2seq loss for a batch."""
        source_texts = self._build_source_texts(samples)
        model_inputs = self._tokenize_sources(source_texts)
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)

        result = {}
        if self.target_key in samples:
            target_texts = [
                self._normalize_text(text) for text in samples[self.target_key]
            ]
            labels = self._tokenize_targets(target_texts).to(self.device)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            result["loss"] = outputs.loss
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            result["logits"] = outputs.logits

        return result

    def generate_text(self, source_texts: Iterable[str], **generate_kwargs) -> List[str]:
        """Generate text for a batch of source prompts."""
        encoded = self._tokenize_sources(list(source_texts))
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=generate_kwargs.pop(
                "max_length", self.generation_max_length
            ),
            **generate_kwargs,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
