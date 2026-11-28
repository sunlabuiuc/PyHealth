"""Text encoders for KeyClass: ``encode(texts)`` returns a finite ``[N, D]`` array."""

from collections.abc import Iterable

import numpy as np


class ClinicalTextEncoder:
    """Frozen Hugging Face encoder with masked mean pooling.

    Embeds phrases, category descriptions and notes for KeyClass. Each text is
    truncated to ``max_length`` tokens (special tokens included), and token
    vectors are averaged over the attention mask. The KeyClass paper uses
    BlueBERT; notes are truncated rather than reduced with the paper's TF-IDF
    text selection. The model is loaded, and downloaded if needed, only when
    the encoder is constructed.

    This is a small class rather than a call to
    ``pyhealth.models.clip_stress_utils.encode_stress_prompts``, which does a
    similar Hugging Face pass for a different model. That helper loads the
    model on every call, averages over padding by default, and rejects empty
    input. KeyClass encodes three sets of texts (phrases, descriptions and
    notes) with one loaded model, needs padding excluded so the result does
    not depend on batch size, and gets empty sets of phrases or notes. The
    ``encode(texts)`` method is the only interface ``KeywordLabeler`` needs,
    so any object with that method can replace this class.

    Args:
        model_name: Hugging Face model ID or local snapshot directory, for
            example ``"bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"``.
        batch_size: Texts encoded per forward pass.
        max_length: Maximum tokens per text, including special tokens.
        device: Torch device, for example ``"cpu"``, ``"cuda"`` or ``"mps"``.

    Raises:
        ValueError: If ``batch_size`` is not positive or ``max_length < 2``.

    Example:
        >>> name = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
        >>> encoder = ClinicalTextEncoder(name)
        >>> encoder.encode(["acute renal failure", "pneumonia"]).shape
        (2, 768)
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        max_length: int = 512,
        device: str = "cpu",
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        if batch_size < 1 or max_length < 2:
            raise ValueError("batch_size must be positive and max_length >= 2")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.model.requires_grad_(False)
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Embeds texts with masked mean pooling.

        Args:
            texts: Texts to embed.

        Returns:
            A float32 array of shape ``[len(texts), hidden_size]``; empty input
            gives shape ``[0, hidden_size]``.
        """
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
