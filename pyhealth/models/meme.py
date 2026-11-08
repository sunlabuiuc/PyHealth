"""MEME: the Multiple Embedding Model for EHR, on clinical pseudo-notes.

This implements Lee et al., "Clinical decision support using pseudo-notes
from multiple streams of EHR data" (npj Digital Medicine 8, 394, 2025;
https://doi.org/10.1038/s41746-025-01777-x; preprint "Emergency Department
Decision Support using Clinical Pseudo-notes", arXiv:2402.00160). Each domain
of an emergency department visit (arrival, triage, medication
reconciliation, vital signs, dispensed medications, diagnosis codes) is
written as a short templated "pseudo-note"
(:class:`~pyhealth.tasks.EDDispositionMIMIC4`). A frozen clinical BERT
(MedBERT) embeds each note; self-attention across the note embeddings feeds
a linear head. In the paper MEME reached an AUROC of 0.991 for ED
disposition on MIMIC-IV-ED, above single-note models and the MC-BEC
benchmark (0.968).

The prediction head is adapted from the reference implementation
(``models/model.py`` in https://github.com/Simonlee711/MEME), which is
distributed under the MIT License:

    Copyright (c) 2023 Simon Lee

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
    NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
    USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn

from ..datasets import SampleDataset
from .base_model import BaseModel
from .llm_ehr_encoder import LLMEHREncoder

READOUTS = ("first", "mean")


class DomainSelfAttention(nn.Module):
    """Single-head self-attention across domain embeddings, then a dense layer.

    The MEME head: queries, keys and values are linear maps of the domain
    embeddings, attention is scaled dot-product across domains, and a dense
    layer with ReLU and dropout follows.

    Args:
        dim: Embedding size.
        dropout: Dropout after the dense layer (0.3 in the paper's code).

    Examples:
        >>> import torch
        >>> layer = DomainSelfAttention(8)
        >>> out, attention = layer(torch.randn(2, 5, 8))
        >>> out.shape, attention.shape
        (torch.Size([2, 5, 8]), torch.Size([2, 5, 5]))
    """

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Attends across domains.

        Args:
            x: ``[batch, domains, dim]`` embeddings.

        Returns:
            The attended ``[batch, domains, dim]`` vectors and the
            ``[batch, domains, domains]`` attention weights.
        """
        scores = self.query(x) @ self.key(x).transpose(1, 2) / self.dim**0.5
        attention = scores.softmax(dim=-1)
        out = torch.relu(self.dense(attention @ self.value(x)))
        return self.dropout(out), attention


class MEME(BaseModel):
    """Multiple Embedding Model for EHR (Lee et al., npj Digital Medicine 2025).

    Each text input of the dataset is one domain "pseudo-note" (see
    :class:`~pyhealth.tasks.EDDispositionMIMIC4`). Every note is embedded by
    the same frozen Hugging Face encoder, taking the ``[CLS]`` token (the
    paper's MedBERT by default). The domain embeddings are stacked, passed
    through :class:`DomainSelfAttention`, and a linear head reads out the
    first domain's attended vector, as in the reference code
    (``readout="mean"`` averages the domains instead). Only the head is
    trained.

    The reference code attends over the concatenated token states of all
    notes (up to 6 x 512 tokens) and reads out the first token; this model
    attends over one ``[CLS]`` vector per note, as the paper describes,
    which lets the embeddings be computed once and cached.

    The notes are embedded by an :class:`~pyhealth.models.LLMEHREncoder`
    with ``pooling="cls"`` and ``normalize=False`` (see
    :attr:`text_encoder`), so embeddings are memoized in memory and, with
    ``cache_dir``, saved on disk, and later epochs and reruns skip the
    encoder. That encoder is not registered as a submodule: checkpoints hold
    the MEME head only.

    Args:
        dataset: The dataset; its ``"text"`` inputs are the domains.
        feature_keys: The text inputs to use, in order. Defaults to every
            ``"text"`` input in schema order. With ``readout="first"`` the
            first one is read out.
        model_name: Hugging Face encoder. Default ``"Charangan/MedBERT"``,
            the paper's.
        max_length: Tokens read per note (512 in the paper).
        dropout: Dropout in the attention head (0.3 in the paper's code).
        readout: ``"first"`` or ``"mean"``.
        batch_size: Notes per encoder call.
        cache_dir: Directory for an on-disk embedding cache, keyed by
            ``cache_name``, ``max_length``, precision and the note text.
        cache_name: The encoder's identity in the cache key. Defaults to
            ``model_name``; required with ``cache_dir`` when ``encoder`` or
            ``embed_fn`` is given.
        memoize: Keep every embedding in memory for later calls.
        torch_dtype: Passed to ``from_pretrained``. Default ``"auto"``.
        encoder: A loaded encoder to use instead of ``model_name`` (with
            ``tokenizer``), e.g. a small model in tests.
        tokenizer: The encoder's tokenizer.
        embed_fn: Alternatively, any ``embed_fn(texts) -> Tensor[n, d]``.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p0", "arrival": "arrived via ambulance.",
        ...      "triage": "acuity score was 1.", "disposition": 1},
        ...     {"patient_id": "p1", "arrival": "arrived via walk in.",
        ...      "triage": "acuity score was 4.", "disposition": 0},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples, input_schema={"arrival": "text", "triage": "text"},
        ...     output_schema={"disposition": "binary"},
        ... )
        >>> model = MEME(  # a stand-in encoder; no download
        ...     dataset, embed_fn=lambda texts: torch.randn(len(texts), 8)
        ... )
        >>> out = model(arrival=["arrived via ambulance."],
        ...             triage=["acuity score was 1."])
        >>> out["y_prob"].shape, out["attention"].shape
        (torch.Size([1, 1]), torch.Size([1, 2, 2]))
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys: Sequence[str] | None = None,
        model_name: str = "Charangan/MedBERT",
        max_length: int = 512,
        dropout: float = 0.3,
        readout: str = "first",
        batch_size: int = 32,
        cache_dir: str | None = None,
        cache_name: str | None = None,
        memoize: bool = True,
        torch_dtype: Any = "auto",
        encoder: nn.Module | None = None,
        tokenizer: Any = None,
        embed_fn: Callable[[Sequence[str]], torch.Tensor] | None = None,
    ):
        super().__init__(dataset)
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, not {readout!r}")
        if feature_keys is None:
            feature_keys = [k for k, v in dataset.input_schema.items() if v == "text"]
        feature_keys = list(feature_keys)
        if not feature_keys:
            raise ValueError("MEME needs at least one text input")
        missing = [k for k in feature_keys if k not in self.feature_keys]
        if missing:
            raise ValueError(f"not inputs of the dataset: {missing}")
        if len(self.label_keys) != 1:
            raise ValueError("MEME supports exactly one label")
        self.domains = feature_keys
        self.label_key = self.label_keys[0]
        self.model_name = model_name
        self.max_length = max_length
        self.readout = readout
        # The frozen encoder, its [CLS] pooling and the embedding caches are
        # LLMEHREncoder's. It is held in a list so it stays out of the module
        # tree: not trained, not saved in checkpoints, not moved by .to().
        self._text_encoder = [
            LLMEHREncoder(
                dataset,
                model_name=model_name,
                max_length=max_length,
                pooling="cls",
                normalize=False,
                batch_size=batch_size,
                cache_dir=cache_dir,
                cache_name=cache_name,
                memoize=memoize,
                torch_dtype=torch_dtype,
                encoder=encoder,
                tokenizer=tokenizer,
                embed_fn=embed_fn,
                text_key=feature_keys[0],
            )
        ]
        self.embedding_dim = self.text_encoder.embedding_dim
        self.attention = DomainSelfAttention(self.embedding_dim, dropout)
        self.fc = nn.Linear(self.embedding_dim, self.label_processor.output_size())

    # ------------------------------------------------------------ embedding
    @property
    def text_encoder(self) -> LLMEHREncoder:
        """The frozen note encoder: an :class:`LLMEHREncoder` with CLS pooling.

        Examples:
            >>> model.text_encoder.pooling  # doctest: +SKIP
            'cls'
        """
        return self._text_encoder[0]

    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        """Embeds notes with the frozen encoder, using the caches.

        Args:
            texts: Pseudo-notes.

        Returns:
            A ``[len(texts), embedding_dim]`` float tensor on the model's
            device.

        Examples:
            >>> model.embed(["At triage: acuity score was 2."]).shape  # doctest: +SKIP
            torch.Size([1, 768])
        """
        text_encoder = self.text_encoder
        if text_encoder.device != self.device:
            text_encoder.to(self.device)
        return text_encoder.embed(texts)

    # -------------------------------------------------------------- forward
    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Embeds each domain's note, attends across domains, and classifies.

        Args:
            **kwargs: A batch: one list of notes per domain, and the label
                when training or evaluating.

        Returns:
            ``logit``, ``y_prob``, ``embed`` (the read-out vector) and
            ``attention`` (``[batch, domains, domains]``), plus ``loss`` and
            ``y_true`` when the label is given.
        """
        notes = []
        for key in self.domains:
            texts = kwargs[key]
            notes.append([texts] if isinstance(texts, str) else list(texts))
        batch = len(notes[0])
        if any(len(n) != batch for n in notes):
            raise ValueError("every domain needs one note per sample")
        # one encoder pass over all notes, domain-major
        flat = [text for domain in notes for text in domain]
        embeddings = self.embed(flat).view(len(self.domains), batch, -1)
        attended, attention = self.attention(embeddings.transpose(0, 1))
        pooled = attended[:, 0] if self.readout == "first" else attended.mean(dim=1)
        logits = self.fc(pooled)
        out = {"logit": logits, "y_prob": self.label_processor.activation(logits),
               "embed": pooled, "attention": attention}
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            out["loss"] = self.label_processor.loss(logits, y_true)
            out["y_true"] = y_true
        return out
