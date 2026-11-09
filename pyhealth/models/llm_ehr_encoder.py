"""A frozen LLM embedding model as an EHR encoder, with a linear head.

This implements the method of Hegselmann et al., "Large language models are
powerful electronic health record encoders" (npj Digital Medicine, 2025;
https://arxiv.org/abs/2502.17403): serialize each patient's record to
Markdown text (:class:`~pyhealth.tasks.EHRTextSerializationTask`), embed the
text with a frozen general-purpose LLM embedding model such as
Qwen3-Embedding, and fit a linear classifier on the embeddings. In the paper
this matched or beat count-based and EHR-specific foundation-model baselines
on EHRSHOT and UK Biobank tasks without any EHR pretraining.

Reference code: https://github.com/stefanhgm/ehrshot-benchmark (Apache-2.0);
this is a reimplementation on PyHealth's model API.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from ..datasets import SampleDataset
from ..tasks.ehr_text_serialization import _name
from .base_model import BaseModel

POOLING_MODES = ("last", "mean", "cls")


def qwen3_instruction(task: str, text: str) -> str:
    """Formats a query the way Qwen3-Embedding expects an instructed input.

    Examples:
        >>> qwen3_instruction("Predict in-hospital mortality", "# EHR")
        'Instruct: Predict in-hospital mortality\\nQuery:# EHR'
    """
    return f"Instruct: {task}\nQuery:{text}"


def _save_atomically(path: str, array: np.ndarray) -> None:
    """Saves ``array`` to ``path`` so readers never see a partial file."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


class LLMEHREncoder(BaseModel):
    """Classifies serialized EHR text with a frozen LLM embedding model.

    Each sample's text is embedded once by a frozen Hugging Face model and
    pooled to one vector (last token by default, as Qwen3-Embedding is
    trained; ``"mean"`` or ``"cls"`` for BERT-style encoders), optionally
    L2-normalized. Only the linear head is trained, which makes this a
    logistic regression on the embeddings, as in the paper. Embeddings are
    memoized in memory (``memoize``) and, with ``cache_dir``, on disk, so
    later epochs and reruns skip the encoder.

    The encoder is not registered as a submodule, so saved checkpoints hold
    the head only, not billions of frozen weights. It follows the model's
    device on each call.

    Args:
        dataset: The dataset; it needs a ``"text"`` input (see
            :class:`~pyhealth.tasks.EHRTextSerializationTask`) and one label.
            Other inputs, e.g. codes kept for a baseline, are ignored.
        model_name: Hugging Face embedding model. Default
            ``"Qwen/Qwen3-Embedding-0.6B"``; the paper's best was
            ``"Qwen/Qwen3-Embedding-8B"``.
        max_length: Tokens read per text. Default 2048; the paper used up to
            8192 for the 8B model.
        pooling: ``"last"`` (last non-padding token), ``"mean"`` or
            ``"cls"``.
        instruction: Optional task instruction, formatted with
            ``instruction_format`` (Qwen3's by default). It is applied to
            the texts before ``encoder`` or ``embed_fn``.
        instruction_format: ``format(instruction, text) -> str``. Its
            module, name and code are part of the cache key.
        normalize: L2-normalize the embeddings.
        batch_size: Texts per encoder call.
        cache_dir: Directory for an on-disk embedding cache, keyed by
            ``cache_name``, the settings and the text.
        cache_name: The embedder's identity in the disk cache key. Defaults
            to ``model_name``; required with ``cache_dir`` when ``encoder``
            or ``embed_fn`` is given, so two embedders sharing a cache
            directory never read each other's vectors.
        memoize: Keep every embedding in memory for later calls (and
            epochs). It holds ``n x embedding_dim`` floats, about 1.6 GB for
            400k records at 1024 dimensions; with ``cache_dir``, pass
            ``False`` to read them from disk instead.
        torch_dtype: Passed to ``from_pretrained``, e.g. ``torch.float16``
            on a GPU. Default ``"auto"``.
        encoder: A loaded encoder to use instead of ``model_name`` (with
            ``tokenizer``), e.g. a small model in tests.
        tokenizer: The encoder's tokenizer.
        embed_fn: Alternatively, any ``embed_fn(texts) -> Tensor[n, d]``;
            then ``model_name``, ``encoder`` and pooling are not used, but
            ``instruction`` is.
        text_key: The input field holding the text. Defaults to the
            dataset's only ``"text"`` input.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p0", "text": "# EHR\\n- Sepsis", "label": 1},
        ...     {"patient_id": "p1", "text": "# EHR\\n- Fracture", "label": 0},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples, input_schema={"text": "text"},
        ...     output_schema={"label": "binary"},
        ... )
        >>> model = LLMEHREncoder(  # a stand-in encoder; no download
        ...     dataset, embed_fn=lambda texts: torch.ones(len(texts), 8)
        ... )
        >>> out = model(text=["# EHR\\n- Sepsis"], label=torch.tensor([[1.0]]))
        >>> out["y_prob"].shape
        torch.Size([1, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        max_length: int = 2048,
        pooling: str = "last",
        instruction: str | None = None,
        instruction_format: Callable[[str, str], str] = qwen3_instruction,
        normalize: bool = True,
        batch_size: int = 8,
        cache_dir: str | None = None,
        cache_name: str | None = None,
        memoize: bool = True,
        torch_dtype: Any = "auto",
        encoder: nn.Module | None = None,
        tokenizer: Any = None,
        embed_fn: Callable[[Sequence[str]], torch.Tensor] | None = None,
        text_key: str | None = None,
    ):
        super().__init__(dataset)
        if pooling not in POOLING_MODES:
            raise ValueError(f"pooling must be one of {POOLING_MODES}, not {pooling!r}")
        if text_key is None:
            texts = [k for k, v in dataset.input_schema.items() if v == "text"]
            if len(texts) != 1:
                raise ValueError("pass text_key: the dataset has "
                                 f"{len(texts)} text inputs")
            text_key = texts[0]
        if text_key not in self.feature_keys:
            raise ValueError(f"{text_key!r} is not an input of the dataset")
        if len(self.label_keys) != 1:
            raise ValueError("LLMEHREncoder supports exactly one label")
        self.text_key = text_key
        self.label_key = self.label_keys[0]
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.instruction = instruction
        self.instruction_format = instruction_format
        # the formatter changes the embedded text, so it is part of the key
        self._format_name = _name(instruction_format) if instruction else ""
        self.normalize = normalize
        self.batch_size = batch_size
        if cache_dir and cache_name is None and (
            encoder is not None or embed_fn is not None
        ):
            raise ValueError(
                "pass cache_name with cache_dir when encoder or embed_fn is "
                "given: model_name does not identify a custom embedder, so "
                "its cached vectors could be read by another one"
            )
        self.cache_dir = cache_dir
        self.cache_name = model_name if cache_name is None else cache_name
        self.memoize = memoize
        self._memo: dict[str, torch.Tensor] = {}
        # kept out of the module tree: not trained, not saved, not moved by
        # .to(); see embed()
        self._embed_fn = embed_fn
        self._encoder: list[nn.Module] = []
        self._tokenizer = tokenizer
        if embed_fn is None:
            if encoder is None:
                from transformers import AutoModel, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, padding_side="left" if pooling == "last" else "right"
                )
                encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype)
                self._tokenizer = tokenizer
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad_(False)
            self._encoder = [encoder]
            # the precision changes the vectors, so it is part of the key
            self._dtype_name = str(next(encoder.parameters()).dtype)
            dim = encoder.config.hidden_size
        else:
            self._dtype_name = ""
            dim = int(embed_fn(["probe"]).shape[-1])
        self.embedding_dim = dim
        self.fc = nn.Linear(dim, self.label_processor.output_size())

    # ------------------------------------------------------------ embedding
    def _cache_key(self, text: str) -> str:
        settings = (f"{self.cache_name}|{self.max_length}|{self.pooling}|"
                    f"{self.instruction}|{self._format_name}|{self.normalize}|"
                    f"{self._dtype_name}|")
        return hashlib.sha256((settings + text).encode()).hexdigest()

    def _encode(self, texts: list[str]) -> torch.Tensor:
        if self.instruction:
            texts = [self.instruction_format(self.instruction, t) for t in texts]
        if self._embed_fn is not None:
            return torch.as_tensor(self._embed_fn(texts), dtype=torch.float32)
        encoder = self._encoder[0]
        device = self.device
        if next(encoder.parameters()).device != device:
            encoder.to(device)
        batch = self._tokenizer(texts, padding=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        hidden = encoder(**batch).last_hidden_state
        mask = batch["attention_mask"]
        if self.pooling == "last":
            # the last non-padding token, whichever side was padded
            positions = torch.arange(mask.shape[1], device=device)
            last = (mask * positions).argmax(dim=1)
            pooled = hidden[torch.arange(hidden.shape[0], device=device), last]
        elif self.pooling == "mean":
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)
        else:
            # the first non-padding token: position 0 only under right padding
            first = mask.argmax(dim=1)
            pooled = hidden[torch.arange(hidden.shape[0], device=device), first]
        return pooled.float()

    @torch.no_grad()
    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        """Embeds texts with the frozen encoder, using the caches.

        Args:
            texts: Serialized records.

        Returns:
            A ``[len(texts), embedding_dim]`` float tensor on the model's
            device.

        Examples:
            >>> model.embed(["# EHR"]).shape  # doctest: +SKIP
            torch.Size([1, 1024])
        """
        keys = [self._cache_key(t) for t in texts]
        found: dict[str, torch.Tensor] = {}  # this call's vectors
        todo: dict[str, str] = {}  # key -> text, in first-seen order
        for key, text in zip(keys, texts):
            if key in found or key in todo:
                continue
            if key in self._memo:
                found[key] = self._memo[key]
                continue
            path = self._cache_path(key)
            if path and os.path.exists(path):
                found[key] = torch.from_numpy(np.load(path))
            else:
                todo[key] = text
        items = list(todo.items())
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            vectors = self._encode([t for _, t in chunk]).cpu()
            if self.normalize:
                vectors = nn.functional.normalize(vectors, dim=-1)
            for (key, _), vector in zip(chunk, vectors):
                found[key] = vector
                path = self._cache_path(key)
                if path:
                    _save_atomically(path, vector.numpy())
        if self.memoize:
            self._memo.update(found)
        return torch.stack([found[k] for k in keys]).to(self.device)

    def _cache_path(self, key: str) -> str | None:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, key[:2], f"{key}.npy")

    # -------------------------------------------------------------- forward
    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Embeds the texts and applies the linear head.

        Returns:
            ``logit``, ``y_prob`` and ``embed``, plus ``loss`` and ``y_true``
            when the label is given.
        """
        texts = kwargs[self.text_key]
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.embed(list(texts))
        logits = self.fc(embeddings)
        out = {"logit": logits, "y_prob": self.label_processor.activation(logits),
               "embed": embeddings}
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            out["loss"] = self.label_processor.loss(logits, y_true)
            out["y_true"] = y_true
        return out
