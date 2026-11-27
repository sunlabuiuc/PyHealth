"""Query-focused extractive EHR summarization model.

Implements diagnosis-conditioned attention over historical clinical
sentences, inspired by McInerney et al. (2020):

"Query-Focused EHR Summarization to Aid Imaging Diagnosis."

Supports pretrained ClinicalBERT embeddings with either a frozen
or end-to-end trainable text encoder.
"""

from collections import OrderedDict
from contextlib import nullcontext

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from pyhealth.models.base_model import BaseModel


class QueryFocusedEHRModel(BaseModel):
    """Diagnosis-conditioned clinical sentence attention model.

    Args:
        dataset: PyHealth SampleDataset created using
            QueryFocusedEHRTask.
        pretrained_model_name: Hugging Face encoder identifier.
        hidden_dim: Dimension for sentence/query projections.
        freeze_encoder: If True, do not update encoder weights.
        max_token_length: Token limit for each sentence/query.
        encoder_batch_size: Maximum number of unique texts encoded
            together in one encoder forward pass.
        cache_embeddings: Whether to cache frozen encoder outputs
            in CPU memory. Defaults to False. Requires
            freeze_encoder=True.
        cache_max_size: Maximum number of unique text embeddings
            stored in the in-memory least-recently-used (LRU) cache.
        dropout: Dropout probability for the classifier.
        encoder: Optional injected encoder for testing.
        tokenizer: Optional injected tokenizer for testing.

    Input:
        sentences: List of sentence lists, one list per patient
            query example.
        query: List of diagnosis-description strings.
        label: Optional binary labels.

    Output:
        logit: [batch_size, 1]
        y_prob: [batch_size, 1]
        loss: Scalar, when labels are supplied.
        y_true: [batch_size, 1], when labels are supplied.
        attention: [batch_size, max_sentences]
        sentence_mask: [batch_size, max_sentences]

    Notes:
        Attention values are learned relevance scores, not
        independently validated clinical relevance labels.

        Caching stores detached, unprojected encoder embeddings
        on CPU. The sentence/query projections and classifier
        remain trainable.

        Caching is unavailable during encoder fine-tuning because
        the encoder's representations change after weight updates.

        The cache is held in memory, is specific to the current
        model instance, and does not persist across program runs.
        Use clear_embedding_cache() to reset cached embeddings.

    Examples:
        >>> from pyhealth.models import QueryFocusedEHRModel
        >>> # Instantiate with an injected lightweight encoder or ClinicalBERT:
        >>> # model = QueryFocusedEHRModel(dataset=sample_dataset, hidden_dim=64)
        >>> # out = model(sentences=[["Fever noted."]], query=["Pneumonia"])
    """

    def __init__(
        self,
        dataset=None,
        pretrained_model_name: str = (
            "emilyalsentzer/Bio_ClinicalBERT"
        ),
        hidden_dim: int = 64,
        freeze_encoder: bool = True,
        max_token_length: int = 128,
        encoder_batch_size: int = 16,
        dropout: float = 0.1,
        encoder: nn.Module | None = None,
        tokenizer=None,
        cache_embeddings: bool = False,
        cache_max_size: int = 50000,
    ):
        super().__init__(dataset=dataset)

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if max_token_length <= 0:
            raise ValueError(
                "max_token_length must be positive."
            )
        if encoder_batch_size <= 0:
            raise ValueError(
                "encoder_batch_size must be positive."
            )
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if not isinstance(cache_embeddings, bool):
            raise TypeError("cache_embeddings must be a boolean.")
        if cache_max_size <= 0:
            raise ValueError("cache_max_size must be positive.")
        if cache_embeddings and not freeze_encoder:
            raise ValueError(
                "Embedding caching requires freeze_encoder=True; "
                "fine-tuning changes the encoder representations."
            )

        if dataset is not None:
            input_keys = set(dataset.input_schema)
            output_keys = set(dataset.output_schema)

            if not {"sentences", "query"} <= input_keys:
                raise ValueError(
                    "Dataset must have 'sentences' "
                    "and 'query' inputs."
                )

            if "label" not in output_keys:
                raise ValueError(
                    "Dataset must provide a 'label' output."
                )

        self.pretrained_model_name = pretrained_model_name
        self.freeze_encoder = freeze_encoder
        self.max_token_length = max_token_length
        self.encoder_batch_size = encoder_batch_size
        self.cache_embeddings = cache_embeddings
        self.cache_max_size = cache_max_size
        self._embedding_cache: OrderedDict[str, Tensor] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(
                pretrained_model_name
            )
        )

        self.encoder = (
            encoder
            if encoder is not None
            else AutoModel.from_pretrained(
                pretrained_model_name
            )
        )

        encoder_hidden_size = self.encoder.config.hidden_size

        self.sentence_projection = nn.Linear(
            encoder_hidden_size, hidden_dim
        )
        self.query_projection = nn.Linear(
            encoder_hidden_size, hidden_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True):
        """Keep a frozen encoder in evaluation mode."""
        super().train(mode)

        if self.freeze_encoder:
            self.encoder.eval()

        return self

    def clear_embedding_cache(self) -> None:
        """Discard cached encoder outputs and reset cache statistics.

        Call this after changing encoder weights or tokenizer behavior.
        The cache is deliberately not included in model checkpoints.
        """
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def embedding_cache_info(self) -> dict[str, int]:
        """Return cache capacity, entries, and hit/miss counts."""
        return {
            "size": len(self._embedding_cache),
            "max_size": self.cache_max_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }

    def _apply(self, fn):
        """Invalidate encoder embeddings when changing module device/dtype."""
        result = super()._apply(fn)
        if hasattr(self, "_embedding_cache"):
            self.clear_embedding_cache()
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Invalidate cached embeddings after restoring model weights."""
        result = super().load_state_dict(
            state_dict, strict=strict, assign=assign
        )
        self.clear_embedding_cache()
        return result

    def _encode_uncached(self, texts: list[str]) -> Tensor:
        """Encode texts in chunks while preserving fine-tuning gradients."""
        chunks = []
        for start in range(0, len(texts), self.encoder_batch_size):
            batch = texts[start:start + self.encoder_batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_token_length,
                return_tensors="pt",
            )
            tokens = {
                key: value.to(self.device)
                for key, value in tokens.items()
            }
            context = torch.no_grad() if self.freeze_encoder else nullcontext()
            with context:
                output = self.encoder(**tokens)
                chunks.append(output.last_hidden_state[:, 0, :])
        return torch.cat(chunks, dim=0)

    def _encode_texts(self, texts: list[str]) -> Tensor:
        """Encode texts, reusing bounded CPU embeddings if enabled.

        Only frozen encoder outputs are cached. Projection layers are never
        cached, so gradient flow into the trainable layers is preserved.
        """
        if not texts:
            raise ValueError("Cannot encode an empty list of texts.")

        if not self.cache_embeddings:
            return self._encode_uncached(texts)

        # Preserve the original input order while encoding each missing
        # unique string only once (including duplicates within this batch).
        unique_texts = list(dict.fromkeys(texts))
        embeddings = {}
        missing = []
        for text in unique_texts:
            cached = self._embedding_cache.get(text)
            if cached is None:
                self._cache_misses += 1
                missing.append(text)
            else:
                self._cache_hits += 1
                self._embedding_cache.move_to_end(text)
                embeddings[text] = cached.to(self.device)

        if missing:
            newly_encoded = self._encode_uncached(missing)
            for text, embedding in zip(missing, newly_encoded):
                embeddings[text] = embedding
                # Keep only detached CPU representations in the cache.
                self._embedding_cache[text] = embedding.detach().cpu().clone()
                self._embedding_cache.move_to_end(text)
                if len(self._embedding_cache) > self.cache_max_size:
                    self._embedding_cache.popitem(last=False)

        return torch.stack([embeddings[text] for text in texts], dim=0)

    def forward(
        self,
        sentences: list[list[str]],
        query: list[str],
        label: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Predict diagnosis presence and sentence relevance."""
        if not sentences or not query:
            raise ValueError(
                "sentences and query cannot be empty."
            )

        # Accept an individual unbatched example.
        if isinstance(query, str):
            query = [query]

        if isinstance(sentences[0], str):
            sentences = [sentences]

        batch_size = len(query)

        if len(sentences) != batch_size:
            raise ValueError(
                "sentences and query batch sizes must match."
            )

        if any(not history for history in sentences):
            raise ValueError(
                "Every example needs at least one sentence."
            )

        if any(
            not isinstance(text, str) or not text.strip()
            for history in sentences
            for text in history
        ):
            raise ValueError(
                "All sentences must be nonempty strings."
            )

        max_sentences = max(
            len(history) for history in sentences
        )

        # Flatten valid sentence inputs before encoding.
        flat_sentences = [
            sentence
            for history in sentences
            for sentence in history
        ]

        # Encode historical sentences and diagnosis queries.
        all_embeddings = self._encode_texts(flat_sentences + query)
        sentence_embeddings = all_embeddings[:len(flat_sentences)]
        query_embeddings = all_embeddings[len(flat_sentences):]

        sentence_features = self.sentence_projection(
            sentence_embeddings
        )
        query_features = self.query_projection(
            query_embeddings
        )

        hidden_dim = sentence_features.shape[-1]

        # Reconstruct padded sentence batches.
        padded = sentence_features.new_zeros(
            batch_size, max_sentences, hidden_dim
        )

        sentence_mask = torch.zeros(
            batch_size,
            max_sentences,
            dtype=torch.bool,
            device=self.device,
        )

        offset = 0
        for i, history in enumerate(sentences):
            count = len(history)

            padded[i, :count] = sentence_features[
                offset:offset + count
            ]
            sentence_mask[i, :count] = True

            offset += count

        # Query-conditioned dot-product attention.
        attention_logits = torch.sum(
            padded * query_features.unsqueeze(1),
            dim=-1,
        )

        # Padding must never receive attention weight.
        attention_logits = attention_logits.masked_fill(
            ~sentence_mask,
            torch.finfo(attention_logits.dtype).min,
        )

        attention = torch.softmax(
            attention_logits, dim=1
        )

        # Weighted patient-history representation.
        context = torch.sum(
            attention.unsqueeze(-1) * padded,
            dim=1,
        )

        combined = torch.cat(
            [context, query_features],
            dim=-1,
        )

        logit = self.classifier(combined)
        y_prob = torch.sigmoid(logit)

        result = {
            "logit": logit,
            "y_prob": y_prob,
            "attention": attention,
            "sentence_mask": sentence_mask,
        }

        if label is not None:
            y_true = torch.as_tensor(
                label,
                dtype=logit.dtype,
                device=logit.device,
            ).reshape(-1, 1)

            if y_true.shape != logit.shape:
                raise ValueError(
                    "Label batch size does not match logits."
                )

            result["loss"] = self.loss_fn(
                logit, y_true
            )
            result["y_true"] = y_true

        return result

    @torch.no_grad()
    def extract_top_k(
        self,
        sentences: list[list[str]],
        query: list[str],
        k: int = 5,
    ) -> list[list[dict]]:
        """Rank and return the most relevant sentences.

        Args:
            sentences: Batch of historical sentence lists.
            query: Batch of diagnosis-description strings.
            k: Maximum number of sentences per example.

        Returns:
            Per-example lists of dictionaries containing
            sentence text, original index, and attention score.
        """
        if k <= 0:
            raise ValueError("k must be positive.")

        if isinstance(query, str):
            query = [query]

        if sentences and isinstance(sentences[0], str):
            sentences = [sentences]

        was_training = self.training
        self.eval()

        try:
            outputs = self.forward(
                sentences=sentences,
                query=query,
            )

            attention = outputs["attention"]
            summaries = []

            for i, history in enumerate(sentences):
                scores = attention[i, :len(history)]

                ranked_indices = torch.argsort(
                    scores, descending=True
                )[:k]

                summaries.append([
                    {
                        "sentence": history[j],
                        "index": j,
                        "score": float(scores[j].item()),
                    }
                    for j in ranked_indices.tolist()
                ])

            return summaries
        finally:
            self.train(was_training)
