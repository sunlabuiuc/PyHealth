"""RASC cross-encoder for value set completion.

Author: Wenchang Qi (NetID: qi14)
Paper: Retrieve, Then Classify: Corpus-Grounded Automation of Clinical Value
    Set Authoring (Mukherjee et al., 2026)
Paper link: https://arxiv.org/abs/2604.14616
Description: The paper's best classifier. A value set title and a candidate
    code's display name are read together as one sentence pair by a BERT
    encoder (SapBERT in the paper), and a linear head on the pooled output
    scores whether the code belongs to the value set. Follows
    model_training/train_cross_encoder.py in https://github.com/mukhes3/RASC,
    and loads the authors' released weights. Optionally adds the retrieval
    score and code system to the logit (late fusion; not in the paper).
"""

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from ..datasets import SampleDataset
from ..processors import TextProcessor
from .base_model import BaseModel

SAPBERT = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
RELEASED_WEIGHTS = "sumitmukherjee2/RASC_cross_encoder"
# Latest commit of RELEASED_WEIGHTS (2026-04-19); pins the released checkpoint.
RELEASED_REVISION = "bce2e4d94a18cbc2c0e34c248dfc3b5b9731a1e9"
_DROPOUT_SETTINGS = (
    "hidden_dropout_prob",
    "attention_probs_dropout_prob",
    "dropout",
    "attention_dropout",
)


class RASCCrossEncoder(BaseModel):
    """Sentence-pair cross-encoder that scores a candidate code for a value set.

    The classifier of Retrieval-Augmented Set Completion (RASC): given a
    value set title and one retrieved candidate code, predict whether the
    code belongs to the value set. Title and display name are tokenized as
    one pair, ``[CLS] title [SEP] display [SEP]``, so every transformer
    layer attends across both. An MLP on separately computed title and code
    embeddings cannot do that, which is the paper's explanation for the
    cross-encoder's higher precision.

    Works with :class:`~pyhealth.tasks.ValueSetPairClassification`, and with
    any dataset that has two ``"text"`` inputs and a binary label.

    Late fusion (not in the paper): every other input of the dataset, e.g.
    the task's ``score`` and ``code_system``, is flattened, concatenated and
    mapped by a bias-free linear layer to a term added to the text logit.
    This equals a linear head on the pooled vector concatenated with those
    features. Its weights start at zero, so an untrained fused model scores
    exactly as the text-only one, and released text-only weights stay valid.

    Differences from the authors' training script: gradient accumulation
    belongs to the training loop, not the model, so use a batch size of 64
    with :class:`~pyhealth.trainer.Trainer` in place of 16 with 4
    accumulation steps; mixed precision is ``mixed_precision=True``
    (bfloat16 in place of fp16 with a gradient scaler).

    Args:
        dataset: Dataset with the two text inputs and one binary label.
        model_name: Hugging Face encoder name or path. Defaults to SapBERT,
            the paper's initialization. ``RELEASED_WEIGHTS`` loads the
            authors' fine-tuned cross-encoder, head included, from the
            authors' own Hugging Face repository.
        revision: Branch, tag or commit of ``model_name`` on the Hugging
            Face Hub. ``None`` (default) loads the latest. Pass
            ``RELEASED_REVISION`` with ``RELEASED_WEIGHTS`` to pin the
            checkpoint evaluated in the example. Ignored for local paths.
        title_key: Input holding the value set title (first segment).
        candidate_key: Input holding the candidate's display name (second
            segment).
        fusion_keys: Tensor inputs added to the logit. ``None`` (default)
            uses every input other than the two text ones; ``()`` turns
            fusion off.
        max_length: Tokens per pair, special tokens included. The longer
            segment is truncated first. Default is 128, as in the paper.
        dropout: Dropout inside the encoder and before the head. Default is
            0.1, as in the authors' script.
        gradient_checkpointing: Recompute activations in the backward pass
            to save memory, as the authors' script does.
        mixed_precision: Run the encoder in 16-bit floats on a GPU
            (bfloat16 autocast, which needs no gradient scaler), about
            twice as fast; the authors' script uses fp16. Ignored on CPU.
            Logits and loss stay in 32 bits. The unit tests run on CPU,
            so the 16-bit path is covered only by the GPU runs reported
            in ``examples/vsac_value_set_completion_rasc.py``.

    Raises:
        ValueError: If the label is not binary, a key is not an input of the
            dataset, a text key is not a ``"text"`` input, a fusion key is
            one, or ``max_length`` is outside what the encoder reads.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import RASCCrossEncoder
        >>> samples = [
        ...     {
        ...         "patient_id": "vs0",
        ...         "title_text": "Type 2 diabetes mellitus",
        ...         "candidate_text": "Type 2 diabetes mellitus without complications",
        ...         "score": [0.91],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "vs0",
        ...         "title_text": "Type 2 diabetes mellitus",
        ...         "candidate_text": "Gestational diabetes mellitus",
        ...         "score": [0.74],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples,
        ...     input_schema={
        ...         "title_text": "text", "candidate_text": "text", "score": "tensor",
        ...     },
        ...     output_schema={"label": "binary"},
        ... )
        >>> model = RASCCrossEncoder(dataset)  # doctest: +SKIP
        >>> out = model(  # doctest: +SKIP
        ...     title_text=["Type 2 diabetes mellitus"],
        ...     candidate_text=["Gestational diabetes mellitus"],
        ...     score=torch.tensor([[0.74]]),
        ...     label=torch.tensor([[0.0]]),
        ... )
        >>> out["y_prob"].shape  # doctest: +SKIP
        torch.Size([1, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        model_name: str = SAPBERT,
        revision: str | None = None,
        title_key: str = "title_text",
        candidate_key: str = "candidate_text",
        fusion_keys: Sequence[str] | None = None,
        max_length: int = 128,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        mixed_precision: bool = False,
    ) -> None:
        super().__init__(dataset=dataset)
        if self.mode != "binary":
            raise ValueError(
                f"RASCCrossEncoder needs one binary label, got mode {self.mode!r}."
            )
        self.label_key = self.label_keys[0]
        processors = dataset.input_processors
        for key in (title_key, candidate_key):
            if not isinstance(processors.get(key), TextProcessor):
                raise ValueError(  # noqa: TRY004
                    f"{key!r} must be a 'text' input of the dataset; inputs are "
                    f"{list(processors)}."
                )
        if fusion_keys is None:
            fusion_keys = [
                key
                for key in self.feature_keys
                if key not in (title_key, candidate_key)
            ]
        for key in fusion_keys:
            if key not in processors or isinstance(processors[key], TextProcessor):
                raise ValueError(f"Fusion key {key!r} must be a non-text input.")
        self.model_name = model_name
        self.revision = revision
        self.title_key = title_key
        self.candidate_key = candidate_key
        self.fusion_keys = list(fusion_keys)

        # Encoders name their dropout settings differently (BERT:
        # hidden_dropout_prob; DistilBERT: dropout), so set those it has.
        config = AutoConfig.from_pretrained(model_name, num_labels=1, revision=revision)
        for name in _DROPOUT_SETTINGS:
            if hasattr(config, name):
                setattr(config, name, dropout)
        if getattr(config, "classifier_dropout", None) is not None:
            config.classifier_dropout = dropout
        # Encoder plus a one-logit head on the pooled output, the layout of
        # the released weights, so they load head included.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config, revision=revision
        )
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

        limit = getattr(config, "max_position_embeddings", 512)
        if not 4 < max_length <= limit:
            raise ValueError(
                f"max_length must be between 5 and {limit} for {model_name}, "
                f"got {max_length}."
            )
        self.max_length = max_length
        self.mixed_precision = mixed_precision

        self.fusion_dim = sum(self._input_size(key) for key in self.fusion_keys)
        self.fusion = None
        if self.fusion_dim:
            self.fusion = nn.Linear(self.fusion_dim, 1, bias=False)
            nn.init.zeros_(self.fusion.weight)

    def _input_size(self, key: str) -> int:
        """Number of values one sample holds for a fusion input.

        Args:
            key: A fusion key.

        Returns:
            The flattened length of the input in the dataset's first sample.
        """
        value = torch.as_tensor(self.dataset[0][key])
        return max(int(value.numel()), 1)

    def encode_pairs(
        self, titles: Sequence[str], candidates: Sequence[str]
    ) -> torch.Tensor:
        """Scores title-candidate pairs from their text alone.

        Args:
            titles: Value set titles, one per pair.
            candidates: Candidate display names, one per pair.

        Returns:
            Text logits of shape ``(len(titles), 1)``, before fusion.

        Raises:
            ValueError: If the two lists differ in length.

        Examples:
            >>> model.encode_pairs(  # doctest: +SKIP
            ...     ["Hypertension"], ["Essential hypertension"]
            ... ).shape
            torch.Size([1, 1])
        """
        if len(titles) != len(candidates):
            raise ValueError(
                f"Got {len(titles)} titles for {len(candidates)} candidates."
            )
        batch = self.tokenizer(
            list(titles),
            list(candidates),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.mixed_precision and self.device.type == "cuda",
        ):
            logits = self.model(**batch).logits
        return logits.float()

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Scores a batch of (title, candidate) pairs.

        Args:
            **kwargs: The batch: lists of strings under the two text keys, a
                tensor of shape ``(batch, ...)`` under each fusion key, and
                optionally the label, of shape ``(batch, 1)``.

        Returns:
            ``logit`` and ``y_prob`` of shape ``(batch, 1)``, and, when the
            label is given, ``loss`` (BCE, weighted by the label's
            ``pos_weight`` if it has one) and ``y_true``.
        """
        logits = self.encode_pairs(kwargs[self.title_key], kwargs[self.candidate_key])
        if self.fusion is not None:
            extra = torch.cat(
                [
                    torch.as_tensor(kwargs[key], dtype=torch.float32)
                    .to(self.device)
                    .reshape(logits.shape[0], -1)
                    for key in self.fusion_keys
                ],
                dim=1,
            )
            logits = logits + self.fusion(extra)
        results = {
            "logit": logits,
            "y_prob": self.label_processor.activation(logits),
        }
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            results["loss"] = self.label_processor.loss(logits, y_true)
            results["y_true"] = y_true
        return results
