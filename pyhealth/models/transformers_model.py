import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..datasets import SampleDataset
from .base_model import BaseModel

DEFAULT_MAX_LENGTH = 256
POOLING_MODES = ("auto", "cls", "mean")
DROPOUT_SETTINGS = (
    "hidden_dropout_prob",
    "attention_probs_dropout_prob",
    "dropout",
    "attention_dropout",
)


class TransformersModel(BaseModel):
    """Text classifier on top of any Hugging Face encoder.

    Each text is tokenized, encoded and pooled to one vector, and a linear
    head maps that vector to the task's label (binary, multiclass or
    multilabel).

    Long texts: an encoder reads at most ``max_length`` tokens at a time
    (256 by default, as before; up to 512 for BERT with
    ``max_length=None``). By default only the first window is read, as
    before.
    Set ``max_chunks`` above 1 (or to ``None`` for no limit) to split each
    text into consecutive windows, encode every window, and max-pool the
    window vectors, so the whole document reaches the classifier. All
    windows of a batch are encoded together, so memory grows with the
    number of windows.

    Args:
        dataset: The dataset to train the model. It must have exactly one
            input field (the text) and one label.
        model_name: Name or path of a Hugging Face encoder, e.g.
            ``"emilyalsentzer/Bio_ClinicalBERT"``.
        dropout: Dropout inside the encoder (whichever of its dropout
            settings it has) and before the linear head. Default is 0.1.
        max_length: Tokens per window, special tokens included. Default
            is 256, as in earlier releases (lowered to the encoder's limit
            if that is smaller). ``None`` uses the encoder's limit (512 for
            BERT), which reads more of each text but costs more memory and
            time: attention grows with the square of the window length.
        max_chunks: Windows read per text. Default is 1: read the first
            ``max_length`` tokens and drop the rest. ``None`` reads the
            whole text. Needs a fast tokenizer when not 1.
        pooling: How a window becomes one vector. ``"auto"`` (default)
            uses the encoder's pooler output when it has one (as earlier
            releases did) and the first token's hidden state otherwise;
            ``"cls"`` uses the first token's hidden state; ``"mean"``
            averages the hidden states of the non-padding tokens.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import TransformersModel
        >>> samples = [
        ...     {"patient_id": "p0", "text": "chest pain and dyspnea", "label": 1},
        ...     {"patient_id": "p1", "text": "routine follow-up visit", "label": 0},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples, input_schema={"text": "text"},
        ...     output_schema={"label": "binary"},
        ... )
        >>> model = TransformersModel(  # doctest: +SKIP
        ...     dataset,
        ...     model_name="emilyalsentzer/Bio_ClinicalBERT",
        ...     max_length=None,  # the encoder's limit: 512 tokens
        ...     max_chunks=None,  # read whole notes in 512-token windows
        ... )
        >>> out = model(  # doctest: +SKIP
        ...     text=["chest pain and dyspnea"], label=torch.tensor([[1.0]])
        ... )
        >>> out["y_prob"].shape  # doctest: +SKIP
        torch.Size([1, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        model_name: str,
        dropout: float = 0.1,
        max_length: int | None = DEFAULT_MAX_LENGTH,
        max_chunks: int | None = 1,
        pooling: str = "auto",
    ):
        super().__init__(
            dataset=dataset,
        )
        self.model_name = model_name
        # Encoders name their dropout settings differently (BERT:
        # hidden_dropout_prob; DistilBERT: dropout), so set those it has.
        config = AutoConfig.from_pretrained(model_name)
        for name in DROPOUT_SETTINGS:
            if hasattr(config, name):
                setattr(config, name, dropout)
        self.model = AutoModel.from_pretrained(model_name, config=config)
        assert (
            len(self.feature_keys) == 1
        ), "Only one feature key is supported if Transformers is initialized"
        self.feature_key = self.feature_keys[0]
        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported if RNN is initialized"
        self.label_key = self.label_keys[0]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        limit = self._encoder_max_length()
        if max_length is None:
            max_length = limit
        elif max_length == DEFAULT_MAX_LENGTH:
            # The default must load every encoder, including short ones.
            max_length = min(max_length, limit)
        if not 2 < max_length <= limit:
            raise ValueError(
                f"max_length must be between 3 and {limit} for {model_name}, "
                f"got {max_length}"
            )
        if max_chunks is not None and max_chunks < 1:
            raise ValueError(f"max_chunks must be at least 1 or None, got {max_chunks}")
        if max_chunks != 1 and not self.tokenizer.is_fast:
            raise ValueError("max_chunks other than 1 needs a fast tokenizer")
        if pooling not in POOLING_MODES:
            raise ValueError(f"pooling must be one of {POOLING_MODES}, got {pooling!r}")
        self.max_length = max_length
        self.max_chunks = max_chunks
        self.pooling = pooling

        output_size = self.get_output_size()
        hidden_dim = self.model.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_size)

    def _encoder_max_length(self) -> int:
        """The longest input the encoder accepts, in tokens.

        The smaller of the tokenizer's ``model_max_length`` and the
        encoder's ``max_position_embeddings``. Neither alone is safe: a
        RoBERTa encoder has 514 position embeddings but reads at most 512
        tokens (two positions are offset by padding), and a tokenizer can
        declare a shorter limit than the encoder's positions when the model
        was trained on shorter inputs, so the later positions are untrained.
        """
        limits = [
            self.tokenizer.model_max_length,
            getattr(self.model.config, "max_position_embeddings", None),
        ]
        # Tokenizers without a configured limit report a huge sentinel value.
        limits = [n for n in limits if isinstance(n, int) and 0 < n < 1_000_000]
        return min(limits) if limits else 512

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encodes texts into one vector each, before dropout and the head.

        Args:
            texts: Raw texts, one per sample.

        Returns:
            A float tensor of shape (len(texts), hidden size).

        Examples:
            >>> model.encode(["chest pain and dyspnea"]).shape  # doctest: +SKIP
            torch.Size([1, 768])
        """
        chunked = self.max_chunks != 1
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_overflowing_tokens=chunked,
        )
        # Row i of the batch is a window of text owner[i]; windows of a text
        # are consecutive and in reading order.
        owner = batch.pop("overflow_to_sample_mapping", None)
        if owner is None:
            owner = torch.arange(len(texts))
        if chunked and self.max_chunks is not None:
            _, counts = torch.unique_consecutive(owner, return_counts=True)
            rank = torch.cat([torch.arange(int(n)) for n in counts])
            keep = rank < self.max_chunks
            batch = {key: value[keep] for key, value in batch.items()}
            owner = owner[keep]
        batch = {key: value.to(self.device) for key, value in batch.items()}

        output = self.model(**batch)
        pooler = getattr(output, "pooler_output", None)
        if self.pooling == "auto" and pooler is not None:
            windows = pooler
        elif self.pooling == "mean":
            mask = batch["attention_mask"].unsqueeze(-1).to(output.last_hidden_state)
            summed = (output.last_hidden_state * mask).sum(dim=1)
            windows = summed / mask.sum(dim=1).clamp(min=1)
        else:
            windows = output.last_hidden_state[:, 0]

        if windows.size(0) == len(texts):
            return windows
        owner = owner.to(self.device).unsqueeze(1).expand_as(windows)
        texts_out = windows.new_full((len(texts), windows.size(1)), float("-inf"))
        return texts_out.scatter_reduce(0, owner, windows, reduce="amax")

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward propagation."""
        embeddings = self.encode(kwargs[self.feature_key])
        logits = self.fc(self.dropout(embeddings))
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }


if __name__ == "__main__":
    from pyhealth.datasets import MedicalTranscriptionsDataset, get_dataloader

    base_dataset = MedicalTranscriptionsDataset(
        root="/srv/local/data/zw12/raw_data/MedicalTranscriptions"
    )

    sample_dataset = base_dataset.set_task()

    train_loader = get_dataloader(sample_dataset, batch_size=16, shuffle=True)

    model = TransformersModel(
        dataset=sample_dataset,
        feature_keys=["transcription"],
        label_key="label",
        mode="multiclass",
        model_name="emilyalsentzer/Bio_ClinicalBERT",
    )

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()
