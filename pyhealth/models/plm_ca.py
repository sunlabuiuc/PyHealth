"""PLM-CA: pretrained language model with class-wise cross-attention.

Contributors: Anvay Patel (apatel3269), Juri Moon (jmoon326)
Paper: Edin et al., "An Unsupervised Approach to Achieve Supervised-Level
    Explainability in Healthcare Records", EMNLP 2024.
Paper link: https://arxiv.org/abs/2406.08958

Description:
    Automated medical coding model used in the paper (Appendix A). A pretrained
    transformer encodes a clinical note in fixed-size chunks. A class-wise
    cross-attention decoder then gives every medical code its own learnable
    query, which attends over all token representations to produce one logit
    per code. The decoder's attention matrix is returned by ``forward`` so that
    attention-based explanations (Attention, AttInGrad) can be computed.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from ..datasets import SampleDataset
from .base_model import BaseModel


class ClassWiseCrossAttention(nn.Module):
    """Class-wise cross-attention decoder from Edin et al. (2024), Appendix A.

    Given token representations ``H`` of shape ``[batch, N, D]``, it computes::

        K = H @ W_key
        V = H @ W_value
        A_j = softmax(C_j @ K^T)              # attention of code j over tokens
        logit_j = layernorm(A_j @ V) @ w_out

    where ``C`` holds one learnable query vector per code.

    Args:
        hidden_dim: Size ``D`` of the incoming token representations.
        num_classes: Number of medical codes ``J``.
        dropout: Dropout applied to the pooled class representations.

    Example:
        >>> import torch
        >>> decoder = ClassWiseCrossAttention(hidden_dim=8, num_classes=3)
        >>> hidden = torch.randn(2, 5, 8)
        >>> mask = torch.ones(2, 5, dtype=torch.long)
        >>> logits, attention = decoder(hidden, mask)
        >>> logits.shape, attention.shape
        (torch.Size([2, 3]), torch.Size([2, 3, 5]))
    """

    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.class_queries = nn.Parameter(torch.empty(num_classes, hidden_dim))
        nn.init.normal_(self.class_queries, std=hidden_dim**-0.5)
        # nn.Softmax (not F.softmax) so interpretability hooks can attach.
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pools token representations into one logit per code.

        Args:
            hidden_states: Token representations of shape ``[batch, N, D]``.
            attention_mask: ``[batch, N]`` tensor, 1 for real tokens and 0
                for padding.

        Returns:
            A tuple ``(logits, attention)`` with shapes ``[batch, J]`` and
            ``[batch, J, N]``. Padding positions get zero attention.
        """
        keys = self.key(hidden_states)
        values = self.value(hidden_states)
        scores = torch.einsum("jd,bnd->bjn", self.class_queries, keys)
        padding = attention_mask[:, None, :] == 0
        scores = scores.masked_fill(padding, torch.finfo(scores.dtype).min)
        attention = self.softmax(scores)
        pooled = torch.bmm(attention, values)
        pooled = self.layer_norm(self.dropout(pooled))
        logits = self.output(pooled).squeeze(-1)
        return logits, attention


class PLMCA(BaseModel):
    """Pretrained language model with class-wise cross-attention (PLM-CA).

    This is the medical coding model from Edin et al. (2024). It replaces the
    label-wise attention of PLM-ICD (Huang et al., 2022) with standard
    cross-attention, which the authors found more stable to train.

    Long notes are tokenized once, truncated to ``max_length`` tokens, padded
    to a multiple of ``chunk_size`` and encoded chunk by chunk. The chunk
    outputs are concatenated before the decoder, so every code can attend to
    every token in the note.

    The model expects a task with exactly one text input (for example
    ``{"text": "text"}``) and one multilabel output, such as
    :class:`~pyhealth.tasks.MIMIC3ICD9Coding`.

    Args:
        dataset: Sample dataset used to read the input/output schemas and the
            number of codes.
        model_name: Hugging Face model id or local path of the encoder and
            tokenizer, for example ``"emilyalsentzer/Bio_ClinicalBERT"``.
        chunk_size: Number of tokens encoded together in one encoder pass.
            Must not exceed the encoder's maximum position embeddings.
        max_length: Maximum number of tokens kept per note. The paper uses
            6000.
        dropout: Dropout applied in the decoder. The paper uses 0.2.

    Raises:
        ValueError: If the task does not have exactly one input and one
            multilabel output, or if ``chunk_size`` or ``max_length`` is
            invalid for the encoder.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {"patient_id": "p0", "text": "chest pain", "icd_codes": ["428.0"]},
        ...     {"patient_id": "p1", "text": "sepsis", "icd_codes": ["038.9"]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"text": "text"},
        ...     output_schema={"icd_codes": "multilabel"},
        ... )
        >>> model = PLMCA(dataset, model_name="emilyalsentzer/Bio_ClinicalBERT")
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> out = model(**batch)
        >>> sorted(out.keys())
        ['attention', 'logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        model_name: str,
        chunk_size: int = 128,
        max_length: int = 6000,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(dataset=dataset)
        if len(self.feature_keys) != 1:
            raise ValueError(
                "PLMCA expects exactly one text input feature, got "
                f"{self.feature_keys}."
            )
        if len(self.label_keys) != 1:
            raise ValueError(
                f"PLMCA expects exactly one label key, got {self.label_keys}."
            )
        if self.mode != "multilabel":
            raise ValueError(
                f"PLMCA supports multilabel tasks only, got mode={self.mode!r}."
            )
        if chunk_size <= 0 or max_length <= 0:
            raise ValueError("chunk_size and max_length must be positive.")

        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        max_positions = getattr(self.encoder.config, "max_position_embeddings", None)
        if max_positions is not None and chunk_size > max_positions:
            raise ValueError(
                f"chunk_size={chunk_size} exceeds the encoder's "
                f"max_position_embeddings={max_positions}."
            )

        hidden_dim = self.encoder.config.hidden_size
        self.decoder = ClassWiseCrossAttention(
            hidden_dim=hidden_dim,
            num_classes=self.get_output_size(),
            dropout=dropout,
        )

    def get_input_embeddings(self) -> nn.Module:
        """Returns the encoder's token embedding layer.

        Interpretability methods use it to turn token ids into the embedding
        matrix ``X`` that gradients are taken with respect to.
        """
        return self.encoder.get_input_embeddings()

    def tokenize(self, texts: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes notes and pads them to a multiple of ``chunk_size``.

        Args:
            texts: One note or a batch of notes.

        Returns:
            ``(input_ids, attention_mask)``, both of shape ``[batch, N]`` on
            the model's device, where ``N`` is a multiple of ``chunk_size``.
        """
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        remainder = (-input_ids.shape[1]) % self.chunk_size
        if remainder:
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids = F.pad(input_ids, (0, remainder), value=pad_id)
            attention_mask = F.pad(attention_mask, (0, remainder), value=0)
        return input_ids.to(self.device), attention_mask.to(self.device)

    def _encode(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encodes embeddings chunk by chunk and re-joins the chunks.

        Args:
            inputs_embeds: ``[batch, N, E]`` token embeddings, ``N`` a
                multiple of ``chunk_size``.
            attention_mask: ``[batch, N]`` padding mask.

        Returns:
            ``[batch, N, D]`` contextualized token representations.
        """
        batch_size, seq_len, embed_dim = inputs_embeds.shape
        if seq_len % self.chunk_size != 0:
            raise ValueError(
                f"Sequence length {seq_len} is not a multiple of "
                f"chunk_size={self.chunk_size}; use PLMCA.tokenize()."
            )
        num_chunks = seq_len // self.chunk_size
        chunk_embeds = inputs_embeds.reshape(
            batch_size * num_chunks, self.chunk_size, embed_dim
        )
        chunk_mask = attention_mask.reshape(batch_size * num_chunks, self.chunk_size)
        # Short notes in a batch end in chunks that are pure padding. Letting
        # each chunk attend to its first position avoids fully masked rows
        # (a NaN risk in some attention kernels). Real chunks are unchanged
        # because padding is on the right, and the decoder still ignores
        # every padded position.
        chunk_mask = chunk_mask.clone()
        chunk_mask[:, 0] = 1
        outputs = self.encoder(inputs_embeds=chunk_embeds, attention_mask=chunk_mask)
        return outputs.last_hidden_state.reshape(batch_size, seq_len, -1)

    def forward_from_embedding(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Runs the model from token embeddings instead of raw text.

        Gradient-based explanations (InputXGrad, AttInGrad) and adversarial
        training (IGR, PGD) need gradients with respect to these embeddings.

        Args:
            inputs_embeds: ``[batch, N, E]`` token embeddings.
            attention_mask: ``[batch, N]`` padding mask.
            **kwargs: May contain the label key with ``[batch, J]`` targets.

        Returns:
            A dict with ``logit`` ``[batch, J]``, ``y_prob`` ``[batch, J]``
            and ``attention`` ``[batch, J, N]``. If labels are given, it also
            has ``loss`` (scalar) and ``y_true`` ``[batch, J]``.
        """
        hidden_states = self._encode(inputs_embeds, attention_mask)
        logits, attention = self.decoder(hidden_states, attention_mask)
        results = {
            "logit": logits,
            "y_prob": self.prepare_y_prob(logits),
            "attention": attention,
        }
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            results["loss"] = self.get_loss_function()(logits, y_true)
            results["y_true"] = y_true
        return results

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Predicts medical codes for a batch of notes.

        Args:
            **kwargs: A dataloader batch. Must contain the text feature key
                and may contain the label key.

        Returns:
            Same dict as :meth:`forward_from_embedding`.
        """
        input_ids, attention_mask = self.tokenize(kwargs[self.feature_key])
        inputs_embeds = self.get_input_embeddings()(input_ids)
        return self.forward_from_embedding(inputs_embeds, attention_mask, **kwargs)
