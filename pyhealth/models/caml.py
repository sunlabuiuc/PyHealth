"""CAML: convolutional attention network for multi-label classification.

Contributors: Juri Moon (jmoon326)
Paper: Mullenbach et al., "Explainable Prediction of Medical Codes from
    Clinical Text", NAACL 2018.
Paper link: https://arxiv.org/abs/1802.05695

Description:
    Automated medical coding model. Word embeddings of a clinical note are
    passed through one convolutional layer with tanh, then every medical code
    attends over all positions with its own attention vector and turns the
    attention-pooled features into one logit. The per-code attention matrix
    is returned by ``forward`` so that attention-based explanations
    (Attention, AttInGrad) can be computed exactly as for
    :class:`~pyhealth.models.PLMCA`.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..datasets import SampleDataset
from .base_model import BaseModel


class CAML(BaseModel):
    """Convolutional Attention for Multi-Label classification (CAML).

    For token embeddings ``X`` of shape ``[batch, N, E]`` it computes::

        H = tanh(conv1d(X))                   # [batch, N, F], same length N
        A_j = softmax(H @ u_j)                # attention of code j over tokens
        v_j = A_j @ H                         # [batch, F]
        logit_j = sum(beta_j * v_j) + b_j

    The note is tokenized with a Hugging Face tokenizer (the same one PLM-CA
    uses), so token offsets of explanations match those of
    :class:`~pyhealth.models.PLMCA`. By default the embedding matrix is
    initialised from that model's word embeddings and fine-tuned. The
    convolution is padded so that its output at position ``n`` is centred
    on token ``n``.

    The model expects a task with exactly one text input (for example
    ``{"text": "text"}``) and one multilabel output, such as
    :class:`~pyhealth.tasks.MIMIC3ICD9Coding`.

    Args:
        dataset: Sample dataset used to read the input/output schemas and the
            number of codes.
        model_name: Hugging Face model id or local path of the tokenizer and,
            if ``pretrained_embeddings``, of the model whose word embeddings
            initialise the embedding layer.
        max_length: Maximum number of tokens kept per note. The paper uses
            2500 words.
        num_filters: Number of convolution filters ``F``. The paper uses 50.
        kernel_size: Convolution width. The paper uses 10.
        dropout: Dropout applied to the embeddings. The paper uses 0.2.
        pretrained_embeddings: If ``True``, copy the word-embedding matrix of
            ``model_name``; otherwise initialise it randomly.
        embed_dim: Embedding size. ``None`` uses the size of ``model_name``'s
            embeddings. With pretrained embeddings it must match that size.

    Raises:
        ValueError: If the task does not have exactly one input and one
            multilabel output, or if a size or the dropout is invalid.

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
        >>> model = CAML(dataset, model_name="emilyalsentzer/Bio_ClinicalBERT")
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> out = model(**batch)
        >>> sorted(out.keys())
        ['attention', 'logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        model_name: str,
        max_length: int = 4096,
        num_filters: int = 50,
        kernel_size: int = 10,
        dropout: float = 0.2,
        pretrained_embeddings: bool = True,
        embed_dim: int | None = None,
    ) -> None:
        super().__init__(dataset=dataset)
        if len(self.feature_keys) != 1:
            raise ValueError(
                "CAML expects exactly one text input feature, got "
                f"{self.feature_keys}."
            )
        if len(self.label_keys) != 1:
            raise ValueError(
                f"CAML expects exactly one label key, got {self.label_keys}."
            )
        if self.mode != "multilabel":
            raise ValueError(
                f"CAML supports multilabel tasks only, got mode={self.mode!r}."
            )
        if min(max_length, num_filters, kernel_size) <= 0:
            raise ValueError(
                "max_length, num_filters and kernel_size must be positive."
            )
        if embed_dim is not None and embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.model_name = model_name
        self.max_length = max_length
        self.kernel_size = kernel_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        pad_id = self.tokenizer.pad_token_id
        if pretrained_embeddings:
            # Only the word embeddings are kept; the transformer is dropped.
            weight = AutoModel.from_pretrained(model_name).get_input_embeddings()
            weight = weight.weight.detach().clone()
            if embed_dim is not None and embed_dim != weight.shape[1]:
                raise ValueError(
                    f"embed_dim={embed_dim} does not match the pretrained "
                    f"embedding size {weight.shape[1]}."
                )
            self.embedding = nn.Embedding.from_pretrained(
                weight, freeze=False, padding_idx=pad_id
            )
        else:
            if embed_dim is None:
                embed_dim = AutoConfig.from_pretrained(model_name).hidden_size
            self.embedding = nn.Embedding(
                len(self.tokenizer), embed_dim, padding_idx=pad_id
            )
        embed_dim = self.embedding.embedding_dim

        num_classes = self.label_processor.output_size()
        self.embed_dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size)
        self.attention_vectors = nn.Parameter(torch.empty(num_classes, num_filters))
        self.output_weights = nn.Parameter(torch.empty(num_classes, num_filters))
        self.output_bias = nn.Parameter(torch.zeros(num_classes))
        for weight in (self.conv.weight, self.attention_vectors, self.output_weights):
            nn.init.xavier_uniform_(weight)
        # nn.Softmax (not F.softmax) so interpretability hooks can attach.
        self.softmax = nn.Softmax(dim=-1)

    def get_input_embeddings(self) -> nn.Module:
        """Returns the token embedding layer.

        Interpretability methods use it to turn token ids into the embedding
        matrix ``X`` that gradients are taken with respect to.
        """
        return self.embedding

    def tokenize(self, texts: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes notes, truncating to ``max_length`` and padding the batch.

        Args:
            texts: One note or a batch of notes.

        Returns:
            ``(input_ids, attention_mask)``, both of shape ``[batch, N]`` on
            the model's device.
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
        return (
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device),
        )

    def _encode(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Convolves token embeddings into per-position features.

        Padding embeddings are zeroed first, so a note's output does not
        depend on how much padding its batch adds. The kernel is padded by
        ``(k - 1) // 2`` on the left and ``k // 2`` on the right, which keeps
        the length at ``N`` and aligns feature ``n`` with token ``n``.

        Args:
            inputs_embeds: ``[batch, N, E]`` token embeddings.
            attention_mask: ``[batch, N]`` padding mask.

        Returns:
            ``[batch, N, F]`` features after tanh.
        """
        x = self.embed_dropout(inputs_embeds)
        x = x * attention_mask[..., None].to(x.dtype)
        left, right = (self.kernel_size - 1) // 2, self.kernel_size // 2
        x = F.pad(x.transpose(1, 2), (left, right))
        return torch.tanh(self.conv(x)).transpose(1, 2)

    def forward_from_embedding(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Runs the model from token embeddings instead of raw text.

        Gradient-based explanations (InputXGrad, AttInGrad) need gradients
        with respect to these embeddings.

        Args:
            inputs_embeds: ``[batch, N, E]`` token embeddings.
            attention_mask: ``[batch, N]`` padding mask.
            **kwargs: May contain the label key with ``[batch, J]`` targets.

        Returns:
            A dict with ``logit`` ``[batch, J]``, ``y_prob`` ``[batch, J]``
            and ``attention`` ``[batch, J, N]`` (0 on padding). If labels are
            given, it also has ``loss`` (scalar) and ``y_true`` ``[batch, J]``.
        """
        hidden = self._encode(inputs_embeds, attention_mask)
        scores = torch.einsum("jf,bnf->bjn", self.attention_vectors, hidden)
        padding = attention_mask[:, None, :] == 0
        scores = scores.masked_fill(padding, torch.finfo(scores.dtype).min)
        attention = self.softmax(scores)
        pooled = torch.bmm(attention, hidden)
        logits = (pooled * self.output_weights).sum(-1) + self.output_bias
        results = {
            "logit": logits,
            "y_prob": self.label_processor.activation(logits),
            "attention": attention,
        }
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            results["loss"] = self.label_processor.loss(logits, y_true)
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
