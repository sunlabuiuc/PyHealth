"""MedTsLLM reconstruction model for time-series anomaly detection.

Contributors: Thi Van Trinh Tran and Zexi Li.
Paper: MedTsLLM, https://arxiv.org/abs/2408.07773.
Port of the authors' ``models/medtsllm.py`` (https://github.com/flixpar/med-ts-llm)
restricted to the paper's anomaly detection configuration ("concat" channels,
linear embedding downsampling). With the same weights and inputs it computes
the same outputs, including the prompt layout (each prompt part tokenized on
its own, prompts left-padded). The LLM is frozen; only adapters train.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.processors import LabelProcessor, RegressionLabelProcessor

# Llama-2-7B, the LLM used for the paper's main results.
DEFAULT_LLM = "meta-llama/Llama-2-7b-hf"
# Larger vocabularies are subsampled to this many word embeddings before the
# prototype mapping, as in the original implementation.
MAX_PROTOTYPE_VOCAB = 100_000


class RevIN(nn.Module):
    """Reversible instance normalization without affine parameters.

    Normalizes each sample and channel over time and undoes it on the output.
    The statistics of the last :meth:`normalize` call are kept for
    :meth:`denormalize`.

    Args:
        eps: Added to the variance for numerical stability.

    Examples:
        >>> revin = RevIN()
        >>> x = torch.randn(2, 8, 3)
        >>> torch.allclose(revin.denormalize(revin.normalize(x)), x, atol=1e-5)
        True
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.mean: torch.Tensor | None = None
        self.stdev: torch.Tensor | None = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes ``x`` of shape ``(batch, time, channels)`` over time.

        Args:
            x: Input windows.

        Returns:
            torch.Tensor: Normalized windows, same shape as ``x``.

        Examples:
            >>> RevIN().normalize(torch.ones(1, 4, 2)).abs().max().item()
            0.0
        """
        self.mean = x.mean(dim=1, keepdim=True).detach()
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        self.stdev = torch.sqrt(var + self.eps).detach()
        return (x - self.mean) / self.stdev

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Undoes the last :meth:`normalize`.

        Args:
            x: Normalized-scale tensor of shape ``(batch, time, channels)``.

        Returns:
            torch.Tensor: ``x`` in the scale of the normalized input.

        Raises:
            RuntimeError: If :meth:`normalize` has not been called.

        Examples:
            >>> revin = RevIN()
            >>> _ = revin.normalize(torch.randn(1, 4, 2))
            >>> revin.denormalize(torch.zeros(1, 4, 2)).shape
            torch.Size([1, 4, 2])
        """
        if self.mean is None or self.stdev is None:
            raise RuntimeError("RevIN.denormalize called before normalize.")
        return x * self.stdev + self.mean


class PatchEmbedding(nn.Module):
    """Cuts series into overlapping patches and embeds each patch.

    The series is right-padded with ``stride`` copies of its last value, so
    a series of length ``T`` gives ``(T - patch_len) // stride + 2`` patches.
    Patches are embedded by a circular 1D convolution over the patch sequence
    (kernel 3, no bias), so each embedding also sees its neighbor patches.

    Args:
        d_model: Embedding size of a patch.
        patch_len: Samples per patch.
        stride: Step between patch starts.
        dropout: Dropout on the embeddings.

    Examples:
        >>> emb = PatchEmbedding(d_model=4, patch_len=16, stride=8, dropout=0.0)
        >>> emb(torch.randn(3, 128)).shape
        torch.Size([3, 16, 4])
    """

    def __init__(
        self, d_model: int, patch_len: int, stride: int, dropout: float
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.token_conv = nn.Conv1d(
            in_channels=patch_len,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(
            self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds series of shape ``(n_series, time)``.

        Args:
            x: Univariate series, one per row.

        Returns:
            torch.Tensor: Patch embeddings ``(n_series, n_patches, d_model)``.

        Examples:
            >>> PatchEmbedding(4, 16, 8, 0.0)(torch.randn(2, 64)).shape
            torch.Size([2, 8, 4])
        """
        pad = x[:, -1:].expand(-1, self.stride)
        patches = torch.cat([x, pad], dim=-1).unfold(-1, self.patch_len, self.stride)
        out = self.token_conv(patches.permute(0, 2, 1)).transpose(1, 2)
        return self.dropout(out)


class ReprogrammingLayer(nn.Module):
    """Cross-attention from patch embeddings onto text prototypes.

    Queries come from the patches, keys and values from the prototypes, so
    each output is a mixture of (projected) prototypes in the LLM's input
    embedding space.

    Args:
        d_model: Size of a patch embedding.
        n_heads: Number of attention heads.
        d_keys: Size of a key per head.
        d_llm: Size of the LLM's embeddings (prototypes and output).
        attention_dropout: Dropout on the attention weights.

    Examples:
        >>> layer = ReprogrammingLayer(8, 2, 4, 16)
        >>> layer(torch.randn(3, 5, 8), torch.randn(10, 16)).shape
        torch.Size([3, 5, 16])
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_keys: int,
        d_llm: int,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Attends from ``target`` (B, L, d_model) onto ``source`` (S, d_llm).

        Args:
            target: Patch embeddings.
            source: Text prototypes, used as keys and values.

        Returns:
            torch.Tensor: Reprogrammed patches of shape ``(B, L, d_llm)``.

        Examples:
            >>> ReprogrammingLayer(8, 2, 4, 16)(
            ...     torch.randn(1, 2, 8), torch.randn(3, 16)).shape
            torch.Size([1, 2, 16])
        """
        b, length, _ = target.shape
        s = source.shape[0]
        h = self.n_heads
        query = self.query_projection(target).view(b, length, h, -1)
        key = self.key_projection(source).view(s, h, -1)
        value = self.value_projection(source).view(s, h, -1)
        scale = 1.0 / query.shape[-1] ** 0.5
        scores = torch.einsum("blhe,she->bhls", query, key)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,she->blhe", attn, value).reshape(b, length, -1)
        return self.out_projection(out)


class MedTsLLMAnomaly(BaseModel):
    """MedTsLLM reconstruction model for time-series anomaly detection.

    Takes windows ``signal`` of shape ``(batch, seq_len, n_features)`` and
    reconstructs them with a frozen LLM. Each window is normalized per
    instance (:class:`RevIN`) and cut into overlapping patches
    (:class:`PatchEmbedding`); the channels of a patch are concatenated, and
    :class:`ReprogrammingLayer` maps each patch into the LLM's input
    embedding space by attending onto text prototypes, learned mixtures of
    the LLM's word embeddings. The prompt embeddings are prepended, and a
    linear head maps the LLM's outputs at the patch positions back to the
    window, after which RevIN is undone. The
    training loss is the mean squared reconstruction error. Labels are not
    used for training: if a per-time-step ``label`` is in the batch it is
    returned as ``anomaly_label`` for evaluation with
    :func:`~pyhealth.metrics.anomaly_detection_metrics`.

    With ``standardize=True`` (the paper's setup), every channel is first
    standardized with the mean and standard deviation of ``dataset`` (the
    training set), and the reconstruction and target are reported on that
    standardized scale.

    Args:
        dataset: Training :class:`~pyhealth.datasets.SampleDataset` with a
            ``"tensor"`` input of shape ``(seq_len, n_features)``, e.g. from
            :class:`~pyhealth.tasks.MITBIHAnomalyDetection`.
        llm_name: Hugging Face model ID or local path of the LLM. Defaults to
            Llama-2-7B. Ignored if ``backbone`` is given.
        llm_layers: If positive, keep only the first ``llm_layers`` layers of
            the LLM. Defaults to 0 (all layers).
        random_backbone: Use the LLM architecture with random weights instead
            of the pretrained ones (still frozen). Seed beforehand.
        torch_dtype: Data type of the LLM weights. Defaults to float32.
        patch_len: Samples per patch. Defaults to 16.
        stride: Step between patches. Defaults to 8.
        d_model: Patch embedding size per channel. Defaults to 32.
        d_ff: Key size per head and size of the LLM output downsampling.
            Defaults to 64.
        n_heads: Heads of the reprogramming attention. Defaults to 8.
        num_tokens: Number of text prototypes. Defaults to 1024.
        dropout: Dropout of the patch embedding and attention. Defaults to 0.1.
        dataset_description: Text after ``"Dataset: "`` in the prompt; empty
            to leave the dataset out. Defaults to ``""``.
        task_description: Text after ``"Task: "``; ``None`` for the original
            reconstruction instruction, empty to leave it out.
        prompt_patient: Add ``"Patient description: ..."`` from the batch's
            ``description`` field (e.g. MIT-BIH header comments).
        standardize: Standardize channels with training-set statistics.
        signal_key: Input field with the windows. Defaults to ``"signal"``.
        backbone: A ready LLM (``transformers`` base model) to use instead of
            loading ``llm_name``, e.g. a tiny model for tests.
        tokenizer: The tokenizer of ``backbone``; required with it.

    Raises:
        ValueError: If the dataset has no such input, the sizes are not
            positive, or only one of ``backbone`` and ``tokenizer`` is given.

    Examples:
        >>> from pyhealth.datasets import MITBIHDataset  # doctest: +SKIP
        >>> ds = MITBIHDataset(root="/data/mitdb/1.0.0")  # doctest: +SKIP
        >>> train, test = ds.set_task(split=ds.anomaly_ratio_split())  # doctest: +SKIP
        >>> model = MedTsLLMAnomaly(train, llm_name="gpt2")  # doctest: +SKIP
        >>> out = model(**next(iter(get_dataloader(test, 4))))  # doctest: +SKIP
        >>> out["y_prob"].shape  # doctest: +SKIP
        torch.Size([4, 128, 2])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        llm_name: str = DEFAULT_LLM,
        llm_layers: int = 0,
        random_backbone: bool = False,
        torch_dtype: torch.dtype = torch.float32,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 32,
        d_ff: int = 64,
        n_heads: int = 8,
        num_tokens: int = 1024,
        dropout: float = 0.1,
        dataset_description: str = "",
        task_description: str | None = None,
        prompt_patient: bool = False,
        standardize: bool = True,
        signal_key: str = "signal",
        backbone: nn.Module | None = None,
        tokenizer: Any = None,
    ) -> None:
        super().__init__(dataset)
        if signal_key not in dataset.input_schema:
            raise ValueError(f"The dataset has no input {signal_key!r}.")
        sizes = (patch_len, stride, d_model, d_ff, n_heads, num_tokens)
        if any(int(v) != v or v <= 0 for v in sizes):
            raise ValueError(
                "patch_len, stride, d_model, d_ff, n_heads and num_tokens "
                "must be positive integers."
            )
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if (backbone is None) != (tokenizer is None):
            raise ValueError("Pass backbone and tokenizer together.")

        self.signal_key = signal_key
        first = torch.as_tensor(dataset[0][signal_key])
        if first.ndim == 1:
            first = first.unsqueeze(-1)
        self.seq_len, self.n_features = int(first.shape[0]), int(first.shape[1])
        if patch_len > self.seq_len:
            raise ValueError("patch_len must not exceed the window length.")
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (self.seq_len - patch_len) // stride + 2
        self.d_ff = d_ff
        self.prompt_patient = prompt_patient
        if task_description is None:
            task_description = (
                f"Reconstruct the past {self.seq_len} steps of data as "
                "accurately as possible using the following information."
            )
        self.dataset_description = dataset_description
        self.task_description = task_description

        mean, std = self._feature_stats(dataset) if standardize else (None, None)
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)

        self.llm, self.tokenizer = self._setup_llm(
            llm_name, llm_layers, random_backbone, torch_dtype, backbone, tokenizer
        )
        self.d_llm = int(self.llm.config.hidden_size)
        vocab = self._prototype_vocab().shape[0]

        # Same construction order as the original, so that a fixed seed gives
        # the same initial adapter weights.
        self.normalize_layer = RevIN()
        self.mapping_layer = nn.Linear(vocab, num_tokens)
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, dropout)
        self.reprogramming_layer = ReprogrammingLayer(
            d_model * self.n_features, n_heads, d_ff, self.d_llm, dropout
        )
        self.output_projection = nn.Linear(
            d_ff * self.n_patches, self.seq_len * self.n_features
        )
        self.embedding_downsample = nn.Linear(self.d_llm, d_ff)
        self._token_cache: dict[str, torch.Tensor] = {}

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """The reconstruction, scored as a regression of the input window.

        The dataset's anomaly labels are for evaluation only.

        Examples:
            >>> model.mode  # doctest: +SKIP
            'regression'
        """
        return {"reconstruction": RegressionLabelProcessor()}

    def _feature_stats(
        self, dataset: SampleDataset
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-channel mean and standard deviation over all training windows.

        Args:
            dataset: The training set.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean and population standard
            deviation, each of shape ``(n_features,)``.
        """
        total = torch.zeros(self.n_features, dtype=torch.float64)
        total_sq = torch.zeros(self.n_features, dtype=torch.float64)
        count = 0
        for sample in dataset:
            x = torch.as_tensor(sample[self.signal_key], dtype=torch.float64)
            x = x.reshape(-1, self.n_features)
            total += x.sum(0)
            total_sq += (x * x).sum(0)
            count += x.shape[0]
        mean = total / count
        std = (total_sq / count - mean * mean).clamp_min(0).sqrt()
        # Constant channels are left unscaled, as in scikit-learn.
        std = torch.where(std > 0, std, torch.ones_like(std))
        return mean.float(), std.float()

    def _setup_llm(
        self,
        llm_name: str,
        llm_layers: int,
        random_backbone: bool,
        torch_dtype: torch.dtype,
        backbone: nn.Module | None,
        tokenizer: Any,
    ) -> tuple[nn.Module, Any]:
        """Loads (or takes) the LLM and tokenizer and freezes the LLM.

        Returns:
            tuple[nn.Module, Any]: The frozen LLM and its tokenizer.
        """
        if backbone is None:
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            config = AutoConfig.from_pretrained(llm_name)
            if 0 < llm_layers < config.num_hidden_layers:
                config.num_hidden_layers = llm_layers
            if random_backbone:
                backbone = AutoModel.from_config(config, torch_dtype=torch_dtype)
            else:
                backbone = AutoModel.from_pretrained(
                    llm_name, config=config, torch_dtype=torch_dtype
                )
            tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        for param in backbone.parameters():
            param.requires_grad = False
        return backbone, tokenizer

    def _prototype_vocab(self) -> torch.Tensor:
        """The (frozen) word embeddings the text prototypes are mixed from."""
        weight = self.llm.get_input_embeddings().weight
        if weight.shape[0] > MAX_PROTOTYPE_VOCAB:
            idx = torch.linspace(
                0, weight.shape[0] - 1, MAX_PROTOTYPE_VOCAB, dtype=torch.long
            )
            weight = weight[idx.to(weight.device)]
        return weight

    def prompt_parts(self, descriptions: Sequence[str] | None, n: int) -> list[list]:
        """Builds the text prompt of each sample as a list of parts.

        Parts are the tokenizer's BOS token, ``"Dataset: ..."``, ``"Patient
        description: ..."``, ``"Task: ..."`` and ``"Time series:"``; empty
        parts are dropped and every part after the first gets a trailing
        space, as in the original implementation.

        Args:
            descriptions: Per-sample patient descriptions, or ``None``.
            n: Batch size.

        Returns:
            list[list]: The prompt parts of each sample.

        Examples:
            >>> model.prompt_parts(None, 1)  # doctest: +SKIP
            [['<|endoftext|>', 'Task: Reconstruct ... ', 'Time series: ']]
        """
        if self.prompt_patient and descriptions is None:
            raise ValueError("prompt_patient needs a 'description' in the batch.")
        bos = self.tokenizer.bos_token or ""
        dataset = (
            f"Dataset: {self.dataset_description}" if self.dataset_description else ""
        )
        task = f"Task: {self.task_description}" if self.task_description else ""
        prompts = []
        for i in range(n):
            patient = ""
            if self.prompt_patient and descriptions[i]:
                patient = f"Patient description: {descriptions[i]}"
            parts = [p for p in (bos, dataset, patient, task, "Time series:") if p]
            prompts.append([p if j == 0 else p + " " for j, p in enumerate(parts)])
        return prompts

    def _tokens(self, text: str) -> torch.Tensor:
        """Token IDs of one prompt part, tokenized on its own (cached)."""
        if text not in self._token_cache:
            ids = self.tokenizer(text, return_tensors="pt").input_ids
            self._token_cache[text] = ids
        return self._token_cache[text].to(self.device)

    def encode_prompts(self, prompts: list[list]) -> torch.Tensor:
        """Embeds prompts with the LLM's embedding table, left-padded.

        Args:
            prompts: Prompt parts per sample, from :meth:`prompt_parts`.

        Returns:
            torch.Tensor: Prompt embeddings of shape ``(B, max_len, d_llm)``.
        """
        embed = self.llm.get_input_embeddings()
        encoded = [
            torch.cat([embed(self._tokens(p)) for p in parts], dim=1)
            for parts in prompts
        ]
        max_len = max(e.shape[1] for e in encoded)
        pad_id = torch.tensor(self.tokenizer.pad_token_id, device=self.device)
        pad = embed(pad_id)
        out = []
        for e in encoded:
            fill = pad.expand(1, max_len - e.shape[1], -1)
            out.append(torch.cat([fill, e], dim=1))
        return torch.cat(out, dim=0)

    def encode_series(self, x: torch.Tensor) -> torch.Tensor:
        """Maps RevIN-normalized windows to LLM input embeddings.

        Args:
            x: Normalized windows of shape ``(B, seq_len, n_features)``.

        Returns:
            torch.Tensor: Reprogrammed patches ``(B, n_patches, d_llm)``.
        """
        b = x.shape[0]
        series = x.permute(0, 2, 1).reshape(b * self.n_features, self.seq_len)
        patches = self.patch_embedding(series)
        # "concat": the channels of a patch are joined into one token.
        patches = patches.reshape(b, self.n_features, self.n_patches, -1)
        patches = patches.permute(0, 2, 1, 3).reshape(b, self.n_patches, -1)
        # The LLM may run in half precision; the adapters are float32.
        vocab = self._prototype_vocab().to(self.mapping_layer.weight.dtype)
        prototypes = self.mapping_layer(vocab.permute(1, 0)).permute(1, 0)
        return self.reprogramming_layer(patches, prototypes)

    def reconstruct(
        self, x: torch.Tensor, descriptions: Sequence[str] | None = None
    ) -> torch.Tensor:
        """Reconstructs (standardized) windows.

        Args:
            x: Windows of shape ``(B, seq_len, n_features)``, already
                standardized if ``standardize`` is on.
            descriptions: Per-sample patient descriptions, if prompted.

        Returns:
            torch.Tensor: Reconstruction, same shape as ``x``.
        """
        b = x.shape[0]
        prompts = self.prompt_parts(descriptions, b)
        prompt_emb = self.encode_prompts(prompts)
        series_emb = self.encode_series(self.normalize_layer.normalize(x))
        hidden = torch.cat([prompt_emb.to(series_emb.dtype), series_emb], dim=1)
        llm_dtype = self.llm.get_input_embeddings().weight.dtype
        out = self.llm(inputs_embeds=hidden.to(llm_dtype)).last_hidden_state
        out = out[:, -self.n_patches :, :].to(x.dtype)
        out = self.embedding_downsample(out)  # [B, n_patches, d_ff]
        out = out.permute(0, 2, 1).reshape(b, -1)  # flattened as [d_ff, n_patches]
        out = self.output_projection(out).view(b, self.seq_len, self.n_features)
        return self.normalize_layer.denormalize(out)

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Reconstructs a batch and computes the MSE loss.

        Args:
            **kwargs: A batch with the ``signal_key`` windows, and optionally
                ``label`` (per-step anomaly labels) and ``description``.

        Returns:
            dict[str, torch.Tensor]: ``loss`` (MSE), ``y_prob`` (the
            reconstruction), ``y_true`` (the standardized input), and
            ``anomaly_label`` if the batch has a per-step ``label``.

        Examples:
            >>> out = model(**batch)  # doctest: +SKIP
            >>> out["loss"].backward()  # doctest: +SKIP
        """
        x = kwargs[self.signal_key].to(self.device, torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.shape[1:] != (self.seq_len, self.n_features):
            raise ValueError(
                f"Expected windows of shape (B, {self.seq_len}, "
                f"{self.n_features}), got {tuple(x.shape)}."
            )
        if self.feature_mean is not None:
            x = (x - self.feature_mean) / self.feature_std
        recon = self.reconstruct(x, kwargs.get("description"))
        out = {"loss": F.mse_loss(recon, x), "y_prob": recon, "y_true": x}
        label = kwargs.get("label")
        if isinstance(label, torch.Tensor) and label.ndim == 2:
            out["anomaly_label"] = label.to(self.device)
        return out
