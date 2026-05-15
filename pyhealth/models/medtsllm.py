# Author: Anton Barchukov
# Paper: Chan et al., "MedTsLLM: Leveraging LLMs for Multimodal
#     Medical Time Series Analysis", MLHC 2024
# Paper link: https://arxiv.org/abs/2408.07773
# Original repo: https://github.com/flixpar/med-ts-llm
# Description: Repurposes a frozen pretrained LLM as a feature
#     extractor for medical time series. Raw signals are patched,
#     projected into the LLM's embedding space via cross-attention
#     (reprogramming layer), and decoded by a lightweight task head.

import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models._medtsllm import (
    FlattenHead,
    PatchEmbedding,
    ReprogrammingLayer,
    RevIN,
    build_prompt,
    encode_prompts,
)

_SUPPORTED_TASKS = {
    "semantic_segmentation",
    "segmentation",
    "anomaly_detection",
    "reconstruction",
    "forecasting",
    "pretraining",
}


class MedTsLLM(BaseModel):
    """MedTsLLM: LLM-based medical time series model.

    Pipeline: RevIN -> PatchEmbedding -> Reprogramming -> LLM -> OutputHead

    Repurposes a frozen pretrained LLM (e.g., GPT-2, Qwen2.5) as a
    feature extractor for medical time series tasks. The LLM weights
    are never updated — only the reprogramming layer, patch embedding,
    and output head are trained (~1-2M parameters).

    Paper: Chan, N. et al. "MedTsLLM: Leveraging LLMs for Multimodal
    Medical Time Series Analysis." MLHC 2024.

    Note: forward-pass task branching (binary segmentation, anomaly
    reconstruction) is not yet wired — only ``semantic_segmentation``
    is fully supported end-to-end. The ``task`` argument currently
    drives the default task-description prompt only.

    Args:
        dataset: PyHealth SampleDataset with ``signal`` input and
            ``label`` output.
        task: One of ``"semantic_segmentation"``, ``"segmentation"``,
            ``"anomaly_detection"``, ``"reconstruction"``,
            ``"forecasting"``, ``"pretraining"``. Drives the default
            task-description prompt. Default
            ``"semantic_segmentation"``.
        seq_len: Input sequence length. Default 512.
        n_features: Number of input channels. Default 1.
        n_classes: Number of output classes for segmentation.
            Default 4.
        backbone: HuggingFace model ID for the LLM backbone.
            Set to ``None`` to use a lightweight replacement
            (for testing without model downloads). Default
            ``"openai-community/gpt2"``.
        d_model: Patch embedding dimension. Default 32.
        d_ff: Feedforward / output head hidden dimension. Default 128.
        n_heads: Attention heads in reprogramming layer. Default 8.
        num_tokens: Number of word prototype tokens. Default 1024.
        patch_len: Length of each patch. Default 16.
        stride: Stride between patches. Default 8.
        dropout: Dropout probability. Default 0.1.
        covariate_mode: ``"univariate"`` or ``"concat"``. Default
            ``"univariate"``.
        reprogramming_layer: Optional module to replace the default
            ``ReprogrammingLayer``. Must accept ``(target, source,
            value)`` and return ``(batch, n_patches, d_llm)``. Use
            ``LinearProjection`` for the no-reprogramming ablation.
        dataset_description: Text description for prompting.
        task_description: Text description for prompting. If empty,
            a task-appropriate default is generated.
        prompt_dataset: Include dataset prompt. Default True.
        prompt_task: Include task prompt. Default True.
        prompt_patient: Include per-patient description (age, sex,
            diagnoses, medications) in prompt. Requires samples to
            include a ``description`` field. Default True.
        prompt_stats: Include per-sample input statistics
            (min/max/median/trend/top-N autocorr lags). Default False
            to match the cs598-pyhealth reference's dtp config; the
            paper's ``input_stats`` prompt is an optional extra.
        n_lags: Number of autocorrelation lags in the stats prompt.
            Default 5.
        llm_dtype: Torch dtype for LLM weights. Default float32.
        word_embeddings: Pre-loaded word embeddings tensor. Required
            when ``backbone`` is None.

    Examples:
        >>> from pyhealth.models import MedTsLLM
        >>> model = MedTsLLM(
        ...     dataset=sample_dataset,
        ...     backbone="openai-community/gpt2",
        ...     seq_len=512,
        ...     n_classes=4,
        ... )
    """

    def __init__(
        self,
        dataset: SampleDataset,
        task: str = "semantic_segmentation",
        seq_len: int = 512,
        n_features: int = 1,
        n_classes: int = 4,
        backbone: Optional[str] = "openai-community/gpt2",
        d_model: int = 32,
        d_ff: int = 128,
        n_heads: int = 8,
        num_tokens: int = 1024,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        covariate_mode: str = "univariate",
        reprogramming_layer: Optional[nn.Module] = None,
        dataset_description: str = "",
        task_description: str = "",
        prompt_dataset: bool = True,
        prompt_task: bool = True,
        prompt_patient: bool = True,
        prompt_stats: bool = False,
        n_lags: int = 5,
        llm_dtype: torch.dtype = torch.float32,
        word_embeddings: Optional[Tensor] = None,
    ):
        super(MedTsLLM, self).__init__(dataset=dataset)

        if task not in _SUPPORTED_TASKS:
            raise ValueError(
                f"task must be one of {sorted(_SUPPORTED_TASKS)}, "
                f"got {task!r}"
            )

        self.task = task
        self.seq_len = seq_len
        self.pred_len = seq_len
        self.n_features = n_features
        self.n_classes = n_classes
        self.d_model = d_model
        self.d_ff = d_ff
        self.covariate_mode = covariate_mode
        self.n_lags = n_lags

        # Compute patch count
        self.n_patches = (seq_len - patch_len) // stride + 2

        # Effective d_model for concat covariate mode
        d_model_effective = d_model
        if covariate_mode == "concat":
            d_model_effective = d_model * n_features

        # Setup LLM backbone or replacement
        if backbone is not None:
            self._setup_llm(backbone, llm_dtype)
        elif word_embeddings is not None:
            self._setup_replacement(word_embeddings)
        else:
            raise ValueError(
                "Either backbone or word_embeddings must be provided."
            )

        # Trainable layers
        self.normalize_layers = RevIN(n_features, affine=False)
        self.patch_embedding = PatchEmbedding(
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            dropout=dropout,
        )
        self.mapping_layer = nn.Linear(self.vocab_size, num_tokens)

        if reprogramming_layer is not None:
            self.reprogramming_layer = reprogramming_layer
        else:
            self.reprogramming_layer = ReprogrammingLayer(
                d_model=d_model_effective,
                n_heads=n_heads,
                d_keys=d_ff,
                d_llm=self.d_llm,
                attention_dropout=dropout,
            )

        self.embedding_downsample = nn.Linear(self.d_llm, d_ff)

        # Output head size depends on task:
        #   semantic_segmentation => one logit per class per step
        #   segmentation          => a single binary logit per step
        #   anomaly_detection /
        #   reconstruction        => one value per feature per step
        #   forecasting /
        #   pretraining           => same as reconstruction
        self.n_outputs_per_step = self._compute_n_outputs_per_step(
            task, n_classes, n_features
        )
        self.output_projection = FlattenHead(
            n_features_in=d_ff * self.n_patches,
            n_outputs=self.n_outputs_per_step * self.pred_len,
        )

        # Prompting config
        self.dataset_description = dataset_description
        self.task_description = (
            task_description
            or self._default_task_description(task, seq_len, self.pred_len)
        )
        self.prompt_config = {
            "dataset": prompt_dataset,
            "task": prompt_task,
            "patient": prompt_patient,
            "stats": prompt_stats,
        }

    @staticmethod
    def _compute_n_outputs_per_step(
        task: str, n_classes: int, n_features: int
    ) -> int:
        """Resolve the per-timestep output dimension from the task."""
        if task == "semantic_segmentation":
            return n_classes
        if task == "segmentation":
            return 1
        if task in (
            "anomaly_detection",
            "reconstruction",
            "forecasting",
            "pretraining",
        ):
            return n_features
        raise ValueError(f"Unsupported task: {task!r}")

    @staticmethod
    def _default_task_description(
        task: str, seq_len: int, pred_len: int
    ) -> str:
        """Generate a task-appropriate default task description.

        Mirrors the original paper implementation's
        ``get_task_description``.
        """
        if task in ("forecasting", "pretraining"):
            return (
                f"Forecast the next {pred_len} steps given the "
                f"previous {seq_len} steps of data."
            )
        if task in ("anomaly_detection", "reconstruction"):
            return (
                f"Reconstruct the past {seq_len} steps of data as "
                "accurately as possible using the following "
                "information."
            )
        if task == "semantic_segmentation":
            return (
                f"Classify the past {seq_len} steps of data as "
                "accurately as possible using the following "
                "information."
            )
        if task == "segmentation":
            return (
                f"Identify the change points in the past {seq_len} "
                "steps of data to segment the sequence."
            )
        return ""

    def parameters(self, recurse: bool = True):
        """Yield only trainable parameters.

        Overrides nn.Module.parameters() to exclude the frozen LLM
        backbone. This prevents PyHealth's Trainer from allocating
        optimizer state (momentum, variance) for frozen parameters,
        saving ~2x the frozen param count in memory.

        For GPT-2 (137M) this saves ~1GB. For Qwen2.5-1.5B it
        saves ~12GB.
        """
        for p in super().parameters(recurse=recurse):
            if p.requires_grad:
                yield p

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        """Yield only trainable named parameters.

        See parameters() for rationale.
        """
        for name, p in super().named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        ):
            if p.requires_grad:
                yield name, p

    def _setup_llm(self, backbone: str, dtype: torch.dtype) -> None:
        """Load a frozen HuggingFace LLM as backbone."""
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        llm_config = AutoConfig.from_pretrained(backbone)
        llm_config.output_hidden_states = True

        self.llm = AutoModel.from_pretrained(
            backbone,
            config=llm_config,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze LLM
        for param in self.llm.parameters():
            param.requires_grad = False

        # Extract word embeddings
        we = self.llm.get_input_embeddings().weight.detach().cpu()
        if we.shape[0] > 100_000:
            inds = torch.linspace(
                0, we.shape[0] - 1, 100_000, dtype=torch.long
            )
            we = we[inds].clone()
        self.word_embeddings = nn.Parameter(we, requires_grad=False)
        self.vocab_size = self.word_embeddings.shape[0]
        self.d_llm = self.word_embeddings.shape[1]
        self._use_llm = True

    def _setup_replacement(self, word_embeddings: Tensor) -> None:
        """Setup a small feedforward network replacing the LLM."""
        self.word_embeddings = nn.Parameter(
            word_embeddings.detach().clone(), requires_grad=False
        )
        self.vocab_size = word_embeddings.shape[0]
        self.d_llm = word_embeddings.shape[1]
        self.llm_replacement = nn.Sequential(
            nn.Linear(self.d_llm, self.d_llm),
            nn.GELU(),
            nn.Linear(self.d_llm, self.d_llm),
        )
        self.tokenizer = None
        self._use_llm = False

    def forward(self, **kwargs) -> Dict[str, Tensor]:
        """Forward pass following PyHealth's BaseModel contract.

        Args:
            **kwargs: Must include ``signal`` tensor of shape
                ``(batch, seq_len)`` or ``(batch, seq_len, n_features)``.
                May include ``label`` tensor for loss computation and
                ``description`` list of per-sample strings for the
                patient prompt.

        Returns:
            Dict with keys: ``logit``, ``y_prob``, and optionally
            ``loss``, ``y_true``.
        """
        signal = kwargs[self.feature_keys[0]].to(self.device)
        if signal.ndim == 2:
            signal = signal.unsqueeze(-1)
        bs = signal.shape[0]

        # Encode time series
        enc_out = self._encode_ts(signal)

        # Build + prepend prompt (only when a real LLM is attached)
        if self._use_llm and any(self.prompt_config.values()):
            prompt_enc = self._build_prompt_embeddings(signal, bs, kwargs)
            enc_out = torch.cat([prompt_enc, enc_out], dim=1)

        # LLM or replacement forward
        if self._use_llm:
            dec_out = self.llm(inputs_embeds=enc_out).last_hidden_state
            dec_out = dec_out.to(device=signal.device, dtype=signal.dtype)
        else:
            dec_out = self.llm_replacement(enc_out)

        # Keep last n_patches outputs
        dec_out = dec_out[:, -self.n_patches :, :]

        # Downsample and project
        dec_out = self.embedding_downsample(dec_out)
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.output_projection(dec_out)

        # Reshape to (bs, pred_len, n_outputs_per_step)
        dec_out = dec_out.view(
            bs, self.pred_len, self.n_outputs_per_step
        )

        label_key = self.label_keys[0] if self.label_keys else None
        y_true = (
            kwargs[label_key].to(self.device)
            if label_key and label_key in kwargs
            else None
        )

        return self._task_head(dec_out, signal, y_true)

    def _task_head(
        self,
        dec_out: Tensor,
        signal: Tensor,
        y_true: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        """Apply task-specific post-processing and loss.

        Branches on ``self.task``:

        - ``semantic_segmentation``: softmax probs, cross-entropy loss.
        - ``segmentation``: binary logit per step, BCE-with-logits loss.
        - ``anomaly_detection`` / ``reconstruction``: RevIN-denormalized
          reconstruction in the original signal space, MSE loss against
          the input signal.
        - ``forecasting`` / ``pretraining``: denormalized prediction,
          MSE loss against the input signal (placeholder — true
          forecasting needs a future-signal label).
        """
        output: Dict[str, Tensor] = {}
        if y_true is not None:
            output["y_true"] = y_true

        if self.task == "semantic_segmentation":
            # dec_out: (bs, pred_len, n_classes)
            logit = dec_out
            output["logit"] = logit
            output["y_prob"] = F.softmax(logit, dim=-1)
            if y_true is not None:
                output["loss"] = F.cross_entropy(
                    logit.view(-1, self.n_classes),
                    y_true.view(-1).long(),
                )
            return output

        if self.task == "segmentation":
            # dec_out: (bs, pred_len, 1) -> (bs, pred_len)
            logit = dec_out.squeeze(-1)
            output["logit"] = logit
            output["y_prob"] = torch.sigmoid(logit)
            if y_true is not None:
                output["loss"] = F.binary_cross_entropy_with_logits(
                    logit, y_true.float()
                )
            return output

        if self.task in (
            "anomaly_detection",
            "reconstruction",
            "forecasting",
            "pretraining",
        ):
            # dec_out: (bs, pred_len, n_features) — denormalize to
            # recover original signal space before computing loss.
            prediction = self.normalize_layers(dec_out, "denorm")
            output["logit"] = prediction
            output["y_prob"] = prediction
            # Train to reconstruct the input signal. For
            # anomaly_detection, labels (anomaly masks) are used at
            # eval time for scoring — not during training.
            output["loss"] = F.mse_loss(prediction, signal)
            return output

        raise ValueError(f"Unsupported task in forward: {self.task!r}")

    def _build_prompt_embeddings(
        self, signal: Tensor, bs: int, kwargs: Dict
    ) -> Tensor:
        """Construct and encode the prompt for the current batch.

        No caching: prompts are rebuilt every forward pass to match
        the original paper implementation. Dataset/task/stats prompts
        are cheap to re-tokenize; patient prompts depend on batch
        contents and can't be cached anyway.
        """
        include_patient = self.prompt_config.get("patient", False)
        bos = getattr(self.tokenizer, "bos_token", None) or ""

        prompt_inputs: Dict = {"x_enc": signal}

        if include_patient:
            description = kwargs.get("description")
            if description is None:
                warnings.warn(
                    "prompt_patient=True but no 'description' field "
                    "provided in batch. Patient prompt will be empty. "
                    "Ensure your task emits 'description' per sample.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                descriptions = [""] * bs
            else:
                descriptions = _coerce_descriptions(description, bs)
            prompt_inputs["descriptions"] = descriptions

        prompts = build_prompt(
            prompt_inputs,
            dataset_description=self.dataset_description,
            task_description=self.task_description,
            include_dataset=self.prompt_config.get("dataset", False),
            include_task=self.prompt_config.get("task", False),
            include_clip=include_patient,
            include_stats=self.prompt_config.get("stats", False),
            n_lags=self.n_lags,
            bos_token=bos,
        )

        with torch.no_grad():
            return encode_prompts(
                prompts,
                self.tokenizer,
                self.llm.get_input_embeddings(),
                signal.device,
            )

    def _encode_ts(self, x_enc: Tensor) -> Tensor:
        """Encode time series: normalize -> patch -> reprogram.

        Args:
            x_enc: (batch, seq_len, n_features).

        Returns:
            Encoded representation: (batch, n_patches, d_llm).
        """
        bs, seq_len, n_features = x_enc.shape

        x_enc = self.normalize_layers(x_enc, "norm")
        x_enc = x_enc.permute(0, 2, 1).contiguous()

        enc_out, _ = self.patch_embedding(x_enc)

        # Project word embeddings to prototypes
        we = self.word_embeddings.to(self.mapping_layer.weight.dtype)
        source_embeddings = self.mapping_layer(
            we.permute(1, 0)
        ).permute(1, 0)

        # Handle covariate modes
        if self.covariate_mode == "concat":
            enc_out = enc_out.reshape(
                bs, n_features, self.n_patches, self.d_model
            )
            enc_out = enc_out.permute(0, 2, 1, 3)
            enc_out = enc_out.reshape(
                bs, self.n_patches, n_features * self.d_model
            )

        enc_out = self.reprogramming_layer(
            enc_out, source_embeddings, source_embeddings
        )

        return enc_out


def _coerce_descriptions(
    description, bs: int
) -> list[str]:
    """Normalize a batch description field into a list[str] of length bs.

    Handles the common shapes produced by PyHealth's default collate
    (a list of strings) as well as scalar-string edge cases.
    """
    if isinstance(description, str):
        return [description] * bs
    if isinstance(description, list):
        return description
    try:
        return list(description)
    except TypeError:
        return [description] * bs
