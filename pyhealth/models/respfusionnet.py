"""
RespFusionNet — a minimal multimodal fusion classifier for respiratory
abnormality prediction on the ICBHI 2017 Respiratory Sound Database.

This model is inspired by the core hypothesis of the RespLLM paper
("Unifying Audio and Text with Multimodal LLMs for Generalized
Respiratory Health Prediction"): combining respiratory audio with
patient/context information is expected to improve respiratory-health
prediction over unimodal inputs. RespFusionNet is intentionally **not**
a reproduction of RespLLM — no OpenBioLLM, no OPERA, no LoRA / PEFT. It
is a small, transparent MLP-fusion network that plugs cleanly into the
existing ``RespiratoryAbnormalityPredictionICBHI`` task and supports a
clean three-way ablation over modality usage (audio only / metadata
only / audio + metadata).

Paper:
    Zhang, Yuwei, et al. "RespLLM: Unifying Audio and Text with Multimodal
    LLMs for Generalized Respiratory Health Prediction." arXiv preprint
    arXiv:2410.05361 (2024).

Paper link:
    https://arxiv.org/abs/2410.05361

Author:
    Andrew Zhao (aazhao2@illinois.edu)
"""

from typing import Dict

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class RespFusionNet(BaseModel):
    """Small multimodal fusion classifier for respiratory abnormality.

    RespFusionNet consumes two fixed-size tensor inputs per sample —
    respiratory audio and a patient-metadata vector — and predicts a
    binary abnormality label. Each enabled modality is encoded by a
    tiny 2-layer MLP; enabled branches are concatenated and fed through
    a dropout + linear classifier head. Both branches can be disabled
    independently for modality-ablation studies.

    This model is a minimal companion to the ICBHI respiratory task
    contribution in PyHealth and is inspired by — but deliberately much
    smaller than — the RespLLM architecture. It is intended to be easy
    to read, easy to test, and cheap to train on a laptop.

    Expected sample schema
    ----------------------
    ``dataset.input_schema`` must contain the configured feature keys
    (``audio_feature_key`` and/or ``metadata_feature_key``) with type
    ``"tensor"``. Each sample must provide:

    - ``signal`` (or custom ``audio_feature_key``): a float tensor of
      shape ``(C, T)`` or ``(T,)``. Anything with the same total number
      of elements is accepted — it is flattened inside the forward
      pass before the audio encoder.
    - ``metadata`` (or custom ``metadata_feature_key``): a float tensor
      of shape ``(D,)``.
    - ``label``: a binary label under a ``BinaryLabelProcessor`` (the
      default for ``output_schema={"label": "binary"}``).

    Args:
        dataset: A PyHealth :class:`SampleDataset` whose ``input_schema``
            contains at least one of ``audio_feature_key`` or
            ``metadata_feature_key`` and whose ``output_schema`` is
            binary. The dataset is used both for
            :class:`~pyhealth.models.BaseModel` helper methods
            (``get_output_size``, ``get_loss_function``,
            ``prepare_y_prob``) and to peek the first sample so that
            ``audio_dim`` and ``metadata_dim`` can be derived
            automatically.
        hidden_dim: Hidden size of each per-modality MLP encoder.
            Default 128.
        dropout: Dropout probability applied inside each encoder and
            before the classifier head. Default 0.1.
        use_audio: If True, enable the audio branch. Default True.
        use_metadata: If True, enable the metadata branch. Default True.
        audio_feature_key: Sample dict key for the audio tensor. Default
            ``"signal"`` — matches the default output of
            :class:`~pyhealth.tasks.RespiratoryAbnormalityPredictionICBHI`.
        metadata_feature_key: Sample dict key for the metadata tensor.
            Default ``"metadata"`` — matches the tensor emitted by
            ``RespiratoryAbnormalityPredictionICBHI`` when
            ``include_metadata_features=True``.
        **kwargs: Reserved for forward-compatible extension points.

    Raises:
        ValueError: If both ``use_audio`` and ``use_metadata`` are False
            (the model would have no inputs).
        ValueError: If an enabled modality's feature key is not present
            in ``dataset.input_schema``.

    Attributes:
        use_audio (bool): Whether the audio branch is active.
        use_metadata (bool): Whether the metadata branch is active.
        audio_feature_key (str): Sample-dict key used for the audio input.
        metadata_feature_key (str): Sample-dict key used for the metadata
            input.
        hidden_dim (int): Per-branch MLP hidden size.
        dropout_p (float): Dropout probability used in encoders + head.
        audio_dim (int): Flattened audio input dimension (0 if audio is
            disabled).
        metadata_dim (int): Metadata input dimension (0 if metadata is
            disabled).
        encoders (nn.ModuleDict): Per-modality 2-layer MLP encoders,
            keyed by ``"audio"`` / ``"metadata"``.
        classifier (nn.Sequential): Dropout + linear head mapping the
            concatenated encoder outputs to ``output_size`` logits
            (1 for binary).

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import RespFusionNet
        >>> samples = [
        ...     {
        ...         "patient_id": "p0", "visit_id": "v0",
        ...         "signal": [0.0] * 32,
        ...         "metadata": [0.45, 1.0, 1.0, 0.0, 0.44, 1.0, 0.15],
        ...         "label": 0,
        ...     },
        ...     {
        ...         "patient_id": "p1", "visit_id": "v1",
        ...         "signal": [0.1] * 32,
        ...         "metadata": [0.20, 1.0, 0.0, 1.0, 0.00, 0.0, 0.10],
        ...         "label": 1,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor", "metadata": "tensor"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="demo",
        ... )
        >>> model = RespFusionNet(dataset=dataset, hidden_dim=16)
        >>> loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        >>> batch = next(iter(loader))
        >>> ret = model(**batch)
        >>> sorted(ret.keys())
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_audio: bool = True,
        use_metadata: bool = True,
        audio_feature_key: str = "signal",
        metadata_feature_key: str = "metadata",
        **kwargs,
    ) -> None:
        super().__init__(dataset)

        if not use_audio and not use_metadata:
            raise ValueError(
                "At least one of use_audio / use_metadata must be True — "
                "RespFusionNet would otherwise have no inputs."
            )
        if len(self.label_keys) != 1:
            raise ValueError(
                "RespFusionNet expects exactly one label key, got "
                f"{self.label_keys!r}."
            )

        schema_keys = set(dataset.input_schema.keys())
        if use_audio and audio_feature_key not in schema_keys:
            raise ValueError(
                f"audio_feature_key={audio_feature_key!r} not found in "
                f"dataset.input_schema keys {sorted(schema_keys)!r}."
            )
        if use_metadata and metadata_feature_key not in schema_keys:
            raise ValueError(
                f"metadata_feature_key={metadata_feature_key!r} not found "
                f"in dataset.input_schema keys {sorted(schema_keys)!r}."
            )

        self.use_audio = bool(use_audio)
        self.use_metadata = bool(use_metadata)
        self.audio_feature_key = audio_feature_key
        self.metadata_feature_key = metadata_feature_key
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.label_key = self.label_keys[0]

        # Peek the first *processed* sample to infer flattened dims. The
        # dataset's ``__getitem__`` returns a dict of already-processed
        # tensors (see ``TensorProcessor.process``), so this is a single
        # O(1) lookup with no additional I/O for in-memory datasets.
        first_sample = dataset[0]
        self.audio_dim = (
            int(first_sample[audio_feature_key].reshape(-1).numel())
            if self.use_audio
            else 0
        )
        self.metadata_dim = (
            int(first_sample[metadata_feature_key].reshape(-1).numel())
            if self.use_metadata
            else 0
        )

        encoders: Dict[str, nn.Module] = {}
        if self.use_audio:
            encoders["audio"] = self._build_encoder(self.audio_dim)
        if self.use_metadata:
            encoders["metadata"] = self._build_encoder(self.metadata_dim)
        self.encoders = nn.ModuleDict(encoders)

        fusion_dim = self.hidden_dim * len(self.encoders)
        output_size = self.get_output_size()
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_p),
            nn.Linear(fusion_dim, output_size),
        )

    def _build_encoder(self, in_dim: int) -> nn.Sequential:
        """Build a 2-layer MLP encoder mapping ``in_dim -> hidden_dim``.

        The encoder uses ``nn.ReLU`` (not the functional form) so that
        gradient-based interpretability hooks can attach cleanly.

        Args:
            in_dim: Input feature dimension for this modality branch.

        Returns:
            A :class:`torch.nn.Sequential` encoder.
        """
        return nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            **kwargs: Batched inputs from
                :func:`~pyhealth.datasets.get_dataloader`. Must contain
                every enabled branch's feature key plus the task's
                label key.

        Returns:
            Dict with:

            - ``logit``: ``(B, output_size)`` raw logits.
            - ``y_prob``: ``(B, output_size)`` probabilities
              (sigmoid for binary).
            - ``loss``: scalar training loss.
            - ``y_true``: the batched label tensor as provided by the
              dataloader.
        """
        parts = []
        if self.use_audio:
            x_audio = kwargs[self.audio_feature_key]
            x_audio = x_audio.reshape(x_audio.shape[0], -1).float()
            parts.append(self.encoders["audio"](x_audio))
        if self.use_metadata:
            x_meta = kwargs[self.metadata_feature_key]
            x_meta = x_meta.reshape(x_meta.shape[0], -1).float()
            parts.append(self.encoders["metadata"](x_meta))

        fused = torch.cat(parts, dim=1)
        logit = self.classifier(fused)

        y_true = kwargs[self.label_key]
        loss_fn = self.get_loss_function()
        # BinaryLabelProcessor emits (B, 1) float labels; match to
        # logit shape regardless of incoming rank.
        loss = loss_fn(logit, y_true.float().reshape(logit.shape))
        y_prob = self.prepare_y_prob(logit)

        return {
            "logit": logit,
            "y_prob": y_prob,
            "loss": loss,
            "y_true": y_true,
        }

    def forward_from_embedding(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Interpretability entry point.

        RespFusionNet already consumes dense tensor features, so this
        method simply delegates to :meth:`forward` to satisfy the
        ``Interpretable`` informal contract described in
        :class:`~pyhealth.models.BaseModel`.
        """
        return self.forward(**kwargs)
