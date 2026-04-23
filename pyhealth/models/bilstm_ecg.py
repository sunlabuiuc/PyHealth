"""
Bidirectional LSTM for 12-lead ECG multi-label classification.

This module provides :class:`BiLSTMECG`, a :class:`~pyhealth.models.BaseModel`
subclass implementing the Bidirectional LSTM architecture benchmarked in:

    Nonaka, N. & Seita, J. (2021). *In-depth Benchmarking of Deep Neural
    Network Architectures for ECG Diagnosis.* Proceedings of Machine Learning
    Research 126:1–19, MLHC 2021.

The paper's best-performing LSTM variant (``lstm_d1_h64``) uses a **single
bidirectional LSTM layer** with ``hidden_size=64``, producing 128-dimensional
hidden states that are projected to the output head.  This implementation
follows the same design but exposes ``hidden_size`` and ``n_layers`` as
constructor arguments so it can be used as a drop-in replacement across the
full ablation grid.

Mathematical framing
--------------------
Given an ECG tensor :math:`X \\in \\mathbb{R}^{B \\times C \\times T}` (batch
size :math:`B`, :math:`C=12` leads, :math:`T` time-steps), the model:

1. Permutes to :math:`(B, T, C)` for sequence-first processing.
2. Passes through a bidirectional LSTM:

   .. math::

       h_t = \\text{BiLSTM}(x_t, h_{t-1})\\quad h_t \\in \\mathbb{R}^{B \\times 2H}

3. Takes the **last** time-step output :math:`h_T \\in \\mathbb{R}^{B \\times 2H}`.
4. Projects with a linear head :math:`W \\in \\mathbb{R}^{2H \\times K}` to produce
   logits for :math:`K` classes.
5. Optimises with **binary cross-entropy with logits** (multi-label):

   .. math::

       \\mathcal{L} = -\\frac{1}{K}\\sum_{k=1}^{K}
           \\bigl[y_k \\log \\sigma(\\hat{y}_k)
           + (1-y_k)\\log(1-\\sigma(\\hat{y}_k))\\bigr]

Paper alignment
---------------
+------------+----------------------------+---------------------------+
| Paper name | Paper setting              | Default in this class     |
+============+============================+===========================+
| lstm_d1_h64| 1 layer, hidden=64         | n_layers=1, hidden_size=64|
+------------+----------------------------+---------------------------+
| lstm_d3_h128| 3 layers, hidden=128      | n_layers=3, hidden_size=128|
+------------+----------------------------+---------------------------+

Signal format expected
----------------------
``feature_keys=["signal"]`` → each batch element is a ``np.ndarray`` of shape
``(12, T)`` loaded by ``SampleSignalDataset`` from a ``.pkl`` file.  :math:`T`
is typically 1000 at 100 Hz or 5000 at 500 Hz.

Authors:
    Anurag Dixit - anuragd2@illinois.edu
    Kent Spillner - kspillne@illinois.edu
    John Wells - jtwells2@illinois.edu
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from pyhealth.models import BaseModel


class BiLSTMECG(BaseModel):
    """Bidirectional LSTM ECG classifier.

    Extends :class:`~pyhealth.models.BaseModel` so it integrates seamlessly
    with :class:`~pyhealth.trainer.Trainer`, :class:`~pyhealth.datasets.PTBXLDataset`,
    and :func:`~pyhealth.metrics.multilabel_metrics_fn`.

    Args:
        dataset: A PyHealth ``SampleSignalDataset`` (or ``Subset``) that
            exposes ``input_info`` with ``"signal"`` → ``{"n_channels": 12}``.
        feature_keys (List[str]): Must be ``["signal"]``.
        label_key (str): Key in the sample dict that holds the label list.
            Use ``"labels"`` to match ``PTBXLMultilabelClassification`` output.
        mode (str): ``"multilabel"`` applies ``BCEWithLogitsLoss``; other modes
            are passed through to :class:`~pyhealth.models.BaseModel`.
        hidden_size (int): LSTM hidden dimension *per direction*.  The
            bidirectional output is ``2 × hidden_size``.  Paper best variant
            uses ``hidden_size=64`` (1 layer).  Defaults to ``64``.
        n_layers (int): Number of stacked LSTM layers.  Paper uses ``1``.
            Defaults to ``1``.
        dropout (float): Dropout probability applied between LSTM layers
            when ``n_layers > 1``.  Defaults to ``0.2``.

    Examples:
        Paper-aligned variant (lstm_d1_h64)::

            >>> from pyhealth.models import BiLSTMECG
            >>> model = BiLSTMECG(
            ...     dataset=sample_dataset,
            ...     feature_keys=["signal"],
            ...     label_key="labels",
            ...     mode="multilabel",
            ...     hidden_size=64,
            ...     n_layers=1,
            ... )

        Deeper variant used in ablation grid::

            >>> model_deep = BiLSTMECG(
            ...     dataset=sample_dataset,
            ...     feature_keys=["signal"],
            ...     label_key="labels",
            ...     mode="multilabel",
            ...     hidden_size=128,
            ...     n_layers=3,
            ... )
    """

    def __init__(
        self,
        dataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        hidden_size: int = 64,
        n_layers: int = 1,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(dataset=dataset)
        self.feature_key = feature_keys[0]
        self.label_key = label_key
        self.mode = mode

        # PTB-XL always has 12 leads; match the hard-coded default in ResNet18ECG
        in_channels: int = 12

        output_size: int = self.get_output_size()

        # ── Bidirectional LSTM ────────────────────────────────────────────────
        # Input:  (B, T, C)  after permute
        # Output: (B, T, hidden_size * 2)  — bidirectional concatenation
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # ── Projection head ───────────────────────────────────────────────────
        # Pool over all time-steps: AdaptiveAvgPool1d(1) → (B, hidden_size*2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    # ------------------------------------------------------------------
    def forward(self, **kwargs) -> dict:  # type: ignore[override]
        """Forward pass.

        Keyword args are the collated batch dict from the DataLoader.
        The key ``self.feature_keys[0]`` (``"signal"``) holds a list of
        ``np.ndarray`` of shape ``(12, T)``.

        Returns:
            dict with keys ``"loss"``, ``"y_prob"``, ``"y_true"``,
            ``"logit"`` — the standard PyHealth model output contract.
        """
        # Input tensor already collated by the dataloader: (B, 12, T)
        x: torch.Tensor = kwargs[self.feature_key].to(self.device)

        # (B, 12, T) → (B, T, 12) for sequence-first LSTM with batch_first=True
        out, _ = self.lstm(x.permute(0, 2, 1))     # (B, T, hidden*2)
        # (B, T, hidden*2) → (B, hidden*2, T) → pool → (B, hidden*2)
        # Matches paper bi_lstm.py: AdaptiveAvgPool1d over ALL timesteps
        pooled = self.pool(out.permute(0, 2, 1)).squeeze(-1)  # (B, hidden*2)
        logits = self.fc(pooled)                   # (B, K)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
