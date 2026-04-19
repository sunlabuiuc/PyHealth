from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.models import BaseModel


class TDLSTMMortality(BaseModel):
    """Temporal-Difference LSTM model for ICU mortality prediction.

    This model is a simplified PyHealth-native reproduction of the temporal
    difference learning idea described in:

        Frost, Li, and Harris. "Robust Real-Time Mortality Prediction in the ICU
        using Temporal Difference Learning." ML4H 2024.

    Compared with the original paper, this implementation intentionally keeps
    the architecture lightweight for reproducibility and contribution readiness:
    - LSTM encoder over fixed-length time-series features
    - binary mortality prediction
    - supervised mode using terminal BCE loss
    - TD mode using bootstrapped future predictions plus terminal BCE anchor

    The model follows the standard PyHealth BaseModel pattern:
    - takes a SampleDataset in the constructor
    - uses feature_key and label_key selected from dataset schemas
    - returns a dictionary containing at least:
        loss, y_prob, y_true, logit
    """

    VALID_TRAINING_MODES = {"td", "supervised"}

    def __init__(
        self,
        dataset,
        feature_key: str,
        label_key: str,
        mode: str = "binary",
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        gamma: float = 0.95,
        alpha_terminal: float = 0.10,
        n_step: int = 1,
        lengths_key: Optional[str] = None,
        embedding_dim: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(dataset=dataset)

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if alpha_terminal < 0.0:
            raise ValueError("alpha_terminal must be non-negative.")
        if n_step <= 0:
            raise ValueError("n_step must be positive.")
        if mode != "binary":
            raise ValueError(
                "TDLSTMMortality currently only supports binary classification."
            )

        if feature_key not in self.feature_keys:
            raise ValueError(
                f"feature_key '{feature_key}' not found in dataset feature_keys: "
                f"{self.feature_keys}"
            )
        if label_key not in self.label_keys:
            raise ValueError(
                f"label_key '{label_key}' not found in dataset label_keys: "
                f"{self.label_keys}"
            )

        self.feature_key = feature_key
        self.label_key = label_key
        self.lengths_key = lengths_key
        self.task_mode = mode

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gamma = gamma
        self.alpha_terminal = alpha_terminal
        self.n_step = n_step
        self.embedding_dim = embedding_dim

        self.training_mode = kwargs.pop("training_mode", "td")
        if self.training_mode not in self.VALID_TRAINING_MODES:
            raise ValueError(
                f"training_mode must be one of {self.VALID_TRAINING_MODES}."
            )

        self.input_dim = self._infer_input_dim_from_dataset()

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(self.hidden_dim, self.get_output_size())

    def _infer_input_dim_from_dataset(self) -> int:
        """Infers feature dimension from dataset samples."""
        try:
            sample = self.dataset[0]
            if self.feature_key in sample:
                x = sample[self.feature_key]
                if torch.is_tensor(x):
                    if x.ndim == 2:
                        return int(x.shape[-1])
                    raise ValueError(
                        f"Processed feature '{self.feature_key}' must be 2D [T, F], "
                        f"got tensor shape {tuple(x.shape)}."
                    )
        except Exception:
            pass

        raw_samples = getattr(self.dataset, "samples", None)
        if raw_samples is not None and len(raw_samples) > 0:
            raw_x = raw_samples[0][self.feature_key]

            if (
                isinstance(raw_x, (list, tuple))
                and len(raw_x) == 2
                and isinstance(raw_x[1], (list, tuple))
                and len(raw_x[1]) > 0
            ):
                first_row = raw_x[1][0]
                if isinstance(first_row, (list, tuple)):
                    return len(first_row)

            if isinstance(raw_x, (list, tuple)) and len(raw_x) > 0:
                first_row = raw_x[0]
                if isinstance(first_row, (list, tuple)):
                    return len(first_row)

        raise ValueError(
            f"Unable to infer input_dim for feature_key '{self.feature_key}' "
            "from dataset."
        )

    def _get_lengths_from_kwargs(self, kwargs: Dict) -> Optional[torch.Tensor]:
        """Gets sequence lengths from kwargs if a lengths_key is configured."""
        if self.lengths_key is None:
            return None
        lengths = kwargs.get(self.lengths_key)
        if lengths is None:
            return None
        if not torch.is_tensor(lengths):
            lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        else:
            lengths = lengths.to(self.device)
        return lengths

    def _prepare_input_tensor(self, x) -> torch.Tensor:
        """Converts batch feature input into a float tensor on model device."""
        if torch.is_tensor(x):
            x_tensor = x.float().to(self.device)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)

        if x_tensor.ndim != 3:
            raise ValueError(
                f"Expected batched input of shape [B, T, F], got {tuple(x_tensor.shape)}."
            )
        return x_tensor

    def _prepare_labels(self, y) -> torch.Tensor:
        """Converts labels into float tensor on model device."""
        if torch.is_tensor(y):
            y_tensor = y.float().to(self.device)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)

        if y_tensor.ndim > 1:
            y_tensor = y_tensor.view(-1)
        return y_tensor

    def encode(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encodes time-series input with an LSTM."""
        if lengths is None:
            encoded, _ = self.lstm(x)
            return encoded

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=x.shape[1],
        )
        return encoded

    def forward_logits(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes per-time-step logits."""
        encoded = self.encode(x, lengths=lengths)
        logits = self.output_layer(encoded).squeeze(-1)
        return logits

    @staticmethod
    def _gather_last_valid_step(
        sequence_tensor: torch.Tensor,
        lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Selects the final valid time step from a [B, T] tensor."""
        if lengths is None:
            return sequence_tensor[:, -1]

        idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(sequence_tensor.size(0), device=sequence_tensor.device)
        return sequence_tensor[batch_idx, idx]

    def build_n_step_targets(
        self,
        target_probs: torch.Tensor,
        y_true: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        gamma: Optional[float] = None,
        n_step: Optional[int] = None,
    ) -> torch.Tensor:
        """Builds n-step TD targets from target-network probabilities."""
        gamma = self.gamma if gamma is None else gamma
        n_step = self.n_step if n_step is None else n_step

        if y_true.ndim > 1:
            y_true = y_true.view(-1)

        batch_size, time_steps = target_probs.shape
        td_targets = torch.zeros_like(target_probs)

        for t in range(time_steps):
            future_idx = t + n_step
            if future_idx < time_steps:
                td_targets[:, t] = (gamma**n_step) * target_probs[:, future_idx]
            else:
                td_targets[:, t] = y_true

        if lengths is None:
            td_targets[:, -1] = y_true
        else:
            last_idx = (lengths - 1).clamp(min=0)
            batch_idx = torch.arange(batch_size, device=target_probs.device)
            td_targets[batch_idx, last_idx] = y_true

        return td_targets

    def compute_supervised_loss(
        self,
        logits_seq: torch.Tensor,
        y_true: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes supervised binary loss on the final valid time step."""
        final_logits = self._gather_last_valid_step(logits_seq, lengths)
        criterion = self.get_loss_function()
        return criterion(final_logits, y_true)

    def compute_td_loss(
        self,
        logits_seq: torch.Tensor,
        x: torch.Tensor,
        y_true: torch.Tensor,
        target_model: "TDLSTMMortality",
        lengths: Optional[torch.Tensor] = None,
        gamma: Optional[float] = None,
        alpha_terminal: Optional[float] = None,
        n_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes TD loss plus terminal BCE anchor."""
        alpha_terminal = (
            self.alpha_terminal if alpha_terminal is None else alpha_terminal
        )

        probs_seq = torch.sigmoid(logits_seq)

        with torch.no_grad():
            target_logits_seq = target_model.forward_logits(x, lengths=lengths)
            target_probs_seq = torch.sigmoid(target_logits_seq)

        td_targets = self.build_n_step_targets(
            target_probs=target_probs_seq,
            y_true=y_true,
            lengths=lengths,
            gamma=gamma,
            n_step=n_step,
        )

        td_mse = F.mse_loss(probs_seq, td_targets)
        terminal_loss = self.compute_supervised_loss(
            logits_seq=logits_seq,
            y_true=y_true,
            lengths=lengths,
        )

        total_loss = td_mse + alpha_terminal * terminal_loss
        return total_loss, td_mse, terminal_loss

    def forward(
        self,
        target_model: Optional["TDLSTMMortality"] = None,
        embed: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass in standard PyHealth format."""
        if self.feature_key not in kwargs:
            raise KeyError(f"Missing feature_key '{self.feature_key}' in batch.")

        if self.label_key not in kwargs:
            raise KeyError(f"Missing label_key '{self.label_key}' in batch.")

        x = self._prepare_input_tensor(kwargs[self.feature_key])
        y_true = self._prepare_labels(kwargs[self.label_key])
        lengths = self._get_lengths_from_kwargs(kwargs)

        logits_seq = self.forward_logits(x, lengths=lengths)
        probs_seq = torch.sigmoid(logits_seq)

        logit = self._gather_last_valid_step(logits_seq, lengths)
        y_prob = self.prepare_y_prob(logit.view(-1, 1)).view(-1)

        ret: Dict[str, torch.Tensor] = {
            "y_true": y_true,
            "logit": logit,
            "y_prob": y_prob,
            "logits_seq": logits_seq,
            "probs_seq": probs_seq,
        }

        if embed:
            encoded = self.encode(x, lengths=lengths)
            if lengths is None:
                embedding = encoded[:, -1, :]
            else:
                idx = (lengths - 1).clamp(min=0)
                batch_idx = torch.arange(encoded.size(0), device=encoded.device)
                embedding = encoded[batch_idx, idx, :]
            ret["embedding"] = embedding

        if self.training_mode == "supervised":
            loss = self.compute_supervised_loss(
                logits_seq=logits_seq,
                y_true=y_true,
                lengths=lengths,
            )
            ret["loss"] = loss
            return ret

        if self.training_mode == "td":
            if target_model is None:
                raise ValueError(
                    "target_model must be provided when training_mode='td'."
                )
            loss, td_loss, terminal_loss = self.compute_td_loss(
                logits_seq=logits_seq,
                x=x,
                y_true=y_true,
                target_model=target_model,
                lengths=lengths,
            )
            ret["loss"] = loss
            ret["td_loss"] = td_loss
            ret["terminal_loss"] = terminal_loss
            return ret

        raise ValueError(
            f"Unsupported training_mode '{self.training_mode}'. "
            f"Expected one of {self.VALID_TRAINING_MODES}."
        )