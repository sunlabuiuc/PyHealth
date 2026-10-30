"""Training helpers for the TD-CNN-LSTM PyHealth contribution.

T34 provides an explicit AdamW optimizer and training step. The standard
PyHealth Trainer does not call ``model.update_target()`` after its optimizer
step, so it must not be used unchanged for TD training.

The full runnable synthetic dataset, evaluation, and delta ablation are planned
for T39-T41. No MIMIC-IV data is needed by the helpers in this module.
"""

import math
from typing import Any, Dict, Mapping, Optional

import torch

from pyhealth.models.td_cnn_lstm import TDCNNLSTM


def make_reference_optimizer(
    model: TDCNNLSTM,
    n_training_batches: int,
    *,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
) -> torch.optim.AdamW:
    """Constructs the AdamW optimizer for the online mortality predictor.

    By default, follows the *released train.py* rather than the paper's
    different learning-rate expression: ``lr = 1 / sqrt(n_parameters)`` and
    ``weight_decay = 1 / (lr * n_training_batches)``. The paper appendix
    instead describes ``1 / n_parameters`` for the learning rate.

    The reference loader's ``segments`` is the number of **full training
    batches per epoch**, ``n_samples // batch_size``. Supply that number here,
    not the number of patients or the total number of epochs. The authors
    also discard the final incomplete batch. If a different data-loader policy
    is used, document the difference in the private reproduction protocol.

    Optional overrides permit a documented departure from the authors' code.
    In both modes only the online predictor's trainable parameters are updated,
    excluding the frozen TD target and non-trainable scaling buffers.

    Args:
        model: Configured TD or supervised ``TDCNNLSTM``.
        n_training_batches: Positive count of full batches in one epoch.
        learning_rate: Optional positive learning rate override.
        weight_decay: Optional non-negative AdamW decay override.

    Returns:
        An ``AdamW`` optimizer covering only online trainable parameters.

    Raises:
        TypeError: If the model or batch count has an invalid type.
        ValueError: If a count or optimizer hyperparameter is invalid.
    """
    if not isinstance(model, TDCNNLSTM):
        raise TypeError("model must be a TDCNNLSTM")
    if isinstance(n_training_batches, bool) or not isinstance(
        n_training_batches, int
    ):
        raise TypeError("n_training_batches must be an integer")
    if n_training_batches <= 0:
        raise ValueError("n_training_batches must be positive")

    parameter_count = model.n_online_trainable_parameters
    if parameter_count <= 0:
        raise ValueError("The online predictor has no trainable parameters")

    if learning_rate is None:
        learning_rate = 1.0 / math.sqrt(parameter_count)
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")

    if weight_decay is None:
        weight_decay = 1.0 / (learning_rate * n_training_batches)
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("weight_decay must be finite and non-negative")

    online_parameters = [
        parameter for parameter in model.online.parameters() if parameter.requires_grad
    ]
    return torch.optim.AdamW(
        online_parameters, lr=float(learning_rate), weight_decay=float(weight_decay)
    )


def training_step(
    model: TDCNNLSTM,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    """Performs one training step, including the TD target update when needed.

    The order is: zero online gradients, run the labeled model forward pass,
    backpropagate the objective-specific loss, update online weights, and then
    perform the target-network soft update for TD training. Supervised mode
    skips the target update because it has no target network.

    Args:
        model: TD or supervised model to optimize.
        optimizer: Optimizer configured for the online predictor.
        batch: Labeled PyHealth batch. TD batches also need matching successor
            state channels. See ``TDCNNLSTM.forward`` for required keys.

    Returns:
        The standard PyHealth output dictionary, with all tensors detached to
        prevent accumulation of autograd graphs in training logs.

    Raises:
        ValueError: If the batch is unlabeled or has invalid inputs.
    """
    model.train()
    optimizer.zero_grad()
    outputs = model(**batch)
    if "loss" not in outputs:
        raise ValueError("training_step requires an observed mortality label")

    outputs["loss"].backward()
    optimizer.step()
    if model.objective == "td":
        model.update_target()
    return {key: value.detach() for key, value in outputs.items()}
