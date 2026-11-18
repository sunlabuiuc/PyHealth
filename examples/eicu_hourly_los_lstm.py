"""Training and ablation extension points for the hourly ICU LOS task.

Contributor: Xiao Luo; UIUC NetID: xiaoluo6.
Paper: Temporal Pointwise Convolutional Networks for Length of Stay Prediction
in the Intensive Care Unit, https://arxiv.org/abs/2007.09483.

STATUS: scaffold. This file does not yet train a model or report results.
The runnable data/task check is:
    python -m examples.tpc_support.checkpoint

TODO(A): integrate existing code for splits, training, metrics and the 2x3
missingness experiment below. Use the fixture generator for smoke checks;
use authorized eICU data for final findings. TODO(A/B): reuse the masking
implementation in repository-root tpc_eicu.py; integrate preprocessing and
complete decay before enabling those settings. Existing working code should
be merged and validated, not written again.
"""

from typing import Any


def train_and_evaluate() -> list[dict[str, Any]]:
    """Run the final experiment once the extension points are complete.

    TODO(A):
        1. Load eICUDataset; split unique patient IDs into train/validation/test.
        2. Fit processors on training patients only; reuse on held-out patients.
        3. Explicitly select RNN_V2(rnn_type="LSTM") and agreed input embeddings.
        4. Train with 0% and 10% extra observation removal, keeping other
           settings controlled. Evaluate each at 0%, 10%, and 20% removal.
        5. Reuse test masks for both models; use seeded, documented repeats.
        6. Return metrics and experimental settings. Raw targets/activated
           predictions are hours; convert together for day-based comparisons.
           Specify a zero-target policy and paper-compatible MSLE convention.
        7. Document measured findings and differences from the original paper.

    Returns:
        One result dictionary per experimental configuration.

    Raises:
        NotImplementedError: Until the training implementation is supplied.
    """
    raise NotImplementedError(
        "TODO(A): implement training and ablations; "
        "TODO(B): complete missingness/decay. "
        "Run python -m examples.tpc_support.checkpoint for data/task validation."
    )


if __name__ == "__main__":
    train_and_evaluate()
