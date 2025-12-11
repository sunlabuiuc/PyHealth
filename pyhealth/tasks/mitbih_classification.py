"""
Task definitions for MIT-BIH Arrhythmia ECG classification.

This module provides a simple classification task function
`mitbih_classification_fn` that can be used with PyHealth's
`BaseDataset.set_task()` API.

The intent is to make it easy to plug MIT-BIH style ECG segments
(187-sample 1D signals with 0–4 arrhythmia labels) into the general
PyHealth pipeline as a signal-classification problem.
"""

from typing import Dict, List, Any

import numpy as np
import torch


def mitbih_classification_fn(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Define a 5-class ECG classification task for MIT-BIH segments.

    Parameters
    ----------
    record : dict
        A single raw record from a MIT-BIH style dataset.  We assume that
        the upstream dataset class (e.g., a `BaseDataset` or `BaseSignalDataset`
        subclass) provides at least the following keys:

        - ``record_id`` (str): unique identifier for the segment.
        - ``signal`` (np.ndarray or torch.Tensor): 1D ECG segment, typically
          of length 187.
        - ``label`` (int): class index in {0, 1, 2, 3, 4}.

        Example::

            {
                "record_id": "train_000123",
                "signal": np.ndarray(shape=(187,)),
                "label": 2,
            }

    Returns
    -------
    List[dict]
        A list of task-level samples.  For MIT-BIH, each raw record
        corresponds to exactly one sample, so the list has length 1.

        Each sample dict contains:

        - ``sample_id`` (str): unique ID for this sample (mirrors record_id).
        - ``signal`` (torch.Tensor): 1D or (1, L) ECG tensor for models.
        - ``label`` (int): 0–4 arrhythmia class.

        These keys are intentionally simple so that downstream models can
        set ``feature_keys=["signal"]`` and ``label_key="label"``.
    """
    # --- basic sanity checks & flexible handling of signal type ---
    record_id = str(record.get("record_id", "unknown"))

    signal = record.get("signal", None)
    if signal is None:
        raise ValueError(
            "mitbih_classification_fn expects `record['signal']` to be present."
        )

    # allow either numpy or torch
    if isinstance(signal, np.ndarray):
        signal_tensor = torch.from_numpy(signal.astype("float32"))
    elif isinstance(signal, torch.Tensor):
        signal_tensor = signal.float()
    else:
        raise TypeError(
            f"`signal` must be np.ndarray or torch.Tensor, got {type(signal)}"
        )

    # ensure 1D, then add channel dimension for Conv1d: (1, L)
    if signal_tensor.ndim == 1:
        signal_tensor = signal_tensor.unsqueeze(0)  # (1, L)
    elif signal_tensor.ndim == 2:
        # accept (1, L) or (L, 1); normalize to (1, L)
        if signal_tensor.shape[0] != 1 and signal_tensor.shape[1] == 1:
            signal_tensor = signal_tensor.squeeze(1).unsqueeze(0)
    else:
        raise ValueError(
            f"`signal` should be 1D or 2D tensor, got shape {signal_tensor.shape}"
        )

    label = int(record.get("label", 0))

    sample = {
        "sample_id": record_id,
        "signal": signal_tensor,
        "label": label,
    }

    return [sample]
