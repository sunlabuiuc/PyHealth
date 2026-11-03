import os
import time
import uuid
import warnings
from typing import List

import numpy as np
import torch


def batch_to_multihot(label: List[List[int]], num_labels: int) -> torch.tensor:
    """Converts label to multihot format.

    Args:
        label: [batch size, *]
        num_labels: total number of labels

    Returns:
        multihot: [batch size, num_labels]
    """
    multihot = torch.zeros((len(label), num_labels))
    for i, l in enumerate(label):
        multihot[i, l] = 1
    return multihot


_REPLACE_ATTEMPTS = 5
"""How many times save_array_atomically tries os.replace before giving up."""


def save_array_atomically(path: str, array: np.ndarray) -> bool:
    """Saves ``array`` to the ``.npy`` file ``path`` so readers never see a
    partial file.

    The array is written with :func:`numpy.save` to a uniquely named
    temporary file in the same directory (created if needed) and then moved
    onto ``path`` with :func:`os.replace`. A reader therefore sees either the
    old file or the complete new one, even when several processes write the
    same path at once. The temporary file is opened with ``"xb"``, so the
    umask applies and the result has the permissions ``np.save(path)`` would
    give it.

    On Windows, replacing a file that another process has open raises
    ``PermissionError``. The move is retried a few times; if it still fails,
    the temporary file is removed, the old file is left in place and a
    ``RuntimeWarning`` is issued instead of raising, because the callers
    (cache writers) still have the array in memory. Any other error removes
    the temporary file and is raised.

    Args:
        path: destination ``.npy`` file.
        array: the array to save.

    Returns:
        True if ``path`` now holds ``array``, False if the move gave up.

    Examples:
        >>> import os, tempfile
        >>> import numpy as np
        >>> from pyhealth.models.utils import save_array_atomically
        >>> path = os.path.join(tempfile.mkdtemp(), "sub", "a.npy")
        >>> save_array_atomically(path, np.arange(3))
        True
        >>> np.load(path)
        array([0, 1, 2])
    """
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # A unique name in the same directory, so os.replace stays on one file
    # system.
    tmp_path = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp_path, "xb") as f:
            np.save(f, array)
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp_path, path)
                return True
            except PermissionError as exc:
                error = exc
                time.sleep(0.05 * (attempt + 1))
        os.remove(tmp_path)
        warnings.warn(
            f"Could not replace {path} ({error}); it may be open in another "
            "process. The old file was left in place.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_ddi_adj(ddi_adj: np.ndarray | torch.Tensor, path: str) -> np.ndarray:
    """Atomically saves a drug-drug interaction (DDI) matrix as a ``.npy`` file.

    GAMENet, MICRON, MoleRec and SafeDrug call this to keep the shared
    ``ddi_adj.npy`` cache file (read by the ``"ddi"`` metric of
    :func:`pyhealth.metrics.multilabel_metrics_fn` when no ``ddi_adj`` is
    passed) up to date. The write goes through :func:`save_array_atomically`,
    so a reader never sees a partial file, and a file locked by another
    process (Windows) gives a warning instead of failing the model's
    constructor.

    Args:
        ddi_adj: the DDI adjacency matrix, as a numpy array or torch tensor.
        path: destination ``.npy`` file.

    Returns:
        The matrix as a numpy array (the model keeps it as ``ddi_adj_matrix``).

    Examples:
        >>> import os, tempfile
        >>> import numpy as np
        >>> from pyhealth.models.utils import save_ddi_adj
        >>> path = os.path.join(tempfile.mkdtemp(), "ddi_adj.npy")
        >>> saved = save_ddi_adj(np.eye(3), path)
        >>> np.array_equal(np.load(path), saved)
        True
    """
    if isinstance(ddi_adj, torch.Tensor):
        ddi_adj = ddi_adj.detach().cpu().numpy()
    ddi_adj = np.asarray(ddi_adj)
    save_array_atomically(path, ddi_adj)
    return ddi_adj

def get_last_visit(hidden_states, mask):
    """Gets the last visit from the sequence model.

    Args:
        hidden_states: [batch size, seq len, hidden_size]
        mask: [batch size, seq len]

    Returns:
        last_visit: [batch size, hidden_size]
    """
    if mask is None:
        return hidden_states[:, -1, :]
    else:
        mask = mask.long()
        last_visit = torch.sum(mask, 1) - 1
        # Clamp to 0 so that samples with an all-zero mask (no valid
        # visits) fall back to the first timestep instead of producing
        # a negative index that would crash torch.gather.
        last_visit = last_visit.clamp(min=0)
        last_visit = last_visit.unsqueeze(-1)
        last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
        last_visit = torch.reshape(last_visit, hidden_states.shape)
        last_hidden_states = torch.gather(hidden_states, 1, last_visit)
        last_hidden_state = last_hidden_states[:, 0, :]
        return last_hidden_state


def get_rightmost_masked_timestep(hidden_states, mask):
    """Gather hidden state at the last True position in ``mask`` per row.

    Unlike :func:`get_last_visit`, this does **not** assume valid tokens form a
    contiguous prefix; it picks the maximum index where ``mask`` is True.
    Use for MPF / CEHR layouts where padding can appear between boundary tokens.

    Args:
        hidden_states: ``[batch, seq_len, hidden_size]``.
        mask: ``[batch, seq_len]`` bool.

    Returns:
        Tensor ``[batch, hidden_size]``.
    """
    if mask is None:
        return hidden_states[:, -1, :]
    batch, seq_len, hidden = hidden_states.shape
    device = hidden_states.device
    idx = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(
        batch, -1
    )
    idx_m = torch.where(mask, idx, torch.full_like(idx, -1))
    last_idx = idx_m.max(dim=1).values.clamp(min=0)
    last_idx = last_idx.view(batch, 1, 1).expand(batch, 1, hidden)
    gathered = torch.gather(hidden_states, 1, last_idx)
    return gathered[:, 0, :]
