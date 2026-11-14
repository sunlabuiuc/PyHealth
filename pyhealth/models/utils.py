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


# how many times save_array_atomically tries os.replace before giving up
_REPLACE_ATTEMPTS = 5


def save_array_atomically(
    path: str, array: np.ndarray, mode: int | None = None
) -> bool:
    """Saves ``array`` to the ``.npy`` file ``path`` so readers never see a
    partial file.

    The array is written with :func:`numpy.save` to a uniquely named
    temporary file in the same directory (created if needed) and then moved
    onto ``path`` with :func:`os.replace`. A reader therefore sees either the
    old file or the complete new one, even when several processes write the
    same path at once. The temporary file is created with :func:`os.open`
    and ``O_EXCL``, so an existing file is never reused. Without ``mode`` it
    is created with ``0o666`` and the umask applies, giving the permissions
    ``np.save(path)`` would. With ``mode`` it is created with those bits and
    then set to exactly ``mode`` with :func:`os.chmod`, in case the umask
    removed some, so the data is never readable more widely. The new file
    always replaces the old one, so an existing file's permissions are not
    kept: pass ``mode`` to choose them.

    On Windows, replacing a file that another process has open raises
    ``PermissionError``. The move is retried a few times; if it still fails,
    the temporary file is removed, the old file is left in place and a
    ``RuntimeWarning`` is issued instead of raising, because the callers
    (cache writers) still have the array in memory. Any other error removes
    the temporary file and is raised.

    Args:
        path: destination ``.npy`` file.
        array: the array to save.
        mode: optional permission bits for the new file, e.g. ``0o600`` for
            files only the owner may read. ``None`` follows the umask.

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
        # O_EXCL so an existing file is never reused; with mode, the file
        # has those bits from the start (0o666 lets the umask apply, as
        # np.save would)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        fd = os.open(tmp_path, flags, 0o666 if mode is None else mode)
        with os.fdopen(fd, "wb") as f:
            if mode is not None:
                os.chmod(tmp_path, mode)  # the umask may have removed bits
            np.save(f, array)
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp_path, path)
                return True
            except PermissionError as exc:
                # Expected on Windows (target open elsewhere). On POSIX,
                # os.replace raises it only for a directory permission
                # problem, which retrying cannot fix; the loop is harmless
                # there and ends in the same warning.
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

    ARMR, GAMENet, MICRON, MoleRec and SafeDrug call this to keep the shared
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


def get_step_mask(x: torch.Tensor) -> torch.Tensor:
    """Marks the steps of a zero-padded numeric sequence that hold data.

    Collate pads shorter sequences with all-zero steps, so a step is padding
    when all its features are zero. Take the mask from the raw values: an
    embedding with a bias maps a padded step to a nonzero vector. A row with
    no nonzero step (an all-zero series) is all zero; the models' mean
    pooling then averages it over all its steps.

    Args:
        x: ``[batch, seq_len, n_features]`` raw values.

    Returns:
        ``[batch, seq_len]`` float mask, 1 for a step with data.

    Examples:
        >>> x = torch.tensor([[[1.0, 2.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
        >>> get_step_mask(x)
        tensor([[1., 0.],
                [0., 0.]])
    """
    return (x != 0).any(dim=-1).float()


def masked_mean_pooling(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the middle dimension, leaving out steps where ``mask`` is 0.

    A row with no valid step (e.g. an all-zero series) is averaged over all
    its steps, as without a mask, so the result is never NaN.

    Args:
        x: ``[batch, seq_len, embedding_dim]`` tensor.
        mask: ``[batch, seq_len]`` tensor, nonzero for a valid step.

    Returns:
        ``[batch, embedding_dim]`` tensor.

    Examples:
        >>> x = torch.tensor([[[1.0], [3.0], [8.0]], [[2.0], [4.0], [6.0]]])
        >>> mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        >>> masked_mean_pooling(x, mask)
        tensor([[2.],
                [4.]])
    """
    mask = mask.to(x.dtype).unsqueeze(-1)
    has_step = mask.sum(dim=1, keepdim=True) > 0
    mask = torch.where(has_step, mask, torch.ones_like(mask))
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


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


def ddi_adjacency(label_vocab: dict[str, int]) -> torch.Tensor:
    """Builds the drug-drug interaction graph over a drug label vocabulary.

    Uses the DDI pairs that GAMENet released (``ATC().get_ddi(gamenet_ddi=True)``)
    converted to ATC level 3, so ``label_vocab`` should hold ATC-3 codes, as
    the MIMIC drug-recommendation tasks produce. Shared by the drug
    recommendation models (GAMENet, SafeDrug, MICRON, MoleRec, ARMR).

    Args:
        label_vocab: drug code to label index.

    Returns:
        Symmetric 0/1 tensor of shape [len(label_vocab), len(label_vocab)].

    Examples:
        >>> adj = ddi_adjacency({"N02B": 0, "B01A": 1, "C03C": 2})  # doctest: +SKIP
        >>> adj.shape  # doctest: +SKIP
        torch.Size([3, 3])
        >>> bool((adj == adj.T).all())  # doctest: +SKIP
        True
    """
    from pyhealth.medcode import ATC

    size = len(label_vocab)
    adj = torch.zeros((size, size))
    for a, b in ATC().get_ddi(gamenet_ddi=True):
        a, b = ATC.convert(a, level=3), ATC.convert(b, level=3)
        if a in label_vocab and b in label_vocab:
            adj[label_vocab[a], label_vocab[b]] = 1
            adj[label_vocab[b], label_vocab[a]] = 1
    return adj


def ehr_cooccurrence_adjacency(drug_sets, label_vocab: dict[str, int]) -> torch.Tensor:
    """Builds the drug co-occurrence graph: drugs given together in a visit.

    Build it from the training samples only, so that evaluation labels do not
    reach the model. Shared by the drug recommendation models (GAMENet, ARMR).

    Args:
        drug_sets: an iterable with one entry per visit, either a list of
            drug codes or a multi-hot tensor over ``label_vocab`` (as a
            processed ``multilabel`` field is). Codes not in ``label_vocab``
            are ignored.
        label_vocab: drug code to label index.

    Returns:
        Symmetric 0/1 tensor of shape [len(label_vocab), len(label_vocab)]
        with an empty diagonal.

    Examples:
        >>> vocab = {"A01A": 0, "B01A": 1, "C03C": 2}
        >>> ehr_cooccurrence_adjacency(
        ...     [["A01A", "B01A"], torch.tensor([0.0, 0.0, 1.0])], vocab
        ... ).tolist()
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    """
    size = len(label_vocab)
    adj = torch.zeros((size, size))
    for drugs in drug_sets:
        if isinstance(drugs, torch.Tensor):  # multi-hot
            ids = torch.nonzero(drugs.flatten()).flatten()
        else:
            ids = torch.tensor(
                sorted({label_vocab[c] for c in drugs if c in label_vocab}),
                dtype=torch.long,
            )
        if ids.numel() > 1:
            adj[ids.unsqueeze(1), ids.unsqueeze(0)] = 1.0
    adj.fill_diagonal_(0.0)
    return adj

def gcn_normalize_adjacency(adj: torch.Tensor, self_loops: bool = True) -> torch.Tensor:
    """Row-normalizes an adjacency matrix for a GCN (Kipf and Welling).

    Computes ``D^-1 (A + I)``: each row is divided by its sum, after adding
    self-loops if ``self_loops``. Rows that sum to zero stay zero. Shared by
    GAMENet's GCN and ARMR.

    Args:
        adj: adjacency tensor of shape [num_nodes, num_nodes].
        self_loops: add the identity before normalizing. Default is True.

    Returns:
        Tensor of the same shape whose non-empty rows sum to 1.

    Examples:
        >>> gcn_normalize_adjacency(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        tensor([[0.5000, 0.5000],
                [0.5000, 0.5000]])
        >>> gcn_normalize_adjacency(torch.zeros(2, 2), self_loops=False)
        tensor([[0., 0.],
                [0., 0.]])
    """
    if self_loops:
        adj = adj + torch.eye(adj.size(0), dtype=adj.dtype, device=adj.device)
    r_inv = adj.sum(dim=1).pow(-1)
    r_inv[torch.isinf(r_inv)] = 0.0
    return r_inv.unsqueeze(1) * adj


def history_to_label_index(
    hist_vocab: dict[str, int], label_vocab: dict[str, int]
) -> torch.Tensor:
    """Maps a drug-history input vocabulary onto the drug label vocabulary.

    ``drugs_hist`` is tokenized with its own input vocabulary, which is not
    the label vocabulary of ``drugs``. The returned table converts history
    indices to label indices. Codes with no label (the processor's
    ``<pad>``/``<unk>`` tokens, or a history code absent from the labels) map
    to ``len(label_vocab)``, a "trash" index to slice away after a scatter.

    Args:
        hist_vocab: history code to input index.
        label_vocab: drug code to label index.

    Returns:
        LongTensor of shape [len(hist_vocab)].

    Examples:
        >>> history_to_label_index(
        ...     {"<pad>": 0, "<unk>": 1, "B01A": 2, "X99X": 3},
        ...     {"N02B": 0, "B01A": 1},
        ... ).tolist()
        [2, 2, 1, 2]
    """
    trash = len(label_vocab)
    mapping = torch.full((len(hist_vocab),), trash, dtype=torch.long)
    for code, idx in hist_vocab.items():
        if code in label_vocab:
            mapping[idx] = label_vocab[code]
    return mapping
