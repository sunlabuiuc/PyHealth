from typing import List

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


def mask_lengths(mask: torch.Tensor) -> torch.Tensor:
    """The length of each sequence: the position of its last valid step + 1.

    A step in the middle with mask 0, such as a visit with no codes, is
    part of the sequence; only the steps after the last valid one are
    padding. Counting the valid steps instead would cut the sequence short
    and drop its last visits.

    Args:
        mask: [batch size, seq len], nonzero for valid steps.

    Returns:
        A long tensor of shape [batch size]; 0 for a row with no valid step.

    Examples:
        >>> mask_lengths(torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]]))
        tensor([3, 2, 0])
    """
    valid = mask.bool()
    steps = torch.arange(1, mask.shape[-1] + 1, device=mask.device)
    return (valid * steps).amax(dim=-1).long()


def get_last_visit(hidden_states, mask):
    """Gets the last visit from the sequence model.

    The last visit is the last step with a nonzero mask, even if an
    earlier step is masked (see :func:`mask_lengths`).

    Args:
        hidden_states: [batch size, seq len, hidden_size]
        mask: [batch size, seq len]

    Returns:
        last_visit: [batch size, hidden_size]

    Examples:
        >>> h = torch.arange(6.0).reshape(1, 3, 2)
        >>> get_last_visit(h, torch.tensor([[1, 0, 1]]))
        tensor([[4., 5.]])
    """
    if mask is None:
        return hidden_states[:, -1, :]
    else:
        last_visit = mask_lengths(mask) - 1
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

    It picks the maximum index where ``mask`` is True, so padding may appear
    between valid tokens (as in MPF / CEHR layouts). :func:`get_last_visit`
    gives the same result for any mask with at least one True entry.

    Args:
        hidden_states: ``[batch, seq_len, hidden_size]``.
        mask: ``[batch, seq_len]`` bool.

    Returns:
        Tensor ``[batch, hidden_size]``.

    Example:
        >>> h = torch.tensor([[[1.0], [2.0], [0.0], [9.0]]])
        >>> m = torch.tensor([[True, True, False, True]])
        >>> get_rightmost_masked_timestep(h, m)
        tensor([[9.]])
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
