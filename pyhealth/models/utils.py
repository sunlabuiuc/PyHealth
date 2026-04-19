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
