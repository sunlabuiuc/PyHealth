from typing import List

import torch
import torch.nn as nn


def get_default_loss_module(mode: str):
    """Get the default loss module for the given mode.

    Args:
        mode: "binary", "multiclass", or "multilabel"

    Returns:
        loss module
    """
    if mode == "binary":
        return nn.BCEWithLogitsLoss()
    elif mode == "multiclass":
        return nn.CrossEntropyLoss()
    elif mode == "multilabel":
        return nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError


def to_multihot(label: List[int], num_classes: int) -> torch.tensor:
    """Convert label to multihot format.

    Args:
        label: [batch size, *]
        num_classes: number of classes

    Returns:
        multihot: [batch size, num_classes]
    """
    multihot = torch.zeros(num_classes)
    multihot[label] = 1
    return multihot


def batch_to_multihot(label: List[List[int]], num_classes: int) -> torch.tensor:
    """Convert label to multihot format.

    Args:
        label: [batch size, *]
        num_classes: number of classes

    Returns:
        multihot: [batch size, num_classes]
    """
    multihot = torch.zeros((len(label), num_classes))
    for i, l in enumerate(label):
        multihot[i, l] = 1
    return multihot
