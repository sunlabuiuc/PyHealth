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


def prepare_label(label, mode):
    pass
