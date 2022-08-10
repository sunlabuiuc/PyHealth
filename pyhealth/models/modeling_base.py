import torch.nn as nn

VALID_MODE = [
    "binary",
    "multiclass",
    "multilabel"
]

# TODO: put common model functions in basemodel class

class BaseModel(nn.Module):
    def __init__(
            self,
            mode: str = "binary",
    ):
        super().__init__()
        if mode not in VALID_MODE:
            raise ValueError(f"mode must be one of {VALID_MODE}")
        self.mode = mode
        if mode in ["binary", "multilabel"]:
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif mode in ["multiclass"]:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"{mode} is not supported")
