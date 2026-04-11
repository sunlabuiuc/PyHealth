from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from torch import Tensor

from pyhealth.datasets import utils as datautils


def agg_loss(loss: torch.Tensor, reduction: str):
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def one_hot_np(labels, K):
    new_labels = np.zeros((len(labels), K))
    new_labels[np.arange(len(labels)), labels] = 1
    return new_labels


class LogLoss(torch.nn.Module):
    """Cross entropy, but takes in the probability instead of the logits"""

    reduction: str

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        clip=1e-10,
    ) -> None:
        super(LogLoss, self).__init__()
        self.register_buffer("weight", weight)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.clip = clip

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        dim = input.dim()
        assert dim == 2, f"Expected 2 dimensions (got {dim})"
        input = input.clip(
            self.clip
        )  # this weight should be trivial, so I won't normalize
        input = -torch.log(input)
        if self.weight is not None:
            input = input * self.weight.unsqueeze(0)
        loss = torch.gather(input, -1, target.unsqueeze(-1)).squeeze(-1)
        return agg_loss(loss, self.reduction)


def prepare_numpy_dataset(
    model,
    dataset,
    keys,
    forward_kwargs=None,
    incl_data_keys=None,
    debug=False,
    batch_size=32,
):
    if forward_kwargs is None:
        forward_kwargs = {}
    if incl_data_keys is None:
        incl_data_keys = []
    loader = datautils.get_dataloader(dataset, batch_size, shuffle=False)

    ret = defaultdict(list)
    with torch.no_grad():
        for _i, data in tqdm.tqdm(
            enumerate(loader), desc=f"retrieving {keys}", total=len(loader)
        ):
            if debug and _i % 10 != 0:
                continue
            data.update(forward_kwargs)
            res = model(**data)
            for key in keys:
                ret[key].append(res[key].detach().cpu().numpy())
            for key in incl_data_keys:
                ret[key].extend(data[key])
    for key in incl_data_keys:
        ret[key] = np.asarray(ret[key])
    for key in keys:
        ret[key] = np.concatenate(ret[key])
    return ret


def extract_embeddings(model, dataset, batch_size=32, device="cpu"):
    """Extract embeddings from a model for a given dataset.

    This function extracts embeddings (features before the final
    classification layer) from a model by calling forward() with
    embed=True flag.

    Parameters
    ----------
    model : BaseModel
        The trained model to extract embeddings from. Model must support
        the embed=True flag in its forward() method.
    dataset : BaseDataset
        The dataset to extract embeddings for.
    batch_size : int, optional
        Batch size for processing, by default 32.
    device : str, optional
        Device to use for computation, by default "cpu".

    Returns
    -------
    np.ndarray
        Embeddings of shape (n_samples, embedding_dim).

    Raises
    ------
    ValueError
        If the model doesn't support embedding extraction via embed=True.

    Examples
    --------
    >>> from pyhealth.datasets import COVID19CXRDataset
    >>> from pyhealth.models import TorchvisionModel
    >>> from pyhealth.calib.utils import extract_embeddings
    >>> dataset = COVID19CXRDataset(root="/path/to/data")
    >>> model = TorchvisionModel(
    ...     dataset=dataset, model_name="resnet18",
    ...     model_config={"weights": "DEFAULT"}
    ... )
    >>> embeddings = extract_embeddings(model, dataset, batch_size=32)
    """
    loader = datautils.get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    all_embeddings = []

    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in loader:
            # Move batch to device
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Request embeddings from model
            batch_device["embed"] = True
            output = model(**batch_device)

            # Check if model returned embeddings
            if "embed" not in output:
                raise ValueError(
                    f"Model {type(model).__name__} does not return "
                    "embeddings. Make sure the model supports the "
                    "embed=True flag in its forward() method."
                )

            # Extract embeddings and convert to numpy
            embeddings = output["embed"].cpu().numpy()
            all_embeddings.append(embeddings)

    return np.concatenate(all_embeddings, axis=0)
