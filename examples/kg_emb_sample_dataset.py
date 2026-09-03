"""Train a TransE model on a synthetic knowledge-graph sample dataset.

This example does not download UMLS. It shows the 2.0-safe path:

1. Build an in-memory :class:`SampleKGDataset` (not ``SampleDataset``).
2. Split into train/val/test folds with :func:`split`.
3. Wrap the train fold in ``torch.utils.data.DataLoader`` using
   :func:`collate_fn_dict_with_padding`. Do not call ``get_dataloader``:
   that helper requires ``litdata.StreamingDataset.set_shuffle()``.
"""

import torch
from torch.utils.data import DataLoader

from pyhealth.datasets import collate_fn_dict_with_padding
from pyhealth.medcode.pretrained_embeddings.kg_emb.datasets import (
    SampleKGDataset,
    split,
)
from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

samples = [
    {
        "triple": (i % 5, i % 2, (i + 1) % 5),
        "ground_truth_head": [i % 5, (i + 1) % 5],
        "ground_truth_tail": [(i + 1) % 5],
        "subsampling_weight": torch.tensor([0.25]),
    }
    for i in range(10)
]

dataset = SampleKGDataset(
    samples=samples,
    dataset_name="toy",
    task_name="link_prediction",
    entity2id={"a": 0, "b": 1, "c": 2, "d": 3, "e": 4},
    relation2id={"treats": 0, "causes": 1},
    negative_sampling=4,
)
print(dataset)
print(dataset.stat())

train, val, test = split(dataset, [0.6, 0.2, 0.2], seed=0)
print("fold sizes", len(train), len(val), len(test))

train_loader = DataLoader(
    train,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn_dict_with_padding,
)

model = TransE(dataset=dataset, e_dim=8, r_dim=8, ns="uniform")
batch = next(iter(train_loader))
out = model(**batch)
print("loss", float(out["loss"]))
out["loss"].backward()
print("backward ok")
