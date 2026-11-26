"""ABIDE I autism vs. control with a graph convolutional network (GCN).

Each scan becomes a brain graph, as in BrainGB: the nodes are the 200
Craddock ROIs, each node's features are its row of the Pearson
connectivity matrix (its "connection profile"), and the edges are the
strongest absolute correlations. ``ABIDEAutismClassification`` builds the
``(200, 200)`` matrix; PyHealth's ``GCN`` embeds every row as one node and
runs graph convolutions over the adjacency passed as ``visit_adj``, then
averages the nodes into the scan's logit.

``BrainGCN`` below only adds that adjacency: it keeps each node's
``--top-percent`` strongest absolute correlations plus a self loop and
row-normalises them, so each layer averages over a node's neighbours.

Data: ABIDE Preprocessed (CC BY-NC-SA 3.0, non-commercial),
http://preprocessed-connectomes-project.org/abide/ ; ``--download``
fetches it from the open ``fcp-indi`` S3 bucket. Example::

    python examples/fmri/abide_autism_gcn.py \\
        --root /data/abide --download --max-subjects 200 --epochs 1

``--max-subjects`` takes the first scans in phenotypic-file order, which
are grouped by site; a small value can give a single-class subset, which
the binary label processor rejects. One epoch shows the pipeline works; it
is not a benchmark result.
"""

from __future__ import annotations

import argparse

from pyhealth.datasets import ABIDEDataset, PatientSplit, get_dataloader
from pyhealth.models import GCN
from pyhealth.tasks import ABIDEAutismClassification
from pyhealth.trainer import Trainer


class BrainGCN(GCN):
    """``GCN`` whose node graph is the sparsified connectivity of each scan.

    Args:
        top_percent: Share of each node's strongest absolute correlations
            kept as edges (0-100).
        **kwargs: Forwarded to :class:`~pyhealth.models.GCN`.
    """

    def __init__(self, *args, top_percent: float = 10.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.top_percent = float(top_percent)

    def forward(self, **kwargs):
        """Add a ``visit_adj`` built from ``connectivity`` and run the GCN."""
        import torch  # after pyhealth; see the note in main()

        corr = kwargs["connectivity"]
        if not isinstance(corr, torch.Tensor):
            corr = torch.as_tensor(corr)
        corr = corr.to(self.device, torch.float32).abs()
        n = corr.shape[-1]
        k = max(1, round(n * self.top_percent / 100.0))
        threshold = corr.topk(k, dim=-1).values[..., -1:]
        eye = torch.eye(n, device=corr.device, dtype=corr.dtype)
        adj = (corr >= threshold).to(corr.dtype) * corr + eye
        kwargs["visit_adj"] = adj / adj.sum(dim=-1, keepdim=True)
        return super().forward(**kwargs)


def main() -> None:
    # torch is imported only after pyhealth: on some macOS envs, importing
    # torch first and then running numpy matmuls in the task segfaults
    # (two OpenMP runtimes).
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--top-percent", type=float, default=10.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset = ABIDEDataset(
        root=args.root, download=args.download, max_subjects=args.max_subjects
    )
    dataset.stats()
    # One scan per subject, so a patient split does not leak.
    train, val, test = dataset.set_task(
        ABIDEAutismClassification(),
        split=PatientSplit(ratios=(0.7, 0.1, 0.2), seed=0),
    )
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    model = BrainGCN(
        dataset=train, embedding_dim=64, nhid=64, top_percent=args.top_percent
    )
    trainer = Trainer(model=model, metrics=["roc_auc", "accuracy"], device=args.device)
    trainer.train(
        train_dataloader=get_dataloader(train, batch_size=32, shuffle=True),
        val_dataloader=get_dataloader(val, batch_size=32),
        epochs=args.epochs,
    )
    print(trainer.evaluate(get_dataloader(test, batch_size=32)))


if __name__ == "__main__":
    main()
