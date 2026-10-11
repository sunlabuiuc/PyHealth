"""Train and evaluate TransE on a small knowledge graph with the 2.0 pipeline.

The example is self-contained: it writes a small synthetic graph with a
learnable structure to a temporary directory instead of downloading UMLS.
Its filtered test metrics should end well above those of a random ranking
(mean rank about 30 for 60 entities); they say nothing about real graphs.
Steps:

1. :class:`BaseKGDataset` reads one ``triples`` table (``head``,
   ``relation``, ``tail``) declared in a YAML config; each triple is one
   record. :class:`UMLSDataset` does the same with a bundled config.
2. ``set_task(KGLinkPrediction(...), split=PatientSplit(...))`` splits the
   triples and fits the processors on the training part only. The
   ``kg_triple`` processor keeps the training-graph dicts that filter the
   training negatives; the ground-truth lists of each sample cover the
   whole graph and filter the ranking at evaluation (filtered setting of
   Bordes et al., 2013).
3. The model reads the numbers of entities and relations from that
   processor; the :class:`~pyhealth.trainer.Trainer` switches between
   training and filtered evaluation through ``model.train()`` /
   ``model.eval()``.

Run from the repository root::

    python examples/kg_emb_link_prediction.py
"""

import tempfile
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import PatientSplit, get_dataloader
from pyhealth.medcode.pretrained_embeddings.kg_emb.datasets import BaseKGDataset
from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE
from pyhealth.medcode.pretrained_embeddings.kg_emb.tasks import KGLinkPrediction
from pyhealth.trainer import Trainer

SEED = 0
N_ENTITIES = 60
N_CLUSTERS = 6
N_RELATIONS = 2
N_TRIPLES = 600
SPLIT_RATIOS = (0.8, 0.1, 0.1)
EPOCHS = 30
BATCH_SIZE = 32

CONFIG = """version: "1.0"
tables:
  triples:
    file_path: "kg.tsv"
    patient_id: null
    timestamp: null
    attributes: [head, relation, tail]
"""


def write_synthetic_graph(root: Path) -> Path:
    """Writes a learnable synthetic graph and its config; returns the config.

    Entities fall into ``N_CLUSTERS`` ordered clusters, and relation ``rk``
    links a random entity of cluster ``c`` to a random entity of cluster
    ``c + k``. A translation model can represent this, so held-out triples
    are predictable from the training ones; on uniformly random triples the
    metrics would stay at chance level whatever the pipeline does.
    """
    rng = np.random.default_rng(SEED)
    size = N_ENTITIES // N_CLUSTERS
    lines = ["head\trelation\ttail"]
    for _ in range(N_TRIPLES):
        k = int(rng.integers(1, N_RELATIONS + 1))
        c = int(rng.integers(0, N_CLUSTERS - k))
        head = c * size + int(rng.integers(size))
        tail = (c + k) * size + int(rng.integers(size))
        lines.append(f"e{head}\tr{k}\te{tail}")
    (root / "kg.tsv").write_text("\n".join(lines) + "\n")
    (root / "kg.yaml").write_text(CONFIG)
    return root / "kg.yaml"


def main() -> None:
    # Training negatives are drawn with NumPy's global generator.
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        config = write_synthetic_graph(root)
        dataset = BaseKGDataset(
            root=str(root), config_path=config, cache_dir=root / "cache"
        )
        print(f"{dataset.num_entities} entities, {dataset.num_relations} relations")

        task = KGLinkPrediction(
            num_entities=dataset.num_entities, num_relations=dataset.num_relations
        )
        train, val, test = dataset.set_task(
            task, split=PatientSplit(ratios=SPLIT_RATIOS, seed=SEED)
        )
        print(f"train/val/test: {len(train)}/{len(val)}/{len(test)} triples")

        model = TransE(dataset=train, e_dim=32, r_dim=32, negative_sampling=16)
        trainer = Trainer(
            model=model,
            metrics=["hits@n", "mean_rank"],
            device="cpu",
            output_path=str(root / "output"),
            exp_name="kg_emb_link_prediction",
        )
        trainer.train(
            train_dataloader=get_dataloader(train, batch_size=BATCH_SIZE, shuffle=True),
            val_dataloader=get_dataloader(val, batch_size=BATCH_SIZE),
            epochs=EPOCHS,
            optimizer_params={"lr": 1e-2},
            monitor="mean_reciprocal_rank",
            monitor_criterion="max",
        )
        scores = trainer.evaluate(get_dataloader(test, batch_size=BATCH_SIZE))
        print("filtered test metrics:", scores)

        # Entity names from ids: the most likely tails of (e0, r1, ?), which
        # should mostly come from the second cluster, e10..e19.
        model.cpu()
        top = model.inference(
            head=dataset.entity2id["e0"], relation=dataset.relation2id["r1"], top_k=5
        )
        print("(e0, r1, ?):", [dataset.id2entity[i] for i in top])


if __name__ == "__main__":
    main()
