"""Train a KG embedding model on UMLS with the PyHealth 2.0 pipeline.

``UMLS_ROOT`` must hold ``graph.txt`` (head, relation and tail separated by
tabs, no header), e.g. downloaded from
https://storage.googleapis.com/pyhealth/umls/graph.txt. Outputs go to
``OUTPUT_DIR``. Ids follow sorted entity names, not the 1.x order, so
checkpoints trained with PyHealth 1.x must not be loaded as they are.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import PatientSplit, get_dataloader
from pyhealth.medcode import InnerMap
from pyhealth.medcode.pretrained_embeddings.kg_emb.datasets import UMLSDataset
from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE
from pyhealth.medcode.pretrained_embeddings.kg_emb.tasks import KGLinkPrediction
from pyhealth.trainer import Trainer

UMLS_ROOT = "path/to/umls"
OUTPUT_DIR = Path("output/umls_transe")
SEED = 0
SPLIT_RATIOS = (0.9, 0.05, 0.05)
NUM_WORKERS = 4


def _main() -> None:
    np.random.seed(SEED)  # training negatives use NumPy's global generator
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    umls_ds = UMLSDataset(root=UMLS_ROOT, num_workers=NUM_WORKERS)
    umls_ds.stats()
    print("Relations in KG:", umls_ds.relation2id)

    # Ids and names, to read the embeddings later.
    with open(OUTPUT_DIR / "id2entity.json", "w") as f:
        json.dump(umls_ds.id2entity, f, indent=2)
    with open(OUTPUT_DIR / "id2relation.json", "w") as f:
        json.dump(umls_ds.id2relation, f, indent=2)

    # Triple-level split; the processors, including the training-graph dicts
    # that filter training negatives, are fitted on the training part only.
    task = KGLinkPrediction(
        num_entities=umls_ds.num_entities, num_relations=umls_ds.num_relations
    )
    train, val, test = umls_ds.set_task(
        task, split=PatientSplit(ratios=SPLIT_RATIOS, seed=SEED)
    )

    model = TransE(dataset=train, e_dim=512, r_dim=512, negative_sampling=64)
    trainer = Trainer(
        model=model,
        metrics=["hits@n", "mean_rank"],
        output_path=str(OUTPUT_DIR),
        exp_name="umls_transe",
    )
    trainer.train(
        train_dataloader=get_dataloader(train, batch_size=8, shuffle=True),
        val_dataloader=get_dataloader(val, batch_size=8),
        epochs=10,
        optimizer_params={"lr": 1e-3},
        monitor="mean_rank",
        monitor_criterion="min",
    )
    print("filtered test metrics:", trainer.evaluate(get_dataloader(test, batch_size=8)))

    with open(OUTPUT_DIR / "entity_embedding.pkl", "wb") as f:
        pickle.dump(model.E_emb, f)
    with open(OUTPUT_DIR / "relation_embedding.pkl", "wb") as f:
        pickle.dump(model.R_emb, f)

    # Head/tail prediction, with the CodeMap mapping codes to free text.
    umls_code_map = InnerMap.load("UMLS")
    model.to("cpu")

    head, relation = "C0000039", "PAR"
    result_eid = model.inference(
        head=umls_ds.entity2id[head], relation=umls_ds.relation2id[relation], top_k=3
    )
    print(f"Input Head: {head} - {umls_code_map.lookup(head)}")
    print(f"Input Relation: {relation}")
    print("Tail Prediction:")
    for idx, eid in enumerate(result_eid):
        tail = umls_ds.id2entity[eid]
        print(f"{idx}: {tail} - {umls_code_map.lookup(tail)}")

    tail, relation = "C5162542", "CHD"
    result_eid = model.inference(
        relation=umls_ds.relation2id[relation], tail=umls_ds.entity2id[tail], top_k=3
    )
    print(f"Input Tail: {tail} - {umls_code_map.lookup(tail)}")
    print(f"Input Relation: {relation}")
    print("Head Prediction:")
    for idx, eid in enumerate(result_eid):
        head = umls_ds.id2entity[eid]
        print(f"{idx}: {head} - {umls_code_map.lookup(head)}")

    head = tail = "C0000039"
    relation = "SY"
    score = model.inference(
        head=umls_ds.entity2id[head],
        relation=umls_ds.relation2id[relation],
        tail=umls_ds.entity2id[tail],
    )
    print(f"Classification Score: {score}")


if __name__ == "__main__":
    _main()
