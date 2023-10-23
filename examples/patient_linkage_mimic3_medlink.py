import numpy as np

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.metrics import ranking_metrics_fn
from pyhealth.models import MedLink
from pyhealth.models.medlink import BM25Okapi
from pyhealth.models.medlink import convert_to_ir_format
from pyhealth.models.medlink import filter_by_candidates
from pyhealth.models.medlink import generate_candidates
from pyhealth.models.medlink import get_bm25_hard_negatives
from pyhealth.models.medlink import get_eval_dataloader
from pyhealth.models.medlink import get_train_dataloader
from pyhealth.models.medlink import tvt_split
from pyhealth.tasks import patient_linkage_mimic3_fn
from pyhealth.trainer import Trainer, logger

"""
IMPORTANT: This implementation differs from the original paper in order to
make it work with the PyHealth framework. Specifically, we do not use the
pre-trained GloVe embeddings. And we only monitor the loss on the validation 
set instead of the ranking metrics. As a result, the performance of this model
is different from the original paper. To reproduce the results in the paper,
please use the official GitHub repo: https://github.com/zzachw/MedLink.
"""

USE_BM25_HARDNEGS = False

""" STEP 1: load data """
base_dataset = MIMIC3Dataset(
    root="/srv/local/data/physionet.org/files/mimiciii/1.4",
    tables=["DIAGNOSES_ICD"],
    code_mapping={"ICD9CM": ("CCSCM", {})},
    dev=False,
    refresh_cache=False,
)
base_dataset.stat()

""" STEP 2: set task """
sample_dataset = base_dataset.set_task(patient_linkage_mimic3_fn)
sample_dataset.stat()
corpus, queries, qrels, corpus_meta, queries_meta = convert_to_ir_format(
    sample_dataset.samples
)
tr_queries, va_queries, te_queries, tr_qrels, va_qrels, te_qrels = tvt_split(
    queries, qrels
)
# generate candidates based on patient identifiers
# (stored in corpus_meta and queries_meta)
te_queries_meta = {k: v for k, v in queries_meta.items() if k in te_queries}
candidates = generate_candidates(corpus_meta, te_queries_meta)
average_matches = int(np.mean([len(v) for v in candidates.values()]))
print(f"Average number of candidates per query: {average_matches}")
# get BM25 hard negatives
if USE_BM25_HARDNEGS:
    bm25_model = BM25Okapi(corpus)
    tr_qrels = get_bm25_hard_negatives(bm25_model, corpus, tr_queries, tr_qrels)
# get data loaders
train_dataloader = get_train_dataloader(
    corpus, tr_queries, tr_qrels, batch_size=32, shuffle=True
)
val_dataloader = get_train_dataloader(
    corpus, va_queries, va_qrels, batch_size=32, shuffle=False
)
test_corpus_dataloader, test_queries_dataloader = get_eval_dataloader(
    corpus, te_queries, batch_size=32
)

""" STEP 3: define model """
model = MedLink(
    dataset=sample_dataset,
    feature_keys=["conditions"],
)

""" STEP 4: define trainer """
trainer = Trainer(model=model)
trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=5,
    monitor="loss",
    monitor_criterion="min",
)

""" STEP 5: evaluate """
results = model.evaluate(test_corpus_dataloader, test_queries_dataloader)
results = filter_by_candidates(results, te_qrels, candidates)
scores = ranking_metrics_fn(te_qrels, results, k_values=[1, 5])
logger.info(f"--- Test ---")
for key in scores.keys():
    logger.info("{}: {:.4f}".format(key, scores[key]))
