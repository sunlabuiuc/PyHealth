import random
from typing import Dict

import numpy as np
import tqdm
from torch.utils.data import DataLoader


def convert_to_ir_format(samples):
    corpus = {}
    queries = {}
    qrels = {}
    corpus_meta = {}
    queries_meta = {}
    for sample in samples:
        corpus[sample["d_visit_id"]] = sample["d_conditions"]
        queries[sample["visit_id"]] = sample["conditions"]
        qrels[sample["visit_id"]] = {sample["d_visit_id"]: 1}
        corpus_meta[sample["d_visit_id"]] = {"age": sample["d_age"],
                                             "identifiers": sample["d_identifiers"]}
        queries_meta[sample["visit_id"]] = {"age": sample["age"],
                                            "identifiers": sample["identifiers"]}
    return corpus, queries, qrels, corpus_meta, queries_meta


def generate_candidates(corpus_meta, queries_meta):
    candidates = {}
    for q_id, q_meta in queries_meta.items():
        age = q_meta["age"]
        identifiers = q_meta["identifiers"]
        matches = []
        for d_id, d_meta in corpus_meta.items():
            if (d_meta["age"] <= age) and (d_meta["identifiers"] == identifiers):
                matches.append(d_id)
        candidates[q_id] = matches
    # solve the problem of empty candidates
    average_matches = int(np.mean([len(v) for v in candidates.values()]))
    for q_id, q_meta in queries_meta.items():
        if len(candidates[q_id]) == 0:
            # random select average_matches candidates
            candidates[q_id] = random.sample(list(corpus_meta.keys()), average_matches)
    return candidates


def filter_by_candidates(results, qrels, candidates):
    filtered_results = {}
    for q_id, scores in results.items():
        c_ids = list(qrels[q_id].keys())
        candidate_ids = candidates[q_id]
        for c_id in c_ids:
            if c_id not in candidate_ids:
                candidate_ids.append(c_id)
        filtered_results[q_id] = {c_id: scores[c_id] for c_id in candidate_ids}
    return filtered_results


def tvt_split(queries, qrels, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    assert train_ratio + val_ratio + test_ratio == 1
    qids = list(queries.keys())
    np.random.shuffle(qids)
    s1 = int(len(qids) * train_ratio)
    s2 = int(len(qids) * (train_ratio + val_ratio))
    train_qids = qids[:s1]
    val_qids = qids[s1:s2]
    test_qids = qids[s2:]
    train_queries = {q_id: queries[q_id] for q_id in train_qids}
    val_queries = {q_id: queries[q_id] for q_id in val_qids}
    test_queries = {q_id: queries[q_id] for q_id in test_qids}
    train_qrels = {q_id: qrels[q_id] for q_id in train_qids}
    val_qrels = {q_id: qrels[q_id] for q_id in val_qids}
    test_qrels = {q_id: qrels[q_id] for q_id in test_qids}
    return train_queries, val_queries, test_queries, train_qrels, val_qrels, test_qrels


def get_bm25_hard_negatives(bm25_model, corpus, queries, qrels):
    qrels_w_neg = {}
    for q_id, q in tqdm.tqdm(queries.items()):
        d_ids = [d_id for d_id in qrels[q_id] if qrels[q_id][d_id] > 0]
        ds = [corpus[d_id] for d_id in d_ids]
        for d_id, d in zip(d_ids, ds):
            scores = bm25_model.get_scores(d)
            for (ned_d_id, neg_s) in sorted(scores.items(), key=lambda x: x[1],
                                            reverse=True):
                if ned_d_id != d_id:
                    qrels_w_neg[q_id] = {d_id: 1, ned_d_id: -1}
                    break
    return qrels_w_neg


def collate_fn(samples):
    outputs = {k: [] for k in samples[0].keys()}
    for sample in samples:
        for k, v in sample.items():
            outputs[k].append(v)
    return outputs


def get_train_dataloader(
    corpus: Dict[str, Dict[str, str]],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    batch_size: int,
    shuffle: bool = True
):
    query_ids = list(queries.keys())
    train_samples = []
    for query_id in query_ids:
        s_q = queries[query_id]
        id_p, s_p, s_n = None, None, None
        assert len(qrels[query_id]) <= 2
        for corpus_id, score in qrels[query_id].items():
            if score == 1:
                id_p = corpus_id
                s_p = corpus[corpus_id]
            if score == -1:
                s_n = corpus[corpus_id]
        if s_n is not None:
            train_samples.append(
                {
                    "query_id": query_id,
                    "id_p": id_p,
                    "s_q": s_q,
                    "s_p": s_p,
                    "s_n": s_n,
                }
            )
        else:
            train_samples.append(
                {
                    "query_id": query_id,
                    "id_p": id_p,
                    "s_q": s_q,
                    "s_p": s_p,
                }
            )
    print("Loaded {} training pairs.".format(len(train_samples)))
    train_dataloader = DataLoader(
        train_samples, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn
    )
    return train_dataloader


def get_eval_dataloader(
    corpus: Dict[str, Dict[str, str]],
    queries: Dict[str, str],
    batch_size: int
):
    corpus_ids = list(corpus.keys())
    eval_samples = []
    for corpus_id in corpus_ids:
        s = corpus[corpus_id]
        eval_samples.append(
            {
                "corpus_id": corpus_id,
                "s": s,
            }
        )
    print("Loaded {} eval corpus.".format(len(eval_samples)))
    eval_corpus_dataloader = DataLoader(
        eval_samples, shuffle=False, batch_size=batch_size, collate_fn=collate_fn
    )

    query_ids = list(queries.keys())
    eval_samples = []
    for query_id in query_ids:
        s = queries[query_id]
        eval_samples.append(
            {
                "query_id": query_id,
                "s": s,
            }
        )
    print("Loaded {} eval queries.".format(len(eval_samples)))
    eval_queries_dataloader = DataLoader(
        eval_samples, batch_size=batch_size, collate_fn=collate_fn
    )
    return eval_corpus_dataloader, eval_queries_dataloader
