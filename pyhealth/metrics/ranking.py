from typing import List, Dict


def ranking_metrics_fn(qrels: Dict[str, Dict[str, int]],
                       results: Dict[str, Dict[str, float]],
                       k_values: List[int]) -> Dict[str, float]:
    import pytrec_eval
    ret = {}

    for k in k_values:
        ret[f"NDCG@{k}"] = 0.0
        ret[f"MAP@{k}"] = 0.0
        ret[f"Recall@{k}"] = 0.0
        ret[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels,
                                               {map_string, ndcg_string, recall_string,
                                                precision_string})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ret[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            ret[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            ret[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            ret[f"P@{k}"] += scores[query_id]["P_" + str(k)]

    for k in k_values:
        ret[f"NDCG@{k}"] = round(ret[f"NDCG@{k}"] / len(scores), 5)
        ret[f"MAP@{k}"] = round(ret[f"MAP@{k}"] / len(scores), 5)
        ret[f"Recall@{k}"] = round(ret[f"Recall@{k}"] / len(scores), 5)
        ret[f"P@{k}"] = round(ret[f"P@{k}"] / len(scores), 5)

    return ret
