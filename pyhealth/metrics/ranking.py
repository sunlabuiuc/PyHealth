from typing import List, Dict


def ranking_metrics_fn(qrels: Dict[str, Dict[str, int]],
                       results: Dict[str, Dict[str, float]],
                       k_values: List[int]) -> Dict[str, float]:
    """Computes metrics for ranking tasks.

    Args:
        qrels: Ground truth. A dictionary of query ids and their corresponding
            relevance judgements. The relevance judgements are a dictionary of
            document ids and their corresponding relevance scores.
        results: Ranked results. A dictionary of query ids and their corresponding
            document scores. The document scores are a dictionary of document ids and
            their corresponding scores.
        k_values: A list of integers specifying the cutoffs for the metrics.

    Returns:
        A dictionary of metrics and their corresponding values.

    Examples:
        >>> qrels = {
        ...     "q1": {"d1": 1, "d2": 0, "d3": 1},
        ...     "q2": {"d1": 1, "d2": 1, "d3": 0}
        ... }
        >>> results = {
        ...     "q1": {"d1": 0.5, "d2": 0.2, "d3": 0.1},
        ...     "q2": {"d1": 0.1, "d2": 0.2, "d3": 0.5}
        ... }
        >>> k_values = [1, 2]
        >>> ranking_metrics_fn(qrels, results, k_values)
        {'NDCG@1': 0.5, 'MAP@1': 0.25, 'Recall@1': 0.25, 'P@1': 0.5, 'NDCG@2': 0.5, 'MAP@2': 0.375, 'Recall@2': 0.5, 'P@2': 0.5}
    """
    try:
        import pytrec_eval
    except:
        raise ImportError("pytrec_eval is not installed. Please install it manually by running \
            'pip install pytrec_eval'.")
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


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
