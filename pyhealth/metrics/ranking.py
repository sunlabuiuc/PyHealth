"""Ranking metrics: ``ranking_metrics_fn`` (TREC style, needs ``pytrec_eval``)
and ``multilabel_ranking_at_k`` (pure NumPy, for multilabel score matrices).
"""

from typing import List, Dict

import numpy as np


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

    See Also:
        :func:`multilabel_ranking_at_k` computes ``P@k``, ``Recall@k`` and
        ``NDCG@k`` for an ``(n_samples, n_labels)`` score matrix in NumPy,
        without ``pytrec_eval``; its docstring lists where the two differ.

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
    except ImportError:
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


def multilabel_ranking_at_k(
    y_true: np.ndarray, y_prob: np.ndarray, k: int
) -> dict[str, float]:
    """Precision, recall and NDCG of each sample's top-``k`` labels.

    Each sample's labels are ranked by score, ties keeping label order. With
    ``hits`` the true labels among the top ``k`` and ``n`` the sample's true
    labels, precision@k is ``hits / k``, recall@k is ``hits / n``, and NDCG@k
    is ``sum(rel_r / log2(r + 1)) / sum_{r <= min(n, k)} 1 / log2(r + 1)``
    for ranks ``r`` from 1. All three are averaged over the samples with
    ``n > 0`` (recall and NDCG are undefined without a true label, and the
    three are kept on the same samples); they are 0 when no sample has one.
    Precision@k and recall@k are those of TRACE's released code
    (``recalls_and_ndcgs_for_ks`` in ``model/trainer.py``, where every
    sample has a true label). NDCG@k here is the standard one; the released
    code weights every rank by ``1 / log2(2) = 1``, so its "NDCG@k" is
    ``hits / min(n, k)``, which equals precision@k whenever ``n >= k``.

    Relation to :func:`ranking_metrics_fn`: the two use the same
    definitions of ``P@k``, ``Recall@k`` and ``NDCG@k`` (binary relevance),
    and give the same values when scores have no ties, every sample has a
    true label and ``k`` is at most the number of labels
    (``tests/core/test_trace.py`` checks hand-computed values, and the two
    functions against each other when ``pytrec_eval`` is installed). This
    function does not call ``ranking_metrics_fn``, which:

    - needs the optional C extension ``pytrec_eval``, which is not a
      PyHealth dependency, while these metrics run in
      :func:`pyhealth.metrics.multilabel_metrics_fn` and the ``Trainer``;
    - breaks score ties by document id in descending string order
      (label ``"9"`` before ``"10"``); here ties keep label order, and
      tied scores are common (e.g. a label-frequency baseline, or
      saturated probabilities);
    - counts samples without a true label as 0 in all three averages;
      here they are left out;
    - divides precision by ``k`` even when ``k`` exceeds the number of
      labels (here ``k`` is clipped to it), and rounds to 5 decimals;
    - also returns ``MAP@k`` and accepts graded relevance.

    Args:
        y_true: Binary labels ``(n_samples, n_labels)``.
        y_prob: Scores of the same shape.
        k: Cutoff (clipped to the number of labels).

    Returns:
        ``{"precision": ..., "recall": ..., "ndcg": ...}``.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        >>> y_prob = np.array(
        ...     [[0.9, 0.8, 0.1, 0.0], [0.1, 0.2, 0.3, 0.4], [0.5, 0.4, 0.3, 0.2]]
        ... )
        >>> out = multilabel_ranking_at_k(y_true, y_prob, 2)
        >>> {m: round(v, 3) for m, v in out.items()}
        {'precision': 0.25, 'recall': 0.25, 'ndcg': 0.307}
    """
    y_true = np.asarray(y_true) > 0
    y_prob = np.asarray(y_prob, dtype=float)
    k = max(1, min(int(k), y_true.shape[1]))
    n_true = y_true.sum(axis=1)
    has_true = n_true > 0
    if not has_true.any():
        return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}
    y_true, y_prob, n_true = y_true[has_true], y_prob[has_true], n_true[has_true]
    top = np.argsort(-y_prob, axis=1, kind="stable")[:, :k]
    hits = np.take_along_axis(y_true, top, axis=1).astype(float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.cumsum(discounts)[np.minimum(n_true, k) - 1]
    return {
        "precision": float(hits.sum(axis=1).mean() / k),
        "recall": float((hits.sum(axis=1) / n_true).mean()),
        "ndcg": float(((hits * discounts).sum(axis=1) / ideal).mean()),
    }


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
