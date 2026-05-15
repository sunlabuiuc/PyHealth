"""Surface-level evaluation metrics for the ReXKG NER and RE tasks.

Entity F1: an entity is correct if its surface text AND type both match
the gold annotation exactly (case-insensitive).

Relation F1: a relation is correct if the subject text, object text, AND
relation type all match the gold annotation exactly (case-insensitive).

This text-based matching avoids tokenisation-index mismatches when comparing
BERT-subword model output against word-tokenised gold annotations.
"""

from typing import Dict, List, Tuple


def _entity_f1(
    pred: List[List[Dict]],
    gold: List[List[Dict]],
) -> Dict[str, float]:
    """Compute micro-averaged entity F1.

    Args:
        pred: One list of entity dicts per document.  Each dict must have
            ``"start"`` (int), ``"end"`` (int), ``"type"`` (str).
        gold: Same structure as *pred* but for gold annotations.

    Returns:
        Dict with keys ``"precision"``, ``"recall"``, ``"f1"``,
        ``"tp"``, ``"fp"``, ``"fn"``.
    """
    tp = fp = fn = 0
    for pred_doc, gold_doc in zip(pred, gold):
        pred_set = {(e["text"].lower(), e["type"]) for e in pred_doc}
        gold_set = {(e["text"].lower(), e["type"]) for e in gold_doc}
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _relation_f1(
    pred: List[List[Dict]],
    gold: List[List[Dict]],
) -> Dict[str, float]:
    """Compute micro-averaged relation F1.

    Args:
        pred: One list of relation dicts per document.  Each dict must have
            ``"subject"`` (dict with ``"start"``, ``"end"``),
            ``"object"`` (dict with ``"start"``, ``"end"``),
            ``"relation"`` (str).
        gold: Same structure as *pred*.

    Returns:
        Dict with keys ``"precision"``, ``"recall"``, ``"f1"``,
        ``"tp"``, ``"fp"``, ``"fn"``.
    """
    tp = fp = fn = 0
    for pred_doc, gold_doc in zip(pred, gold):
        def _rel_key(r: Dict) -> Tuple:
            return (
                r["subject"]["text"].lower(),
                r["object"]["text"].lower(),
                r["relation"],
            )

        pred_set = {_rel_key(r) for r in pred_doc}
        gold_set = {_rel_key(r) for r in gold_doc}
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def rexkg_metrics(
    pred_entities: List[List[Dict]],
    gold_entities: List[List[Dict]],
    pred_relations: List[List[Dict]],
    gold_relations: List[List[Dict]],
) -> Dict[str, Dict[str, float]]:
    """Compute both entity and relation F1 for a batch of documents.

    Args:
        pred_entities: Predicted entities per document (from
            :meth:`~pyhealth.models.ReXKGModel.predict_entities`).
        gold_entities: Gold entity annotations per document; each entity
            dict must have ``"start"``, ``"end"``, ``"type"``.
        pred_relations: Predicted relations per document (from
            :meth:`~pyhealth.models.ReXKGModel.predict_relations`).
        gold_relations: Gold relation annotations per document; each dict
            must have ``"subject"`` (with ``"start"``/``"end"``),
            ``"object"`` (with ``"start"``/``"end"``), ``"relation"``.

    Returns:
        Dict with keys ``"entity"`` and ``"relation"``, each containing
        ``"precision"``, ``"recall"``, ``"f1"``, ``"tp"``, ``"fp"``, ``"fn"``.

    Example::

        >>> from pyhealth.metrics.rexkg import rexkg_metrics
        >>> gold_ents = [[{"start": 0, "end": 1, "type": "anatomy"}]]
        >>> pred_ents = [[{"start": 0, "end": 1, "type": "anatomy"}]]
        >>> gold_rels = [[]]
        >>> pred_rels = [[]]
        >>> rexkg_metrics(pred_ents, gold_ents, pred_rels, gold_rels)
        {'entity': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 1, 'fp': 0, 'fn': 0}, 'relation': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}}
    """
    return {
        "entity": _entity_f1(pred_entities, gold_entities),
        "relation": _relation_f1(pred_relations, gold_relations),
    }
