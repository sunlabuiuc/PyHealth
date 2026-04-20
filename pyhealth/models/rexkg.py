"""
PyHealth model for radiology knowledge graph extraction (ReXKG).

Implements a span-based named entity recognition (NER) model and a pairwise
relation extractor on top of a BERT backbone, following the PURE architecture
used in the original ReXKG paper.  A lightweight KG builder is included to
convert model predictions into a structured knowledge graph compatible with the
rest of the ReXKG pipeline.

ReXKG paper: (please cite if you use this module)
    Li, Z., et al. "ReXKG: A Structured Radiology Report Knowledge Graph for
    Chest X-ray Analysis." arXiv:2408.14397 (2024).

ReXKG paper link:
    https://arxiv.org/abs/2408.14397

Authors:
    Aaron Miller (aaronm6@illinois.edu)
    Kathryn Thompson (kyt3@illinois.edu)
    Pushpendra Tiwari (pkt3@illinois.edu)
"""

import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.tasks.rexkg_extraction import NER_LABELS, RELATION_TYPES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NER_LABEL2ID: Dict[str, int] = {lbl: i + 1 for i, lbl in enumerate(NER_LABELS)}
_NER_ID2LABEL: Dict[int, str] = {v: k for k, v in _NER_LABEL2ID.items()}
_NER_ID2LABEL[0] = "O"  # index 0 → outside / non-entity

_REL_LABEL2ID: Dict[str, int] = {lbl: i + 1 for i, lbl in enumerate(RELATION_TYPES)}
_REL_ID2LABEL: Dict[int, str] = {v: k for k, v in _REL_LABEL2ID.items()}
_REL_ID2LABEL[0] = "none"


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class _SpanNERHead(nn.Module):
    """Span-based NER classification head.

    For each candidate span ``[start, end]`` in a sentence this head
    produces a distribution over :data:`NER_LABELS` + the ``"O"`` class.

    The span representation is constructed by concatenating:
    * the BERT hidden state of the *start* token,
    * the BERT hidden state of the *end* token, and
    * a learned width embedding for the span length.

    Args:
        hidden_size (int): BERT hidden dimension.
        num_ner_labels (int): Number of entity types (including ``"O"``).
        width_embedding_dim (int): Dimension for the span-width embedding.
        max_span_length (int): Maximum span length (in tokens).
        dropout (float): Dropout probability applied before the classifier.
    """

    def __init__(
        self,
        hidden_size: int,
        num_ner_labels: int,
        width_embedding_dim: int = 150,
        max_span_length: int = 8,
        span_hidden_dim: int = 150,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.width_embedding = nn.Embedding(
            max_span_length + 1, width_embedding_dim
        )
        span_input_dim = hidden_size * 2 + width_embedding_dim
        # Mirror the original checkpoint's 2-part structure:
        #   network: Linear(span_input_dim → span_hidden_dim) → ReLU → Dropout → Linear(hidden → hidden)
        #   output:  Linear(span_hidden_dim → num_ner_labels)
        # This allows the NER checkpoint to load with exact key matches.
        self.network = nn.Sequential(
            nn.Linear(span_input_dim, span_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(span_hidden_dim, span_hidden_dim),
        )
        self.output = nn.Linear(span_hidden_dim, num_ner_labels)

    def forward(
        self,
        sequence_output: torch.Tensor,
        spans: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NER logits for a batch of spans.

        Args:
            sequence_output (torch.Tensor): BERT output of shape
                ``(batch, seq_len, hidden_size)``.
            spans (torch.Tensor): Span indices of shape
                ``(batch, num_spans, 3)`` where the last dim is
                ``[start, end, width]``.

        Returns:
            torch.Tensor: Logits of shape ``(batch, num_spans, num_ner_labels)``.
        """
        start_idx = spans[:, :, 0]  # (B, S)
        end_idx = spans[:, :, 1]    # (B, S)
        width = spans[:, :, 2]      # (B, S)

        batch_size, seq_len, hidden = sequence_output.size()
        # Gather start / end representations
        start_rep = sequence_output[
            torch.arange(batch_size).unsqueeze(1), start_idx
        ]  # (B, S, H)
        end_rep = sequence_output[
            torch.arange(batch_size).unsqueeze(1), end_idx
        ]  # (B, S, H)

        width_rep = self.width_embedding(width)  # (B, S, W)
        span_rep = torch.cat([start_rep, end_rep, width_rep], dim=-1)
        return self.output(self.network(span_rep))


class _PairwiseRelationHead(nn.Module):
    """Pairwise relation classification head.

    For each ordered entity pair ``(subject, object)`` this head predicts
    a distribution over :data:`RELATION_TYPES` + the ``"none"`` class.

    The pair representation is the concatenation of both entity span vectors,
    each produced by averaging the BERT token states within the span.

    Args:
        hidden_size (int): BERT hidden dimension.
        num_rel_labels (int): Number of relation types (including ``"none"``).
        dropout (float): Dropout probability applied before the classifier.
    """

    def __init__(
        self,
        hidden_size: int,
        num_rel_labels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_rel_labels),
        )

    def forward(
        self,
        sequence_output: torch.Tensor,
        entity_spans: List[List[Tuple[int, int]]],
    ) -> List[torch.Tensor]:
        """Compute relation logits for all entity pairs in the batch.

        Args:
            sequence_output (torch.Tensor): BERT output of shape
                ``(batch, seq_len, hidden_size)``.
            entity_spans (List[List[Tuple[int, int]]]): For each sentence in
                the batch a list of ``(start, end)`` index tuples identifying
                predicted entity spans.

        Returns:
            List[torch.Tensor]: One logit tensor per sentence of shape
                ``(num_pairs, num_rel_labels)``.  An empty list is returned for
                sentences with fewer than two predicted entities.
        """
        results = []
        for i, spans in enumerate(entity_spans):
            if len(spans) < 2:
                results.append(
                    torch.zeros(
                        0,
                        self.classifier[-1].out_features,
                        device=sequence_output.device,
                        dtype=sequence_output.dtype,
                    )
                )
                continue

            # Average pool tokens for each entity span
            span_reps = []
            for start, end in spans:
                rep = sequence_output[i, start : end + 1].mean(dim=0)
                span_reps.append(rep)
            span_reps_t = torch.stack(span_reps)  # (E, H)

            # All ordered pairs (subject, object)
            n = len(span_reps)
            sub_idx = [s for s in range(n) for _ in range(n) if s != _]
            obj_idx = [o for s in range(n) for o in range(n) if s != o]
            sub_idx = []
            obj_idx = []
            for s in range(n):
                for o in range(n):
                    if s != o:
                        sub_idx.append(s)
                        obj_idx.append(o)

            pairs = torch.cat(
                [span_reps_t[sub_idx], span_reps_t[obj_idx]], dim=-1
            )  # (P, 2H)
            results.append(self.classifier(pairs))
        return results


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class ReXKGModel(BaseModel):
    """Radiology Knowledge Graph extraction model (ReXKG).

    Jointly trains a span-based NER head and a pairwise relation extraction
    head on top of a shared BERT encoder.  After inference, :meth:`build_kg`
    converts the predicted entities and relations into a structured knowledge
    graph in the format used by the rest of the ReXKG pipeline.

    Architecture (PURE-based):

    * **Encoder**: pre-trained BERT (``bert-base-uncased`` by default).
    * **NER head**: span representation (start + end + width embedding) fed
      through a two-layer MLP → entity-type logits.
    * **RE head**: average-pooled entity pair representation fed through a
      two-layer MLP → relation-type logits.
    * **KG builder**: entity merging + deduplication + size-relation
      extraction → JSON-serialisable KG.

    Args:
        dataset (SampleDataset): The processed sample dataset produced by
            :meth:`~pyhealth.datasets.CheXpertPlusDataset.set_task`.
        bert_model_name (str): HuggingFace model id for the BERT encoder.
            Defaults to ``"bert-base-uncased"``.
        max_span_length (int): Maximum span length considered during NER.
            Defaults to ``8``.
        width_embedding_dim (int): Dimension of the span-width embedding.
            Defaults to ``150``.
        ner_dropout (float): Dropout for the NER head.  Defaults to ``0.1``.
        rel_dropout (float): Dropout for the RE head.  Defaults to ``0.1``.
        context_window (int): Number of tokens to include as left/right
            context when encoding a sentence.  Defaults to ``100``.

    Example::
        >>> from pyhealth.datasets import CheXpertPlusDataset
        >>> from pyhealth.tasks import RadiologyKGExtractionTask
        >>> from pyhealth.models import ReXKGModel
        >>>
        >>> base = CheXpertPlusDataset(root="/path/to/chexpert_plus")
        >>> samples = base.set_task(RadiologyKGExtractionTask())
        >>> model = ReXKGModel(dataset=samples)
        >>> kg = model.build_kg(reports=["No acute cardiopulmonary process."])
    """

    def __init__(
        self,
        dataset: Optional[SampleDataset] = None,
        bert_model_name: str = "bert-base-uncased",
        max_span_length: int = 8,
        width_embedding_dim: int = 150,
        span_hidden_dim: int = 150,
        ner_dropout: float = 0.1,
        rel_dropout: float = 0.1,
    ) -> None:
        super().__init__(dataset)

        self.bert_model_name = bert_model_name
        self.max_span_length = max_span_length

        # BERT encoder
        self.encoder = AutoModel.from_pretrained(bert_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        hidden_size: int = self.encoder.config.hidden_size

        # Task heads
        num_ner_labels = len(NER_LABELS) + 1  # +1 for "O"
        num_rel_labels = len(RELATION_TYPES) + 1  # +1 for "none"
        self.ner_head = _SpanNERHead(
            hidden_size=hidden_size,
            num_ner_labels=num_ner_labels,
            width_embedding_dim=width_embedding_dim,
            max_span_length=max_span_length,
            span_hidden_dim=span_hidden_dim,
            dropout=ner_dropout,
        )
        self.rel_head = _PairwiseRelationHead(
            hidden_size=hidden_size,
            num_rel_labels=num_rel_labels,
            dropout=rel_dropout,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run the BERT encoder and return the last hidden states.

        Args:
            input_ids (torch.Tensor): Token ids of shape ``(B, L)``.
            attention_mask (torch.Tensor): Attention mask of shape ``(B, L)``.

        Returns:
            torch.Tensor: BERT last hidden states of shape ``(B, L, H)``.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def _enumerate_spans(
        self, seq_len: int
    ) -> List[Tuple[int, int, int]]:
        """Enumerate all valid spans ``(start, end, width)`` for a sequence.

        Args:
            seq_len (int): Number of tokens in the sequence (excluding
                ``[CLS]`` / ``[SEP]``).

        Returns:
            List[Tuple[int, int, int]]: Each tuple is ``(start, end, width)``
            with ``1 <= width <= max_span_length``.
        """
        spans = []
        for start in range(seq_len):
            for end in range(start, min(start + self.max_span_length, seq_len)):
                width = end - start + 1
                spans.append((start, end, width))
        return spans

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass returning NER and relation logits.

        Expects the following keys in ``kwargs``:

        * ``input_ids`` (torch.Tensor): ``(B, L)``
        * ``attention_mask`` (torch.Tensor): ``(B, L)``
        * ``spans`` (torch.Tensor): ``(B, num_spans, 3)`` — each row is
          ``[start, end, width]`` (pre-computed by the collate function).
        * ``ner_labels`` (torch.Tensor, optional): ``(B, num_spans)`` NER
          label indices.  When provided the NER loss is included.
        * ``entity_spans`` (List[List[Tuple[int,int]]], optional): Gold entity
          span ``(start, end)`` pairs per sentence (1-indexed, CLS-inclusive).
          Required for RE training.
        * ``rel_labels`` (List[torch.Tensor], optional): For each sentence,
          a 1-D tensor of relation-type label ids for every ordered
          ``(subject, object)`` pair over ``entity_spans``.  Label ``0``
          means ``"none"``.  Required for RE training.

        Returns:
            Dict[str, torch.Tensor]: A dict with keys:

            * ``"ner_logits"`` — shape ``(B, num_spans, num_ner_labels)``
            * ``"ner_loss"`` — scalar, present only when ``ner_labels`` given
            * ``"rel_loss"`` — scalar, present only when ``rel_labels`` given
            * ``"loss"``     — sum of ner_loss + rel_loss (for Trainer)
        """
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        spans = kwargs["spans"]

        sequence_output = self._encode(input_ids, attention_mask)
        ner_logits = self.ner_head(sequence_output, spans)

        out: Dict[str, torch.Tensor] = {"ner_logits": ner_logits}
        total_loss = None

        if "ner_labels" in kwargs:
            ner_labels = kwargs["ner_labels"]
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            B, S, C = ner_logits.size()
            ner_loss = loss_fct(
                ner_logits.view(B * S, C), ner_labels.view(B * S)
            )
            out["ner_loss"] = ner_loss
            total_loss = ner_loss

        if "entity_spans" in kwargs and "rel_labels" in kwargs:
            entity_spans = kwargs["entity_spans"]   # List[List[(start,end)]]
            rel_labels_list = kwargs["rel_labels"]  # List[Tensor]
            rel_logits_list = self.rel_head(sequence_output, entity_spans)
            rel_loss_fct = nn.CrossEntropyLoss()
            rel_losses = []
            for logits, labels in zip(rel_logits_list, rel_labels_list):
                if logits.numel() > 0 and labels.numel() > 0:
                    rel_losses.append(rel_loss_fct(logits, labels.to(logits.device)))
            if rel_losses:
                rel_loss = torch.stack(rel_losses).mean()
                out["rel_loss"] = rel_loss
                total_loss = rel_loss if total_loss is None else total_loss + rel_loss

        if total_loss is not None:
            out["loss"] = total_loss  # pyhealth.Trainer expects "loss"

        return out

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_entities(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[Dict]]:
        """Run NER inference on a list of radiology report texts.

        Args:
            texts (List[str]): Raw report strings.
            batch_size (int): Number of reports per GPU batch.

        Returns:
            List[List[Dict]]: One list per report; each dict contains:

            * ``"start"`` (int): Start token index.
            * ``"end"`` (int): End token index.
            * ``"text"`` (str): Surface form of the entity.
            * ``"type"`` (str): Predicted entity type from :data:`NER_LABELS`.
        """
        self.eval()
        device = self.device
        all_entities: List[List[Dict]] = []

        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start : batch_start + batch_size]

            encoding = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            offset_mapping = encoding["offset_mapping"]

            sequence_output = self._encode(input_ids, attention_mask)

            for i, text in enumerate(batch_texts):
                seq_len = int(attention_mask[i].sum().item()) - 2  # strip CLS/SEP
                if seq_len <= 0:
                    all_entities.append([])
                    continue

                raw_spans = self._enumerate_spans(seq_len)
                if not raw_spans:
                    all_entities.append([])
                    continue

                span_tensor = torch.tensor(
                    raw_spans, dtype=torch.long, device=device
                ).unsqueeze(0)  # (1, S, 3), token indices exclude CLS/SEP
                ner_span_tensor = span_tensor.clone()
                ner_span_tensor[..., 0:2] += 1  # shift start/end to align with CLS-inclusive sequence_output

                ner_logits = self.ner_head(
                    sequence_output[i].unsqueeze(0), ner_span_tensor
                )  # (1, S, num_labels)
                pred_ids = ner_logits.squeeze(0).argmax(dim=-1)  # (S,)

                entities = []
                offsets = offset_mapping[i]  # (L, 2)
                for span_idx, (start_tok, end_tok, _width) in enumerate(raw_spans):
                    label_id = pred_ids[span_idx].item()
                    if label_id == 0:
                        continue  # "O" — not an entity
                    # Map token indices → char offsets
                    char_start = offsets[start_tok + 1][0].item()  # +1 for CLS
                    char_end = offsets[end_tok + 1][1].item()
                    entity_text = text[char_start:char_end].strip()
                    if entity_text:
                        entities.append(
                            {
                                "start": start_tok,
                                "end": end_tok,
                                "text": entity_text,
                                "type": _NER_ID2LABEL[label_id],
                            }
                        )
                all_entities.append(entities)

        return all_entities

    @torch.no_grad()
    def predict_relations(
        self,
        texts: List[str],
        entity_lists: List[List[Dict]],
        batch_size: int = 32,
    ) -> List[List[Dict]]:
        """Run relation extraction inference given predicted entities.

        Args:
            texts (List[str]): Raw report strings (same order as
                ``entity_lists``).
            entity_lists (List[List[Dict]]): NER predictions from
                :meth:`predict_entities`.
            batch_size (int): Number of reports per GPU batch.

        Returns:
            List[List[Dict]]: One list per report; each dict contains:

            * ``"subject"`` (Dict): Subject entity.
            * ``"object"`` (Dict): Object entity.
            * ``"relation"`` (str): Predicted relation type.
        """
        self.eval()
        device = self.device
        all_relations: List[List[Dict]] = []

        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start : batch_start + batch_size]
            batch_entities = entity_lists[batch_start : batch_start + batch_size]

            encoding = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            sequence_output = self._encode(input_ids, attention_mask)

            entity_spans_batch = [
                [(e["start"] + 1, e["end"] + 1) for e in ents]  # +1 for CLS
                for ents in batch_entities
            ]
            rel_logits_batch = self.rel_head(sequence_output, entity_spans_batch)

            for i, (entities, rel_logits) in enumerate(
                zip(batch_entities, rel_logits_batch)
            ):
                relations: List[Dict] = []
                if len(entities) < 2 or rel_logits.numel() == 0:
                    all_relations.append(relations)
                    continue

                pred_ids = rel_logits.argmax(dim=-1)
                pair_idx = 0
                for s in range(len(entities)):
                    for o in range(len(entities)):
                        if s == o:
                            continue
                        rel_id = pred_ids[pair_idx].item()
                        if rel_id != 0:  # 0 → "none"
                            relations.append(
                                {
                                    "subject": entities[s],
                                    "object": entities[o],
                                    "relation": _REL_ID2LABEL[rel_id],
                                }
                            )
                        pair_idx += 1
                all_relations.append(relations)

        return all_relations

    # ------------------------------------------------------------------
    # KG builder
    # ------------------------------------------------------------------

    def build_kg(
        self,
        reports: List[str],
        patient_ids: Optional[List[str]] = None,
        batch_size: int = 32,
    ) -> Dict:
        """Run the full ReXKG pipeline on a list of radiology reports.

        Executes NER → relation extraction → KG construction in sequence and
        returns a structured knowledge graph dict.

        Args:
            reports (List[str]): Raw radiology report texts.
            patient_ids (Optional[List[str]]): Optional study identifiers used
                as keys in the output KG.  If ``None``, integer indices are
                used.
            batch_size (int): Batch size for BERT inference.

        Returns:
            Dict: A knowledge graph with the following top-level keys:

            * ``"nodes"`` (List[Dict]): Unique entity nodes, each with
              ``"id"``, ``"text"``, and ``"type"``.
            * ``"edges"`` (List[Dict]): Relation edges, each with
              ``"subject_id"``, ``"object_id"``, and ``"relation"``.
            * ``"subgraphs"`` (Dict[str, Dict]): Per-study subgraph containing
              ``"entities"`` and ``"relations"`` for that study.

        Example::
            >>> kg = model.build_kg(
            ...     reports=["Mild cardiomegaly is present."],
            ...     patient_ids=["patient01/study1/view1.jpg"],
            ... )
            >>> kg["nodes"][0]
            {'id': 0, 'text': 'cardiomegaly', 'type': 'Observation'}
        """
        if patient_ids is None:
            patient_ids = [str(i) for i in range(len(reports))]
        elif len(patient_ids) != len(reports):
            raise ValueError(
                "patient_ids and reports must have the same length; "
                f"got {len(patient_ids)} patient_ids and {len(reports)} reports."
            )

        entity_lists = self.predict_entities(reports, batch_size=batch_size)
        relation_lists = self.predict_relations(
            reports, entity_lists, batch_size=batch_size
        )

        # Build global node registry (deduplicate by normalised surface form)
        node_text_to_id: Dict[str, int] = {}
        nodes: List[Dict] = []

        def _get_or_add_node(entity: Dict) -> int:
            key = entity["text"].lower().strip()
            if key not in node_text_to_id:
                node_id = len(nodes)
                node_text_to_id[key] = node_id
                nodes.append(
                    {"id": node_id, "text": entity["text"], "type": entity["type"]}
                )
            return node_text_to_id[key]

        edges: List[Dict] = []
        seen_edges = set()
        subgraphs: Dict[str, Dict] = {}

        for pid, entities, relations in zip(
            patient_ids, entity_lists, relation_lists
        ):
            sub_entity_ids = [_get_or_add_node(e) for e in entities]
            sub_edges = []
            for rel in relations:
                subj_id = _get_or_add_node(rel["subject"])
                obj_id = _get_or_add_node(rel["object"])
                edge_key = (subj_id, obj_id, rel["relation"])
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edge = {
                        "subject_id": subj_id,
                        "object_id": obj_id,
                        "relation": rel["relation"],
                    }
                    edges.append(edge)
                    sub_edges.append(edge)

            subgraphs[pid] = {
                "entities": [nodes[i] for i in sub_entity_ids],
                "relations": sub_edges,
            }

        return {"nodes": nodes, "edges": edges, "subgraphs": subgraphs}

    def save_kg(self, kg: Dict, output_path: str) -> None:
        """Serialize a KG dict produced by :meth:`build_kg` to a JSON file.

        Args:
            kg (Dict): KG dict as returned by :meth:`build_kg`.
            output_path (str): Destination file path (will be created or
                overwritten).
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(kg, f, indent=2, ensure_ascii=False)
        logger.info("KG saved to %s", output_path)

    def evaluate(
        self,
        texts: List[str],
        gold_entities: List[List[Dict]],
        gold_relations: Optional[List[List[Dict]]] = None,
        batch_size: int = 32,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate entity and relation extraction against gold annotations.

        Computes span-level micro-averaged precision, recall, and F1 for both
        NER and (optionally) RE.  A predicted entity is correct only when its
        span *and* type exactly match the gold annotation.  A predicted
        relation is correct when subject span, object span, *and* relation
        type all match exactly.

        Args:
            texts (List[str]): Raw report strings in the same order as the
                gold annotation lists.
            gold_entities (List[List[Dict]]): Gold entity annotations; one
                list per document.  Each entity dict must contain:

                * ``"start"`` (int): Token start index.
                * ``"end"`` (int): Token end index.
                * ``"type"`` (str): Entity type string.

            gold_relations (List[List[Dict]], optional): Gold relation
                annotations; one list per document.  Each relation dict must
                contain:

                * ``"subject"`` (Dict): Dict with ``"start"`` and ``"end"``.
                * ``"object"`` (Dict): Dict with ``"start"`` and ``"end"``.
                * ``"relation"`` (str): Relation type string.

                When *None*, relation F1 is computed against empty gold sets
                (all predicted relations will count as false positives).
            batch_size (int): Batch size passed to :meth:`predict_entities`
                and :meth:`predict_relations`.

        Returns:
            Dict with two keys:

            * ``"entity"`` — dict with ``precision``, ``recall``, ``f1``,
              ``tp``, ``fp``, ``fn``.
            * ``"relation"`` — same structure (zero if *gold_relations* is
              ``None``).

        Example::

            >>> metrics = model.evaluate(texts, gold_entities, gold_relations)
            >>> print(f"Entity F1: {metrics['entity']['f1']:.3f}")
            >>> print(f"Relation F1: {metrics['relation']['f1']:.3f}")
        """
        from pyhealth.metrics.rexkg import rexkg_metrics

        pred_entities = self.predict_entities(texts, batch_size=batch_size)
        pred_relations = self.predict_relations(
            texts, pred_entities, batch_size=batch_size
        )

        if gold_relations is None:
            gold_relations = [[] for _ in texts]

        return rexkg_metrics(
            pred_entities, gold_entities, pred_relations, gold_relations
        )
