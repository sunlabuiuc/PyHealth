"""Sentence-level knowledge-distillation transformer (Kim et al., CHIL 2024).

This module implements the student-side model from:

    Kyungsu Kim, Junhyun Park, Saul Langarica, Adham M. Alkhadrawi, Synho Do.
    "Integrating ChatGPT into Secure Hospital Networks: A Case Study on
    Improving Radiology Report Analysis". PMLR 248:72-87, 2024.
    https://proceedings.mlr.press/v248/kim24a.html

The paper distills a cloud language model (GPT-3.5) into an on-premises
BERT-family student. The student is trained sentence-by-sentence on teacher
labels ``{normal, abnormal, uncertain}`` (paper Eq. 3) with a combined
cross-entropy + supervised-contrastive objective (paper Eq. 5). At inference,
a document-level anomaly probability is the maximum sentence-level abnormal
probability across the report (paper Eq. 4).

This PyHealth model reproduces the trainable student: loss, forward pass,
and a document-aggregation helper. The GPT-3.5 teacher labeling pipeline
(paper Appendix B) is intentionally out of scope.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ..datasets import SampleDataset
from .base_model import BaseModel


_VALID_DOC_AGG = ("max", "topk_mean", "attn")


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute the supervised contrastive loss of Khosla et al. (NeurIPS 2020).

    The loss encourages embeddings of samples sharing a class label to lie
    close in representation space and to be far from embeddings of other
    classes. It is the second term of the paper's Eq. 5::

        L_cont = - log E_{v in B_y} [ exp(sim(z, z_v) / tau) /
                                      sum_{k in B} exp(sim(z, z_k) / tau) ]

    If no class in the mini-batch has at least two positive examples (i.e.,
    there are no valid ``(anchor, positive)`` pairs), the function returns a
    zero-valued tensor that still carries ``features``'s autograd history so
    the overall loss remains differentiable.

    Args:
        features: Latent features of shape ``(batch, dim)``. Typically the
            pre-classifier ``[CLS]`` hidden state.
        labels: Class indices of shape ``(batch,)`` with dtype ``torch.long``.
        temperature: Softmax temperature ``tau``. The paper uses 0.07
            (following Khosla et al.).

    Returns:
        A scalar tensor with the mean supervised contrastive loss over
        anchors that have at least one same-class positive.

    Example:
        >>> features = torch.randn(8, 16, requires_grad=True)
        >>> labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
        >>> loss = supervised_contrastive_loss(features, labels)
        >>> loss.backward()
    """
    if features.dim() != 2:
        raise ValueError(
            f"features must be 2-D (batch, dim); got shape {tuple(features.shape)}"
        )
    if labels.dim() != 1 or labels.size(0) != features.size(0):
        raise ValueError(
            "labels must be 1-D with the same batch size as features; "
            f"got labels shape {tuple(labels.shape)}, features shape "
            f"{tuple(features.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0; got {temperature}")

    batch_size = features.size(0)
    zero = features.sum() * 0.0

    # Need at least one class with >= 2 positives to form an anchor-positive pair.
    unique, counts = torch.unique(labels, return_counts=True)
    if (counts >= 2).sum().item() == 0:
        return zero

    z = F.normalize(features, p=2, dim=1)
    logits = torch.matmul(z, z.t()) / temperature
    # Numerical stability: subtract per-row max before exponentiating.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    positive_mask = label_eq & ~self_mask
    denom_mask = ~self_mask

    exp_logits = torch.exp(logits) * denom_mask.to(logits.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_per_anchor = positive_mask.sum(dim=1)
    valid_anchor = pos_per_anchor > 0
    if valid_anchor.sum().item() == 0:
        return zero

    log_prob_pos = (log_prob * positive_mask.to(log_prob.dtype)).sum(dim=1)
    mean_log_prob_pos = log_prob_pos[valid_anchor] / pos_per_anchor[valid_anchor]
    return -mean_log_prob_pos.mean()


class SentenceKDTransformer(BaseModel):
    """HuggingFace BERT student with CE + supervised-contrastive KD loss.

    This model reproduces the student architecture from Kim et al. (CHIL
    2024). A pretrained BERT-family encoder maps each input text to a ``[CLS]``
    hidden state, a linear head produces logits over the task's classes
    (e.g. ``{normal, abnormal, uncertain}`` for sentence-level KD), and the
    training objective combines standard cross-entropy with the supervised
    contrastive loss of Khosla et al. (2020) weighted by ``lam`` (paper
    Eq. 5). The ``[CLS]`` hidden state is used directly as the contrastive
    feature (paper Sec. 4.5), not ``pooler_output``.

    A :meth:`document_predict` helper implements the paper's Eq. 4 inference:
    the document's anomaly probability is aggregated from per-sentence
    abnormal probabilities via ``max`` (paper default), top-k mean, or a
    temperature-controlled soft-max (``"attn"``) — the latter two are novel
    ablation modes not present in the paper.

    Args:
        dataset: Fitted :class:`~pyhealth.datasets.SampleDataset`. Must expose
            exactly one feature key (the input text) and one multiclass or
            binary label key.
        model_name: HuggingFace model identifier for the encoder backbone.
            The paper's best-performing backbone is
            ``"StanfordAIMI/RadBERT"`` (RadBERT-RoBERTa-4m, Table 2); other
            domain-specific BERTs such as ``"emilyalsentzer/Bio_ClinicalBERT"``
            or ``"dmis-lab/biobert-base-cased-v1.2"`` are drop-in.
        dropout: Dropout probability applied before the classification head
            and inside the backbone.
        lam: Weight ``lambda`` on the contrastive loss term (paper Eq. 5).
            Setting ``lam=0`` disables the contrastive term and recovers a
            pure cross-entropy student.
        temperature: Temperature ``tau`` for the contrastive loss.
        max_length: Maximum tokenizer length per input text. Sentences in
            MIMIC-CXR reports are short; 128 is a safe default.
        doc_agg: Aggregation mode for :meth:`document_predict`. One of
            ``"max"`` (paper Eq. 4), ``"topk_mean"``, or ``"attn"``.
        abnormal_class_index: Class index treated as the positive / anomaly
            class by :meth:`document_predict`. Defaults to ``None``; when
            ``None`` the helper resolves it from the dataset's label
            vocabulary by looking up ``"abnormal"``, falling back to the
            last index.

    Attributes:
        feature_key: Name of the single text feature key.
        label_key: Name of the single label key.
        tokenizer: HuggingFace ``PreTrainedTokenizer`` for ``model_name``.
        model: The backbone ``AutoModel``.
        fc: ``nn.Linear`` classification head mapping the encoder hidden
            dimension to ``get_output_size()``.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {"patient_id": "p0", "sentence": "No acute findings.",
        ...      "label": "normal"},
        ...     {"patient_id": "p1", "sentence": "Large pleural effusion.",
        ...      "label": "abnormal"},
        ...     {"patient_id": "p2", "sentence": "Possible consolidation.",
        ...      "label": "uncertain"},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"sentence": "text"},
        ...     output_schema={"label": "multiclass"},
        ...     dataset_name="demo",
        ... )
        >>> model = SentenceKDTransformer(
        ...     dataset=dataset, model_name="prajjwal1/bert-tiny",
        ...     lam=1.0, temperature=0.07,
        ... )  # doctest: +SKIP
        >>> loader = get_dataloader(dataset, batch_size=3, shuffle=False)
        >>> batch = next(iter(loader))  # doctest: +SKIP
        >>> out = model(**batch)  # doctest: +SKIP
        >>> sorted(out.keys())  # doctest: +SKIP
        ['embed', 'logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        model_name: str = "StanfordAIMI/RadBERT",
        dropout: float = 0.1,
        lam: float = 1.0,
        temperature: float = 0.07,
        max_length: int = 128,
        doc_agg: str = "max",
        abnormal_class_index: Optional[int] = None,
    ) -> None:
        super().__init__(dataset=dataset)

        if len(self.feature_keys) != 1:
            raise ValueError(
                "SentenceKDTransformer expects exactly one feature key; got "
                f"{self.feature_keys}"
            )
        if len(self.label_keys) != 1:
            raise ValueError(
                "SentenceKDTransformer expects exactly one label key; got "
                f"{self.label_keys}"
            )
        if doc_agg not in _VALID_DOC_AGG:
            raise ValueError(
                f"doc_agg must be one of {_VALID_DOC_AGG}; got {doc_agg!r}"
            )
        if lam < 0:
            raise ValueError(f"lam must be >= 0; got {lam}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {temperature}")
        if max_length <= 0:
            raise ValueError(f"max_length must be > 0; got {max_length}")

        self.feature_key: str = self.feature_keys[0]
        self.label_key: str = self.label_keys[0]
        self.model_name: str = model_name
        self.lam: float = float(lam)
        self.temperature: float = float(temperature)
        self.max_length: int = int(max_length)
        self.doc_agg: str = doc_agg

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        hidden_dim = self.model.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, self.get_output_size())

        self._abnormal_class_index: Optional[int] = abnormal_class_index
        # Optional soft temperature for the "attn" doc-agg mode. Chosen small
        # enough that "attn" behaves close to "max" on confident sentences
        # while remaining differentiable-like across the full document.
        self._attn_temperature: float = 0.1

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _encode(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a list of strings and return ``(logits, cls_features)``.

        Args:
            texts: Python sequence of raw input strings.

        Returns:
            A tuple ``(logits, cls_features)`` where:

            - ``logits`` has shape ``(batch, num_classes)``.
            - ``cls_features`` is the pre-classifier ``[CLS]`` hidden state
              of shape ``(batch, hidden_dim)``, used for the contrastive
              loss and for downstream embedding analysis.
        """
        enc = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        outputs = self.model(**enc)
        cls_features = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(self.dropout(cls_features))
        return logits, cls_features

    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Run a training / evaluation step.

        Args:
            **kwargs: Must contain ``self.feature_key`` (a batched list of
                raw strings) and ``self.label_key`` (a ``torch.Tensor`` of
                class indices). If ``embed=True`` is passed, the output
                also includes the pre-classifier ``[CLS]`` hidden state
                under key ``"embed"``.

        Returns:
            A dict with keys ``loss``, ``y_prob``, ``y_true``, ``logit``,
            and (if requested) ``embed``. ``loss`` is the combined
            cross-entropy + ``lam`` * supervised-contrastive loss from
            paper Eq. 5.
        """
        texts = kwargs[self.feature_key]
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, torch.Tensor):
            raise TypeError(
                "SentenceKDTransformer expects raw strings for the feature "
                "input; received a tensor. Ensure the task uses a 'text' "
                "processor so strings pass through untokenized."
            )

        logits, cls_features = self._encode(texts)
        y_true = kwargs[self.label_key].to(self.device)

        ce_loss = self.get_loss_function()(logits, y_true)

        if self.lam > 0 and y_true.dim() == 1:
            con_loss = supervised_contrastive_loss(
                cls_features, y_true, temperature=self.temperature
            )
            loss = ce_loss + self.lam * con_loss
        else:
            loss = ce_loss

        y_prob = self.prepare_y_prob(logits)
        result: Dict[str, torch.Tensor] = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            result["embed"] = cls_features
        return result

    # ------------------------------------------------------------------
    # Document-level inference (paper Eq. 4 and novel ablation modes)
    # ------------------------------------------------------------------
    def _resolve_abnormal_class_index(self) -> int:
        """Return the class index representing the anomaly class.

        Resolution order:

        1. Constructor arg ``abnormal_class_index`` if provided.
        2. The label vocabulary entry for the string ``"abnormal"``.
        3. The last index in the label vocabulary (compatibility fallback
           so that plain binary setups default to ``1`` = positive).
        """
        if self._abnormal_class_index is not None:
            return int(self._abnormal_class_index)

        try:
            processor = self.dataset.output_processors[self.label_key]
            vocab = getattr(processor, "label_vocab", None)
            if isinstance(vocab, dict):
                for key, idx in vocab.items():
                    if isinstance(key, str) and key.lower() == "abnormal":
                        return int(idx)
                num_classes = len(vocab)
                if num_classes > 0:
                    return num_classes - 1
        except Exception:
            pass

        out_size = self.get_output_size() or 1
        return max(0, out_size - 1)

    def document_predict(
        self,
        sentences: Sequence[str],
        doc_agg: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Aggregate per-sentence predictions into a document-level prediction.

        Paper Eq. 4 aggregates with ``max`` — a document is deemed abnormal
        as soon as any of its sentences is abnormal. This helper exposes
        two additional aggregation modes (``"topk_mean"`` and ``"attn"``)
        that are new to this PyHealth implementation and used in the
        accompanying ablation study.

        Args:
            sentences: Raw sentence strings for a single document.
            doc_agg: Optional override of ``self.doc_agg``. Must be one of
                ``"max"`` (paper), ``"topk_mean"``, ``"attn"``.
            top_k: ``k`` for the ``"topk_mean"`` aggregation.

        Returns:
            A dict with:

            - ``pa``: Document-level probability of being abnormal.
            - ``pn``: ``1 - pa``.
            - ``per_sentence_probs``: Tensor of shape
              ``(num_sentences, num_classes)``.
            - ``abnormal_index``: The class index used as the anomaly class.

        Raises:
            ValueError: If ``sentences`` is empty or ``doc_agg`` is invalid.
        """
        if len(sentences) == 0:
            raise ValueError("document_predict requires at least one sentence")

        mode = self.doc_agg if doc_agg is None else doc_agg
        if mode not in _VALID_DOC_AGG:
            raise ValueError(
                f"doc_agg must be one of {_VALID_DOC_AGG}; got {mode!r}"
            )
        abnormal_idx = self._resolve_abnormal_class_index()

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                logits, _ = self._encode(list(sentences))
                per_sentence_probs = F.softmax(logits, dim=-1)
        finally:
            if was_training:
                self.train()

        abnormal_probs = per_sentence_probs[:, abnormal_idx]
        if mode == "max":
            pa = abnormal_probs.max()
        elif mode == "topk_mean":
            k = min(int(top_k), abnormal_probs.numel())
            pa = torch.topk(abnormal_probs, k=k).values.mean()
        else:  # "attn": softmax-weighted average of sentence abnormal probs
            weights = F.softmax(abnormal_probs / self._attn_temperature, dim=0)
            pa = (weights * abnormal_probs).sum()

        pa_val = float(pa.clamp(min=0.0, max=1.0).item())
        return {
            "pa": pa_val,
            "pn": float(max(0.0, 1.0 - pa_val)),
            "per_sentence_probs": per_sentence_probs,
            "abnormal_index": abnormal_idx,
        }


if __name__ == "__main__":
    from pyhealth.datasets import create_sample_dataset, get_dataloader

    samples = [
        {"patient_id": f"p{i}", "sentence": s, "label": lab}
        for i, (s, lab) in enumerate([
            ("no acute intrathoracic abnormality", "normal"),
            ("large pleural effusion on the right", "abnormal"),
            ("possible early consolidation", "uncertain"),
            ("lungs are clear", "normal"),
            ("opacity consistent with pneumonia", "abnormal"),
            ("unclear if atelectasis vs effusion", "uncertain"),
        ])
    ]
    ds = create_sample_dataset(
        samples=samples,
        input_schema={"sentence": "text"},
        output_schema={"label": "multiclass"},
        dataset_name="sentence-kd-smoke",
    )
    mdl = SentenceKDTransformer(
        dataset=ds, model_name="prajjwal1/bert-tiny",
        lam=1.0, temperature=0.07, max_length=32,
    )
    loader = get_dataloader(ds, batch_size=6, shuffle=False)
    out = mdl(**next(iter(loader)))
    print({k: tuple(v.shape) if hasattr(v, "shape") else v
           for k, v in out.items()})
    doc = mdl.document_predict([s["sentence"] for s in samples])
    print("doc:", doc["pa"], doc["pn"], doc["per_sentence_probs"].shape)
