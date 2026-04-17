"""BulkRNABert model for cancer type classification and survival prediction.

PyHealth implementation of BulkRNABert from:
    Gélard et al. (2024). BulkRNABert: Cancer prognosis from bulk RNA-seq
    based language models. https://doi.org/10.1101/2024.06.18.599483

Pretrained weights are released by InstaDeepAI on HuggingFace at
``InstaDeepAI/BulkRNABert``. The HuggingFace weights are Flax/JAX-based;
when ``AutoModel.from_pretrained`` cannot load a PyTorch checkpoint the model
automatically falls back to a scratch encoder whose architecture matches the
paper (4 blocks, 8 heads, dim=256) but uses a global linear projection
(``num_genes → embedding_dim``) before a single-token transformer pass so
that unit tests remain fast on CPU.  To use the published pretrained weights,
convert the Flax checkpoint to PyTorch and pass the local path as
``pretrained_model_name``.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel

try:
    from transformers import AutoModel

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class BulkRNABertMLP(nn.Module):
    """MLP head shared by both the classification and survival tasks.

    Implements a two-hidden-layer MLP with SELU activation, dropout, and
    layer normalisation as described in BulkRNABert (Gélard et al., 2024).
    The final linear layer has no activation and outputs raw logits or
    log-risk scores.

    Args:
        input_dim: Input dimension (encoder embedding size, typically 256).
        hidden_dims: Ordered list of hidden layer widths.  Classification
            uses ``[256, 128]``; survival uses ``[512, 256]``.
        output_dim: Output dimension.  ``num_classes`` for classification,
            ``1`` for survival.
        dropout: Dropout probability applied after each hidden activation.
            Defaults to ``0.1``.

    Examples:
        >>> mlp = BulkRNABertMLP(input_dim=256, hidden_dims=[256, 128], output_dim=33)
        >>> x = torch.randn(4, 256)
        >>> mlp(x).shape
        torch.Size([4, 33])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.SELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(h),
                ]
            )
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Pass ``x`` through the MLP and return raw logits.

        Args:
            x: Input tensor of shape ``(batch_size, input_dim)``.

        Returns:
            Output tensor of shape ``(batch_size, output_dim)``.
        """
        return self.net(x)


class CoxPartialLikelihoodLoss(nn.Module):
    """Negative Cox partial log-likelihood loss for survival analysis.

    Implements the discrete-time Cox loss used in BulkRNABert, following
    DeepSurv (Katzman et al., 2018).  The batch loss is::

        L = -mean_over_events( risk_i - log( sum_{j: T_j >= T_i} exp(risk_j) ) )

    Inputs are sorted by survival time descending so that the cumulative
    sum of ``exp(risk)`` equals the at-risk set for each event.  When the
    batch contains no observed events the loss is zero (no gradient signal).

    Examples:
        >>> loss_fn = CoxPartialLikelihoodLoss()
        >>> risk = torch.randn(4)
        >>> times = torch.tensor([100.0, 200.0, 300.0, 400.0])
        >>> events = torch.ones(4)
        >>> loss_fn(risk, times, events).ndim
        0
    """

    def forward(
        self,
        risk_scores: torch.FloatTensor,
        survival_times: torch.FloatTensor,
        events: torch.FloatTensor,
    ) -> torch.Tensor:
        """Compute the negative Cox partial log-likelihood.

        Args:
            risk_scores: Predicted log-risk scores of shape
                ``(batch_size, 1)`` or ``(batch_size,)``.
            survival_times: Observed or censored survival times of shape
                ``(batch_size,)``.
            events: Event indicators of shape ``(batch_size,)``.
                ``1`` = event occurred, ``0`` = censored.

        Returns:
            Scalar loss tensor.  Returns ``tensor(0.0)`` when all samples
            are censored.
        """
        risk = risk_scores.squeeze(-1)

        order = torch.argsort(survival_times, descending=True)
        risk = risk[order]
        events = events[order]

        log_cumsum_exp = torch.logcumsumexp(risk, dim=0)

        event_mask = events.bool()
        if event_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=risk.device)

        loss = -(risk[event_mask] - log_cumsum_exp[event_mask]).mean()
        return loss


class BulkRNABert(BaseModel):
    """BulkRNABert: transformer encoder for cancer prognosis from RNA-seq.

    Loads a pretrained BERT-style encoder trained on bulk RNA-seq data and
    fine-tunes it for cancer type classification or survival prediction.
    Optionally applies IA3 parameter-efficient fine-tuning, which adds
    learned scaling vectors to attention value projections and feed-forward
    intermediate activations while keeping the encoder weights frozen.

    This is a PyHealth implementation of BulkRNABert from:
        Gélard et al. (2024). BulkRNABert: Cancer prognosis from bulk RNA-seq
        based language models. https://doi.org/10.1101/2024.06.18.599483

    Note:
        The ``InstaDeepAI/BulkRNABert`` HuggingFace checkpoint is Flax/JAX-
        based.  When a PyTorch checkpoint is unavailable (the common case),
        the model falls back to a scratch encoder: a global linear projection
        from ``num_genes → embedding_dim`` followed by a single-token pass
        through a 4-block, 8-head transformer.  This preserves the paper's
        encoder architecture while keeping CPU inference fast enough for unit
        tests.  To use the published pretrained weights, supply a local path
        to a converted PyTorch checkpoint via ``pretrained_model_name``.

    Args:
        dataset: Optional PyHealth ``SampleDataset``.  Not required for
            standalone use; pass ``None`` (the default) to instantiate the
            model without a dataset.
        num_classes: Number of output classes for classification.  Use
            ``33`` for pan-cancer TCGA classification.  Ignored when
            ``task="survival"``.  Defaults to ``33``.
        task: Either ``"classification"`` or ``"survival"``.  Determines
            which head and loss function are used.  Defaults to
            ``"classification"``.
        pretrained_model_name: HuggingFace model identifier or local path
            for pretrained BulkRNABert weights.  Defaults to
            ``"InstaDeepAI/BulkRNABert"``.
        use_ia3: If ``True``, inject IA3 scaling vectors into all
            transformer layers and freeze the base encoder.  Only the IA3
            vectors and task head are trained.  Defaults to ``False``.
        freeze_encoder: If ``True``, freeze all encoder parameters.  Useful
            for linear probing.  Automatically set to ``True`` when
            ``use_ia3=True``.  Defaults to ``False``.
        dropout: Dropout probability for MLP heads.  Defaults to ``0.1``.
        embedding_dim: Dimension of the encoder output.  Should match the
            pretrained model (``256`` for BulkRNABert).  Defaults to
            ``256``.
        num_genes: Number of genes in the input expression vector.  Used
            only by the scratch encoder to size ``input_proj``.  Defaults
            to ``19042`` (the TCGA/GTEx common gene set from the paper).

    Examples:
        >>> model = BulkRNABert(num_classes=33, task="classification")
        >>> x = torch.randn(4, 19042)
        >>> out = model(x)
        >>> out["logits"].shape
        torch.Size([4, 33])

        >>> model = BulkRNABert(task="survival")
        >>> out = model(x)
        >>> out["risk_score"].shape
        torch.Size([4, 1])
    """

    def __init__(
        self,
        dataset: Optional[SampleDataset] = None,
        num_classes: int = 33,
        task: str = "classification",
        pretrained_model_name: str = "InstaDeepAI/BulkRNABert",
        use_ia3: bool = False,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
        embedding_dim: int = 256,
        num_genes: int = 19042,
    ) -> None:
        super().__init__(dataset=dataset)

        assert task in (
            "classification",
            "survival",
        ), f"task must be 'classification' or 'survival', got {task!r}"

        self.task = task
        self.num_classes = num_classes
        self.use_ia3 = use_ia3
        self.embedding_dim = embedding_dim
        self.num_genes = num_genes

        # ----- encoder -----
        self._encoder_is_scratch: bool
        self.input_proj: Optional[nn.Linear]

        if _TRANSFORMERS_AVAILABLE:
            try:
                self.encoder = AutoModel.from_pretrained(pretrained_model_name)
                self._encoder_is_scratch = False
                self.input_proj = None
                logger.info(
                    "Loaded pretrained BulkRNABert encoder from %s",
                    pretrained_model_name,
                )
            except Exception as exc:
                logger.warning(
                    "Could not load pretrained encoder from %s (%s). "
                    "Falling back to scratch encoder.",
                    pretrained_model_name,
                    exc,
                )
                self._build_scratch_encoder(embedding_dim, dropout, num_genes)
        else:
            logger.warning("transformers package not available. Using scratch encoder.")
            self._build_scratch_encoder(embedding_dim, dropout, num_genes)

        # ----- freeze encoder -----
        if freeze_encoder or use_ia3:
            for param in self.encoder.parameters():
                param.requires_grad = False
            if self.input_proj is not None:
                for param in self.input_proj.parameters():
                    param.requires_grad = False

        # ----- IA3 vectors -----
        if use_ia3:
            self._ia3_vectors: nn.ParameterList = nn.ParameterList()
            self._inject_ia3()

        # ----- task head -----
        if task == "classification":
            self.head = BulkRNABertMLP(
                input_dim=embedding_dim,
                hidden_dims=[256, 128],
                output_dim=num_classes,
                dropout=dropout,
            )
            self.loss_fn: nn.Module = nn.CrossEntropyLoss()
        else:
            self.head = BulkRNABertMLP(
                input_dim=embedding_dim,
                hidden_dims=[512, 256],
                output_dim=1,
                dropout=dropout,
            )
            self.loss_fn = CoxPartialLikelihoodLoss()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_scratch_encoder(
        self, embedding_dim: int, dropout: float, num_genes: int
    ) -> None:
        """Build the scratch fallback encoder.

        Projects the full gene-expression vector globally to ``embedding_dim``
        via a learned linear layer, then refines the representation with a
        4-block, 8-head transformer applied to the single compressed token.
        This matches the paper's encoder depth while remaining fast on CPU
        (the transformer processes a sequence of length 1, not 19 042).

        Args:
            embedding_dim: Hidden dimension of the transformer.
            dropout: Dropout probability used in each encoder layer.
            num_genes: Number of input gene features.
        """
        self.input_proj = nn.Linear(num_genes, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self._encoder_is_scratch = True

    def _inject_ia3(self) -> None:
        """Inject IA3 learned scaling vectors into transformer layers.

        For each transformer layer, registers two ``nn.Parameter`` vectors:

        * ``ia3_v`` (shape ``embedding_dim``) — scales the output of the
          multi-head self-attention sublayer (approximates value-vector
          scaling from the paper).
        * ``ia3_ff`` (shape ``dim_feedforward``) — scales the output of the
          first feed-forward linear (intermediate activations).

        Vectors are initialised to ``ones`` so the model starts as a
        no-op and learns the scaling during fine-tuning.  Hooks are
        registered on the relevant submodules so that the encoder weights
        themselves are never modified.

        Note:
            For HuggingFace BERT-style encoders, attribute paths
            (``encoder.encoder.layer``, ``attention.self``, etc.) depend on
            the specific model class.  If automatic injection fails an
            informational warning is logged.  The model continues to
            function correctly with ``use_ia3=False``.
        """
        if self._encoder_is_scratch:
            for layer in self.encoder.layers:
                d_v = self.embedding_dim
                d_ff: int = layer.linear1.out_features

                ia3_v = nn.Parameter(torch.ones(d_v))
                ia3_ff = nn.Parameter(torch.ones(d_ff))
                self._ia3_vectors.extend([ia3_v, ia3_ff])

                layer.self_attn.register_forward_hook(
                    self._make_ia3_attention_hook(ia3_v)
                )
                layer.linear1.register_forward_hook(self._make_ia3_ff_hook(ia3_ff))
        else:
            # HuggingFace BERT-style model
            try:
                for layer in self.encoder.encoder.layer:
                    d_k: int = layer.attention.self.key.weight.shape[0]
                    d_v_hf: int = layer.attention.self.value.weight.shape[0]
                    d_ff_hf: int = layer.intermediate.dense.weight.shape[0]

                    ia3_k = nn.Parameter(torch.ones(d_k))
                    ia3_v = nn.Parameter(torch.ones(d_v_hf))
                    ia3_ff = nn.Parameter(torch.ones(d_ff_hf))
                    self._ia3_vectors.extend([ia3_k, ia3_v, ia3_ff])

                    layer.attention.self.register_forward_hook(
                        self._make_ia3_attention_hook(ia3_v)
                    )
                    layer.intermediate.register_forward_hook(
                        self._make_ia3_ff_hook(ia3_ff)
                    )
            except AttributeError as exc:
                logger.warning(
                    "IA3 injection failed for HuggingFace encoder (%s). "
                    "Inspect model.encoder.named_modules() to find correct "
                    "attribute paths.",
                    exc,
                )

    def _make_ia3_attention_hook(self, ia3_v: nn.Parameter):
        """Return a forward hook that scales the attention output by ``ia3_v``.

        Args:
            ia3_v: Learned scaling vector of shape ``(embedding_dim,)``.

        Returns:
            A hook function compatible with ``register_forward_hook``.
        """

        def hook(module, input, output):  # noqa: ANN001
            if isinstance(output, tuple):
                return (output[0] * ia3_v, *output[1:])
            return output * ia3_v

        return hook

    def _make_ia3_ff_hook(self, ia3_ff: nn.Parameter):
        """Return a forward hook that scales FFN intermediate activations.

        Args:
            ia3_ff: Learned scaling vector of shape ``(dim_feedforward,)``.

        Returns:
            A hook function compatible with ``register_forward_hook``.
        """

        def hook(module, input, output):  # noqa: ANN001
            return output * ia3_ff

        return hook

    def _get_embedding(self, gene_expression: torch.FloatTensor) -> torch.FloatTensor:
        """Extract a fixed-size patient embedding from the encoder.

        For the scratch encoder the gene-expression vector is projected to
        ``embedding_dim`` via ``input_proj``, then refined by the transformer
        acting on that single token.

        For a loaded HuggingFace encoder, continuous ``[0, 1]`` values are
        discretised into 64 integer bins (matching the paper's tokenisation
        strategy) before being passed as ``input_ids``.  The last hidden state
        is mean-pooled across the gene dimension.

        Args:
            gene_expression: Normalised gene expression tensor of shape
                ``(batch_size, num_genes)``.  Values should be in ``[0, 1]``.

        Returns:
            Patient embedding tensor of shape ``(batch_size, embedding_dim)``.
        """
        if self._encoder_is_scratch:
            # (B, G) → (B, D) via linear projection
            projected = self.input_proj(gene_expression)  # type: ignore[misc]
            # Treat as a single-token sequence for the transformer
            hidden = self.encoder(projected.unsqueeze(1))  # (B, 1, D)
            embedding = hidden.squeeze(1)  # (B, D)
        else:
            # Discretise continuous values to 64 bins as per the paper
            gene_bins = (gene_expression * 63).long().clamp(0, 63)
            outputs = self.encoder(input_ids=gene_bins)
            hidden = outputs.last_hidden_state  # (B, G, D)
            embedding = hidden.mean(dim=1)  # (B, D)

        return embedding

    def forward(
        self,
        gene_expression: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        survival_times: Optional[torch.FloatTensor] = None,
        events: Optional[torch.FloatTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of BulkRNABert.

        Encodes gene expression through the pretrained transformer, then
        passes the mean-pooled patient embedding through the task-specific
        MLP head.

        Args:
            gene_expression: Gene expression tensor of shape
                ``(batch_size, num_genes)``.  Values should be
                log10(1+TPM) normalised and max-normalised to ``[0, 1]``.
            labels: Class labels of shape ``(batch_size,)`` for
                classification.  When provided the cross-entropy loss is
                included in the output dict.  Pass ``None`` for inference.
            survival_times: Survival times of shape ``(batch_size,)`` for
                the survival task.  Required for Cox loss computation.
            events: Event indicators of shape ``(batch_size,)`` for the
                survival task.  ``1`` = event (death), ``0`` = censored.
                Required for Cox loss computation.

        Returns:
            For ``task="classification"``:
                * ``"logits"``: shape ``(batch_size, num_classes)``
                * ``"loss"``: scalar cross-entropy loss
                  *(only present when* ``labels`` *is provided)*

            For ``task="survival"``:
                * ``"risk_score"``: shape ``(batch_size, 1)``
                * ``"loss"``: scalar Cox partial likelihood loss
                  *(only present when both* ``survival_times`` *and*
                  ``events`` *are provided)*

        Examples:
            >>> model = BulkRNABert(num_classes=33, task="classification")
            >>> x = torch.randn(4, 19042)
            >>> out = model(x)
            >>> out["logits"].shape
            torch.Size([4, 33])
            >>> out = model(x, labels=torch.randint(0, 33, (4,)))
            >>> "loss" in out
            True
        """
        embedding = self._get_embedding(gene_expression)

        if self.task == "classification":
            logits = self.head(embedding)
            output: Dict[str, torch.Tensor] = {"logits": logits}
            if labels is not None:
                output["loss"] = self.loss_fn(logits, labels)
            return output
        else:
            risk_score = self.head(embedding)
            output = {"risk_score": risk_score}
            if survival_times is not None and events is not None:
                output["loss"] = self.loss_fn(risk_score, survival_times, events)
            return output
