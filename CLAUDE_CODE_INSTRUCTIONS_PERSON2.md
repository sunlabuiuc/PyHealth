# Claude Code Instructions: BulkRNABert Model Implementation
# Person 2 — BulkRNABert PyHealth Contribution

---

## Your role and context

You are helping implement the model component of a PyHealth contribution for a
Deep Learning for Healthcare course final project at UIUC. The project reproduces
BulkRNABert (Gélard et al., 2024), a BERT-style transformer pretrained on bulk
RNA-seq data for cancer type classification and survival prediction.

We are working directly inside a fork of https://github.com/sunlabuiuc/PyHealth.
All files you create go directly into the correct locations in this fork. The
final PR will target the PyHealth main branch.

Before writing anything, read the following files that already exist in this repo:
  - pyhealth/models/base_model.py — understand BaseModel's interface
  - pyhealth/models/__init__.py — understand how models are exported
  - pyhealth/models/ — read 1-2 existing model implementations to understand patterns
  - docs/api/models.rst — understand the index format before modifying it
  - docs/api/models/ — read one existing RST file to understand the format

Also read the attached paper PDF. Focus on:
  - Section 2.1 (Language model pre-training) — tokenization and architecture
  - Section 2.2 (Cancer type classification) — MLP head architecture
  - Section 2.3 (Survival analysis) — Cox head and loss function
  - Section 2.4 (IA3 fine-tuning) — exactly how IA3 modifies attention layers
  - Table 1 — target metrics for classification (weighted F1 = 0.942)
  - Table 2 — target metrics for survival (C-index = 0.765)

---

## What you are building

| Action | File                                                        |
|--------|-------------------------------------------------------------|
| CREATE | pyhealth/models/bulk_rnabert.py                             |
| CREATE | tests/test_bulk_rnabert.py                                  |
| CREATE | docs/api/models/pyhealth.models.bulk_rnabert.rst            |
| MODIFY | docs/api/models.rst (add entry alphabetically)              |
| MODIFY | pyhealth/models/__init__.py (export BulkRNABert)            |

---

## Interface contract with Person 1

Person 1's task (TCGARNASeqCancerTypeClassification) produces sample dicts:

```python
{
    "patient_id": "TCGA-A2-A0T2",
    "gene_expression": torch.FloatTensor,  # shape (19042,), values in [0, 1]
    "label": 2,                            # int 0-32
}
```

After PyHealth's DataLoader batches these, your forward method receives:

```python
gene_expression: torch.FloatTensor  # shape (batch_size, 19042)
labels: torch.LongTensor            # shape (batch_size,), optional
```

CRITICAL: The key name "gene_expression" in your forward signature must exactly
match the key in Person 1's output dict. Do not rename it.

You do NOT need Person 1's code to be done to build and test your model.
Use random synthetic tensors throughout development:

```python
x = torch.randn(4, 19042)
model = BulkRNABert(num_classes=33, task="classification")
output = model(x)
assert output["logits"].shape == (4, 33)
```

---

## Understanding the paper's architecture

### What BulkRNABert is

BulkRNABert is a BERT-style encoder-only transformer pretrained on bulk RNA-seq
data using Masked Language Modeling. Each RNA-seq sample (19,042 gene expression
values) is treated like a sentence, with each gene's expression value tokenized
into one of 64 discrete bins. The model learns gene co-regulatory patterns
without labels.

The pretrained weights are publicly released on HuggingFace by InstaDeep.
You do NOT pretrain from scratch. You load the pretrained weights and
fine-tune on downstream tasks.

### Model architecture (from Section 2 of the paper)

Transformer encoder:
  - 4 transformer blocks
  - 8 attention heads per block
  - Embedding dimension: 256 (n_embedding = 256)
  - Gene2Vec-initialized gene embeddings replace positional encodings
  - Input: gene expression vector of shape (batch_size, 19042)
  - Output of last self-attention layer: shape (batch_size, 19042, 256)
  - Mean pool across gene dimension → patient embedding: shape (batch_size, 256)

Classification head (Section 2.2):
  - MLP with two hidden layers: 256 → 128
  - SELU activation
  - Dropout
  - LayerNorm
  - Final linear: 128 → num_classes (33 for pan-cancer)
  - Loss: cross-entropy

Survival head (Section 2.3):
  - Same architecture as classification head but slightly larger:
    hidden layers [512, 256]
  - Output: scalar log-risk score, shape (batch_size, 1)
  - Loss: negative Cox partial log-likelihood

IA3 fine-tuning (Section 2.4):
  - Parameter-efficient fine-tuning method
  - Adds learned scaling vectors to attention K, V, and feed-forward layers
  - Attention becomes: softmax(Q(lk ⊙ K) / sqrt(dk)) (lv ⊙ V)
  - FFN becomes: (lff ⊙ gamma(W1 x)) W2
  - lk, lv, lff are learned nn.Parameter vectors
  - Only these vectors (and the task head) are trained; encoder weights are frozen
  - Represents less than 0.07% of total parameters

---

## HuggingFace model loading

The authors released pretrained weights on HuggingFace. Before writing code,
search HuggingFace for the correct model identifier:
  https://huggingface.co/InstaDeepAI

Look for a model named something like "nucleotide-transformer" or check the
authors' GitHub for the exact HuggingFace model string:
  https://github.com/instadeepai/multiomics-open-research

The model likely loads with:
```python
from transformers import AutoModel, AutoConfig
config = AutoConfig.from_pretrained("InstaDeepAI/YOUR-MODEL-NAME")
encoder = AutoModel.from_pretrained("InstaDeepAI/YOUR-MODEL-NAME")
```

If you cannot find the exact HuggingFace identifier, implement the model
architecture from scratch using the paper's specifications (4 blocks, 8 heads,
dim=256) and add a note in the docstring that pretrained weights should be
loaded from the authors' repo once the exact HuggingFace path is confirmed.
The implementation must still be correct and runnable either way.

---

## File 1: pyhealth/models/bulk_rnabert.py

### Class structure overview

The file contains:
  1. Helper class: BulkRNABertMLP (the shared MLP head used for both tasks)
  2. Helper class: CoxPartialLikelihoodLoss (the survival loss function)
  3. Main class: BulkRNABert(BaseModel)

---

### Helper class: BulkRNABertMLP

```python
class BulkRNABertMLP(nn.Module):
    """MLP head used for both classification and survival tasks.

    Implements a two-hidden-layer MLP with SELU activation, dropout,
    and layer normalization as described in BulkRNABert (Gélard et al., 2024).

    Args:
        input_dim: Input dimension (embedding size from encoder, typically 256).
        hidden_dims: List of hidden layer sizes. Classification uses [256, 128],
            survival uses [512, 256].
        output_dim: Output dimension. num_classes for classification, 1 for survival.
        dropout: Dropout probability. Defaults to 0.1.
    """
```

Implementation:
  - nn.Sequential with alternating Linear → SELU → Dropout → LayerNorm blocks
  - Final Linear(hidden_dims[-1], output_dim) with no activation
  - forward(x) returns raw logits / log-risk scores

---

### Helper class: CoxPartialLikelihoodLoss

```python
class CoxPartialLikelihoodLoss(nn.Module):
    """Negative Cox partial log-likelihood loss for survival analysis.

    Implements the loss used in BulkRNABert for survival prediction,
    following DeepSurv (Katzman et al., 2018).

    The loss for a batch is:
        L = -sum over events i of (risk_i - log(sum of exp(risk_j) for j at risk))
        divided by number of events

    Args:
        None

    Note:
        Expects inputs sorted by survival time descending for efficient
        at-risk set computation. Handles batches with no events gracefully
        by returning a zero loss tensor.
    """
```

Implementation:
```python
def forward(
    self,
    risk_scores: torch.FloatTensor,   # shape (batch_size, 1) or (batch_size,)
    survival_times: torch.FloatTensor, # shape (batch_size,)
    events: torch.FloatTensor,         # shape (batch_size,), 1=event, 0=censored
) -> torch.Tensor:
    # Flatten risk scores to (batch_size,)
    risk = risk_scores.squeeze(-1)

    # Sort by survival time descending
    order = torch.argsort(survival_times, descending=True)
    risk = risk[order]
    events = events[order]

    # Compute log of cumulative sum of exp(risk) for at-risk set
    log_cumsum_exp = torch.logcumsumexp(risk, dim=0)

    # Only sum over actual events
    event_mask = events.bool()
    if event_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=risk.device)

    loss = -(risk[event_mask] - log_cumsum_exp[event_mask]).mean()
    return loss
```

---

### Main class: BulkRNABert(BaseModel)

```python
class BulkRNABert(BaseModel):
    """BulkRNABert: transformer-based model for cancer prognosis from RNA-seq.

    Loads a pretrained BERT-style encoder trained on bulk RNA-seq data and
    fine-tunes it for either cancer type classification or survival prediction.
    Supports IA3 parameter-efficient fine-tuning as described in the paper.

    This model is a PyHealth implementation of BulkRNABert from:
        Gélard et al. (2024). BulkRNABert: Cancer prognosis from bulk RNA-seq
        based language models. https://doi.org/10.1101/2024.06.18.599483

    Args:
        num_classes: Number of output classes for classification. Use 33 for
            pan-cancer TCGA classification. Ignored when task="survival".
        task: Either "classification" or "survival". Determines which head
            and loss function are used.
        pretrained_model_name: HuggingFace model identifier for pretrained
            BulkRNABert weights. Defaults to the InstaDeep release.
        use_ia3: If True, inject IA3 scaling vectors into all transformer
            attention layers and freeze the base encoder weights. Only the
            IA3 vectors and task head are trained. Defaults to False.
        freeze_encoder: If True, freeze all encoder parameters. Useful for
            linear probing baseline. Defaults to False. Automatically set
            to True when use_ia3=True.
        dropout: Dropout probability for MLP heads. Defaults to 0.1.
        embedding_dim: Dimension of the encoder output. Should match the
            pretrained model (256 for BulkRNABert). Defaults to 256.

    Examples:
        >>> # Classification with IA3 fine-tuning
        >>> model = BulkRNABert(num_classes=33, task="classification", use_ia3=True)
        >>> x = torch.randn(4, 19042)
        >>> output = model(x)
        >>> output["logits"].shape
        torch.Size([4, 33])

        >>> # Survival prediction
        >>> model = BulkRNABert(task="survival")
        >>> output = model(x)
        >>> output["risk_score"].shape
        torch.Size([4, 1])
    """
```

#### __init__ implementation:

```python
def __init__(
    self,
    num_classes: int = 33,
    task: str = "classification",
    pretrained_model_name: str = "InstaDeepAI/YOUR-MODEL-NAME",
    use_ia3: bool = False,
    freeze_encoder: bool = False,
    dropout: float = 0.1,
    embedding_dim: int = 256,
) -> None:
    super().__init__()

    assert task in ("classification", "survival"), \
        f"task must be 'classification' or 'survival', got {task!r}"

    self.task = task
    self.num_classes = num_classes
    self.use_ia3 = use_ia3
    self.embedding_dim = embedding_dim

    # Load pretrained encoder
    try:
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
    except Exception:
        # Fallback: build encoder from scratch with paper's architecture
        # 4 blocks, 8 heads, dim=256 — use transformers TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=8, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self._encoder_is_scratch = True
    else:
        self._encoder_is_scratch = False

    # Freeze encoder if requested or if using IA3
    if freeze_encoder or use_ia3:
        for param in self.encoder.parameters():
            param.requires_grad = False

    # Inject IA3 vectors if requested
    if use_ia3:
        self._ia3_vectors = nn.ParameterList()
        self._inject_ia3()

    # Build task head
    if task == "classification":
        self.head = BulkRNABertMLP(
            input_dim=embedding_dim,
            hidden_dims=[256, 128],
            output_dim=num_classes,
            dropout=dropout,
        )
        self.loss_fn = nn.CrossEntropyLoss()
    else:  # survival
        self.head = BulkRNABertMLP(
            input_dim=embedding_dim,
            hidden_dims=[512, 256],
            output_dim=1,
            dropout=dropout,
        )
        self.loss_fn = CoxPartialLikelihoodLoss()
```

#### _inject_ia3 implementation:

IA3 works by multiplying intermediate activations by learned scalar vectors.
For each transformer layer, register three nn.Parameter vectors:

```python
def _inject_ia3(self) -> None:
    """Inject IA3 learned scaling vectors into all transformer attention layers.

    For each transformer layer, registers three Parameter vectors:
      - ia3_k: scales the attention key vectors, shape (d_k,)
      - ia3_v: scales the attention value vectors, shape (d_v,)
      - ia3_ff: scales the feed-forward intermediate activations, shape (d_ff,)

    These are registered as module parameters so they are included in
    optimizer updates while the frozen encoder weights are not.

    Note:
        The actual scaling is applied via forward hooks registered on each
        attention and feed-forward sublayer. This avoids modifying the
        underlying HuggingFace model's forward methods directly.
    """
    # Implementation depends on the encoder architecture.
    # For HuggingFace BERT-style models, iterate over encoder.layer:
    #
    # for layer in self.encoder.encoder.layer:
    #     d_k = layer.attention.self.key.weight.shape[0]
    #     d_v = layer.attention.self.value.weight.shape[0]
    #     d_ff = layer.intermediate.dense.weight.shape[0]
    #
    #     ia3_k = nn.Parameter(torch.ones(d_k))
    #     ia3_v = nn.Parameter(torch.ones(d_v))
    #     ia3_ff = nn.Parameter(torch.ones(d_ff))
    #
    #     self._ia3_vectors.extend([ia3_k, ia3_v, ia3_ff])
    #
    #     # Register forward hooks to apply scaling
    #     layer.attention.self.register_forward_hook(
    #         self._make_ia3_attention_hook(ia3_k, ia3_v)
    #     )
    #     layer.intermediate.register_forward_hook(
    #         self._make_ia3_ff_hook(ia3_ff)
    #     )
    #
    # The exact attribute paths (encoder.encoder.layer, attention.self, etc.)
    # depend on the HuggingFace model class. Inspect the model's named_modules()
    # to find the correct paths before hardcoding them.
    #
    # If the encoder was built from scratch (self._encoder_is_scratch = True),
    # adapt the attribute paths accordingly for nn.TransformerEncoderLayer.
    pass

def _make_ia3_attention_hook(self, ia3_k, ia3_v):
    """Returns a forward hook that applies IA3 scaling to K and V."""
    def hook(module, input, output):
        # output is typically (context_layer, attention_probs)
        # Scale attention output by ia3_v
        # Exact implementation depends on HuggingFace model internals
        return output
    return hook

def _make_ia3_ff_hook(self, ia3_ff):
    """Returns a forward hook that applies IA3 scaling to FFN activations."""
    def hook(module, input, output):
        return output * ia3_ff
    return hook
```

NOTE ON IA3: The exact implementation of _inject_ia3 depends on which
HuggingFace model class is loaded. After loading the model, run:
```python
for name, module in model.encoder.named_modules():
    print(name, type(module).__name__)
```
to discover the exact layer names and adapt the hook registration accordingly.
IA3 is an optional enhancement — implement it as best effort. The model must
work correctly with use_ia3=False even if the IA3 injection is incomplete.

#### _get_embedding implementation:

```python
def _get_embedding(
    self, gene_expression: torch.FloatTensor
) -> torch.FloatTensor:
    """Extract mean-pooled patient embedding from encoder.

    Passes gene expression through the encoder and mean-pools the last
    hidden state across the gene dimension to produce a fixed-size
    patient embedding.

    Args:
        gene_expression: Normalized gene expression tensor of shape
            (batch_size, num_genes). Values should be in [0, 1].

    Returns:
        Patient embedding tensor of shape (batch_size, embedding_dim).
    """
    if self._encoder_is_scratch:
        # nn.TransformerEncoder expects (batch, seq, dim)
        # Need to embed input first — add a simple linear projection
        # from num_genes to embedding_dim if not already present
        hidden = self.encoder(gene_expression.unsqueeze(-1).expand(
            -1, -1, self.embedding_dim
        ))
    else:
        # HuggingFace model — pass input_ids or inputs_embeds
        # BulkRNABert takes tokenized gene expressions as input_ids
        # Gene expressions are already binned by Person 3's preprocessing
        # but if not, we need to bin here. For now pass as is and adapt
        # based on actual model input requirements.
        outputs = self.encoder(inputs_embeds=gene_expression.unsqueeze(-1))
        hidden = outputs.last_hidden_state  # (batch, num_genes, embedding_dim)

    # Mean pool across gene sequence dimension
    embedding = hidden.mean(dim=1)  # (batch, embedding_dim)
    return embedding
```

#### forward implementation:

```python
def forward(
    self,
    gene_expression: torch.FloatTensor,
    labels: Optional[torch.LongTensor] = None,
    survival_times: Optional[torch.FloatTensor] = None,
    events: Optional[torch.FloatTensor] = None,
) -> Dict[str, torch.Tensor]:
    """Forward pass of BulkRNABert.

    Encodes gene expression using the pretrained transformer, mean-pools
    across the gene dimension, and passes through the task-specific head.

    Args:
        gene_expression: Gene expression tensor of shape (batch_size, num_genes).
            Values should be log10(1+TPM) normalized and max-normalized to [0, 1].
        labels: Class labels of shape (batch_size,) for classification.
            Required for loss computation during training. Pass None for inference.
        survival_times: Survival times of shape (batch_size,) for survival task.
            Required for Cox loss computation during training.
        events: Event indicators of shape (batch_size,) for survival task.
            1 = event occurred (death), 0 = censored. Required for Cox loss.

    Returns:
        Dict containing:
            For classification:
                - "logits": shape (batch_size, num_classes)
                - "loss": scalar cross-entropy loss (only if labels provided)
            For survival:
                - "risk_score": shape (batch_size, 1)
                - "loss": scalar Cox loss (only if survival_times and events provided)

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
        logits = self.head(embedding)  # (batch, num_classes)
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels)
        return output

    else:  # survival
        risk_score = self.head(embedding)  # (batch, 1)
        output = {"risk_score": risk_score}
        if survival_times is not None and events is not None:
            output["loss"] = self.loss_fn(risk_score, survival_times, events)
        return output
```

---

## File 2: tests/test_bulk_rnabert.py

### Hard rules
- NO real data anywhere
- All tests complete under 1 second total
- Use small synthetic tensors only
- NUM_FAKE_GENES = 19042 is fine here since we're just making random tensors
  (no CSV reading, just torch.randn) — it's fast
- Use batch_size = 4 throughout

### Tests to write

```
test_classification_model_instantiates
  model = BulkRNABert(num_classes=33, task="classification")
  assert isinstance(model, BulkRNABert)

test_survival_model_instantiates
  model = BulkRNABert(task="survival")
  assert isinstance(model, BulkRNABert)

test_invalid_task_raises
  with pytest.raises(AssertionError):
      BulkRNABert(task="invalid")

test_classification_forward_output_shape
  model = BulkRNABert(num_classes=33, task="classification")
  x = torch.randn(4, 19042)
  out = model(x)
  assert out["logits"].shape == (4, 33)
  assert "loss" not in out  # no labels provided

test_classification_forward_with_labels
  model = BulkRNABert(num_classes=33, task="classification")
  x = torch.randn(4, 19042)
  labels = torch.randint(0, 33, (4,))
  out = model(x, labels=labels)
  assert "loss" in out
  assert out["loss"].ndim == 0  # scalar
  assert out["logits"].shape == (4, 33)

test_survival_forward_output_shape
  model = BulkRNABert(task="survival")
  x = torch.randn(4, 19042)
  out = model(x)
  assert out["risk_score"].shape == (4, 1)
  assert "loss" not in out

test_survival_forward_with_labels
  model = BulkRNABert(task="survival")
  x = torch.randn(4, 19042)
  times = torch.rand(4) * 1000
  events = torch.randint(0, 2, (4,)).float()
  out = model(x, survival_times=times, events=events)
  assert "loss" in out
  assert out["loss"].ndim == 0

test_gradients_flow_classification
  model = BulkRNABert(num_classes=33, task="classification")
  x = torch.randn(4, 19042, requires_grad=False)
  labels = torch.randint(0, 33, (4,))
  out = model(x, labels=labels)
  out["loss"].backward()
  # At least one parameter should have a gradient
  has_grad = any(
      p.grad is not None and p.grad.abs().sum() > 0
      for p in model.parameters() if p.requires_grad
  )
  assert has_grad

test_gradients_flow_survival
  Same as above but for survival task with times and events

test_freeze_encoder_flag
  model = BulkRNABert(num_classes=33, task="classification", freeze_encoder=True)
  for name, param in model.encoder.named_parameters():
      assert not param.requires_grad, f"{name} should be frozen"
  # Head parameters should still be trainable
  for name, param in model.head.named_parameters():
      assert param.requires_grad, f"{name} should require grad"

test_ia3_only_ia3_params_trainable
  model = BulkRNABert(num_classes=33, task="classification", use_ia3=True)
  for name, param in model.encoder.named_parameters():
      assert not param.requires_grad, f"Encoder param {name} should be frozen"
  # IA3 vectors should be trainable
  if len(list(model._ia3_vectors)) > 0:
      for i, vec in enumerate(model._ia3_vectors):
          assert vec.requires_grad, f"IA3 vector {i} should require grad"

test_cox_loss_with_all_events
  loss_fn = CoxPartialLikelihoodLoss()
  risk = torch.randn(4)
  times = torch.tensor([100.0, 200.0, 300.0, 400.0])
  events = torch.ones(4)
  loss = loss_fn(risk, times, events)
  assert loss.ndim == 0
  assert not torch.isnan(loss)

test_cox_loss_with_no_events
  loss_fn = CoxPartialLikelihoodLoss()
  risk = torch.randn(4)
  times = torch.tensor([100.0, 200.0, 300.0, 400.0])
  events = torch.zeros(4)  # all censored
  loss = loss_fn(risk, times, events)
  assert not torch.isnan(loss)  # should handle gracefully

test_mlp_head_output_shape
  mlp = BulkRNABertMLP(input_dim=256, hidden_dims=[256, 128], output_dim=33)
  x = torch.randn(4, 256)
  out = mlp(x)
  assert out.shape == (4, 33)

test_5_cohort_forward
  # Paper evaluates on 5-cohort subset (BRCA, BLCA, GBMLGG, LUAD, UCEC)
  model = BulkRNABert(num_classes=5, task="classification")
  x = torch.randn(4, 19042)
  out = model(x)
  assert out["logits"].shape == (4, 5)
```

---

## File 3: docs/api/models/pyhealth.models.bulk_rnabert.rst

Read an existing RST file in docs/api/models/ before writing this.
Standard format:

```rst
pyhealth.models.bulk\_rnabert
==============================

.. automodule:: pyhealth.models.bulk_rnabert
   :members:
   :undoc-members:
   :show-inheritance:
```

---

## File 4: docs/api/models.rst

Read this file before editing it. Add to the toctree in alphabetical order:

```
pyhealth.models.bulk_rnabert
```

---

## File 5: pyhealth/models/__init__.py

Read this file before editing it. Add in the same style as existing exports:

```python
from .bulk_rnabert import BulkRNABert
```

---

## Code style requirements — graded

- PEP8, 88-character line length
- snake_case for variables and functions, PascalCase for classes
- Type hints on every argument and return value
- Google-style docstrings on every class and method

Run black when done:
```bash
black pyhealth/models/bulk_rnabert.py tests/test_bulk_rnabert.py
```

---

## Verification commands

```bash
# Model imports correctly
python -c "from pyhealth.models.bulk_rnabert import BulkRNABert; print('OK')"

# Public export works
python -c "from pyhealth.models import BulkRNABert; print('OK')"

# All tests pass and are fast
pytest tests/test_bulk_rnabert.py -v

# Style
black --check pyhealth/models/bulk_rnabert.py tests/test_bulk_rnabert.py

# YAML (not applicable here but run for the full suite)
pytest tests/ -v
```

---

## Build order — do these in sequence

1. Read all reference files in the repo first (base_model.py, existing models,
   models/__init__.py, docs/api/models.rst, one existing RST)
2. Search HuggingFace for the correct BulkRNABert model identifier
3. Write CoxPartialLikelihoodLoss — standalone, testable immediately
4. Write BulkRNABertMLP — standalone, testable immediately
5. Write BulkRNABert with use_ia3=False first — get the basic forward pass working
6. Write tests for the basic model — all should pass before moving on
7. Implement IA3 injection — this is the hardest part, adapt to actual model structure
8. Add IA3-related tests
9. Write RST docs file
10. Update models.rst index and __init__.py exports

---

## What NOT to do

- Do NOT rename "gene_expression" in the forward signature — must match Person 1
- Do NOT pretrain from scratch — load HuggingFace weights
- Do NOT use real data in tests
- Do NOT skip the Cox loss implementation — survival is a core part of the paper
- Do NOT make tests slow — keep them all under 1 second total
- Do NOT skip docstrings — explicitly graded
- Do NOT write the model without reading base_model.py first

---

## Common issues to watch for

HuggingFace model input format: BulkRNABert takes tokenized gene expressions
(binned into 64 bins) not raw float values. The paper preprocesses values into
integer bin IDs. If the HuggingFace model expects input_ids (integers), you may
need to add a binning step in _get_embedding. Check the model's from_pretrained
config to see what input format it expects before assuming.

IA3 attribute paths: The exact attribute paths into the HuggingFace model
(e.g. encoder.encoder.layer vs encoder.layers) vary by model class. Always
inspect with named_modules() before hardcoding paths.

BaseModel inheritance: Read base_model.py carefully. If it defines abstract
methods, you must implement all of them or you will get a TypeError on
instantiation.

Missing __init__.py export: Test with:
  from pyhealth.models import BulkRNABert
If this fails, you forgot to update __init__.py.

---

## Paper reference

Paper: Gélard et al. (2024). BulkRNABert: Cancer prognosis from bulk RNA-seq
based language models.
DOI: https://doi.org/10.1101/2024.06.18.599483
Code: https://github.com/instadeepai/multiomics-open-research
HuggingFace: https://huggingface.co/InstaDeepAI
