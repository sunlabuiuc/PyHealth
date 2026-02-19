UnifiedMultimodalEmbeddingModel
--------------------------------

.. autoclass:: pyhealth.models.UnifiedMultimodalEmbeddingModel
   :members:
   :undoc-members:
   :show-inheritance:

**Overview**

``UnifiedMultimodalEmbeddingModel`` embeds heterogeneous temporal features
— codes, clinical text, images, and continuous vitals — into a **single,
time-aligned sequence** ``(B, S_total, E')`` ready for any downstream sequence
model (Transformer, Mamba, RNN, etc.).

All input processors must be :class:`~pyhealth.processors.TemporalFeatureProcessor`
subclasses. Non-temporal processors are rejected early with a clear error message.

**Pipeline Position**

.. code-block:: text

    datasets → tasks → processors → models.embeddings → models
                                          ↑
                           UnifiedMultimodalEmbeddingModel lives here

**Input / Output**

- **Input:** ``dict[field_name, {"value": Tensor, "time": Tensor, "mask": Tensor (opt)}]``
  — one sub-dict per temporal feature, as produced by :func:`~pyhealth.datasets.collate_temporal`.
- **Output:** ``dict`` with keys:

  - ``"sequence"`` — ``(B, S_total, E')`` temporally-sorted event embeddings
  - ``"time"``     — ``(B, S_total)`` timestamps in hours
  - ``"mask"``     — ``(B, S_total)`` validity mask (1 = real event, 0 = padding)
  - ``"type_ids"`` — ``(B, S_total)`` modality index per event

**Modality Encoders**

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Modality
     - Encoder
     - Notes
   * - ``CODE``
     - ``nn.Embedding(vocab_size, E')``
     - vocab_size from ``processor.value_dim()`` after ``fit()``
   * - ``TEXT``
     - ``AutoModel.from_pretrained(...)`` + optional ``nn.Linear``
     - processor must have ``tokenizer_model`` set; projection added if hidden_size ≠ E'
   * - ``IMAGE``
     - ResNet-18 (torchvision) or fallback CNN
     - falls back to 3-layer Conv if torchvision unavailable
   * - ``NUMERIC`` / ``SIGNAL``
     - ``nn.Linear(n_features, E')``
     - n_features from ``processor.value_dim()`` after ``fit()``

**Temporal Alignment Algorithm**

1. For each field, encode events → ``(B, N_i, E')``
2. Retrieve timestamps → ``(B, N_i)``
3. Concatenate across all fields → ``(B, S_total, E')``
4. Sort events along dim=1 by timestamp (ascending)
5. Add ``SinusoidalTimeEmbedding(time)`` + learned ``type_embedding(modality_idx)``

**Example Usage**

.. code-block:: python

    from pyhealth.models import UnifiedMultimodalEmbeddingModel
    from pyhealth.datasets import create_sample_dataset, collate_temporal
    from torch.utils.data import DataLoader

    # Build a dataset with two temporal modalities
    dataset = create_sample_dataset(
        samples=patient_samples,
        input_schema={
            "conditions": "stagenet",          # CODE modality
            "notes": (
                "tuple_time_text",             # TEXT modality
                {"tokenizer_model": "emilyalsentzer/Bio_ClinicalBERT"},
            ),
        },
        output_schema={"label": "binary"},
    )

    # Build the unified embedding model
    model = UnifiedMultimodalEmbeddingModel(
        processors=dataset.input_processors,   # all must be TemporalFeatureProcessor
        embedding_dim=128,
    )

    # Use collate_temporal as the DataLoader collate_fn
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_temporal)

    for batch in loader:
        inputs = {k: batch[k] for k in dataset.input_processors}
        out = model(inputs)
        # out["sequence"]: (8, S_total, 128)  — codes + notes interleaved by time
        # out["mask"]:     (8, S_total)
        break

**Using with a Downstream Sequence Model**

.. code-block:: python

    import torch.nn as nn
    from pyhealth.models import UnifiedMultimodalEmbeddingModel

    class UnifiedTransformer(nn.Module):
        def __init__(self, processors, embedding_dim=128, n_heads=4, n_layers=2):
            super().__init__()
            self.embed = UnifiedMultimodalEmbeddingModel(
                processors=processors,
                embedding_dim=embedding_dim,
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim, nhead=n_heads, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Linear(embedding_dim, 1)

        def forward(self, inputs):
            out  = self.embed(inputs)
            seq  = out["sequence"]  # (B, S, E)
            mask = out["mask"]      # (B, S)  1=valid
            # TransformerEncoder uses True=ignore, so invert
            key_mask = (mask == 0)
            encoded  = self.transformer(seq, src_key_padding_mask=key_mask)
            pooled   = (encoded * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            return self.head(pooled)  # (B, 1)

**Registering Custom Modalities**

To add a new modality, subclass :class:`~pyhealth.processors.TemporalFeatureProcessor`
and return the corresponding :class:`~pyhealth.processors.ModalityType`:

.. code-block:: python

    from pyhealth.processors.base_processor import ModalityType, TemporalFeatureProcessor

    class MyAudioProcessor(TemporalFeatureProcessor):
        def modality(self) -> ModalityType:
            return ModalityType.AUDIO   # triggers NotImplementedError in UnifiedMultimodalEmbeddingModel
                                        # until you add an encoder branch

        def value_dim(self) -> int:
            return self.n_samples       # raw waveform length

        def process(self, value) -> dict:
            ...
            return {"value": audio_tensor, "time": time_tensor}

.. autoclass:: pyhealth.models.SinusoidalTimeEmbedding
   :members:
   :undoc-members:
   :show-inheritance:
