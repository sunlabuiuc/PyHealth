TextEmbedding
-------------

.. autoclass:: pyhealth.models.TextEmbedding
   :members:
   :undoc-members:
   :show-inheritance:

**Overview**

``TextEmbedding`` encodes clinical text using pretrained transformers (default: Bio_ClinicalBERT)
for multimodal fusion with EHR data and medical images.

**Input/Output**

- **Input:** List of text strings (clinical notes)
- **Output:** ``(batch, seq_len, embedding_dim)`` embeddings + ``(batch, seq_len)`` mask

**Key Features**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Feature
     - Description
   * - **128-token chunking**
     - Splits long clinical notes into non-overlapping chunks
   * - **Pooling modes**
     - none (all tokens), cls ([CLS] per chunk), mean (average per chunk)
   * - **Performance guardrails**
     - max_chunks parameter prevents OOM on very long texts
   * - **Mask output**
     - Boolean tensors compatible with TransformerLayer
   * - **Backward compatibility**
     - return_mask=False for legacy code

**Example Usage**

.. code-block:: python

    from pyhealth.models import TextEmbedding

    # Initialize encoder
    encoder = TextEmbedding(
        embedding_dim=128,
        chunk_size=128,
        pooling="none",
        freeze=True,  # Freeze transformer for multimodal fusion
    )

    # Encode clinical notes
    texts = [
        "Patient presents with chest pain and shortness of breath.",
        "Follow-up visit for diabetes management. Blood glucose stable."
    ]
    embeddings, mask = encoder(texts)
    print(embeddings.shape)  # (2, 8, 128)
    print(mask.shape)        # (2, 8)

**Chunking Behavior**

Long texts are automatically split into chunks:

.. code-block:: python

    # Long clinical note
    long_note = "Patient admitted with symptoms. " * 100

    embeddings, mask = encoder([long_note])
    print(embeddings.shape)  # (1, 254, 128) - multiple chunks concatenated
    print(mask.sum())        # 254 valid tokens

**Pooling Modes**

.. code-block:: python

    # All token embeddings
    enc_none = TextEmbedding(pooling="none")
    emb, _ = enc_none([text])  # (1, T, 128)

    # [CLS] token per chunk
    enc_cls = TextEmbedding(pooling="cls")
    emb, _ = enc_cls([text])   # (1, C, 128) where C = num_chunks

    # Mean-pooled per chunk
    enc_mean = TextEmbedding(pooling="mean")
    emb, _ = enc_mean([text])  # (1, C, 128)

**Multimodal Fusion**

The sequence output format enables easy fusion with other modalities:

.. code-block:: python

    # Text embeddings: (B, T, E)
    text_emb, text_mask = text_model(clinical_notes)
    
    # Vision embeddings: (B, P, E)
    vision_emb = vision_model({"image": xrays})["image"]
    
    # EHR embeddings: (B, S, E)
    ehr_emb = ehr_model({"codes": codes})["codes"]
    
    # Concatenate for Transformer fusion
    combined = torch.cat([text_emb, vision_emb, ehr_emb], dim=1)
    # -> (B, T+P+S, E)
