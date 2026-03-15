VisionEmbeddingModel
--------------------

.. autoclass:: pyhealth.models.VisionEmbeddingModel
   :members:
   :undoc-members:
   :show-inheritance:

**Overview**

``VisionEmbeddingModel`` converts medical images to sequences of patch embeddings
suitable for attention-based fusion with other modalities (EHR, text).

**Input/Output**

- **Input:** ``(batch, channels, height, width)`` image tensors
- **Output:** ``(batch, num_tokens, embedding_dim)`` sequence embeddings

**Supported Backbones**

.. list-table::
   :header-rows: 1
   :widths: 15 20 15 50

   * - Backbone
     - Output Tokens
     - Parameters
     - Description
   * - ``patch``
     - 196 + CLS
     - ~300K
     - ViT-style patch projection (lightweight)
   * - ``cnn``
     - 49 + CLS
     - ~100K
     - Small CNN encoder with good inductive bias
   * - ``resnet18``
     - 49 + CLS
     - ~11M
     - Pretrained ResNet-18 backbone
   * - ``resnet50``
     - 49 + CLS
     - ~24M
     - Pretrained ResNet-50 backbone

**Example Usage**

.. code-block:: python

    from pyhealth.models import VisionEmbeddingModel
    from pyhealth.datasets import create_sample_dataset

    # Create model
    model = VisionEmbeddingModel(
        dataset=sample_dataset,
        embedding_dim=128,
        backbone="resnet18",
        pretrained=True,
        use_cls_token=True,
    )

    # Forward pass: (B, C, H, W) -> (B, num_tokens, E)
    embeddings = model({"chest_xray": images})
    print(embeddings["chest_xray"].shape)  # (16, 50, 128)

    # Get output info
    info = model.get_output_info("chest_xray")
    # {'num_patches': 49, 'num_tokens': 50, 'embedding_dim': 128, ...}

**Multimodal Fusion**

The sequence output format enables easy fusion with other modalities:

.. code-block:: python

    # Vision embeddings: (B, P, E)
    vision_emb = vision_model({"image": images})["image"]
    
    # Text embeddings: (B, T, E)  
    text_emb = text_model({"notes": text})["notes"]
    
    # EHR embeddings: (B, S, E)
    ehr_emb = ehr_model({"codes": codes})["codes"]
    
    # Concatenate for Transformer fusion
    combined = torch.cat([vision_emb, text_emb, ehr_emb], dim=1)
    # -> (B, P+T+S, E)
