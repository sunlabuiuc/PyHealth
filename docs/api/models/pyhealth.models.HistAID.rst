pyhealth.models.HistAID
=======================

Historical Sequential Transformers for AI-augmented Diagnostics (HIST-AID). This model implements a multi-modal architecture designed to replicate the clinical workflow of a radiologist by integrating current visual evidence with longitudinal medical history.

The architecture utilizes a **Vision Transformer (ViT)** for image feature extraction and a **BERT-Base encoder** for processing sequential radiology reports. These representations are fused via a **Transformer-based fusion layer** using cross-modal self-attention to generate context-aware diagnostic predictions.

.. autoclass:: pyhealth.models.hist_aid.HistAID
    :members:
    :undoc-members:
    :show-inheritance: