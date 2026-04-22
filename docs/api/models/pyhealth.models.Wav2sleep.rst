pyhealth.models.Wav2Sleep
===================================

Wav2Sleep is a **multimodal sleep stage classification model** that processes physiological signals 
(ECG, PPG, respiratory signals) to predict sleep stages (Wake, N1, N2, N3, REM).

Paper-faithful implementation of:
  - Transformer-based fusion for multimodal aggregation
  - Dilated CNN sequence mixer for temporal modeling

Architecture Overview
---------------------

The model follows a four-stage pipeline:

1. **Modality-Specific Encoders**: Each signal type (ECG, PPG) is processed by a dedicated 
   convolutional encoder that extracts temporal features per epoch.

2. **Transformer-Based Fusion**: Modalities are aggregated using a transformer fusion module 
   that handles missing modalities gracefully.

3. **Dilated CNN Sequence Mixer**: Temporal dependencies across sleep epochs are captured 
   using exponentially dilated convolutions.

4. **Classification Head**: A linear layer maps features to sleep stage predictions.

Supported Signal Types
----------------------

- **ECG**: Electrocardiogram (1024 samples per epoch)
- **PPG**: Photoplethysmography (1024 samples per epoch)
- **ABD**: Abdominal breathing (256 samples per epoch)
- **THX**: Thoracic breathing (256 samples per epoch)
- **EOG_L/EOG_R**: Left/Right electrooculography (4096 samples per epoch)

Classes
-------

.. autoclass:: pyhealth.models.Wav2Sleep
    :members:
    :undoc-members:
    :show-inheritance:

Supporting Classes
~~~~~~~~~~~~~~~~~

Signal Encoders
^^^^^^^^^^^^^^^

.. autoclass:: pyhealth.models.ConvBlock
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.ConvLayer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.SignalEncoder
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.SignalEncoders
    :members:
    :undoc-members:
    :show-inheritance:

Fusion Modules
^^^^^^^^^^^^^^

.. autoclass:: pyhealth.models.TransformerFusion
    :members:
    :undoc-members:
    :show-inheritance:

Temporal Modeling
^^^^^^^^^^^^^^^^^

.. autoclass:: pyhealth.models.DilatedCNNSequenceMixer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.DilatedConvBlock
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.TemporalConvBlock
    :members:
    :undoc-members:
    :show-inheritance:

Normalization Layers
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyhealth.models.ConvLayerNorm
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.ConvRMSNorm
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.ConvGroupNorm
    :members:
    :undoc-members:
    :show-inheritance:

Usage Example
-------------

.. code-block:: python

    from pyhealth.models import Wav2Sleep
    from pyhealth.datasets import SampleDataset

    # Create model with paper-faithful components
    model = Wav2Sleep(
        dataset=train_dataset,
        embedding_dim=128,
        hidden_dim=128,
        num_classes=5,
        num_fusion_heads=4,
        num_fusion_layers=2,
        num_temporal_layers=5,
        use_paper_faithful=True,
    )

    # Forward pass
    output = model(ecg=ecg_signals, ppg=ppg_signals, labels=labels)
    # Returns: {'loss': tensor, 'y_prob': ..., 'y_true': ..., 'logit': ...}

Notes
-----

- At least one modality (ECG or PPG) must be present in the input.
- Missing modalities are handled gracefully by the fusion module.
- The dilated CNN sequence mixer provides an exponentially growing receptive field 
  for capturing long-range sleep cycle dependencies.
