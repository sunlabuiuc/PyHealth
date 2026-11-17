pyhealth.models.ConVIRT
=======================

ConVIRT contrastive pretraining of an EHR LSTM and a chest X-ray ResNet, and
a linear probe on the frozen encoders, as evaluated in MedMod.

Reference: Elsharief et al., "MedMod: Multimodal Benchmark for Medical
Prediction Tasks with Electronic Health Records and Chest X-Ray Scans."
CHIL 2025. https://proceedings.mlr.press/v287/elsharief25a.html

.. autoclass:: pyhealth.models.ConVIRT
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.ConVIRTLinearProbe
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.convirt.info_nce_loss
