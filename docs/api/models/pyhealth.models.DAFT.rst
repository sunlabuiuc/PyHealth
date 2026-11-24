pyhealth.models.DAFT
====================

Overview
--------

DAFT (Dynamic Affine Feature Map Transform) fuses clinical time series (EHR)
and chest X-rays (CXR) by predicting a per-channel scale and shift for one
modality from the other. ``direction="cxr_to_ehr"`` (default) follows the DAFT
baseline of MedMod, where the ResNet feature map conditions the hidden
sequence of the EHR LSTM. ``direction="ehr_to_cxr"`` follows the original DAFT
paper, where the EHR state conditions a ResNet feature map. Samples without an
image are handled through ``cxr_mask``.

It takes the ``ehr``, ``cxr`` and ``cxr_mask`` inputs of
:class:`~pyhealth.tasks.EHRCXRMortalityMIMIC4` and
:class:`~pyhealth.tasks.EHRCXRPhenotypingMIMIC4`.

References:

- Polsterl et al., "Combining 3D image and tabular data via the dynamic affine
  feature map transform." MICCAI 2021.
- Elsharief et al., "MedMod: Multimodal Benchmark for Medical Prediction Tasks
  with Electronic Health Records and Chest X-Ray Scans." CHIL 2025.

API Reference
-------------

.. autoclass:: pyhealth.models.DAFTBlock
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyhealth.models.DAFTLayer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyhealth.models.DAFT
   :members:
   :undoc-members:
   :show-inheritance:
