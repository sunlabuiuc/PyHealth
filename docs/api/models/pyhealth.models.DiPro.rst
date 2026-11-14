pyhealth.models.DiPro
=====================

Overview
--------

DiPro identifies disease progression between two chest X-rays from anatomical
regions and the EHR recorded around them, by separating static from dynamic
image features and aligning them with the EHR at several time scales. Use it
with :class:`~pyhealth.tasks.DiseaseProgressionMIMIC4`.

DiPro as published works on anatomical region crops: build the task with
``region_crops=True``. The model also accepts whole images, as a single
region, but results in that mode are not those of DiPro as published.

Reference: Liu et al., "Multimodal Disease Progression Modeling via
Spatiotemporal Disentanglement and Multiscale Alignment." NeurIPS 2025.

API Reference
-------------

.. autoclass:: pyhealth.models.DiPro
   :members:
   :undoc-members:
   :show-inheritance:
