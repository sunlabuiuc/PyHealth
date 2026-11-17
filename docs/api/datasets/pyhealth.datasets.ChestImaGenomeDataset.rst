pyhealth.datasets.ChestImaGenomeDataset
=======================================

Overview
--------

Chest ImaGenome adds anatomical scene graphs to the frontal chest X-rays in
MIMIC-CXR: automatically built silver graphs for every image, and a manually
validated gold subset of 500 patients. Access is credentialed, so download it
from `PhysioNet <https://physionet.org/content/chest-imagenome/1.0.0/>`_ after
signing the data use agreement, then unzip ``silver_dataset/scene_graph.zip``.

For more information see Wu et al., "Chest ImaGenome Dataset for Clinical
Reasoning", NeurIPS 2021 (`arXiv <https://arxiv.org/abs/2108.00316>`_). The
fields kept follow DiPro (Liu et al., NeurIPS 2025).

The dataset writes four tables keyed by MIMIC-CXR ``subject_id``:
``silver_images``, ``silver_objects`` (anatomical boxes), ``silver_comparisons``
(a region compared between a current and a previous image) and
``gold_comparisons``. ``study_id`` and ``dicom_id`` join to
:class:`~pyhealth.datasets.MIMIC4CXRDataset`. It loads no images and derives no
progression labels.

API Reference
-------------

.. autoclass:: pyhealth.datasets.ChestImaGenomeDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.chest_imagenome.prepare_chest_imagenome
