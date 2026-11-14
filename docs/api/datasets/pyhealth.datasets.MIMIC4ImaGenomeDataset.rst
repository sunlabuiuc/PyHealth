pyhealth.datasets.MIMIC4ImaGenomeDataset
========================================

MIMIC-IV events together with the Chest ImaGenome tables, joined by patient.
It composes :class:`~pyhealth.datasets.MIMIC4Dataset` and
:class:`~pyhealth.datasets.ChestImaGenomeDataset` and is used by
:class:`~pyhealth.tasks.DiseaseProgressionMIMIC4`.

The data come from PhysioNet and need credentialed access:

- `MIMIC-IV <https://physionet.org/content/mimiciv/>`_: ``ehr_root`` (and
  ``note_root`` or ``ed_root`` if used)
- `MIMIC-CXR-JPG <https://physionet.org/content/mimic-cxr-jpg/>`_: ``cxr_root``
- `Chest ImaGenome 1.0.0 <https://physionet.org/content/chest-imagenome/1.0.0/>`_:
  ``imagenome_root`` (Wu et al., NeurIPS Datasets and Benchmarks 2021)

.. autoclass:: pyhealth.datasets.mimic4_imagenome.MIMIC4ImaGenomeDataset
    :members:
    :undoc-members:
    :show-inheritance:
