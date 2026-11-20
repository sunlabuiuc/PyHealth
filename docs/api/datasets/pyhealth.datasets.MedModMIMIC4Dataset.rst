pyhealth.datasets.MedModMIMIC4Dataset
=====================================

MIMIC-IV v1.0 and MIMIC-CXR-JPG 2.0.0 as used by MedMod (Elsharief et al.,
"MedMod: Multimodal Benchmark for Medical Prediction Tasks with Electronic
Health Records and Chest X-Ray Scans", CHIL 2025, PMLR 287;
`code <https://github.com/nyuad-cai/MedMod>`__) for in-hospital mortality,
with MedMod's fixed train / val / test split. Use it with
:class:`~pyhealth.tasks.MedModInHospitalMortality`.

Access to MIMIC-IV v1.0 and MIMIC-CXR-JPG 2.0.0 needs PhysioNet credentialing.
Without it, ``pyhealth.datasets.medmod.synthetic.create_synthetic_samples``
builds synthetic samples with the same contract.

.. autoclass:: pyhealth.datasets.MedModMIMIC4Dataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.datasets.MedModSplit
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.split_by_medmod

MedSigLIP image embeddings
--------------------------

For the extension, :mod:`pyhealth.datasets.medmod.medsiglip` replaces the
ResNet-34 image features with those of a frozen ``google/medsiglip-448``
image encoder. Each image is embedded once and saved to one ``.npz`` file,
where the embeddings are looked up by image file name.

.. automodule:: pyhealth.datasets.medmod.medsiglip
    :members:
