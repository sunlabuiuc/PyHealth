pyhealth.datasets.MedModMIMIC4Dataset
=====================================

MIMIC-IV v1.0 and MIMIC-CXR-JPG 2.0.0 as used by MedMod (Elsharief et al.,
"MedMod: Multimodal Benchmark for Medical Prediction Tasks with Electronic
Health Records and Chest X-Ray Scans", CHIL 2025, PMLR 287;
`paper <https://proceedings.mlr.press/v287/elsharief25a.html>`__,
`code <https://github.com/nyuad-cai/MedMod>`__) for in-hospital mortality,
with MedMod's fixed train / val / test split. Use it with
:class:`~pyhealth.tasks.MedModInHospitalMortality`.

Access to `MIMIC-IV v1.0 <https://physionet.org/content/mimiciv/1.0/>`__
and `MIMIC-CXR-JPG 2.0.0 <https://physionet.org/content/mimic-cxr-jpg/2.0.0/>`__
needs PhysioNet credentialing.

.. autoclass:: pyhealth.datasets.MedModMIMIC4Dataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.datasets.MedModSplit
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.split_by_medmod
