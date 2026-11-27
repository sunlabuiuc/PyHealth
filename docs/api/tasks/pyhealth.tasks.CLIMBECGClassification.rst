pyhealth.tasks.CLIMBECGClassification
=====================================

CLIMB's multi-label ECG diagnosis task (Dai et al., `CLIMB: Data Foundations
for Large Scale Multimodal Clinical Foundation Models
<https://arxiv.org/abs/2503.07667>`_, ICML 2025) over PTB-XL, CPSC 2018 and
Georgia (:class:`~pyhealth.datasets.CardiologyDataset`) and Chapman-Shaoxing
(:class:`~pyhealth.datasets.ChapmanShaoxingDataset`), with CLIMB's seven
classes. Labels and the train / validation split of each recording come from
CLIMB's split files (https://github.com/DDVD233/CLIMB, ``src/datasets/ecg``),
read by :func:`~pyhealth.tasks.read_climb_ecg_splits`.

.. autoclass:: pyhealth.tasks.CLIMBECGClassification
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.read_climb_ecg_splits

.. autodata:: pyhealth.tasks.climb_ecg.CLIMB_ECG_CLASSES

.. autodata:: pyhealth.tasks.climb_ecg.ECG_JEPA_LEADS

.. autodata:: pyhealth.tasks.climb_ecg.ECG_LEADS
