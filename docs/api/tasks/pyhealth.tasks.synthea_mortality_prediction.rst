pyhealth.tasks.synthea_mortality_prediction
===========================================

Overview
--------

Mortality prediction after inpatient discharge for the Synthea synthetic
EHR dataset. Predicts whether a patient will die within a configurable
prediction window after their latest inpatient encounter discharge.

This task replicates the Mortality-Disch experiment from the CEHR-GAN-BERT
paper (Poulain et al., MLHC 2022, Section A.2).

API Reference
-------------

.. autoclass:: pyhealth.tasks.MortalityPredictionSynthea
    :members:
    :undoc-members:
    :show-inheritance:
