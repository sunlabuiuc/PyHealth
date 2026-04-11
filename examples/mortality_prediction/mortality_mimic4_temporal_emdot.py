"""
EMDOT Temporal Evaluation for In-Hospital Mortality on MIMIC-IV.

Reproduces the temporal evaluation framework from:
    Zhou, H.; Chen, Y.; and Lipton, Z. C. 2023. "Evaluating Model Performance
    in Medical Datasets Over Time." CHIL 2023. PMLR 209:498–508.

This script demonstrates that standard random train-test splits overestimate
real-world model performance by ignoring temporal distribution shift. 

Two EMDOT training regimes are evaluated across simulated deployment years:
    - All-historical: train on all data up to deployment year t
    - Sliding window: train only on recent data within a window before t

This example shows:
1. Loading MIMIC-IV data with chronological admission year tagging.
2. Applying the InHospitalMortalityTemporalMIMIC4 task to create temporal samples.
3. Establishing a baseline with simple random splits for comparison.
4. Evaluating logistic regression under the EMDOT all-historical specification.
5. Evaluating logistic regression under the EMDOT sliding window specification.
6. Comparing performance across temporal splits to highlight distribution shift effects.

Usage:
    python examples/mortality_prediction/mimic4_mortality_temporal_emdot.py
"""