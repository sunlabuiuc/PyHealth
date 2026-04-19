pyhealth.tasks.ecg_delineation
===============================

ECG wave delineation task for the LUDB dataset. Segments each 10-second ECG lead signal into background (0), P wave (1), QRS complex (2), and T wave (3) regions. Designed for replication of Park et al., "Benchmarking ECG Delineation using Deep Neural Network-based Semantic Segmentation Models," CHIL 2025.

.. autofunction:: pyhealth.tasks.ecg_delineation_ludb_fn
