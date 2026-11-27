pyhealth.tasks.MedModInHospitalMortality
========================================

In-hospital mortality from the first 48 hours of an ICU stay and the latest
chest X-ray in that window, reproducing MedMod (Elsharief et al., 2025) and its
released code: one sample per ICU stay with ``ehr`` (48 x 76), ``image_path``
and ``label``. Use it with :class:`~pyhealth.datasets.MedModMIMIC4Dataset`.

.. autoclass:: pyhealth.tasks.MedModInHospitalMortality
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.processors.medmod_ehr_processor.MedModEHRProcessor
    :members:
    :undoc-members:
    :show-inheritance:

Per-model settings
------------------

MedMod's models read their data through two code paths, and the task
reproduces each. Both use the same cohort, labels and split; only the EHR rows
at Hours == 0 and the image transform differ.

.. list-table::
   :header-rows: 1

   * - Model
     - ``hours_zero``
     - Image transform
     - Why
   * - Early Fusion
     - ``"keep"``
     - ImageNet-normalized
     - MedMod's supervised ``fusion_main.py:13-16`` imports ``datasets.*`` and
       ``ehr_utils.*``, which exist only in MedFuse; MedFuse's
       ``datasets/ehr_dataset.py`` keeps every row and
       ``datasets/cxr_dataset.py`` normalizes.
   * - ConVIRT (pretraining, linear evaluation)
     - ``"drop"``
     - pixels in [0, 1]
     - MedMod's ``src/data/datasets.py:137-163`` keeps 0 < t only; the ConVIRT
       scripts pass ``--transforms_cxr simclrv2``.

Paper vs code
-------------

Where MedMod's paper and released code disagree, the task follows the code,
which produced the reported numbers. The bin width and the normalizer remain
settings of the task (``timestep``, ``normalizer``).

.. list-table::
   :header-rows: 1

   * - Rule
     - MedMod paper
     - Released code
     - Default here
   * - Bin width
     - §3.2.2 "every two hours" (MedFuse §4.1.1 too)
     - 1 h (``--timestep`` default 1.0, ``src/parser.py:30``)
     - 1 h (``timestep``)
   * - Normalizer
     - §3.2.2 "task-specific", computed on the dataset
     - the phenotyping file ``ph_ts1.0`` (an ``is`` comparison in ``ehr_funcs``)
     - ``ph_ts1.0``; ``normalizer=None`` fits one on the train split
   * - Split
     - §3.2.4 random 70/10/20 by patient
     - fixed subject lists
     - the fixed lists, which reproduce Table 2

Two checks on the real data (image-paired test set, 1,241 stays):

- MedFuse's released EHR checkpoint (``checkpoints/mortality/uni_ehr_all``),
  evaluated frozen: test AUROC 0.833 (95% CI 0.802-0.860) with 1 h +
  ``ph_ts1.0``, against 0.735 with 1 h + a normalizer fitted on the train split
  and 0.719 with 2 h + fitted; MedFuse reports 0.825 (0.793-0.852). So a model
  that uses MedFuse's encoders frozen needs the code's settings. MedFuse split
  with the same subject lists, so its training data holds none of these test
  patients.
- MedMod's EHR LSTM trained from scratch (5 seeds): 1 h vs 2 h with the same
  normalizer makes no difference (Δ AUROC +0.000 ± 0.006); ``ph_ts1.0`` scores
  0.009 higher than a fitted normalizer (0.816 vs 0.807; paper Table 3: 0.829).
