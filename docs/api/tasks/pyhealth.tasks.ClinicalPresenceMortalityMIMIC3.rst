pyhealth.tasks.ClinicalPresenceMortalityMIMIC3
==============================================

Clinical presence is the interaction between patients and the healthcare
system that generates observational data. Which laboratory tests are ordered,
and when, reflects a patient's health, but also their access to care and their
clinicians' practice, so missingness can differ between groups of patients.
`Jeanselme et al. (ML4H 2022)
<https://proceedings.mlr.press/v193/jeanselme22a.html>`_, "Imputation
Strategies Under Clinical Presence: Impact on Algorithmic Fairness", show that
imputation strategies that perform alike on the whole population can still
treat marginalised groups differently, and that no strategy consistently
reduces the gaps. This task reproduces their MIMIC-III experiment: death within
8 days of admission, predicted from the labs of the first 24 hours, imputed at
the cohort level. Black, female and publicly insured patients are then compared
at a fixed care capacity with
:func:`~pyhealth.metrics.fairness_utils.capacity_fairness_metrics`. The task
follows the authors' `reference code
<https://github.com/Jeanselme/ClinicalPresenceFairness>`_.

Cohort
------

Load the dataset with ``MIMIC3Dataset(root, tables=["labevents"])``; the
``patients`` and ``admissions`` tables are added automatically.

- **Index admission.** Each adult's last hospital admission. Age is the days
  from birth to admission divided by 365 and must exceed 18; patients over 89,
  whose birth dates MIMIC-III shifts by about 300 years, stay in.
- **Stay.** The index admission must last at least 24 hours. A patient whose
  last admission is shorter is dropped, even if an earlier one qualifies.
- **Features.** The latest value of each of 67 lab tests charted strictly
  between admission and 24 hours after it. The 67 tests pool 87 ``LABEVENTS``
  itemids, as the reference selects them from MIMIC-Extract's variable map
  (Wang et al., 2020). If one lab is charted more than once at the same time,
  all those values are dropped. A patient needs at least one lab value.
- **Label.** ``mortality`` is 1 for in-hospital death within 8 days of
  admission.
- **Groups.** ``black`` (ethnicity contains "BLACK"), ``female``, and
  ``public_insurance`` (any insurance other than Private, so Self Pay counts as
  public). 1 marks the marginalised group. The flags travel with each sample
  for evaluation and are never features.

Imputation needs the whole cohort, so the task's ``pre_filter`` runs it once,
before the samples are written. It assigns a seeded patient split, 80/10/10 by
default, fits the imputer and the scaler on the training patients only, and
transforms the validation and test patients separately, so the training
features never depend on them.

On MIMIC-III v1.4 the cohort has 36,313 patients, 6.75% of them positive, with
40.6 of the 67 labs missing on average. The reference code gives 36,296; the
differences are listed below.

.. note::

   The duplicate rule comes from the reference, and it leaves
   ``Prothrombin time`` almost empty. MIMIC-Extract maps both INR and PT to
   that lab, and the two are charted together, so nearly every value is a
   duplicate. On MIMIC-III v1.4 only 44 patients keep a value, so models
   effectively see 66 labs.

Strategies
----------

``imputation`` and ``add_missing_indicators`` select the strategy.

.. list-table::
   :header-rows: 1
   :widths: 40 26 18 16

   * - Strategy
     - ``imputation``
     - ``add_missing_indicators``
     - Default ``n_draws``
   * - Median
     - ``"median"``
     - ``False``
     - 1
   * - MICE
     - ``"mice"``
     - ``False``
     - 10
   * - Group MICE
     - ``"group_mice"``
     - ``False``
     - 10
   * - Group MICE Missing
     - ``"group_mice"``
     - ``True``
     - 10
   * - Median Missing, MICE Missing (paper appendix)
     - ``"median"``, ``"mice"``
     - ``True``
     - 1, 10
   * - MissForest, Group MissForest, with or without Missing (not in the paper)
     - ``"missforest"``, ``"group_missforest"``
     - either
     - 1
   * - Corrected MICE, Corrected Group MICE (sensitivity analysis, not in the
       paper)
     - ``"corrected_mice"``, ``"group_corrected_mice"``
     - ``False``
     - 10

- **MICE** follows the reference. It starts from the training medians, then
  for ``max_iter`` sweeps (default 10) regresses each lab on all the others by
  least squares, least-missing lab first, and draws each missing value from
  N(prediction, sd of the absolute residuals). Each draw is a separate sample.
  As in the paper, train one model per draw and average the predicted
  probabilities.
- **Group** strategies add the three flags as predictors of the imputation
  only.
- **Missing** strategies append one indicator per lab, 1 where the lab is
  missing, after imputation, so the imputers never see them.
- **MissForest-style** strategies chain scikit-learn's ``ExtraTreesRegressor``
  in an ``IterativeImputer``: 50 trees, the square root of the number of
  features per split, at least 5 samples per leaf, and one draw;
  ``forest_params`` overrides these settings. They stop on scikit-learn's
  ``tol``, not on Stekhoven and Bühlmann's rule, and on MIMIC-III they run all
  ``max_iter`` sweeps.
- **Corrected MICE** exists because linear MICE is ill-posed for rare labs. On
  MIMIC-III, six labs have fewer observed training patients than the 66
  predictors, and about 0.05% of imputed values land far outside the training
  range, up to hundreds of standard deviations from the mean. Those labs also
  fit their few rows exactly, which makes the noise of most labs collapse to
  almost zero after the first sweep. The corrected strategies fit a lab by its
  training mean when it has fewer than 10 observed values per predictor, draw
  noise from N(0, sd of the residuals), and keep imputed values within the
  training range.

Samples
-------

Each cohort patient gives ``n_draws`` samples, which differ only in ``labs``,
``draw`` and ``n_extreme_imputations``.

.. list-table::
   :header-rows: 1
   :widths: 24 26 50

   * - Key
     - Type
     - Content
   * - ``patient_id``
     - ``str``
     - MIMIC-III ``SUBJECT_ID``
   * - ``hadm_id``
     - ``str``
     - ``HADM_ID`` of the index admission
   * - ``labs``
     - float32 tensor of shape (67,), or (134,) with indicators
     - The imputed labs in ``feature_names()`` order, followed by the
       indicators if ``add_missing_indicators``. Every column is standardised
       on the training patients unless ``standardize=False``.
   * - ``mortality``
     - float32 tensor of shape (1,)
     - The label, 0 or 1
   * - ``split``
     - ``str``
     - ``"train"``, ``"val"`` or ``"test"``
   * - ``draw``
     - ``int``
     - The imputation draw, from 0 to ``n_draws - 1``
   * - ``black``, ``female``, ``public_insurance``
     - ``int``
     - 1 for the marginalised group
   * - ``n_lab_orders``
     - ``int``
     - Distinct times at which the patient has a lab value in the window
   * - ``n_observed_labs``
     - ``int``
     - Labs with a value in the window, out of 67
   * - ``n_extreme_imputations``
     - ``int``
     - This draw's imputed labs beyond the training range ± 3 sd

.. warning::

   Select samples by their ``split`` field, and by ``draw``, and never re-split
   them, for example with ``split_by_patient``. The imputer and the scaler
   were fit on the training patients, so a new split would let them see
   evaluation patients. The split depends only on the cohort, ``seed`` and
   ``split_ratios``, so all strategies with the same ``seed`` and ratios share
   their test patients.

Differences from the Reference Implementation
---------------------------------------------

The task follows the reference code, which produced the paper's numbers, also
where the paper's text differs from it: the stay is anchored at the hospital
admission, not at the ICU admission as the text says. It departs from the code
(commit ``fcb580b``) in these places:

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * - Topic
     - Reference code
     - This task
   * - Index admission
     - The patient's last row in ``ADMISSIONS.csv``, which is not always the
       latest admission
     - The latest admission time. On MIMIC-III v1.4, 88 patients (0.24%) get a
       different admission.
   * - Label
     - Death time minus admission time under 8 days
     - ``hospital_expire_flag`` is 1 and the discharge time is under 8 days
       after admission, because PyHealth's MIMIC-III config does not load
       ``DEATHTIME``. 5 labels differ.
   * - Labs of the admission
     - Matched by ``HADM_ID`` and the window
     - Matched by the window only, because the config does not load
       ``HADM_ID`` for ``LABEVENTS``. 37 patients move into or out of the
       cohort.
   * - Split
     - The imputer is fit on 80% of patients; the classifier is trained on
       64.8%, tuned on 7.2% and evaluated on the other 28%
     - 80/10/10. The imputer and the scaler are fit on the training 80%, which
       is also the classifier's training set.
   * - Missingness indicators
     - Appended before MICE, so they are also imputation predictors
     - Appended after imputation, as the paper's text says and as the authors'
       later commit ``3fc2b65`` does
   * - Group MICE
     - Drops only the last of the three flag columns, so the Black flag and a
       constant column reach the classifier
     - Drops all three flags
   * - Sex flag
     - Compares ``ETHNICITY`` with ``"M"``, so it is always False (fixed in
       ``3fc2b65``)
     - ``gender == "F"``
   * - MICE noise
     - Drawn from NumPy's unseeded global random state
     - The same noise model, seeded per draw
   * - MICE regressions
     - scikit-learn's ``LinearRegression``; labs are visited in order of
       missingness over all patients
     - The normal equations, which agree wherever a regression is well-posed;
       labs are visited in order of missingness over the training patients,
       which moves 2 of 67 positions

On MIMIC-III v1.4 these give 36,313 patients against the reference's 36,296.
For the 36,192 patients that both cohorts share with the same index admission,
every lab value, lab-order count and group flag matches.

Task Class
----------

.. autoclass:: pyhealth.tasks.clinical_presence_mortality_mimic3.ClinicalPresenceMortalityMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:
