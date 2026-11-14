pyhealth.tasks.KnowRareICUMortalityMIMIC4
=========================================

ICU mortality from the first 24 hours of the ICU stay on MIMIC-IV, with each
stay's primary condition, for KnowRare (Zhu et al., npj Digital Medicine
2025). It is the MIMIC-IV counterpart of the paper's eICU ICU-mortality task
and reuses the cohort and features of
:class:`~pyhealth.tasks.YAIBMortalityMIMIC4`. Pairs with
:class:`~pyhealth.models.KnowRare`; :func:`select_rare_conditions` applies the
paper's rule for choosing the rare conditions to evaluate.

.. autoclass:: pyhealth.tasks.KnowRareICUMortalityMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.knowrare_mimic4.select_rare_conditions

.. autofunction:: pyhealth.tasks.knowrare_mimic4.icd_category
