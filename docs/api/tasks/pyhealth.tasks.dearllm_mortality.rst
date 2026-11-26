pyhealth.tasks.dearllm_mortality
================================

Mortality prediction for :class:`~pyhealth.models.DearLLM` (Xu et al., AAAI
2025) on MIMIC-III and MIMIC-IV. Following the paper, each patient with at
least two admissions gives one sample: the earlier admissions are the input
and the outcome of the last admission is the label. The sample also carries
the patient's feature prior graph (nodes, CF-IPF node weights and
LLM-deduced edges). :func:`compute_ipf` and
:func:`normalize_perplexity_graphs` prepare the task's inputs from the
training patients.

.. autoclass:: pyhealth.tasks.DearLLMMortalityPredictionMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.DearLLMMortalityPredictionMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.dearllm_mortality.compute_ipf

.. autofunction:: pyhealth.tasks.dearllm_mortality.compute_cf_ipf

.. autofunction:: pyhealth.tasks.dearllm_mortality.normalize_perplexity_graphs
