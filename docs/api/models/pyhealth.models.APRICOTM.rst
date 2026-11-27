pyhealth.models.APRICOTM
========================

Overview
--------

APRICOT-M (Contreras et al., `Real-time prediction of intensive care unit
patient acuity and therapy requirements using state-space modelling
<https://doi.org/10.1038/s41467-025-62121-1>`_, Nature Communications 16,
7315, 2025) predicts, every 4 hours of an ICU stay, the patient's acuity in
the next 4 hours (discharge, stable, unstable or deceased), transitions
between stable and unstable, and the onset or stop of mechanical
ventilation, vasopressors and continuous renal replacement therapy.

The model reads the clinical events of the last 4 hours as (time, variable,
value) triplets, so sparse, irregular measurements need no resampling or
imputation. Time and value pass through position-wise layers, the variable
through an embedding; a stack of Mamba (selective state-space) blocks
contextualizes the events, top-k pooling over time and an MLP summarize
them, a static-feature embedding is added, and one head per label gives its
probability.

This is a clean reimplementation from the paper. The authors' code
(https://github.com/iheallab/apricotM) is GPL-3.0, so none of it was
copied; it was only read to confirm details. The Mamba blocks are
:class:`~pyhealth.models.SelectiveSSM`, the pure PyTorch block shared with
:class:`~pyhealth.models.ARMR`, run with its chunked parallel selective scan,
so the CUDA ``mamba_ssm`` package is not needed and the model runs on CPU.

Inputs and outputs
------------------

The input is a ``sequence_v2`` series with one token and one value per
event, such as the ``events`` field of
:class:`~pyhealth.tasks.ICUAcuityMIMIC4`, plus an optional static tensor
field. The label is a tensor of ``K`` binary labels (12 for that task, in the
order of ``ICUAcuityMIMIC4.LABELS``). ``mode`` is ``"multilabel"``; the model
returns ``logit``, ``y_prob`` and, with labels, ``loss`` (binary
cross-entropy) and ``y_true``. Pass ``pos_weight="balanced"`` with the
training split for the class weighting of the paper. The example notebook
``examples/apricotm_mimic4_acuity.ipynb`` trains it on MIMIC-IV.

API
---

.. autoclass:: pyhealth.models.APRICOTM
   :members:
   :show-inheritance:

.. autoclass:: pyhealth.models.SelectiveSSM
   :members:
   :show-inheritance:

.. autofunction:: pyhealth.models.selective_ssm.selective_scan
