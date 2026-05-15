pyhealth.models.RespFusionNet
=============================

A small multimodal fusion classifier that combines fixed-size respiratory
audio features with patient-metadata features via tiny MLP encoders and
concatenation fusion. Designed for the ICBHI respiratory-abnormality
task and intended as a minimal, transparent baseline for modality
ablation (audio only / metadata only / audio + metadata), inspired by
the RespLLM hypothesis that combining audio and context improves
respiratory-health prediction over unimodal inputs.

RespFusionNet is explicitly **not** a reproduction of RespLLM — it does
not include OpenBioLLM, OPERA, LoRA, or PEFT. It is intended as a clean,
readable companion model to the
:class:`~pyhealth.tasks.RespiratoryAbnormalityPredictionICBHI` task.
Enable ``include_metadata_features=True`` on the task to surface the
fixed-size ``metadata`` tensor that this model consumes alongside
``signal``.

.. autoclass:: pyhealth.models.RespFusionNet
    :members:
    :undoc-members:
    :show-inheritance:
