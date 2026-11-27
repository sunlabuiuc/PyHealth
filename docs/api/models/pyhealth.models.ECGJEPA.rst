pyhealth.models.ECGJEPA
=======================

Overview
--------

ECG-JEPA is a ViT encoder for 12-lead ECGs pretrained with a joint-embedding
predictive objective. It reads 8 leads (I, II, V1-V6) of 10 s at 250 Hz
(2,500 samples), cuts each lead into 50 patches of 50 samples, and uses
cross-pattern attention (CroPA): a patch attends to the patches of its own lead
and to the patches of every lead at the same time. ``ECGJEPA`` puts a linear
head on the mean token and can be fine-tuned or, with ``freeze_encoder=True``,
used as a linear probe. The encoder is ported from the authors' code (MIT
licence) with its parameter names, so the published checkpoints load as they
are. CLIMB (Dai et al., ICML 2025) uses it as its ECG encoder; see
:class:`~pyhealth.tasks.CLIMBECGClassification`.

Paper: S. Kim, `ECG-JEPA: Learning General Representation of 12-Lead
Electrocardiogram With a Joint-Embedding Predictive Architecture
<https://arxiv.org/abs/2410.08559>`_, 2024. Code and checkpoints:
https://github.com/sehunfromdaegu/ECG_JEPA.

API Reference
-------------

.. autoclass:: pyhealth.models.ECGJEPA
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.ECGJEPAEncoder
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.ecg_jepa.sincos_pos_embed

.. autofunction:: pyhealth.models.ecg_jepa.cross_pattern_mask
