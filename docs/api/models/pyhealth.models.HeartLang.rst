pyhealth.models.HeartLang
===================================

HeartLang: an ECG language model that reads an ECG as a sentence of
heartbeat "words", pre-trained on MIMIC-IV-ECG
(`Jin et al., ICLR 2025 <https://openreview.net/forum?id=6Hz1Ko087B>`_).
``pretrained=True`` loads the authors' checkpoint from
`Hugging Face <https://huggingface.co/PKUDigitalHealth/HeartLang>`_; use
:class:`~pyhealth.tasks.PTBXLHeartLangClassification` to build its inputs.

.. autoclass:: pyhealth.models.HeartLang
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.HeartLangBackbone
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.ecg_to_heartbeat_sentence

.. autofunction:: pyhealth.models.heartlang.heartlang_state_dict

.. autofunction:: pyhealth.models.heartlang.load_heartlang_checkpoint
