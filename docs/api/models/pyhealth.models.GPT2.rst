pyhealth.models.GPT2
===================================

A decoder-only GPT-2 baseline for unconditional synthetic EHR generation,
wrapped as a PyHealth :class:`~pyhealth.models.BaseModel`. Patient visit-code
sequences are serialized into causal-LM token streams and modeled
autoregressively.

Reference:
    Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019).
    *Language Models are Unsupervised Multitask Learners.* OpenAI.
    https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

.. autoclass:: pyhealth.models.GPT2
    :members:
    :undoc-members:
    :show-inheritance:
