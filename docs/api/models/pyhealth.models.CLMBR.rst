pyhealth.models.CLMBR
=====================

CLMBR-T-base, the EHRSHOT foundation model (Wornow et al., "EHRSHOT: An EHR
Benchmark for Few-Shot Evaluation of Foundation Models", NeurIPS 2023),
reimplemented in plain PyTorch from femr 0.2.3: age-based rotary embeddings,
local causal attention (``attention_width`` 496) with a banded mask, RMSNorm,
and femr's ``dictionary.msgpack`` tokenizer. It loads the gated
``StanfordShahLab/clmbr-t-base`` weights (``model.safetensors``) without femr
or xformers. It pairs with :class:`~pyhealth.tasks.EHRFoundationModelTask`, and all
tokenization happens in ``forward()``.

Access: request access on the Hugging Face model page (CC-BY-NC-4.0, a
current CITI certificate, and the EHRSHOT Credentialed Health Data License),
then run ``huggingface-cli login``. The EHRSHOT data is on Redivis under its
data-use agreement.

.. autoclass:: pyhealth.models.CLMBR
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.CLMBRTokenizer
    :members:

.. autoclass:: pyhealth.models.CLMBRTransformer
    :members:

.. autoclass:: pyhealth.models.CLMBRConfig
    :members:

.. autofunction:: pyhealth.models.clmbr.load_clmbr_weights

.. autofunction:: pyhealth.models.clmbr.local_causal_mask

.. autofunction:: pyhealth.models.clmbr.age_rotary_embedding
