pyhealth.models.MedFlamingo
===================================

MedFlamingo: multimodal medical few-shot learner.

This reference covers the visual resampler, the gated cross-attention
building block, and the complete MedFlamingo model used in the VQA-RAD
integration branch.

**Paper:** Moor et al. "Med-Flamingo: a Multimodal Medical Few-shot Learner" ML4H 2023.

.. note::

   ``forward()`` follows the PyHealth training contract for dataset-backed
   classification-style use, while ``generate()`` provides the multimodal
   prompting path for direct medical VQA generation.

PerceiverResampler
------------------

.. autoclass:: pyhealth.models.medflamingo.PerceiverResampler
    :members:
    :undoc-members:
    :show-inheritance:

MedFlamingoLayer
----------------

.. autoclass:: pyhealth.models.medflamingo.MedFlamingoLayer
    :members:
    :undoc-members:
    :show-inheritance:

MedFlamingo
-----------

.. autoclass:: pyhealth.models.MedFlamingo
    :members:
    :undoc-members:
    :show-inheritance:
