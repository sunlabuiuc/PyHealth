pyhealth.models.TRACE
=====================

TRACE: a time-aware Transformer for intra-visit clinical event nowcasting.
A visit trajectory of event codes and times, followed by a mask token, is
encoded with a timestamp embedding (decay and periodic parts) and Transformer
layers whose attention is denoised by a learnable smoothed mask; the mask
token's final state predicts the laboratory events of the next blood draw
(`Liang et al., WWW 2025 Companion <https://arxiv.org/abs/2503.23072>`_).

Pairs with :class:`~pyhealth.tasks.LabNowcastingMIMIC4` and
:class:`~pyhealth.tasks.LabNowcastingMIMIC3`. See
``examples/trace_mimic4_lab_nowcasting.ipynb`` for a run on the MIMIC-IV
demo and on full MIMIC-IV.

.. autoclass:: pyhealth.models.TRACE
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.TRACETimeEmbedding
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.TRACEAttention
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.TRACELayer
    :members:
    :undoc-members:
    :show-inheritance:
