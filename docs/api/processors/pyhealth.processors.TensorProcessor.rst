pyhealth.processors.TensorProcessor
===================================

Processor for tensor data.

``size()`` returns the width of the feature's last dimension seen in ``fit()``
(1 for scalars), or ``None`` before fitting.

Custom tensor processors
------------------------

Subclass ``TensorProcessor`` to transform numeric features, for example to select
columns, impute or scale. Models such as ``MLP`` size their input layer from the
already-processed samples, falling back to ``size()``, and never call
``process()`` on a sample a second time, so ``process()`` does not need to be
idempotent. If ``process()`` changes the width, override ``size()`` to return the
output width. See ``examples/custom_tensor_processor.py``.

In PyHealth 2.0.2 and earlier, ``EmbeddingModel`` called ``process()`` again on
already-processed samples to infer the width, which broke processors that are not
idempotent.

.. autoclass:: pyhealth.processors.TensorProcessor
    :members:
    :undoc-members:
    :show-inheritance:
