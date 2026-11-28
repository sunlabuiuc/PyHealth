pyhealth.models.SequenceTransformerNetwork
==========================================

Sequence Transformer Networks for clinical time series
(`Oh, Wang and Wiens, MLHC 2018 <https://proceedings.mlr.press/v85/oh18a.html>`_):
Per-sample time warp and value shift/scale, followed by a 1D CNN.
See ``examples/mimic3_mortality_sequence_transformer.py`` for the demo experiment.
It uses YAIB ICU mortality with hourly bins 0 through 24, rather than the
paper's 48-hour in-hospital mortality task. Only the dynamic features are used.
The demo results test the implementation and do not reproduce the paper's
full-data benchmark.

The localizer uses the 1D convolutions shown in the paper; the authors' released
code uses 2D convolutions. This implementation also uses a continuous leaky clamp
and dropout after the classifier's hidden layer. These choices differ from the
released code and are described in the model docstring.

.. autoclass:: pyhealth.models.SequenceTransformerNetwork
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.SequenceTransformerLayer
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.temporal_sample
