pyhealth.models.MAS
===================

MAS fits a linear time-varying Cox model to counting-process survival data.
Each row contains standardized features from the start of an interval and a
``survival_interval`` label with start time, stop time, and event indicator.
Use one batch per split to preserve the complete Cox risk sets. The model
uses Breslow ties and returns log-relative hazards in ``y_prob``, where higher
scores indicate higher risk.

The Mayo PBC example fits the four-biomarker adaptation using
:class:`~pyhealth.datasets.PBCSeqDataset` and
:class:`~pyhealth.tasks.LongitudinalLiverSurvival`. It reports enrollment
concordance on held-out patients.

.. code-block:: console

   python examples/pbcseq_liver_survival_mas.py --root /path/to/pbcseq-directory

.. autoclass:: pyhealth.models.MAS
   :members: forward
