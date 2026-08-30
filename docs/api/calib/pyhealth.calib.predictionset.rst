pyhealth.calib.predictionset
===================================

Prediction set constructors that provide set-valued predictions with statistical
coverage guarantees. These methods are based on conformal prediction and related
techniques for uncertainty quantification.

``BaseConformal``, ``LABEL``, ``ClusterLabel``, ``CovariateLabel``, and
``NeighborhoodLabel`` all accept a ``score_type`` argument selecting the
nonconformity/conformity score used for calibration and set construction:
either ``"threshold"`` (the default, unchanged from prior releases) or
``"aps"`` (Adaptive Prediction Sets, Romano, Sesia, and Candes 2020), which
adapts the prediction set size to the model's per-input confidence. See
:mod:`pyhealth.calib.predictionset.scores` for the exact score formulas.
``SCRIB`` and ``FavMac`` are not included since their calibration
procedures aren't a score-then-quantile pattern.

Available Methods
-----------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   pyhealth.calib.predictionset.BaseConformal
   pyhealth.calib.predictionset.LABEL
   pyhealth.calib.predictionset.SCRIB
   pyhealth.calib.predictionset.FavMac
   pyhealth.calib.predictionset.CovariateLabel
   pyhealth.calib.predictionset.ClusterLabel
   pyhealth.calib.predictionset.NeighborhoodLabel

BaseConformal (Standard Split Conformal Prediction)
----------------------------------------------------

.. autoclass:: pyhealth.calib.predictionset.BaseConformal
   :members:
   :undoc-members:
   :show-inheritance:

LABEL (Least Ambiguous Set-valued Classifier)
----------------------------------------------

.. autoclass:: pyhealth.calib.predictionset.LABEL
   :members:
   :undoc-members:
   :show-inheritance:

SCRIB (Set-classifier with Class-specific Risk Bounds)
-------------------------------------------------------

.. autoclass:: pyhealth.calib.predictionset.SCRIB
   :members:
   :undoc-members:
   :show-inheritance:

FavMac (Fast Value-Maximizing Prediction Sets)
-----------------------------------------------

.. autoclass:: pyhealth.calib.predictionset.FavMac
   :members:
   :undoc-members:
   :show-inheritance:

CovariateLabel (Covariate Shift Adaptive)
------------------------------------------

.. autoclass:: pyhealth.calib.predictionset.CovariateLabel
   :members:
   :undoc-members:
   :show-inheritance:

ClusterLabel (K-means Cluster-based Conformal)
----------------------------------------------

.. note::

   ClusterLabel is an instance of Mondrian conformal prediction (Vovk,
   Lindsay, Nouretdinov, and Gammerman, "Mondrian confidence machine,"
   Technical report, Royal Holloway University of London, 2003) using
   K-means clusters as the category function -- not itself a specific
   published method, but a pyhealth-original combination of a standard
   technique with the Mondrian framework. Coverage holds independently
   *within each cluster* (a strictly stronger guarantee than plain
   marginal coverage). Fitting K-means on train+cal embeddings combined
   (rather than train only, with calibration points assigned via
   ``.predict()``) was checked via Monte Carlo simulation and found not
   to introduce measurable coverage bias -- see the class docstring's
   ``Note`` for details.

.. autoclass:: pyhealth.calib.predictionset.ClusterLabel
   :members:
   :undoc-members:
   :show-inheritance:

NeighborhoodLabel (Neighborhood Conformal Prediction)
-----------------------------------------------------

.. autoclass:: pyhealth.calib.predictionset.NeighborhoodLabel
   :members:
   :undoc-members:
   :show-inheritance:

Helper Functions
----------------

.. autofunction:: pyhealth.calib.predictionset.covariate.fit_kde

Score Functions
---------------

Shared, pluggable nonconformity/conformity score implementations backing
the ``score_type`` argument described above.

.. autofunction:: pyhealth.calib.predictionset.scores.all_class_nc_scores
.. autofunction:: pyhealth.calib.predictionset.scores.all_class_conformity_scores
.. autofunction:: pyhealth.calib.predictionset.scores.true_class_nc_scores
.. autofunction:: pyhealth.calib.predictionset.scores.true_class_conformity_scores
