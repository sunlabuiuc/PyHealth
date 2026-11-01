pyhealth.calib.predictionset
===================================

Prediction set constructors that provide set-valued predictions with statistical
coverage guarantees. These methods are based on conformal prediction and related
techniques for uncertainty quantification.

``BaseConformal``, ``LABEL``, ``ClusterLabel``, ``CovariateLabel``,
``LabelShift``, ``RC3P``, and ``NeighborhoodLabel`` all accept a ``score_type`` argument selecting the
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
   pyhealth.calib.predictionset.LabelShift
   pyhealth.calib.predictionset.RC3P
   pyhealth.calib.predictionset.ClusterLabel
   pyhealth.calib.predictionset.NeighborhoodLabel
   pyhealth.calib.predictionset.EnbPI

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

SCRIB's threshold search assumes the calibration set's empty-prediction
handling (``fill_max``) matches what is applied at inference time; both
``calibrate()`` and ``forward()`` resolve and use the same ``fill_max``
value, so calibration is never optimized against behavior that inference
doesn't actually apply.

.. autoclass:: pyhealth.calib.predictionset.SCRIB
   :members:
   :undoc-members:
   :show-inheritance:

FavMac (Fast Value-Maximizing Prediction Sets)
-----------------------------------------------

.. note::

   FavMac's threshold formulas are verified directly against the paper's
   Eq. 18 (expected cost control) and Eq. 21 (violation control,
   Appendix B.3 Algorithm 5 / Theorem 4.6). Like standard split-conformal
   prediction, both guarantees assume exchangeability between the
   calibration data and the point being predicted -- FavMac does not
   correct for covariate shift; see
   :class:`~pyhealth.calib.predictionset.CovariateLabel` for a method
   that does.

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

LabelShift (Label Shift Adaptive)
---------------------------------

.. autoclass:: pyhealth.calib.predictionset.LabelShift
   :members:
   :undoc-members:
   :show-inheritance:

RC3P (Rank Calibrated Class-conditional Conformal Prediction)
-------------------------------------------------------------

.. autoclass:: pyhealth.calib.predictionset.RC3P
   :members:
   :undoc-members:
   :show-inheritance:

ClusterLabel (K-means Cluster-based Conformal)
----------------------------------------------

.. autoclass:: pyhealth.calib.predictionset.ClusterLabel
   :members:
   :undoc-members:
   :show-inheritance:

NeighborhoodLabel (Neighborhood Conformal Prediction)
-----------------------------------------------------

.. note::

   ``NeighborhoodLabel.calibrate()`` excludes each calibration point from
   its own k-nearest-neighbor set during the alpha_tilde search
   (leave-one-out), matching test time, where a test point is never its
   own neighbor. Including it lets a point's own score lower its own
   threshold and causes under-coverage at test time. With leave-one-out
   the lowest-scoring calibration point is never covered, so calibration
   coverage is at most (N-1)/N for N calibration points, whatever
   ``k_neighbors`` is. If the target is unreachable, ``calibrate()`` emits
   a ``UserWarning``: above (N-1)/N it needs more calibration data or a
   larger alpha; below it, a larger ``k_neighbors`` can also help.

.. autoclass:: pyhealth.calib.predictionset.NeighborhoodLabel
   :members:
   :undoc-members:
   :show-inheritance:

EnbPI (Ensemble Batch Prediction Intervals)
--------------------------------------------

EnbPI builds prediction intervals for time series regression rather than
prediction sets. It trains its own bootstrap ensemble from a ``model_fn``
factory, so it uses no separate calibration split. Intervals are returned as
``y_lower`` and ``y_upper`` next to the point prediction ``y_prob``. In
deployment, call ``forward`` for intervals and ``update`` once labels are
revealed. ``predict_sequential`` is a convenience method for evaluation: it
runs the paper's online predict, reveal and update loop over a fully labeled
test stream. Score the results with
:func:`pyhealth.metrics.regression_metrics_fn`, passing ``y_lower`` and
``y_upper`` and asking for ``miscoverage_pi``, ``interval_width`` or
``rolling_miscoverage_pi@W``.

.. autoclass:: pyhealth.calib.predictionset.EnbPI
   :members: calibrate, forward, update, reset, predict_sequential
   :show-inheritance:

Interval Interface and Time Series Helpers
------------------------------------------

``IntervalPredictor`` fixes only the public surface of interval methods,
``calibrate``, ``forward`` and ``update``. Each time series method implements
its own online loop, using the helper functions below where they fit.

.. autoclass:: pyhealth.calib.base_classes.IntervalPredictor
   :members: calibrate, forward, update

.. autofunction:: pyhealth.calib.predictionset.timeseries.conformal_quantile

Helper Functions
----------------

.. autofunction:: pyhealth.calib.predictionset.covariate.fit_kde
.. autofunction:: pyhealth.calib.predictionset.bbse
.. autofunction:: pyhealth.calib.predictionset.em

Score Functions
---------------

Shared, pluggable nonconformity/conformity score implementations backing
the ``score_type`` argument described above.

.. autofunction:: pyhealth.calib.predictionset.scores.all_class_nc_scores
.. autofunction:: pyhealth.calib.predictionset.scores.all_class_conformity_scores
.. autofunction:: pyhealth.calib.predictionset.scores.true_class_nc_scores
.. autofunction:: pyhealth.calib.predictionset.scores.true_class_conformity_scores
