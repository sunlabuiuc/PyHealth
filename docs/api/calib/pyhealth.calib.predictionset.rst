pyhealth.calib.predictionset
===================================

Prediction set constructors that provide set-valued predictions with statistical 
coverage guarantees. These methods are based on conformal prediction and related 
techniques for uncertainty quantification.

Available Methods
-----------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   pyhealth.calib.predictionset.LABEL
   pyhealth.calib.predictionset.SCRIB
   pyhealth.calib.predictionset.FavMac
   pyhealth.calib.predictionset.CovariateLabel
   pyhealth.calib.predictionset.ClusterLabel
   pyhealth.calib.predictionset.NeighborhoodLabel

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
