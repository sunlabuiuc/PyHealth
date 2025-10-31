pyhealth.calib.calibration
===================================

Model calibration methods for adjusting predicted probabilities to better reflect 
true confidence levels. These methods are essential for reliable uncertainty 
quantification in healthcare AI applications.

Available Methods
-----------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   pyhealth.calib.calibration.TemperatureScaling
   pyhealth.calib.calibration.HistogramBinning
   pyhealth.calib.calibration.DirichletCalibration
   pyhealth.calib.calibration.KCal

Temperature Scaling
-------------------

.. autoclass:: pyhealth.calib.calibration.TemperatureScaling
   :members:
   :undoc-members:
   :show-inheritance:

Histogram Binning
-----------------

.. autoclass:: pyhealth.calib.calibration.HistogramBinning
   :members:
   :undoc-members:
   :show-inheritance:

Dirichlet Calibration
---------------------

.. autoclass:: pyhealth.calib.calibration.DirichletCalibration
   :members:
   :undoc-members:
   :show-inheritance:

KCal (Kernel-Based Calibration)
--------------------------------

.. autoclass:: pyhealth.calib.calibration.KCal
   :members:
   :undoc-members:
   :show-inheritance:
