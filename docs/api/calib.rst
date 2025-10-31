Calibration and Uncertainty Quantification
===========================================

This module provides post-hoc calibration methods and prediction set constructors for 
uncertainty quantification in healthcare AI models. All methods can be combined with 
any trained PyHealth model.

Overview
--------

**Model Calibration Methods** adjust predicted probabilities to better reflect true 
confidence levels:

- :class:`~pyhealth.calib.calibration.TemperatureScaling`: Simple and effective logit scaling
- :class:`~pyhealth.calib.calibration.HistogramBinning`: Non-parametric binning approach
- :class:`~pyhealth.calib.calibration.DirichletCalibration`: Matrix-based recalibration
- :class:`~pyhealth.calib.calibration.KCal`: Kernel-based full calibration

**Prediction Set Constructors** provide set-valued predictions with coverage guarantees:

- :class:`~pyhealth.calib.predictionset.LABEL`: Conformal prediction with bounded error
- :class:`~pyhealth.calib.predictionset.SCRIB`: Class-specific risk control
- :class:`~pyhealth.calib.predictionset.FavMac`: Value-maximizing sets with cost control
- :class:`~pyhealth.calib.predictionset.CovariateLabel`: Covariate shift adaptive conformal

Quick Links
-----------

- :doc:`calib/usage_guide` - Practical examples and best practices
- :doc:`calib/pyhealth.calib.calibration` - Model calibration API reference
- :doc:`calib/pyhealth.calib.predictionset` - Prediction set API reference

Module Contents
---------------

.. toctree::
    :maxdepth: 3

    calib/usage_guide
    calib/pyhealth.calib.calibration
    calib/pyhealth.calib.predictionset

