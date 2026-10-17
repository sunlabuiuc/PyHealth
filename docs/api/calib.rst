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
- :class:`~pyhealth.calib.predictionset.CovariateLabel`: Covariate shift adaptive conformal prediction with a finite-sample correction for the calibration/test weighting
- :class:`~pyhealth.calib.predictionset.ClusterLabel`: K-means cluster-based conformal prediction
- :class:`~pyhealth.calib.predictionset.NeighborhoodLabel`: Neighborhood Conformal Prediction (NCP)

Getting Started
---------------

New to calibration and uncertainty quantification? Check out this complete example:

**Browse all examples online**: https://github.com/sunlabuiuc/PyHealth/tree/master/examples

- **Example**: ``examples/covid19cxr_conformal.py`` - Comprehensive conformal prediction workflow demonstrating:

  - Training a ResNet-18 model on COVID-19 chest X-ray classification
  - Applying conventional conformal prediction with **LABEL**
  - Using covariate shift adaptive conformal prediction with **CovariateLabel**
  - Comparing coverage guarantees and efficiency between methods
  - Understanding when to use each method based on distribution shift

This example shows the complete pipeline from model training to uncertainty-aware predictions with formal coverage guarantees.

.. note::

   ``CovariateLabel``'s finite-sample correction implements Corollary 1 of
   Tibshirani, Barber, Candes, and Ramdas, "Conformal Prediction Under
   Covariate Shift" (NeurIPS 2019, https://arxiv.org/abs/1904.06019): the
   test point's reserved probability mass must be inserted as an actual
   point in the weighted empirical distribution (at the conservative
   extreme), not merely folded into the normalizing denominator -- doing
   only the latter silently under-covers relative to the target coverage
   level.

   Corollary 1 also defines the threshold *per test point*, using that
   point's own likelihood ratio w(x). Pass ``test_embeddings`` to
   ``CovariateLabel.forward()`` to get this exact per-point threshold;
   omitting it falls back to a single threshold computed from the *mean*
   calibration likelihood ratio, which is only an approximation of the
   paper's guarantee (a ``UserWarning`` is raised when this fallback is
   used).

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

