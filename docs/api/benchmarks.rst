Benchmarks
==========

This module runs benchmarks. You describe each run as an
:class:`~pyhealth.benchmarks.Evaluation` holding its parameters, and a
:class:`~pyhealth.benchmarks.Benchmark` runs a batch of evaluations and returns
one :class:`~pyhealth.benchmarks.Result` per evaluation.

Overview
--------

**CPBench** (experimental) evaluates the prediction set constructors from
:doc:`calib` on PyHealth datasets, tasks and models. Each evaluation names the
dataset, task, model, method, metrics and protocol. CPBench trains the model,
calibrates the method and scores its prediction sets on the test split. With a
cache directory, evaluations that differ only in method or metrics reuse one
trained model instead of retraining it. An evaluation that fails while running
is returned as a failed Result instead of stopping the batch. Supported
methods:

- :class:`~pyhealth.calib.predictionset.BaseConformal`: Standard split conformal prediction
- :class:`~pyhealth.calib.predictionset.LABEL`: Conformal prediction with bounded error
- :class:`~pyhealth.calib.predictionset.SCRIB`: Class-specific risk control
- :class:`~pyhealth.calib.predictionset.FavMac`: Value-maximizing sets with cost control
- :class:`~pyhealth.calib.predictionset.CovariateLabel`: Covariate shift adaptive conformal prediction
- :class:`~pyhealth.calib.predictionset.LabelShift`: Label shift adaptive conformal prediction
- :class:`~pyhealth.calib.predictionset.RC3P`: Rank-calibrated class-conditional conformal prediction
- :class:`~pyhealth.calib.predictionset.ClusterLabel`: K-means cluster-based conformal prediction
- :class:`~pyhealth.calib.predictionset.NeighborhoodLabel`: Neighborhood Conformal Prediction (NCP)

Getting Started
---------------

New to CPBench? Check out this complete example:

**Browse all examples online**: https://github.com/sunlabuiuc/PyHealth/tree/master/examples

- **Example**: ``examples/cpbench/synthetic.py`` - Conformal prediction methods compared on synthetic data, demonstrating:

  - Describing one evaluation per method, with the same model and split settings
  - Comparing BaseConformal, LABEL, RC3P, SCRIB, NeighborhoodLabel and FavMac
  - Reading each method's status and metrics, such as coverage and set size, from its Result

Run it from the repository root with:

.. code-block:: console

    python -m examples.cpbench.synthetic

Module Contents
---------------

.. toctree::
    :maxdepth: 3

    benchmarks/pyhealth.benchmarks.cpbench
