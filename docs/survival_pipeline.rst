Survival datasets, tasks and benchmarks
=======================================

GBSG and SUPPORT tasks connect native datasets to PyHealth's existing
SurvivalMDN. The task interface uses separate duration/event tensors. The model
supplies its censored likelihood to the standard Trainer with ``mode=None``.

Dataset preparation
-------------------

GBSG needs a local ``gbsg.csv`` prepared with ``prepare_gbsg_csv`` from the
separately obtained DeepSurv HDF5 release. Install ``h5py`` only when converting
HDF5; prepared-CSV loading does not need it. SUPPORT uses the existing
``Support2Dataset`` and a local ``support2.csv`` with patient-number header ``sno``.
Download the authors' released split arrays separately from
`Survival-MDN <https://github.com/XintianHan/Survival-MDN>`_. No source data or
reference implementation is distributed with these examples.

Preprocessing and time units
----------------------------

The released GBSG protocol keeps raw covariates and uses
``months / cohort_maximum + 0.002``. This deliberately uses the full cohort's
maximum to reproduce the released convention. SUPPORT fits its feature
standardization on training patients once and uses ``days / 365.25 + 0.001``.
Neither path passes a model-side ``time_scale``. The task defaults instead keep
source months/days, allowing explicitly chosen alternative protocols.

Predictions require positive grid times in the same units as the labels. Save
the feature/time preprocessing and model configuration with every checkpoint.

Manual acceptance checks
------------------------

From a source checkout, run the download-free smoke examples::

    python -m examples.survival_benchmark.run_survival_mdn --dataset synthetic --epochs 2 --device cpu --out /tmp/mdn-smoke
    python -m examples.survival_benchmark.run_baselines --dataset synthetic --epochs 2 --out /tmp/baseline-smoke

Baseline backends are optional. Install their requirements from
``examples/survival_benchmark/requirements.txt`` to run Cox and DeepSurv.

Expected inputs are six artificial patients with seven features, separate
duration/event labels, and disjoint two-patient train/validation/test groups.
One censored patient is retained in training. Expected outputs include finite
losses/metrics, non-increasing survival curves, exact observed-time survival
probabilities for calibration, and checkpoints with reconstruction metadata.
Tiny-cohort scores only demonstrate interfaces.
Each group needs two patients for the training BatchNorm and curve comparisons,
so this three-way smoke example uses six rather than five artificial patients.

Identical completed requests verify and reuse their files. Changing settings,
inputs, executable source or runtime creates a new result directory. Incomplete
or changed results raise an error without overwriting them. The example README
documents explicit restart and restoration commands.

Real-data checks use local CSVs and released arrays::

    python -m examples.survival_pipeline --dataset gbsg --root /data/gbsg --released-dir /data/Survival-MDN/data/gbsg --out /tmp/gbsg-pilot
    python -m examples.survival_pipeline --dataset support2 --root /data/support2 --released-dir /data/Survival-MDN/data/support --out /tmp/support-pilot

Each pilot uses one full validation batch, checks finite losses/gradients and
reloads the best checkpoint. Real data is never required by CI tests.
Pass ``--device cuda`` for CUDA validation on a suitable GPU.

API reference
-------------

* :doc:`api/datasets/pyhealth.datasets.GBSGDataset`
* :doc:`api/datasets/pyhealth.datasets.Support2Dataset`
* :doc:`api/tasks/pyhealth.tasks.survival_time_to_event`
* :doc:`api/metrics/pyhealth.metrics.survival_curve`
